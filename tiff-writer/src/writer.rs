//! Main TiffWriter: orchestrates multi-IFD streaming writes.

use std::io::{Seek, SeekFrom, Write};

use tiff_core::{ByteOrder, Tag};

use crate::builder::ImageBuilder;
use crate::compress;
use crate::encoder;
use crate::error::{Error, Result};
use crate::sample::TiffWriteSample;

const CLASSIC_TIFF_LIMIT: u64 = u32::MAX as u64;

fn checked_len_u64(len: usize, context: &str) -> Result<u64> {
    u64::try_from(len).map_err(|_| Error::Other(format!("{context} length exceeds u64::MAX")))
}

fn checked_add_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::Other(format!("{context} overflow")))
}

fn classic_offset_u32(offset: u64) -> Result<u32> {
    u32::try_from(offset).map_err(|_| Error::ClassicOffsetOverflow { offset })
}

fn classic_byte_count_u32(byte_count: u64) -> Result<u32> {
    u32::try_from(byte_count).map_err(|_| Error::ClassicByteCountOverflow { byte_count })
}

/// TIFF format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffVariant {
    Classic,
    BigTiff,
    /// Exact auto-selection.
    ///
    /// The writer records block data first and chooses Classic TIFF vs
    /// BigTIFF from the finalized file layout during `finish()`.
    Auto,
}

/// Configuration for the TIFF writer.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    pub byte_order: ByteOrder,
    pub variant: TiffVariant,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            byte_order: ByteOrder::LittleEndian,
            variant: TiffVariant::Auto,
        }
    }
}

impl WriteOptions {
    /// Construct exact auto-selection options.
    ///
    /// The `estimated_bytes` parameter is retained for source compatibility,
    /// but the writer now decides from the exact finalized layout instead of
    /// an upfront size heuristic.
    pub fn auto(_estimated_bytes: u64) -> Self {
        Self {
            byte_order: ByteOrder::LittleEndian,
            variant: TiffVariant::Auto,
        }
    }
}

/// Handle identifying a specific image within the writer.
#[derive(Debug, Clone)]
pub struct ImageHandle {
    pub(crate) index: usize,
}

/// Write state for one IFD.
struct IfdState {
    builder: ImageBuilder,
    block_records: Vec<Option<(u64, u64)>>,
}

/// A streaming TIFF/BigTIFF file writer.
pub struct TiffWriter<W: Write + Seek> {
    sink: W,
    byte_order: ByteOrder,
    requested_variant: TiffVariant,
    header_offset: u64,
    images: Vec<IfdState>,
    finalized: bool,
}

impl<W: Write + Seek> TiffWriter<W> {
    /// Create a new TIFF writer.
    ///
    /// Image data is streamed immediately. The final IFD chain and the header
    /// are emitted during `finish()`, which allows `TiffVariant::Auto` to
    /// choose Classic TIFF vs BigTIFF from the exact completed layout.
    pub fn new(mut sink: W, options: WriteOptions) -> Result<Self> {
        let header_offset = sink.stream_position()?;
        let reserved_header_len = match options.variant {
            TiffVariant::Classic => encoder::header_len(false),
            TiffVariant::BigTiff | TiffVariant::Auto => encoder::header_len(true),
        };
        sink.write_all(&[0; encoder::BIGTIFF_HEADER_LEN as usize][..reserved_header_len as usize])?;

        Ok(Self {
            sink,
            byte_order: options.byte_order,
            requested_variant: options.variant,
            header_offset,
            images: Vec::new(),
            finalized: false,
        })
    }

    /// Add an image (IFD) to the file.
    pub fn add_image(&mut self, builder: ImageBuilder) -> Result<ImageHandle> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        builder.validate()?;

        let index = self.images.len();
        self.images.push(IfdState {
            block_records: vec![None; builder.block_count()],
            builder,
        });

        Ok(ImageHandle { index })
    }

    /// Write raw bytes between the reserved header area and the image data.
    ///
    /// When `TiffVariant::Auto` is used, the writer reserves the 16-byte
    /// BigTIFF header footprint up front so the finalized header can switch
    /// variants without relocating block data.
    pub fn write_header_prefix(&mut self, bytes: &[u8]) -> Result<()> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        if !self.images.is_empty() {
            return Err(Error::Other(
                "header prefix bytes must be written before adding images".into(),
            ));
        }

        self.sink.seek(SeekFrom::End(0))?;
        let prefix_end = checked_add_u64(
            self.sink.stream_position()?,
            checked_len_u64(bytes.len(), "header prefix")?,
            "header prefix size",
        )?;
        if matches!(self.requested_variant, TiffVariant::Classic) {
            classic_offset_u32(prefix_end)?;
        }
        self.sink.write_all(bytes)?;
        Ok(())
    }

    /// Write a single strip or tile for the given image.
    pub fn write_block<T: TiffWriteSample>(
        &mut self,
        handle: &ImageHandle,
        block_index: usize,
        samples: &[T],
    ) -> Result<()> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        let state = self
            .images
            .get(handle.index)
            .ok_or(Error::Other("invalid image handle".into()))?;

        let total_blocks = state.builder.block_count();
        if block_index >= total_blocks {
            return Err(Error::BlockIndexOutOfRange {
                index: block_index,
                total: total_blocks,
            });
        }

        let expected = state.builder.block_sample_count(block_index);
        if samples.len() != expected {
            return Err(Error::BlockSizeMismatch {
                index: block_index,
                expected,
                actual: samples.len(),
            });
        }

        let compressed = if matches!(state.builder.compression, tiff_core::Compression::Lerc) {
            let opts = state.builder.lerc_options.unwrap_or_default();
            let block_width = state.builder.block_row_width() as u32;
            let block_height = state.builder.block_height(block_index);
            let depth = state.builder.block_samples_per_pixel() as u32;
            compress::compress_block_lerc(
                samples,
                block_width,
                block_height,
                depth,
                &opts,
                block_index,
            )?
        } else {
            compress::compress_block(
                samples,
                self.byte_order,
                state.builder.compression,
                state.builder.predictor,
                state.builder.block_samples_per_pixel(),
                state.builder.block_row_width(),
                block_index,
            )?
        };

        self.write_block_raw(handle, block_index, &compressed)
    }

    /// Write a pre-compressed block (bypass the compression pipeline).
    pub fn write_block_raw(
        &mut self,
        handle: &ImageHandle,
        block_index: usize,
        compressed_bytes: &[u8],
    ) -> Result<()> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }

        let state = self
            .images
            .get(handle.index)
            .ok_or(Error::Other("invalid image handle".into()))?;
        let total = state.builder.block_count();
        if block_index >= total {
            return Err(Error::BlockIndexOutOfRange {
                index: block_index,
                total,
            });
        }

        let offset = self.sink.seek(SeekFrom::End(0))?;
        let byte_count = checked_len_u64(compressed_bytes.len(), "block payload")?;
        if matches!(self.requested_variant, TiffVariant::Classic) {
            classic_offset_u32(offset)?;
            classic_byte_count_u32(byte_count)?;
        }

        self.sink.write_all(compressed_bytes)?;

        let state = self
            .images
            .get_mut(handle.index)
            .ok_or(Error::Other("invalid image handle".into()))?;
        state.block_records[block_index] = Some((offset, byte_count));
        Ok(())
    }

    /// Write a block whose on-disk bytes include a prefix and/or suffix that
    /// must not be reflected in the TIFF block offset/byte-count arrays.
    pub fn write_block_raw_segmented(
        &mut self,
        handle: &ImageHandle,
        block_index: usize,
        prefix: &[u8],
        payload: &[u8],
        suffix: &[u8],
    ) -> Result<()> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }

        let state = self
            .images
            .get(handle.index)
            .ok_or(Error::Other("invalid image handle".into()))?;
        let total = state.builder.block_count();
        if block_index >= total {
            return Err(Error::BlockIndexOutOfRange {
                index: block_index,
                total,
            });
        }

        let start = self.sink.seek(SeekFrom::End(0))?;
        let prefix_len = checked_len_u64(prefix.len(), "block prefix")?;
        let byte_count = checked_len_u64(payload.len(), "block payload")?;
        let suffix_len = checked_len_u64(suffix.len(), "block suffix")?;
        let offset = checked_add_u64(start, prefix_len, "block offset")?;
        let end = checked_add_u64(
            checked_add_u64(offset, byte_count, "segmented block size")?,
            suffix_len,
            "segmented block size",
        )?;
        if matches!(self.requested_variant, TiffVariant::Classic) {
            classic_offset_u32(offset)?;
            classic_byte_count_u32(byte_count)?;
            classic_offset_u32(end)?;
        }

        self.sink.write_all(prefix)?;
        self.sink.write_all(payload)?;
        self.sink.write_all(suffix)?;

        let state = self
            .images
            .get_mut(handle.index)
            .ok_or(Error::Other("invalid image handle".into()))?;
        state.block_records[block_index] = Some((offset, byte_count));
        Ok(())
    }

    fn choose_is_bigtiff(&mut self) -> Result<bool> {
        match self.requested_variant {
            TiffVariant::Classic => {
                self.ensure_classic_layout()?;
                Ok(false)
            }
            TiffVariant::BigTiff => Ok(true),
            TiffVariant::Auto => Ok(!self.classic_layout_fits()?),
        }
    }

    fn classic_layout_fits(&mut self) -> Result<bool> {
        for state in &self.images {
            for &(offset, byte_count) in state.block_records.iter().flatten() {
                if offset > CLASSIC_TIFF_LIMIT || byte_count > CLASSIC_TIFF_LIMIT {
                    return Ok(false);
                }
            }
        }

        let mut current = self.sink.seek(SeekFrom::End(0))?;
        for state in &self.images {
            let tags = state.builder.build_tags(false);
            current = checked_add_u64(
                current,
                encoder::estimate_ifd_size(self.byte_order, false, &tags),
                "classic IFD layout",
            )?;
            if current > CLASSIC_TIFF_LIMIT {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn ensure_classic_layout(&mut self) -> Result<()> {
        for state in &self.images {
            for &(offset, byte_count) in state.block_records.iter().flatten() {
                classic_offset_u32(offset)?;
                classic_byte_count_u32(byte_count)?;
            }
        }

        let mut current = self.sink.seek(SeekFrom::End(0))?;
        for state in &self.images {
            let tags = state.builder.build_tags(false);
            current = checked_add_u64(
                current,
                encoder::estimate_ifd_size(self.byte_order, false, &tags),
                "classic IFD layout",
            )?;
            classic_offset_u32(current)?;
        }

        Ok(())
    }

    fn write_final_ifds(
        &mut self,
        is_bigtiff: bool,
    ) -> Result<Vec<(Vec<Tag>, encoder::IfdWriteResult)>> {
        let mut results = Vec::with_capacity(self.images.len());
        for state in &self.images {
            let tags = state.builder.build_tags(is_bigtiff);
            let (offsets_tag_code, byte_counts_tag_code) = state.builder.offset_tag_codes();
            let ifd_result = encoder::write_ifd(
                &mut self.sink,
                self.byte_order,
                is_bigtiff,
                &tags,
                offsets_tag_code,
                byte_counts_tag_code,
                state.builder.block_count(),
            )?;
            results.push((tags, ifd_result));
        }
        Ok(results)
    }

    /// Finalize the TIFF file. Emits the IFD chain and patches the header.
    pub fn finish(mut self) -> Result<W> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        self.finalized = true;

        for state in &self.images {
            let total = state.builder.block_count();
            let written = state
                .block_records
                .iter()
                .filter(|record| record.is_some())
                .count();
            if written != total {
                return Err(Error::IncompleteImage { written, total });
            }
        }

        let is_bigtiff = self.choose_is_bigtiff()?;

        self.sink.seek(SeekFrom::Start(self.header_offset))?;
        encoder::write_header(&mut self.sink, self.byte_order, is_bigtiff)?;

        self.sink.seek(SeekFrom::End(0))?;
        let ifd_results = self.write_final_ifds(is_bigtiff)?;

        for (img_idx, state) in self.images.iter().enumerate() {
            let offsets: Vec<u64> = state
                .block_records
                .iter()
                .map(|record| record.unwrap().0)
                .collect();
            let byte_counts: Vec<u64> = state
                .block_records
                .iter()
                .map(|record| record.unwrap().1)
                .collect();

            let (tags, ifd_result) = &ifd_results[img_idx];
            let (offsets_tag_code, byte_counts_tag_code) = state.builder.offset_tag_codes();

            if offsets.len() == 1 {
                if let Some(off) = encoder::find_inline_tag_value_offset(
                    ifd_result.ifd_offset,
                    is_bigtiff,
                    tags,
                    offsets_tag_code,
                ) {
                    self.sink.seek(SeekFrom::Start(off))?;
                    if is_bigtiff {
                        self.sink
                            .write_all(&self.byte_order.write_u64(offsets[0]))?;
                    } else {
                        self.sink.write_all(
                            &self.byte_order.write_u32(classic_offset_u32(offsets[0])?),
                        )?;
                    }
                }
                if let Some(off) = encoder::find_inline_tag_value_offset(
                    ifd_result.ifd_offset,
                    is_bigtiff,
                    tags,
                    byte_counts_tag_code,
                ) {
                    self.sink.seek(SeekFrom::Start(off))?;
                    if is_bigtiff {
                        self.sink
                            .write_all(&self.byte_order.write_u64(byte_counts[0]))?;
                    } else {
                        self.sink.write_all(
                            &self
                                .byte_order
                                .write_u32(classic_byte_count_u32(byte_counts[0])?),
                        )?;
                    }
                }
            } else {
                if let Some(off) = ifd_result.offsets_tag_data_offset {
                    encoder::patch_block_offsets(
                        &mut self.sink,
                        self.byte_order,
                        is_bigtiff,
                        off,
                        &offsets,
                    )?;
                }
                if let Some(off) = ifd_result.byte_counts_tag_data_offset {
                    encoder::patch_block_byte_counts(
                        &mut self.sink,
                        self.byte_order,
                        is_bigtiff,
                        off,
                        &byte_counts,
                    )?;
                }
            }

            if img_idx == 0 {
                encoder::patch_first_ifd(
                    &mut self.sink,
                    self.header_offset,
                    self.byte_order,
                    is_bigtiff,
                    ifd_result.ifd_offset,
                )?;
            } else {
                let prev = &ifd_results[img_idx - 1].1;
                encoder::patch_next_ifd(
                    &mut self.sink,
                    self.byte_order,
                    is_bigtiff,
                    prev.next_ifd_pointer_offset,
                    ifd_result.ifd_offset,
                )?;
            }
        }

        self.sink.seek(SeekFrom::End(0))?;
        Ok(self.sink)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Seek, SeekFrom, Write};

    use super::*;
    use crate::builder::ImageBuilder;

    #[derive(Default)]
    struct CountingSink {
        pos: u64,
        len: u64,
    }

    impl Write for CountingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.pos += buf.len() as u64;
            self.len = self.len.max(self.pos);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Seek for CountingSink {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            let next = match pos {
                SeekFrom::Start(offset) => offset as i128,
                SeekFrom::End(delta) => self.len as i128 + delta as i128,
                SeekFrom::Current(delta) => self.pos as i128 + delta as i128,
            };
            if next < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "negative seek in CountingSink",
                ));
            }
            self.pos = next as u64;
            self.len = self.len.max(self.pos);
            Ok(self.pos)
        }
    }

    #[test]
    fn auto_promotes_to_bigtiff_from_the_final_layout() {
        let mut writer = TiffWriter::new(CountingSink::default(), WriteOptions::default()).unwrap();
        let handle = writer
            .add_image(ImageBuilder::new(1, 1).sample_type::<u8>().strips(1))
            .unwrap();

        writer
            .sink
            .seek(SeekFrom::Start(CLASSIC_TIFF_LIMIT + 1))
            .unwrap();
        writer.write_block_raw(&handle, 0, &[1]).unwrap();

        assert!(writer.choose_is_bigtiff().unwrap());
    }

    #[test]
    fn auto_keeps_classic_for_small_layouts() {
        let mut writer = TiffWriter::new(Cursor::new(Vec::new()), WriteOptions::default()).unwrap();
        let handle = writer
            .add_image(ImageBuilder::new(1, 1).sample_type::<u8>().strips(1))
            .unwrap();
        writer.write_block(&handle, 0, &[7u8]).unwrap();

        assert!(!writer.choose_is_bigtiff().unwrap());
    }

    #[test]
    fn write_block_raw_validates_before_mutating_sink() {
        let mut writer = TiffWriter::new(Cursor::new(Vec::new()), WriteOptions::default()).unwrap();
        let handle = writer
            .add_image(ImageBuilder::new(1, 1).sample_type::<u8>().strips(1))
            .unwrap();

        let len_before = writer.sink.get_ref().len();
        let err = writer.write_block_raw(&handle, 1, &[1, 2, 3]).unwrap_err();

        assert!(matches!(err, Error::BlockIndexOutOfRange { .. }));
        assert_eq!(writer.sink.get_ref().len(), len_before);
    }

    #[test]
    fn write_block_raw_segmented_validates_before_mutating_sink() {
        let mut writer = TiffWriter::new(Cursor::new(Vec::new()), WriteOptions::default()).unwrap();
        let handle = writer
            .add_image(ImageBuilder::new(1, 1).sample_type::<u8>().strips(1))
            .unwrap();

        let len_before = writer.sink.get_ref().len();
        let err = writer
            .write_block_raw_segmented(&handle, 1, &[1, 2], &[3, 4], &[5, 6])
            .unwrap_err();

        assert!(matches!(err, Error::BlockIndexOutOfRange { .. }));
        assert_eq!(writer.sink.get_ref().len(), len_before);
    }
}
