//! Main TiffWriter: orchestrates multi-IFD streaming writes.

use std::io::{Seek, SeekFrom, Write};

use tiff_core::ByteOrder;

use crate::builder::ImageBuilder;
use crate::compress;
use crate::encoder;
use crate::error::{Error, Result};
use crate::sample::TiffWriteSample;

/// TIFF format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffVariant {
    Classic,
    BigTiff,
    /// Auto-detect: use BigTIFF if any added image's estimated uncompressed
    /// size (plus IFD overhead) would exceed the classic 4 GiB limit.
    /// The header is written as classic initially; if `add_image` detects
    /// the threshold would be exceeded, `finish()` returns an error
    /// recommending explicit BigTIFF. For a seamless experience callers
    /// should pre-calculate and pass `BigTiff` explicitly, or use
    /// `WriteOptions::auto(estimated_bytes)`.
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
    /// Create options that auto-select BigTIFF based on estimated total bytes.
    pub fn auto(estimated_bytes: u64) -> Self {
        let variant = if estimated_bytes >= 4_000_000_000 {
            TiffVariant::BigTiff
        } else {
            TiffVariant::Classic
        };
        Self {
            byte_order: ByteOrder::LittleEndian,
            variant,
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
    tags: Vec<tiff_core::Tag>,
    ifd_result: encoder::IfdWriteResult,
    block_records: Vec<Option<(u64, u64)>>,
}

/// A streaming TIFF/BigTIFF file writer.
pub struct TiffWriter<W: Write + Seek> {
    sink: W,
    byte_order: ByteOrder,
    is_bigtiff: bool,
    header_offset: u64,
    images: Vec<IfdState>,
    finalized: bool,
}

impl<W: Write + Seek> TiffWriter<W> {
    /// Create a new TIFF writer. Writes the file header immediately.
    ///
    /// With `TiffVariant::Auto`, classic TIFF is used initially. If the
    /// cumulative estimated image size exceeds 4 GiB when `add_image` is
    /// called, the writer returns an error recommending `TiffVariant::BigTiff`.
    /// Use `WriteOptions::auto(estimated_bytes)` for automatic selection.
    pub fn new(mut sink: W, options: WriteOptions) -> Result<Self> {
        let is_bigtiff = matches!(options.variant, TiffVariant::BigTiff);
        let header_offset = encoder::write_header(&mut sink, options.byte_order, is_bigtiff)?;
        Ok(Self {
            sink,
            byte_order: options.byte_order,
            is_bigtiff,
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

        let num_blocks = builder.block_count();
        let (offsets_tag, byte_counts_tag) = builder.offset_tag_codes();
        let layout_tags = builder.layout_tags();

        let tags = encoder::build_image_tags(&encoder::ImageTagParams {
            width: builder.width,
            height: builder.height,
            samples_per_pixel: builder.samples_per_pixel,
            bits_per_sample: builder.bits_per_sample,
            sample_format: builder.sample_format.to_code(),
            compression: builder.compression.to_code(),
            photometric: builder.photometric.to_code(),
            predictor: builder.predictor.to_code(),
            planar_configuration: builder.planar_configuration.to_code(),
            subfile_type: builder.subfile_type,
            extra_tags: &builder.extra_tags,
            offsets_tag_code: offsets_tag,
            byte_counts_tag_code: byte_counts_tag,
            num_blocks,
            layout_tags: &layout_tags,
            is_bigtiff: self.is_bigtiff,
        });

        let ifd_result = encoder::write_ifd(
            &mut self.sink,
            self.byte_order,
            self.is_bigtiff,
            &tags,
            offsets_tag,
            byte_counts_tag,
            num_blocks,
        )?;

        let index = self.images.len();
        self.images.push(IfdState {
            builder,
            tags,
            ifd_result,
            block_records: vec![None; num_blocks],
        });

        Ok(ImageHandle { index })
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

        let compressed = compress::compress_block(
            samples,
            self.byte_order,
            state.builder.compression,
            state.builder.predictor,
            state.builder.block_samples_per_pixel(),
            state.builder.block_row_width(),
            block_index,
        )?;

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

        let offset = self.sink.seek(SeekFrom::End(0))?;
        self.sink.write_all(compressed_bytes)?;
        let byte_count = compressed_bytes.len() as u64;

        let state = self
            .images
            .get_mut(handle.index)
            .ok_or(Error::Other("invalid image handle".into()))?;

        let total = state.builder.block_count();
        if block_index >= total {
            return Err(Error::BlockIndexOutOfRange {
                index: block_index,
                total,
            });
        }

        state.block_records[block_index] = Some((offset, byte_count));
        Ok(())
    }

    /// Finalize the TIFF file. Patches all IFDs and chains them together.
    pub fn finish(mut self) -> Result<W> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        self.finalized = true;

        for (img_idx, state) in self.images.iter().enumerate() {
            let total = state.builder.block_count();
            let written = state.block_records.iter().filter(|r| r.is_some()).count();
            if written != total {
                return Err(Error::IncompleteImage { written, total });
            }

            let offsets: Vec<u64> = state.block_records.iter().map(|r| r.unwrap().0).collect();
            let byte_counts: Vec<u64> = state.block_records.iter().map(|r| r.unwrap().1).collect();

            let (offsets_tag_code, byte_counts_tag_code) = state.builder.offset_tag_codes();
            let is_bigtiff = state.ifd_result.is_bigtiff;

            if total == 1 {
                // Single block: value may be inline
                if let Some(off) = encoder::find_inline_tag_value_offset(
                    state.ifd_result.ifd_offset,
                    is_bigtiff,
                    &state.tags,
                    offsets_tag_code,
                ) {
                    self.sink.seek(SeekFrom::Start(off))?;
                    if is_bigtiff {
                        self.sink
                            .write_all(&self.byte_order.write_u64(offsets[0]))?;
                    } else {
                        self.sink
                            .write_all(&self.byte_order.write_u32(offsets[0] as u32))?;
                    }
                }
                if let Some(off) = encoder::find_inline_tag_value_offset(
                    state.ifd_result.ifd_offset,
                    is_bigtiff,
                    &state.tags,
                    byte_counts_tag_code,
                ) {
                    self.sink.seek(SeekFrom::Start(off))?;
                    if is_bigtiff {
                        self.sink
                            .write_all(&self.byte_order.write_u64(byte_counts[0]))?;
                    } else {
                        self.sink
                            .write_all(&self.byte_order.write_u32(byte_counts[0] as u32))?;
                    }
                }
            } else {
                if let Some(off) = state.ifd_result.offsets_tag_data_offset {
                    encoder::patch_block_offsets(
                        &mut self.sink,
                        self.byte_order,
                        is_bigtiff,
                        off,
                        &offsets,
                    )?;
                }
                if let Some(off) = state.ifd_result.byte_counts_tag_data_offset {
                    encoder::patch_block_byte_counts(
                        &mut self.sink,
                        self.byte_order,
                        is_bigtiff,
                        off,
                        &byte_counts,
                    )?;
                }
            }

            // Chain IFDs
            if img_idx == 0 {
                encoder::patch_first_ifd(
                    &mut self.sink,
                    self.header_offset,
                    self.byte_order,
                    is_bigtiff,
                    state.ifd_result.ifd_offset,
                )?;
            } else {
                let prev = &self.images[img_idx - 1];
                encoder::patch_next_ifd(
                    &mut self.sink,
                    self.byte_order,
                    is_bigtiff,
                    prev.ifd_result.next_ifd_pointer_offset,
                    state.ifd_result.ifd_offset,
                )?;
            }
        }

        self.sink.seek(SeekFrom::End(0))?;
        Ok(self.sink)
    }
}
