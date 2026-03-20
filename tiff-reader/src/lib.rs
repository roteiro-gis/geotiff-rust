//! Pure-Rust, read-only TIFF and BigTIFF decoder.
//!
//! Supports:
//! - **TIFF** (classic): `II`/`MM` byte order mark + version 42
//! - **BigTIFF**: `II`/`MM` byte order mark + version 43
//! - **Sources**: mmap, in-memory bytes, or any custom random-access source
//!
//! # Example
//!
//! ```no_run
//! use tiff_reader::TiffFile;
//!
//! let file = TiffFile::open("image.tif").unwrap();
//! println!("byte order: {:?}", file.byte_order());
//! println!("IFD count: {}", file.ifd_count());
//!
//! let ifd = file.ifd(0).unwrap();
//! println!("  width: {}", ifd.width());
//! println!("  height: {}", ifd.height());
//! println!("  bits per sample: {:?}", ifd.bits_per_sample());
//!
//! let pixels: ndarray::ArrayD<u16> = file.read_image(0).unwrap();
//! ```

pub mod cache;
pub mod error;
pub mod filters;
pub mod header;
pub mod ifd;
pub mod io;
pub mod source;
pub mod strip;
pub mod tag;
pub mod tile;

use std::path::Path;
use std::sync::Arc;

use cache::BlockCache;
use error::{Error, Result};
use ndarray::{ArrayD, IxDyn};
use source::{BytesSource, MmapSource, SharedSource, TiffSource};

pub use error::Error as TiffError;
pub use header::ByteOrder;
pub use ifd::{Ifd, RasterLayout};
pub use tag::{Tag, TagValue};

/// Configuration for opening a TIFF file.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Maximum bytes held in the decoded strip/tile cache.
    pub block_cache_bytes: usize,
    /// Maximum number of cached strips/tiles.
    pub block_cache_slots: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            block_cache_bytes: 64 * 1024 * 1024,
            block_cache_slots: 257,
        }
    }
}

/// A memory-mapped TIFF file handle.
pub struct TiffFile {
    source: SharedSource,
    header: header::TiffHeader,
    ifds: Vec<ifd::Ifd>,
    block_cache: Arc<BlockCache>,
    gdal_structural_metadata: Option<GdalStructuralMetadata>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GdalStructuralMetadata {
    block_leader_size_as_u32: bool,
    block_trailer_repeats_last_4_bytes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Window {
    pub row_off: usize,
    pub col_off: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Window {
    pub(crate) fn is_empty(self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    pub(crate) fn row_end(self) -> usize {
        self.row_off + self.rows
    }

    pub(crate) fn col_end(self) -> usize {
        self.col_off + self.cols
    }

    pub(crate) fn output_len(self, layout: &RasterLayout) -> Result<usize> {
        self.cols
            .checked_mul(self.rows)
            .and_then(|pixels| pixels.checked_mul(layout.pixel_stride_bytes()))
            .ok_or_else(|| Error::InvalidImageLayout("window size overflows usize".into()))
    }
}

impl GdalStructuralMetadata {
    fn from_prefix(bytes: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(bytes).ok()?;
        if !text.contains("GDAL_STRUCTURAL_METADATA_SIZE=") {
            return None;
        }

        Some(Self {
            block_leader_size_as_u32: text.contains("BLOCK_LEADER=SIZE_AS_UINT4"),
            block_trailer_repeats_last_4_bytes: text
                .contains("BLOCK_TRAILER=LAST_4_BYTES_REPEATED"),
        })
    }

    pub(crate) fn unwrap_block<'a>(
        &self,
        raw: &'a [u8],
        byte_order: ByteOrder,
        offset: u64,
    ) -> Result<&'a [u8]> {
        if self.block_leader_size_as_u32 {
            if raw.len() < 4 {
                return Ok(raw);
            }
            let declared_len = match byte_order {
                ByteOrder::LittleEndian => u32::from_le_bytes(raw[..4].try_into().unwrap()),
                ByteOrder::BigEndian => u32::from_be_bytes(raw[..4].try_into().unwrap()),
            } as usize;
            if let Some(payload_end) = 4usize.checked_add(declared_len) {
                if payload_end <= raw.len() {
                    if self.block_trailer_repeats_last_4_bytes {
                        let trailer_end = payload_end.checked_add(4).ok_or_else(|| {
                            Error::InvalidImageLayout("GDAL block trailer overflows usize".into())
                        })?;
                        if trailer_end <= raw.len() {
                            let expected = &raw[payload_end - 4..payload_end];
                            let trailer = &raw[payload_end..trailer_end];
                            if expected != trailer {
                                return Err(Error::InvalidImageLayout(format!(
                                    "GDAL block trailer mismatch at offset {offset}"
                                )));
                            }
                        }
                    }
                    return Ok(&raw[4..payload_end]);
                }
            }
        }

        if self.block_trailer_repeats_last_4_bytes && raw.len() >= 8 {
            let split = raw.len() - 4;
            if raw[split - 4..split] == raw[split..] {
                return Ok(&raw[..split]);
            }
        }

        Ok(raw)
    }
}

const GDAL_STRUCTURAL_METADATA_PREFIX: &str = "GDAL_STRUCTURAL_METADATA_SIZE=";

/// Types that can be read directly from a decoded TIFF raster.
pub trait TiffSample: Clone + 'static {
    fn matches_layout(layout: &RasterLayout) -> bool;
    fn decode_many(bytes: &[u8]) -> Vec<Self>;
    fn type_name() -> &'static str;
}

macro_rules! impl_tiff_sample {
    ($ty:ty, $format:expr, $bits:expr, $chunk:expr, $from_ne:expr) => {
        impl TiffSample for $ty {
            fn matches_layout(layout: &RasterLayout) -> bool {
                layout.sample_format == $format && layout.bits_per_sample == $bits
            }

            fn decode_many(bytes: &[u8]) -> Vec<Self> {
                bytes.chunks_exact($chunk).map($from_ne).collect()
            }

            fn type_name() -> &'static str {
                stringify!($ty)
            }
        }
    };
}

impl_tiff_sample!(u8, 1, 8, 1, |chunk: &[u8]| chunk[0]);
impl_tiff_sample!(i8, 2, 8, 1, |chunk: &[u8]| chunk[0] as i8);
impl_tiff_sample!(u16, 1, 16, 2, |chunk: &[u8]| u16::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(i16, 2, 16, 2, |chunk: &[u8]| i16::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(u32, 1, 32, 4, |chunk: &[u8]| u32::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(i32, 2, 32, 4, |chunk: &[u8]| i32::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(f32, 3, 32, 4, |chunk: &[u8]| f32::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(u64, 1, 64, 8, |chunk: &[u8]| u64::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(i64, 2, 64, 8, |chunk: &[u8]| i64::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(f64, 3, 64, 8, |chunk: &[u8]| f64::from_ne_bytes(
    chunk.try_into().unwrap()
));

impl TiffFile {
    /// Open a TIFF file from disk using memory-mapped I/O.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, OpenOptions::default())
    }

    /// Open a TIFF file from disk with explicit decoder options.
    pub fn open_with_options<P: AsRef<Path>>(path: P, options: OpenOptions) -> Result<Self> {
        let source: SharedSource = Arc::new(MmapSource::open(path.as_ref())?);
        Self::from_source_with_options(source, options)
    }

    /// Open a TIFF file from an owned byte buffer (WASM-compatible).
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        Self::from_bytes_with_options(data, OpenOptions::default())
    }

    /// Open a TIFF file from bytes with explicit decoder options.
    pub fn from_bytes_with_options(data: Vec<u8>, options: OpenOptions) -> Result<Self> {
        let source: SharedSource = Arc::new(BytesSource::new(data));
        Self::from_source_with_options(source, options)
    }

    /// Open a TIFF file from an arbitrary random-access source.
    pub fn from_source(source: SharedSource) -> Result<Self> {
        Self::from_source_with_options(source, OpenOptions::default())
    }

    /// Open a TIFF file from an arbitrary random-access source with options.
    pub fn from_source_with_options(source: SharedSource, options: OpenOptions) -> Result<Self> {
        let header_len = usize::try_from(source.len().min(16)).unwrap_or(16);
        let header_bytes = source.read_exact_at(0, header_len)?;
        let header = header::TiffHeader::parse(&header_bytes)?;
        let gdal_structural_metadata = parse_gdal_structural_metadata(source.as_ref());
        let ifds = ifd::parse_ifd_chain(source.as_ref(), &header)?;
        Ok(Self {
            source,
            header,
            ifds,
            block_cache: Arc::new(BlockCache::new(
                options.block_cache_bytes,
                options.block_cache_slots,
            )),
            gdal_structural_metadata,
        })
    }

    /// Returns the byte order of the TIFF file.
    pub fn byte_order(&self) -> ByteOrder {
        self.header.byte_order
    }

    /// Returns `true` if this is a BigTIFF file.
    pub fn is_bigtiff(&self) -> bool {
        self.header.is_bigtiff()
    }

    /// Returns the number of IFDs (images/pages) in the file.
    pub fn ifd_count(&self) -> usize {
        self.ifds.len()
    }

    /// Returns the IFD at the given index.
    pub fn ifd(&self, index: usize) -> Result<&Ifd> {
        self.ifds.get(index).ok_or(Error::IfdNotFound(index))
    }

    /// Returns all parsed IFDs.
    pub fn ifds(&self) -> &[Ifd] {
        &self.ifds
    }

    /// Returns the raw file bytes.
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        self.source.as_slice()
    }

    /// Returns the backing source.
    pub fn source(&self) -> &dyn TiffSource {
        self.source.as_ref()
    }

    /// Decode an image into native-endian interleaved sample bytes.
    pub fn read_image_bytes(&self, ifd_index: usize) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        let layout = ifd.raster_layout()?;
        self.decode_window_bytes(
            ifd,
            Window {
                row_off: 0,
                col_off: 0,
                rows: layout.height,
                cols: layout.width,
            },
        )
    }

    /// Decode a pixel window into native-endian interleaved sample bytes.
    pub fn read_window_bytes(
        &self,
        ifd_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        let layout = ifd.raster_layout()?;
        let window = validate_window(&layout, row_off, col_off, rows, cols)?;
        self.decode_window_bytes(ifd, window)
    }

    fn decode_window_bytes(&self, ifd: &Ifd, window: Window) -> Result<Vec<u8>> {
        if window.is_empty() {
            return Ok(Vec::new());
        }

        if ifd.is_tiled() {
            tile::read_window(
                self.source.as_ref(),
                ifd,
                self.byte_order(),
                &self.block_cache,
                window,
                self.gdal_structural_metadata.as_ref(),
            )
        } else {
            strip::read_window(
                self.source.as_ref(),
                ifd,
                self.byte_order(),
                &self.block_cache,
                window,
                self.gdal_structural_metadata.as_ref(),
            )
        }
    }

    /// Decode a window into a typed ndarray.
    ///
    /// Single-band rasters are returned as shape `[rows, cols]`.
    /// Multi-band rasters are returned as shape `[rows, cols, samples_per_pixel]`.
    pub fn read_window<T: TiffSample>(
        &self,
        ifd_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        let layout = ifd.raster_layout()?;
        let window = validate_window(&layout, row_off, col_off, rows, cols)?;
        if !T::matches_layout(&layout) {
            return Err(Error::TypeMismatch {
                expected: T::type_name(),
                actual: format!(
                    "sample_format={} bits_per_sample={}",
                    layout.sample_format, layout.bits_per_sample
                ),
            });
        }

        let decoded = self.decode_window_bytes(ifd, window)?;
        let values = T::decode_many(&decoded);
        let shape = if layout.samples_per_pixel == 1 {
            vec![window.rows, window.cols]
        } else {
            vec![window.rows, window.cols, layout.samples_per_pixel]
        };
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
            Error::InvalidImageLayout(format!("failed to build ndarray from decoded raster: {e}"))
        })
    }

    /// Decode an image into a typed ndarray.
    ///
    /// Single-band rasters are returned as shape `[height, width]`.
    /// Multi-band rasters are returned as shape `[height, width, samples_per_pixel]`.
    pub fn read_image<T: TiffSample>(&self, ifd_index: usize) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        let layout = ifd.raster_layout()?;
        if !T::matches_layout(&layout) {
            return Err(Error::TypeMismatch {
                expected: T::type_name(),
                actual: format!(
                    "sample_format={} bits_per_sample={}",
                    layout.sample_format, layout.bits_per_sample
                ),
            });
        }

        self.read_window(ifd_index, 0, 0, layout.height, layout.width)
    }
}

fn validate_window(
    layout: &RasterLayout,
    row_off: usize,
    col_off: usize,
    rows: usize,
    cols: usize,
) -> Result<Window> {
    let row_end = row_off
        .checked_add(rows)
        .ok_or_else(|| Error::InvalidImageLayout("window row range overflows usize".into()))?;
    let col_end = col_off
        .checked_add(cols)
        .ok_or_else(|| Error::InvalidImageLayout("window column range overflows usize".into()))?;
    if row_end > layout.height || col_end > layout.width {
        return Err(Error::InvalidImageLayout(format!(
            "window [{row_off}..{row_end}, {col_off}..{col_end}) exceeds raster bounds {}x{}",
            layout.height, layout.width
        )));
    }
    Ok(Window {
        row_off,
        col_off,
        rows,
        cols,
    })
}

fn parse_gdal_structural_metadata(source: &dyn TiffSource) -> Option<GdalStructuralMetadata> {
    let available_len = usize::try_from(source.len().checked_sub(8)?).ok()?;
    if available_len == 0 {
        return None;
    }

    let probe_len = available_len.min(64);
    let probe = source.read_exact_at(8, probe_len).ok()?;
    let total_len = parse_gdal_structural_metadata_len(&probe)?;
    if total_len == 0 || total_len > available_len {
        return None;
    }

    let bytes = source.read_exact_at(8, total_len).ok()?;
    GdalStructuralMetadata::from_prefix(&bytes)
}

fn parse_gdal_structural_metadata_len(bytes: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(bytes).ok()?;
    let newline_index = text.find('\n')?;
    let header = &text[..newline_index];
    let value = header.strip_prefix(GDAL_STRUCTURAL_METADATA_PREFIX)?;
    let digits: String = value.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let payload_len: usize = digits.parse().ok()?;
    newline_index.checked_add(1)?.checked_add(payload_len)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{
        parse_gdal_structural_metadata, parse_gdal_structural_metadata_len, GdalStructuralMetadata,
        TiffFile, GDAL_STRUCTURAL_METADATA_PREFIX,
    };
    use crate::source::{BytesSource, TiffSource};

    fn le_u16(value: u16) -> [u8; 2] {
        value.to_le_bytes()
    }

    fn le_u32(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    fn build_stripped_tiff(
        width: u32,
        height: u32,
        image_data: &[u8],
        overrides: &[(u16, u16, u32, Vec<u8>)],
    ) -> Vec<u8> {
        let mut entries = BTreeMap::new();
        entries.insert(256, (4, 1, le_u32(width).to_vec()));
        entries.insert(257, (4, 1, le_u32(height).to_vec()));
        entries.insert(258, (3, 1, [8, 0, 0, 0].to_vec()));
        entries.insert(259, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(273, (4, 1, Vec::new()));
        entries.insert(277, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(278, (4, 1, le_u32(height).to_vec()));
        entries.insert(279, (4, 1, le_u32(image_data.len() as u32).to_vec()));
        for &(tag, ty, count, ref value) in overrides {
            entries.insert(tag, (ty, count, value.clone()));
        }

        let ifd_offset = 8u32;
        let ifd_size = 2 + entries.len() * 12 + 4;
        let mut next_data_offset = ifd_offset as usize + ifd_size;
        let image_offset = next_data_offset as u32;
        next_data_offset += image_data.len();

        let mut data = Vec::with_capacity(next_data_offset);
        data.extend_from_slice(b"II");
        data.extend_from_slice(&le_u16(42));
        data.extend_from_slice(&le_u32(ifd_offset));
        data.extend_from_slice(&le_u16(entries.len() as u16));

        let mut deferred = Vec::new();
        for (tag, (ty, count, value)) in entries {
            data.extend_from_slice(&le_u16(tag));
            data.extend_from_slice(&le_u16(ty));
            data.extend_from_slice(&le_u32(count));
            if tag == 273 {
                data.extend_from_slice(&le_u32(image_offset));
            } else if value.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..value.len()].copy_from_slice(&value);
                data.extend_from_slice(&inline);
            } else {
                let offset = next_data_offset as u32;
                data.extend_from_slice(&le_u32(offset));
                next_data_offset += value.len();
                deferred.push(value);
            }
        }
        data.extend_from_slice(&le_u32(0));
        data.extend_from_slice(image_data);
        for value in deferred {
            data.extend_from_slice(&value);
        }
        data
    }

    fn build_tiled_tiff(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[&[u8]],
    ) -> Vec<u8> {
        let mut entries = BTreeMap::new();
        entries.insert(256, (4, 1, le_u32(width).to_vec()));
        entries.insert(257, (4, 1, le_u32(height).to_vec()));
        entries.insert(258, (3, 1, [8, 0, 0, 0].to_vec()));
        entries.insert(259, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(277, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(322, (4, 1, le_u32(tile_width).to_vec()));
        entries.insert(323, (4, 1, le_u32(tile_height).to_vec()));
        entries.insert(
            325,
            (
                4,
                tiles.len() as u32,
                tiles
                    .iter()
                    .flat_map(|tile| le_u32(tile.len() as u32))
                    .collect(),
            ),
        );

        let ifd_offset = 8u32;
        let ifd_size = 2 + (entries.len() + 1) * 12 + 4;
        let mut tile_data_offset = ifd_offset as usize + ifd_size;
        let tile_offsets: Vec<u32> = tiles
            .iter()
            .map(|tile| {
                let offset = tile_data_offset as u32;
                tile_data_offset += tile.len();
                offset
            })
            .collect();
        entries.insert(
            324,
            (
                4,
                tile_offsets.len() as u32,
                tile_offsets
                    .iter()
                    .flat_map(|offset| le_u32(*offset))
                    .collect(),
            ),
        );

        let mut next_data_offset = tile_data_offset;
        let mut data = Vec::with_capacity(next_data_offset);
        data.extend_from_slice(b"II");
        data.extend_from_slice(&le_u16(42));
        data.extend_from_slice(&le_u32(ifd_offset));
        data.extend_from_slice(&le_u16(entries.len() as u16));

        let mut deferred = Vec::new();
        for (tag, (ty, count, value)) in entries {
            data.extend_from_slice(&le_u16(tag));
            data.extend_from_slice(&le_u16(ty));
            data.extend_from_slice(&le_u32(count));
            if value.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..value.len()].copy_from_slice(&value);
                data.extend_from_slice(&inline);
            } else {
                let offset = next_data_offset as u32;
                data.extend_from_slice(&le_u32(offset));
                next_data_offset += value.len();
                deferred.push(value);
            }
        }
        data.extend_from_slice(&le_u32(0));
        for tile in tiles {
            data.extend_from_slice(tile);
        }
        for value in deferred {
            data.extend_from_slice(&value);
        }
        data
    }

    fn build_multi_strip_tiff(width: u32, rows: &[&[u8]]) -> Vec<u8> {
        let mut entries = BTreeMap::new();
        entries.insert(256, (4, 1, le_u32(width).to_vec()));
        entries.insert(257, (4, 1, le_u32(rows.len() as u32).to_vec()));
        entries.insert(258, (3, 1, [8, 0, 0, 0].to_vec()));
        entries.insert(259, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(277, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(278, (4, 1, le_u32(1).to_vec()));
        entries.insert(
            279,
            (
                4,
                rows.len() as u32,
                rows.iter()
                    .flat_map(|row| le_u32(row.len() as u32))
                    .collect(),
            ),
        );

        let ifd_offset = 8u32;
        let ifd_size = 2 + (entries.len() + 1) * 12 + 4;
        let mut strip_data_offset = ifd_offset as usize + ifd_size;
        let strip_offsets: Vec<u32> = rows
            .iter()
            .map(|row| {
                let offset = strip_data_offset as u32;
                strip_data_offset += row.len();
                offset
            })
            .collect();
        entries.insert(
            273,
            (
                4,
                strip_offsets.len() as u32,
                strip_offsets
                    .iter()
                    .flat_map(|offset| le_u32(*offset))
                    .collect(),
            ),
        );

        let mut next_data_offset = strip_data_offset;
        let mut data = Vec::with_capacity(next_data_offset);
        data.extend_from_slice(b"II");
        data.extend_from_slice(&le_u16(42));
        data.extend_from_slice(&le_u32(ifd_offset));
        data.extend_from_slice(&le_u16(entries.len() as u16));

        let mut deferred = Vec::new();
        for (tag, (ty, count, value)) in entries {
            data.extend_from_slice(&le_u16(tag));
            data.extend_from_slice(&le_u16(ty));
            data.extend_from_slice(&le_u32(count));
            if value.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..value.len()].copy_from_slice(&value);
                data.extend_from_slice(&inline);
            } else {
                let offset = next_data_offset as u32;
                data.extend_from_slice(&le_u32(offset));
                next_data_offset += value.len();
                deferred.push(value);
            }
        }
        data.extend_from_slice(&le_u32(0));
        for row in rows {
            data.extend_from_slice(row);
        }
        for value in deferred {
            data.extend_from_slice(&value);
        }
        data
    }

    struct CountingSource {
        bytes: Vec<u8>,
        reads: AtomicUsize,
    }

    impl CountingSource {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                reads: AtomicUsize::new(0),
            }
        }

        fn reset_reads(&self) {
            self.reads.store(0, Ordering::SeqCst);
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    impl TiffSource for CountingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_exact_at(&self, offset: u64, len: usize) -> crate::error::Result<Vec<u8>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let start =
                usize::try_from(offset).map_err(|_| crate::error::Error::OffsetOutOfBounds {
                    offset,
                    length: len as u64,
                    data_len: self.len(),
                })?;
            let end = start
                .checked_add(len)
                .ok_or(crate::error::Error::OffsetOutOfBounds {
                    offset,
                    length: len as u64,
                    data_len: self.len(),
                })?;
            if end > self.bytes.len() {
                return Err(crate::error::Error::OffsetOutOfBounds {
                    offset,
                    length: len as u64,
                    data_len: self.len(),
                });
            }
            Ok(self.bytes[start..end].to_vec())
        }
    }

    #[test]
    fn reads_stripped_u8_image() {
        let data = build_stripped_tiff(2, 2, &[1, 2, 3, 4], &[]);
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[2, 2]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn reads_horizontal_predictor_u16_strip() {
        let encoded = [1, 0, 1, 0, 2, 0];
        let data = build_stripped_tiff(
            3,
            1,
            &encoded,
            &[
                (258, 3, 1, [16, 0, 0, 0].to_vec()),
                (317, 3, 1, [2, 0, 0, 0].to_vec()),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u16>(0).unwrap();
        assert_eq!(image.shape(), &[1, 3]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1, 2, 4]);
    }

    #[test]
    fn reads_stripped_u8_window() {
        let data = build_multi_strip_tiff(
            4,
            &[
                &[1, 2, 3, 4],
                &[5, 6, 7, 8],
                &[9, 10, 11, 12],
                &[13, 14, 15, 16],
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let window = file.read_window::<u8>(0, 1, 1, 2, 2).unwrap();
        assert_eq!(window.shape(), &[2, 2]);
        let (values, offset) = window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![6, 7, 10, 11]);
    }

    #[test]
    fn reads_tiled_u8_window() {
        let data = build_tiled_tiff(
            4,
            4,
            2,
            2,
            &[
                &[1, 2, 5, 6],
                &[3, 4, 7, 8],
                &[9, 10, 13, 14],
                &[11, 12, 15, 16],
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let window = file.read_window::<u8>(0, 1, 1, 2, 2).unwrap();
        assert_eq!(window.shape(), &[2, 2]);
        let (values, offset) = window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![6, 7, 10, 11]);
    }

    #[test]
    fn windowed_tiled_reads_only_intersecting_blocks() {
        let data = build_tiled_tiff(
            4,
            4,
            2,
            2,
            &[
                &[1, 2, 5, 6],
                &[3, 4, 7, 8],
                &[9, 10, 13, 14],
                &[11, 12, 15, 16],
            ],
        );
        let source = Arc::new(CountingSource::new(data));
        let file = TiffFile::from_source(source.clone()).unwrap();
        source.reset_reads();

        let window = file.read_window::<u8>(0, 0, 0, 2, 2).unwrap();
        let (values, offset) = window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1, 2, 5, 6]);
        assert_eq!(source.reads(), 1);
    }

    #[test]
    fn unwraps_gdal_structural_metadata_block() {
        let metadata = GdalStructuralMetadata::from_prefix(
            b"GDAL_STRUCTURAL_METADATA_SIZE=000174 bytes\nBLOCK_LEADER=SIZE_AS_UINT4\nBLOCK_TRAILER=LAST_4_BYTES_REPEATED\n",
        )
        .unwrap();

        let payload = [1u8, 2, 3, 4];
        let mut block = Vec::new();
        block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        block.extend_from_slice(&payload);
        block.extend_from_slice(&payload[payload.len() - 4..]);

        let unwrapped = metadata
            .unwrap_block(&block, crate::ByteOrder::LittleEndian, 256)
            .unwrap();
        assert_eq!(unwrapped, payload);
    }

    #[test]
    fn rejects_gdal_structural_metadata_trailer_mismatch() {
        let metadata = GdalStructuralMetadata::from_prefix(
            b"GDAL_STRUCTURAL_METADATA_SIZE=000174 bytes\nBLOCK_LEADER=SIZE_AS_UINT4\nBLOCK_TRAILER=LAST_4_BYTES_REPEATED\n",
        )
        .unwrap();

        let block = [
            4u8, 0, 0, 0, //
            1, 2, 3, 4, //
            4, 3, 2, 1,
        ];

        let error = metadata
            .unwrap_block(&block, crate::ByteOrder::LittleEndian, 512)
            .unwrap_err();
        assert!(error.to_string().contains("GDAL block trailer mismatch"));
    }

    #[test]
    fn parses_gdal_structural_metadata_before_binary_prefix_data() {
        let rest = "LAYOUT=IFDS_BEFORE_DATA\nBLOCK_ORDER=ROW_MAJOR\nBLOCK_LEADER=SIZE_AS_UINT4\nBLOCK_TRAILER=LAST_4_BYTES_REPEATED\nKNOWN_INCOMPATIBLE_EDITION=NO\n";
        let prefix = format!(
            "{GDAL_STRUCTURAL_METADATA_PREFIX}{:06} bytes\n{rest}",
            rest.len()
        );

        let mut bytes = vec![0u8; 8];
        bytes.extend_from_slice(prefix.as_bytes());
        bytes.extend_from_slice(&[0xff, 0x00, 0x80, 0x7f]);

        let source = BytesSource::new(bytes);
        let metadata = parse_gdal_structural_metadata(&source).unwrap();
        assert!(metadata.block_leader_size_as_u32);
        assert!(metadata.block_trailer_repeats_last_4_bytes);
    }

    #[test]
    fn parses_gdal_structural_metadata_declared_length_as_header_plus_payload() {
        let rest = "LAYOUT=IFDS_BEFORE_DATA\nBLOCK_ORDER=ROW_MAJOR\n";
        let prefix = format!(
            "{GDAL_STRUCTURAL_METADATA_PREFIX}{:06} bytes\n{rest}",
            rest.len()
        );
        assert_eq!(
            parse_gdal_structural_metadata_len(prefix.as_bytes()),
            Some(prefix.len())
        );
    }

    #[test]
    fn leaves_payload_only_gdal_block_unchanged() {
        let metadata = GdalStructuralMetadata {
            block_leader_size_as_u32: true,
            block_trailer_repeats_last_4_bytes: true,
        };
        let payload = [0x80u8, 0x1a, 0xcf, 0x68, 0x43, 0x9a, 0x11, 0x08];
        let unwrapped = metadata
            .unwrap_block(&payload, crate::ByteOrder::LittleEndian, 570)
            .unwrap();
        assert_eq!(unwrapped, payload);
    }
}
