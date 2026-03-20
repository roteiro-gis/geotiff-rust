//! Pure-Rust, read-only TIFF and BigTIFF file decoder.
//!
//! Supports:
//! - **TIFF** (classic): `II`/`MM` byte order mark + version 42
//! - **BigTIFF**: `II`/`MM` byte order mark + version 43
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
//! ```

pub mod cache;
pub mod error;
pub mod filters;
pub mod header;
pub mod ifd;
pub mod io;
pub mod strip;
pub mod source;
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
}

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
                bytes
                    .chunks_exact($chunk)
                    .map($from_ne)
                    .collect()
            }

            fn type_name() -> &'static str {
                stringify!($ty)
            }
        }
    };
}

impl_tiff_sample!(u8, 1, 8, 1, |chunk: &[u8]| chunk[0]);
impl_tiff_sample!(i8, 2, 8, 1, |chunk: &[u8]| chunk[0] as i8);
impl_tiff_sample!(u16, 1, 16, 2, |chunk: &[u8]| u16::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(i16, 2, 16, 2, |chunk: &[u8]| i16::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(u32, 1, 32, 4, |chunk: &[u8]| u32::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(i32, 2, 32, 4, |chunk: &[u8]| i32::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(f32, 3, 32, 4, |chunk: &[u8]| f32::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(u64, 1, 64, 8, |chunk: &[u8]| u64::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(i64, 2, 64, 8, |chunk: &[u8]| i64::from_ne_bytes(chunk.try_into().unwrap()));
impl_tiff_sample!(f64, 3, 64, 8, |chunk: &[u8]| f64::from_ne_bytes(chunk.try_into().unwrap()));

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
        let ifds = ifd::parse_ifd_chain(source.as_ref(), &header)?;
        Ok(Self {
            source,
            header,
            ifds,
            block_cache: Arc::new(BlockCache::new(
                options.block_cache_bytes,
                options.block_cache_slots,
            )),
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
        if ifd.is_tiled() {
            tile::read_image(self.source.as_ref(), ifd, self.byte_order(), &self.block_cache)
        } else {
            strip::read_image(self.source.as_ref(), ifd, self.byte_order(), &self.block_cache)
        }
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

        let decoded = self.read_image_bytes(ifd_index)?;
        let values = T::decode_many(&decoded);
        let shape = if layout.samples_per_pixel == 1 {
            vec![layout.height, layout.width]
        } else {
            vec![layout.height, layout.width, layout.samples_per_pixel]
        };
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
            Error::InvalidImageLayout(format!("failed to build ndarray from decoded raster: {e}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::TiffFile;

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
}
