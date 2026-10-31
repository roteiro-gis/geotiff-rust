//! Pure-Rust, read-only TIFF and BigTIFF decoder.
//!
//! Supports:
//! - **TIFF** (classic): `II`/`MM` byte order mark + version 42
//! - **BigTIFF**: `II`/`MM` byte order mark + version 43
//! - **Sources**: file-backed random access, opt-in mmap, in-memory bytes, or any custom random-access source
//! - **Reads**: full rasters, windows, and single storage-domain bands
//! - **Compression**: None, Deflate, LZW, PackBits, LERC, JPEG (feature), ZSTD (feature), WebP (feature)
//!
//! TIFF-side `LERC+DEFLATE` is supported unconditionally. TIFF-side
//! `LERC+ZSTD` requires the default `zstd` feature.
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
//! let samples: ndarray::ArrayD<u16> = file.read_image(0).unwrap();
//! ```

mod block_decode;
pub mod cache;
pub mod error;
pub mod filters;
pub mod header;
pub mod ifd;
pub mod io;
mod pixel;
pub mod source;
pub mod strip;
pub mod tag;
pub mod tile;

use std::path::Path;
use std::sync::Arc;

use cache::BlockCache;
use error::{Error, Result};
use ndarray::{ArrayD, IxDyn};
use source::{BytesSource, FileSource, MmapSource, SharedSource, TiffSource};

pub use error::Error as TiffError;
pub use header::ByteOrder;
pub use ifd::{Ifd, ParseBudgets, RasterLayout};
pub use tag::{Tag, TagValue};
pub use tiff_core::constants;
pub use tiff_core::sample::TiffSample;
pub use tiff_core::TagType;
pub use tiff_core::{
    ColorMap, ColorModel, ExtraSample, InkSet, PhotometricInterpretation, YCbCrPositioning,
};

const DEFAULT_DECODE_OUTPUT_BYTES: usize = 1024 * 1024 * 1024;

/// Configuration for opening a TIFF file.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Maximum bytes held in the decoded strip/tile cache.
    pub block_cache_bytes: usize,
    /// Maximum number of cached strips/tiles.
    pub block_cache_slots: usize,
    /// Maximum IFDs, tag entries, and per-tag/aggregate tag-value bytes parsed from metadata.
    pub parse_budgets: ParseBudgets,
    /// Maximum bytes allocated for a single decoded output buffer.
    pub decode_output_bytes: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            block_cache_bytes: 64 * 1024 * 1024,
            block_cache_slots: 257,
            parse_budgets: ParseBudgets::default(),
            decode_output_bytes: DEFAULT_DECODE_OUTPUT_BYTES,
        }
    }
}

/// A TIFF file handle.
pub struct TiffFile {
    source: SharedSource,
    header: header::TiffHeader,
    ifds: Vec<ifd::Ifd>,
    parse_budgets: ParseBudgets,
    decode_output_bytes: usize,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeReadOptions<'a> {
    pub decode_output_bytes: usize,
    pub gdal_structural_metadata: Option<&'a GdalStructuralMetadata>,
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
        let pixel_stride = layout.checked_pixel_stride_bytes()?;
        self.cols
            .checked_mul(self.rows)
            .and_then(|pixels| pixels.checked_mul(pixel_stride))
            .ok_or_else(|| Error::InvalidImageLayout("window size overflows usize".into()))
    }

    pub(crate) fn band_output_len(self, layout: &RasterLayout) -> Result<usize> {
        self.cols
            .checked_mul(self.rows)
            .and_then(|pixels| pixels.checked_mul(layout.bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout("window band size overflows usize".into()))
    }
}

pub(crate) fn checked_layout_add(lhs: usize, rhs: usize, context: &'static str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidImageLayout(format!("{context} overflows usize")))
}

pub(crate) fn checked_layout_mul(lhs: usize, rhs: usize, context: &'static str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidImageLayout(format!("{context} overflows usize")))
}

/// Copy `count` samples between rows that space them differently, as moving a
/// band out of interleaved pixels or a plane into them does.
///
/// This runs for every sample of a read while holding the output lock, so each
/// supported sample width gets its own fixed-size copy.
pub(crate) fn copy_strided_samples(
    dest: &mut [u8],
    dest_stride: usize,
    src: &[u8],
    src_stride: usize,
    bytes_per_sample: usize,
    count: usize,
) {
    fn copy<const BYTES: usize>(
        dest: &mut [u8],
        dest_stride: usize,
        src: &[u8],
        src_stride: usize,
        count: usize,
    ) {
        for (dest, src) in dest
            .chunks_mut(dest_stride)
            .zip(src.chunks(src_stride))
            .take(count)
        {
            dest[..BYTES].copy_from_slice(&src[..BYTES]);
        }
    }

    match bytes_per_sample {
        1 => copy::<1>(dest, dest_stride, src, src_stride, count),
        2 => copy::<2>(dest, dest_stride, src, src_stride, count),
        4 => copy::<4>(dest, dest_stride, src, src_stride, count),
        8 => copy::<8>(dest, dest_stride, src, src_stride, count),
        _ => {
            for (dest, src) in dest
                .chunks_mut(dest_stride)
                .zip(src.chunks(src_stride))
                .take(count)
            {
                dest[..bytes_per_sample].copy_from_slice(&src[..bytes_per_sample]);
            }
        }
    }
}

pub(crate) fn allocate_decode_output(output_len: usize, budget: usize) -> Result<Vec<u8>> {
    let mut output = allocate_decode_output_capacity(output_len, budget)?;
    output.resize(output_len, 0);
    Ok(output)
}

pub(crate) fn allocate_decode_output_capacity(output_len: usize, budget: usize) -> Result<Vec<u8>> {
    validate_decode_output_len(output_len, budget)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|error| Error::DecodeOutputAllocationFailed {
            requested: output_len,
            reason: error.to_string(),
        })?;
    Ok(output)
}

pub(crate) fn copy_decode_output(bytes: &[u8], budget: usize) -> Result<Vec<u8>> {
    let mut output = allocate_decode_output_capacity(bytes.len(), budget)?;
    output.extend_from_slice(bytes);
    Ok(output)
}

pub(crate) fn validate_decode_output_len(output_len: usize, budget: usize) -> Result<()> {
    if output_len > budget {
        return Err(Error::DecodeOutputTooLarge {
            requested: output_len,
            limit: budget,
        });
    }
    Ok(())
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

pub(crate) fn read_block_payload(
    source: &dyn TiffSource,
    offset: u64,
    byte_count: u64,
    byte_count_limit: usize,
    index: usize,
) -> Result<Vec<u8>> {
    let len = validate_block_byte_count(index, byte_count, byte_count_limit)?;
    if let Some(bytes) = source.as_slice() {
        let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds {
            offset,
            length: byte_count,
            data_len: bytes.len() as u64,
        })?;
        let end = start.checked_add(len).ok_or(Error::OffsetOutOfBounds {
            offset,
            length: byte_count,
            data_len: bytes.len() as u64,
        })?;
        if end > bytes.len() {
            return Err(Error::OffsetOutOfBounds {
                offset,
                length: byte_count,
                data_len: bytes.len() as u64,
            });
        }
        Ok(bytes[start..end].to_vec())
    } else {
        source.read_exact_at(offset, len)
    }
}

pub(crate) fn read_gdal_block_payload(
    source: &dyn TiffSource,
    metadata: &GdalStructuralMetadata,
    byte_order: ByteOrder,
    offset: u64,
    byte_count: u64,
    byte_count_limit: usize,
    index: usize,
) -> Result<Vec<u8>> {
    let payload_len = validate_block_byte_count(index, byte_count, byte_count_limit)?;

    // GDAL's COG ghost area wraps each block in a 4-byte size leader (plus an
    // optional repeated 4-byte trailer) while the IFD offset points at the
    // payload itself. Read the wrapped copy first; it wins outright when its
    // unwrapped payload has exactly the declared length.
    let wrapped_result = (metadata.block_leader_size_as_u32 && offset >= 4).then(|| {
        read_wrapped_gdal_block(
            source,
            metadata,
            byte_order,
            offset,
            byte_count,
            byte_count_limit,
            index,
        )
    });
    if let Some(Ok(payload)) = &wrapped_result {
        if payload.len() == payload_len {
            return Ok(payload.clone());
        }
    }

    // Otherwise a successful direct read of the declared payload range wins,
    // with a mismatched-but-readable wrapped payload as the last resort.
    let direct_result = source
        .read_exact_at(offset, payload_len)
        .and_then(|raw| Ok(metadata.unwrap_block(&raw, byte_order, offset)?.to_vec()));
    match direct_result {
        Ok(payload) => {
            if payload.len() > byte_count_limit {
                return Err(block_byte_count_too_large(
                    index,
                    payload.len() as u64,
                    byte_count_limit,
                ));
            }
            Ok(payload)
        }
        Err(direct_error) => match wrapped_result {
            Some(Ok(payload)) => Ok(payload),
            Some(Err(wrapped_error)) => Err(wrapped_error),
            None => Err(direct_error),
        },
    }
}

/// Read and unwrap the leader-prefixed copy of a GDAL ghost-area block.
fn read_wrapped_gdal_block(
    source: &dyn TiffSource,
    metadata: &GdalStructuralMetadata,
    byte_order: ByteOrder,
    offset: u64,
    byte_count: u64,
    byte_count_limit: usize,
    index: usize,
) -> Result<Vec<u8>> {
    let wrapper_extra = if metadata.block_trailer_repeats_last_4_bytes {
        8u64
    } else {
        4u64
    };
    let wrapped_offset = offset - 4;
    let wrapped_len = byte_count.checked_add(wrapper_extra).ok_or_else(|| {
        Error::InvalidImageLayout("GDAL wrapped block length overflows u64".into())
    })?;
    let len = usize::try_from(wrapped_len).map_err(|_| Error::OffsetOutOfBounds {
        offset: wrapped_offset,
        length: wrapped_len,
        data_len: source.len(),
    })?;
    let raw = source.read_exact_at(wrapped_offset, len)?;
    let payload = metadata.unwrap_block(&raw, byte_order, wrapped_offset)?;
    if payload.len() > byte_count_limit {
        return Err(block_byte_count_too_large(
            index,
            payload.len() as u64,
            byte_count_limit,
        ));
    }
    Ok(payload.to_vec())
}

fn validate_block_byte_count(
    index: usize,
    byte_count: u64,
    byte_count_limit: usize,
) -> Result<usize> {
    let len = usize::try_from(byte_count)
        .map_err(|_| block_byte_count_too_large(index, byte_count, byte_count_limit))?;
    if len > byte_count_limit {
        return Err(block_byte_count_too_large(
            index,
            byte_count,
            byte_count_limit,
        ));
    }
    Ok(len)
}

fn block_byte_count_too_large(index: usize, byte_count: u64, byte_count_limit: usize) -> Error {
    Error::DecompressionFailed {
        index,
        reason: format!(
            "encoded block byte count {byte_count} exceeds TIFF block read budget {byte_count_limit}"
        ),
    }
}

const GDAL_STRUCTURAL_METADATA_PREFIX: &str = "GDAL_STRUCTURAL_METADATA_SIZE=";

// TiffSample trait and impls are provided by tiff-core and re-exported above.

impl TiffFile {
    /// Open a TIFF file from disk using safe file-backed I/O.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, OpenOptions::default())
    }

    /// Open a TIFF file from disk using safe file-backed I/O with explicit decoder options.
    pub fn open_with_options<P: AsRef<Path>>(path: P, options: OpenOptions) -> Result<Self> {
        let source: SharedSource = Arc::new(FileSource::open(path.as_ref())?);
        Self::from_source_with_options(source, options)
    }

    /// Open a TIFF file from disk using memory-mapped I/O.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the mapped file is not mutated or
    /// truncated while the returned `TiffFile` is alive. This includes writes
    /// through other file handles and writes from other processes.
    pub unsafe fn open_mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        unsafe { Self::open_mmap_with_options(path, OpenOptions::default()) }
    }

    /// Open a TIFF file from disk using memory-mapped I/O with explicit decoder options.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the mapped file is not mutated or
    /// truncated while the returned `TiffFile` is alive. This includes writes
    /// through other file handles and writes from other processes.
    pub unsafe fn open_mmap_with_options<P: AsRef<Path>>(
        path: P,
        options: OpenOptions,
    ) -> Result<Self> {
        let source: SharedSource = Arc::new(unsafe { MmapSource::open(path.as_ref())? });
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
        let ifds =
            ifd::parse_ifd_chain_with_budgets(source.as_ref(), &header, options.parse_budgets)?;
        Ok(Self {
            source,
            header,
            ifds,
            parse_budgets: options.parse_budgets,
            decode_output_bytes: options.decode_output_bytes,
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

    /// Returns the raw file bytes when the source exposes a resident immutable slice.
    ///
    /// This returns `Some` for in-memory and memory-mapped sources. It returns
    /// `None` for the default safe file-backed source.
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        self.source.as_slice()
    }

    /// Returns the backing source.
    pub fn source(&self) -> &dyn TiffSource {
        self.source.as_ref()
    }

    fn decode_read_options(&self) -> DecodeReadOptions<'_> {
        DecodeReadOptions {
            decode_output_bytes: self.decode_output_bytes,
            gdal_structural_metadata: self.gdal_structural_metadata.as_ref(),
        }
    }

    /// Parse an IFD at an arbitrary file offset.
    pub fn read_ifd_at_offset(&self, offset: u64) -> Result<Ifd> {
        ifd::parse_ifd_at_with_budgets(
            self.source.as_ref(),
            &self.header,
            offset,
            self.parse_budgets,
        )
    }

    /// Decode an image into native-endian interleaved storage sample bytes.
    pub fn read_image_bytes(&self, ifd_index: usize) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_image_bytes_from_ifd(ifd)
    }

    /// Decode an arbitrary IFD into native-endian interleaved storage sample bytes.
    pub fn read_image_bytes_from_ifd(&self, ifd: &Ifd) -> Result<Vec<u8>> {
        let layout = ifd.raster_layout()?;
        self.decode_window_sample_bytes(
            ifd,
            Window {
                row_off: 0,
                col_off: 0,
                rows: layout.height,
                cols: layout.width,
            },
        )
    }

    /// Decode an image into native-endian interleaved color-decoded pixel bytes.
    pub fn read_decoded_image_bytes(&self, ifd_index: usize) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_decoded_image_bytes_from_ifd(ifd)
    }

    /// Decode an arbitrary IFD into native-endian interleaved color-decoded
    /// pixel bytes.
    pub fn read_decoded_image_bytes_from_ifd(&self, ifd: &Ifd) -> Result<Vec<u8>> {
        let layout = ifd.decoded_raster_layout()?;
        self.decode_window_pixel_bytes(
            ifd,
            Window {
                row_off: 0,
                col_off: 0,
                rows: layout.height,
                cols: layout.width,
            },
        )
    }

    /// Decode a pixel window into native-endian interleaved storage sample
    /// bytes.
    pub fn read_window_bytes(
        &self,
        ifd_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_window_bytes_from_ifd(ifd, row_off, col_off, rows, cols)
    }

    /// Decode a pixel window into native-endian interleaved color-decoded pixel
    /// bytes.
    pub fn read_decoded_window_bytes(
        &self,
        ifd_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_decoded_window_bytes_from_ifd(ifd, row_off, col_off, rows, cols)
    }

    /// Decode a pixel window from an arbitrary IFD into native-endian
    /// interleaved storage sample bytes.
    pub fn read_window_bytes_from_ifd(
        &self,
        ifd: &Ifd,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let layout = ifd.raster_layout()?;
        let window = validate_window(&layout, row_off, col_off, rows, cols)?;
        self.decode_window_sample_bytes(ifd, window)
    }

    /// Decode a pixel window from an arbitrary IFD into native-endian
    /// interleaved color-decoded pixel bytes.
    pub fn read_decoded_window_bytes_from_ifd(
        &self,
        ifd: &Ifd,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let layout = ifd.decoded_raster_layout()?;
        let window = validate_window(&layout, row_off, col_off, rows, cols)?;
        self.decode_window_pixel_bytes(ifd, window)
    }

    /// Decode a single storage-domain band into native-endian sample bytes.
    pub fn read_band_bytes(&self, ifd_index: usize, band_index: usize) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_band_bytes_from_ifd(ifd, band_index)
    }

    /// Decode a single storage-domain band from an arbitrary IFD into
    /// native-endian sample bytes.
    pub fn read_band_bytes_from_ifd(&self, ifd: &Ifd, band_index: usize) -> Result<Vec<u8>> {
        let layout = ifd.raster_layout()?;
        self.read_band_window_bytes_from_ifd(ifd, band_index, 0, 0, layout.height, layout.width)
    }

    /// Decode a pixel window from one storage-domain band into native-endian
    /// sample bytes.
    pub fn read_band_window_bytes(
        &self,
        ifd_index: usize,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_band_window_bytes_from_ifd(ifd, band_index, row_off, col_off, rows, cols)
    }

    /// Decode a pixel window from one storage-domain band in an arbitrary IFD
    /// into native-endian sample bytes.
    pub fn read_band_window_bytes_from_ifd(
        &self,
        ifd: &Ifd,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        let layout = ifd.raster_layout()?;
        validate_band_index(&layout, band_index)?;
        let window = validate_window(&layout, row_off, col_off, rows, cols)?;
        self.decode_window_sample_band_bytes(ifd, window, band_index)
    }

    fn decode_window_sample_bytes(&self, ifd: &Ifd, window: Window) -> Result<Vec<u8>> {
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
                self.decode_read_options(),
            )
        } else {
            strip::read_window(
                self.source.as_ref(),
                ifd,
                self.byte_order(),
                &self.block_cache,
                window,
                self.decode_read_options(),
            )
        }
    }

    fn decode_window_sample_band_bytes(
        &self,
        ifd: &Ifd,
        window: Window,
        band_index: usize,
    ) -> Result<Vec<u8>> {
        if window.is_empty() {
            return Ok(Vec::new());
        }

        let layout = ifd.raster_layout()?;
        validate_band_index(&layout, band_index)?;
        if ifd.is_tiled() {
            tile::read_window_band(
                self.source.as_ref(),
                ifd,
                self.byte_order(),
                &self.block_cache,
                window,
                band_index,
                self.decode_read_options(),
            )
        } else {
            strip::read_window_band(
                self.source.as_ref(),
                ifd,
                self.byte_order(),
                &self.block_cache,
                window,
                band_index,
                self.decode_read_options(),
            )
        }
    }

    fn decode_window_pixel_bytes(&self, ifd: &Ifd, window: Window) -> Result<Vec<u8>> {
        let storage_layout = ifd.raster_layout()?;
        let sample_bytes = self.decode_window_sample_bytes(ifd, window)?;
        let (_, pixels) = pixel::decode_pixels(
            ifd,
            &storage_layout,
            window.cols,
            window.rows,
            &sample_bytes,
            self.decode_output_bytes,
        )?;
        Ok(pixels)
    }

    /// Decode a window into a typed ndarray of storage-domain samples.
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
        self.read_window_from_ifd(ifd, row_off, col_off, rows, cols)
    }

    /// Decode a window from an arbitrary IFD into a typed ndarray of
    /// storage-domain samples.
    pub fn read_window_from_ifd<T: TiffSample>(
        &self,
        ifd: &Ifd,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
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

        let decoded = self.decode_window_sample_bytes(ifd, window)?;
        let values = T::decode_many(&decoded);
        let shape = if layout.samples_per_pixel == 1 {
            vec![window.rows, window.cols]
        } else {
            vec![window.rows, window.cols, layout.samples_per_pixel]
        };
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
            Error::InvalidImageLayout(format!("failed to build ndarray from storage raster: {e}"))
        })
    }

    /// Decode a window into a typed ndarray of color-decoded pixels.
    ///
    /// Single-channel decoded rasters are returned as shape `[rows, cols]`.
    /// Multi-channel decoded rasters are returned as shape `[rows, cols, channels]`.
    pub fn read_decoded_window<T: TiffSample>(
        &self,
        ifd_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_decoded_window_from_ifd(ifd, row_off, col_off, rows, cols)
    }

    /// Decode a window from an arbitrary IFD into a typed ndarray of
    /// color-decoded pixels.
    pub fn read_decoded_window_from_ifd<T: TiffSample>(
        &self,
        ifd: &Ifd,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let layout = ifd.decoded_raster_layout()?;
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

        let decoded = self.decode_window_pixel_bytes(ifd, window)?;
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

    /// Decode one storage-domain band into a typed `[height, width]` ndarray.
    pub fn read_band<T: TiffSample>(
        &self,
        ifd_index: usize,
        band_index: usize,
    ) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_band_from_ifd(ifd, band_index)
    }

    /// Decode one storage-domain band from an arbitrary IFD into a typed
    /// `[height, width]` ndarray.
    pub fn read_band_from_ifd<T: TiffSample>(
        &self,
        ifd: &Ifd,
        band_index: usize,
    ) -> Result<ArrayD<T>> {
        let layout = ifd.raster_layout()?;
        self.read_band_window_from_ifd(ifd, band_index, 0, 0, layout.height, layout.width)
    }

    /// Decode a window from one storage-domain band into a typed
    /// `[rows, cols]` ndarray.
    pub fn read_band_window<T: TiffSample>(
        &self,
        ifd_index: usize,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_band_window_from_ifd(ifd, band_index, row_off, col_off, rows, cols)
    }

    /// Decode a window from one storage-domain band in an arbitrary IFD into a
    /// typed `[rows, cols]` ndarray.
    pub fn read_band_window_from_ifd<T: TiffSample>(
        &self,
        ifd: &Ifd,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let layout = ifd.raster_layout()?;
        validate_band_index(&layout, band_index)?;
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

        let decoded = self.decode_window_sample_band_bytes(ifd, window, band_index)?;
        let values = T::decode_many(&decoded);
        ArrayD::from_shape_vec(IxDyn(&[window.rows, window.cols]), values).map_err(|e| {
            Error::InvalidImageLayout(format!("failed to build ndarray from band raster: {e}"))
        })
    }

    /// Decode an image into a typed ndarray of storage-domain samples.
    ///
    /// Single-band rasters are returned as shape `[height, width]`.
    /// Multi-band rasters are returned as shape `[height, width, samples_per_pixel]`.
    pub fn read_image<T: TiffSample>(&self, ifd_index: usize) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_image_from_ifd(ifd)
    }

    /// Decode an arbitrary IFD into a typed ndarray of storage-domain samples.
    pub fn read_image_from_ifd<T: TiffSample>(&self, ifd: &Ifd) -> Result<ArrayD<T>> {
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

        self.read_window_from_ifd(ifd, 0, 0, layout.height, layout.width)
    }

    /// Decode an image into a typed ndarray of color-decoded pixels.
    ///
    /// Single-channel decoded rasters are returned as shape `[height, width]`.
    /// Multi-channel decoded rasters are returned as shape
    /// `[height, width, channels]`.
    pub fn read_decoded_image<T: TiffSample>(&self, ifd_index: usize) -> Result<ArrayD<T>> {
        let ifd = self.ifd(ifd_index)?;
        self.read_decoded_image_from_ifd(ifd)
    }

    /// Decode an arbitrary IFD into a typed ndarray of color-decoded pixels.
    pub fn read_decoded_image_from_ifd<T: TiffSample>(&self, ifd: &Ifd) -> Result<ArrayD<T>> {
        let layout = ifd.decoded_raster_layout()?;
        if !T::matches_layout(&layout) {
            return Err(Error::TypeMismatch {
                expected: T::type_name(),
                actual: format!(
                    "sample_format={} bits_per_sample={}",
                    layout.sample_format, layout.bits_per_sample
                ),
            });
        }

        self.read_decoded_window_from_ifd(ifd, 0, 0, layout.height, layout.width)
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

fn validate_band_index(layout: &RasterLayout, band_index: usize) -> Result<()> {
    if band_index >= layout.samples_per_pixel {
        return Err(Error::BandIndexOutOfBounds {
            index: band_index,
            band_count: layout.samples_per_pixel,
        });
    }
    Ok(())
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
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        copy_strided_samples, parse_gdal_structural_metadata, parse_gdal_structural_metadata_len,
        Error, GdalStructuralMetadata, OpenOptions, ParseBudgets, TiffFile,
        GDAL_STRUCTURAL_METADATA_PREFIX,
    };
    use crate::source::{BytesSource, TiffSource};
    use flate2::{write::ZlibEncoder, Compression as FlateCompression};
    use std::io::Write;

    fn le_u16(value: u16) -> [u8; 2] {
        value.to_le_bytes()
    }

    fn le_u32(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    fn le_u64(value: u64) -> [u8; 8] {
        value.to_le_bytes()
    }

    fn temp_tiff_path(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "geotiff-rust-{test_name}-{}-{nanos}.tif",
            std::process::id()
        ))
    }

    fn bigtiff_header(first_ifd_offset: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"II");
        bytes.extend_from_slice(&le_u16(43));
        bytes.extend_from_slice(&le_u16(8));
        bytes.extend_from_slice(&le_u16(0));
        bytes.extend_from_slice(&le_u64(first_ifd_offset));
        bytes
    }

    fn inline_short(value: u16) -> Vec<u8> {
        let mut bytes = [0u8; 4];
        bytes[..2].copy_from_slice(&le_u16(value));
        bytes.to_vec()
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

    /// Build a classic TIFF whose chain holds one 1x1 uncompressed IFD per
    /// pixel value, with every tag value stored inline.
    fn build_multi_ifd_tiff(pixel_values: &[u8]) -> Vec<u8> {
        const IFD_ENTRY_COUNT: usize = 8;
        const IFD_SIZE: usize = 2 + IFD_ENTRY_COUNT * 12 + 4;
        const IFD_STRIDE: usize = IFD_SIZE + 1; // one 1x1 u8 strip per IFD

        let mut data = Vec::with_capacity(8 + pixel_values.len() * IFD_STRIDE);
        data.extend_from_slice(b"II");
        data.extend_from_slice(&le_u16(42));
        data.extend_from_slice(&le_u32(8));

        for (index, &pixel) in pixel_values.iter().enumerate() {
            let ifd_offset = 8 + index * IFD_STRIDE;
            debug_assert_eq!(data.len(), ifd_offset);
            let image_offset = (ifd_offset + IFD_SIZE) as u32;
            let next_ifd_offset = if index + 1 < pixel_values.len() {
                (ifd_offset + IFD_STRIDE) as u32
            } else {
                0
            };

            data.extend_from_slice(&le_u16(IFD_ENTRY_COUNT as u16));
            for (tag, ty, value) in [
                (256u16, 4u16, le_u32(1).to_vec()),
                (257, 4, le_u32(1).to_vec()),
                (258, 3, inline_short(8)),
                (259, 3, inline_short(1)),
                (273, 4, le_u32(image_offset).to_vec()),
                (277, 3, inline_short(1)),
                (278, 4, le_u32(1).to_vec()),
                (279, 4, le_u32(1).to_vec()),
            ] {
                data.extend_from_slice(&le_u16(tag));
                data.extend_from_slice(&le_u16(ty));
                data.extend_from_slice(&le_u32(1));
                let mut inline = [0u8; 4];
                inline[..value.len()].copy_from_slice(&value);
                data.extend_from_slice(&inline);
            }
            data.extend_from_slice(&le_u32(next_ifd_offset));
            data.push(pixel);
        }
        data
    }

    #[test]
    fn strided_copy_moves_one_band_between_interleaved_and_planar_rows() {
        for bytes_per_sample in [1usize, 2, 3, 4, 8] {
            let pixel_stride = bytes_per_sample * 3;
            let count = 5;
            let interleaved: Vec<u8> = (0..pixel_stride * count).map(|v| v as u8).collect();

            // Extract the middle band from interleaved pixels.
            let mut band = vec![0u8; bytes_per_sample * count];
            copy_strided_samples(
                &mut band,
                bytes_per_sample,
                &interleaved[bytes_per_sample..],
                pixel_stride,
                bytes_per_sample,
                count,
            );
            let expected: Vec<u8> = interleaved
                .chunks_exact(pixel_stride)
                .flat_map(|pixel| pixel[bytes_per_sample..bytes_per_sample * 2].to_vec())
                .collect();
            assert_eq!(band, expected, "{bytes_per_sample}-byte extract");

            // Insert it back, leaving the other bands untouched.
            let mut output = vec![0xffu8; interleaved.len()];
            copy_strided_samples(
                &mut output[bytes_per_sample..],
                pixel_stride,
                &band,
                bytes_per_sample,
                bytes_per_sample,
                count,
            );
            for (index, pixel) in output.chunks_exact(pixel_stride).enumerate() {
                assert_eq!(
                    pixel[bytes_per_sample..bytes_per_sample * 2],
                    expected[index * bytes_per_sample..(index + 1) * bytes_per_sample],
                );
                assert!(pixel[..bytes_per_sample].iter().all(|&value| value == 0xff));
                assert!(pixel[bytes_per_sample * 2..]
                    .iter()
                    .all(|&value| value == 0xff));
            }
        }
    }

    #[test]
    fn strided_copy_stops_after_count_samples() {
        let mut dest = [0u8; 6];
        copy_strided_samples(&mut dest, 2, &[1, 2, 3, 4, 5, 6], 2, 2, 2);
        assert_eq!(dest, [1, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn block_cache_does_not_collide_between_chain_index_and_ifd_offset() {
        // Ten chained IFDs: chain IFD #8 shares its numeric index with the
        // file offset (8) of the first IFD.
        let pixel_values: Vec<u8> = (0..10u8).map(|index| 100 + index).collect();
        let file = TiffFile::from_bytes(build_multi_ifd_tiff(&pixel_values)).unwrap();
        assert_eq!(file.ifd_count(), 10);

        // Populate the block cache from chain IFD 8 first.
        assert_eq!(file.read_image_bytes(8).unwrap(), vec![pixel_values[8]]);

        // Re-parsing the first IFD by offset must decode its own strip, not
        // return the cached strip of chain IFD 8.
        let first_ifd = file.read_ifd_at_offset(8).unwrap();
        assert_eq!(
            file.read_image_bytes_from_ifd(&first_ifd).unwrap(),
            vec![pixel_values[0]]
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn build_lerc2_header_v2(
        width: u32,
        height: u32,
        valid_pixel_count: u32,
        image_type: i32,
        max_z_error: f64,
        z_min: f64,
        z_max: f64,
        payload_len: usize,
    ) -> Vec<u8> {
        let blob_size = 58 + 4 + payload_len;
        let mut bytes = Vec::with_capacity(blob_size);
        bytes.extend_from_slice(b"Lerc2 ");
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
        bytes.extend_from_slice(&image_type.to_le_bytes());
        bytes.extend_from_slice(&max_z_error.to_le_bytes());
        bytes.extend_from_slice(&z_min.to_le_bytes());
        bytes.extend_from_slice(&z_max.to_le_bytes());
        bytes
    }

    #[allow(clippy::too_many_arguments)]
    fn build_lerc2_header_v4(
        width: u32,
        height: u32,
        depth: u32,
        valid_pixel_count: u32,
        image_type: i32,
        max_z_error: f64,
        z_min: f64,
        z_max: f64,
        payload_len: usize,
    ) -> Vec<u8> {
        let blob_size = 66 + 4 + payload_len;
        let mut bytes = Vec::with_capacity(blob_size);
        bytes.extend_from_slice(b"Lerc2 ");
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&depth.to_le_bytes());
        bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
        bytes.extend_from_slice(&image_type.to_le_bytes());
        bytes.extend_from_slice(&max_z_error.to_le_bytes());
        bytes.extend_from_slice(&z_min.to_le_bytes());
        bytes.extend_from_slice(&z_max.to_le_bytes());
        bytes
    }

    fn finalize_lerc2_v4_with_checksum(mut bytes: Vec<u8>) -> Vec<u8> {
        let blob_size = bytes.len() as i32;
        bytes[34..38].copy_from_slice(&blob_size.to_le_bytes());
        let checksum = fletcher32(&bytes[14..blob_size as usize]);
        bytes[10..14].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    fn fletcher32(bytes: &[u8]) -> u32 {
        let mut sum1 = 0xffffu32;
        let mut sum2 = 0xffffu32;
        let mut words = bytes.len() / 2;
        let mut index = 0usize;

        while words > 0 {
            let chunk = words.min(359);
            words -= chunk;
            for _ in 0..chunk {
                sum1 += (bytes[index] as u32) << 8;
                index += 1;
                sum2 += sum1 + bytes[index] as u32;
                sum1 += bytes[index] as u32;
                index += 1;
            }
            sum1 = (sum1 & 0xffff) + (sum1 >> 16);
            sum2 = (sum2 & 0xffff) + (sum2 >> 16);
        }

        if bytes.len() & 1 != 0 {
            sum1 += (bytes[index] as u32) << 8;
            sum2 += sum1;
        }

        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
        (sum2 << 16) | (sum1 & 0xffff)
    }

    fn encode_mask_rle(mask: &[u8]) -> Vec<u8> {
        let bitset_len = mask.len().div_ceil(8);
        let mut bitset = vec![0u8; bitset_len];
        for (index, &value) in mask.iter().enumerate() {
            if value != 0 {
                bitset[index >> 3] |= 1 << (7 - (index & 7));
            }
        }

        let mut encoded = Vec::with_capacity(bitset_len + 4);
        encoded.extend_from_slice(&(bitset_len as i16).to_le_bytes());
        encoded.extend_from_slice(&bitset);
        encoded.extend_from_slice(&i16::MIN.to_le_bytes());
        encoded
    }

    fn build_lerc_tiff(
        width: u32,
        height: u32,
        image_data: &[u8],
        bits_per_sample: u16,
        sample_format: u16,
        samples_per_pixel: u16,
        lerc_parameters: Option<[u32; 2]>,
    ) -> Vec<u8> {
        let mut overrides = vec![
            (258u16, 3u16, 1u32, inline_short(bits_per_sample)),
            (259u16, 3u16, 1u32, inline_short(34887)),
            (277u16, 3u16, 1u32, inline_short(samples_per_pixel)),
            (279u16, 4u16, 1u32, le_u32(image_data.len() as u32).to_vec()),
        ];
        if sample_format != 1 {
            overrides.push((339u16, 3u16, 1u32, inline_short(sample_format)));
        }
        if let Some([version, additional_compression]) = lerc_parameters {
            overrides.push((
                50674u16,
                4u16,
                2u32,
                [version, additional_compression]
                    .into_iter()
                    .flat_map(le_u32)
                    .collect(),
            ));
        }
        build_stripped_tiff(width, height, image_data, &overrides)
    }

    fn build_tiled_tiff(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[&[u8]],
    ) -> Vec<u8> {
        build_tiled_tiff_with_overrides(width, height, tile_width, tile_height, tiles, &[])
    }

    fn build_tiled_tiff_with_overrides(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[&[u8]],
        overrides: &[(u16, u16, u32, Vec<u8>)],
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
        for &(tag, ty, count, ref value) in overrides {
            entries.insert(tag, (ty, count, value.clone()));
        }

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

    fn build_planar_stripped_tiff(width: u32, height: u32, planes: &[&[u8]]) -> Vec<u8> {
        let mut entries = BTreeMap::new();
        entries.insert(256, (4, 1, le_u32(width).to_vec()));
        entries.insert(257, (4, 1, le_u32(height).to_vec()));
        entries.insert(258, (3, 1, [8, 0, 0, 0].to_vec()));
        entries.insert(259, (3, 1, [1, 0, 0, 0].to_vec()));
        entries.insert(262, (3, 1, [2, 0, 0, 0].to_vec()));
        entries.insert(277, (3, 1, inline_short(planes.len() as u16)));
        entries.insert(278, (4, 1, le_u32(height).to_vec()));
        entries.insert(284, (3, 1, [2, 0, 0, 0].to_vec()));
        entries.insert(
            279,
            (
                4,
                planes.len() as u32,
                planes
                    .iter()
                    .flat_map(|plane| le_u32(plane.len() as u32))
                    .collect(),
            ),
        );

        let ifd_offset = 8u32;
        let ifd_size = 2 + (entries.len() + 1) * 12 + 4;
        let mut strip_data_offset = ifd_offset as usize + ifd_size;
        let strip_offsets: Vec<u32> = planes
            .iter()
            .map(|plane| {
                let offset = strip_data_offset as u32;
                strip_data_offset += plane.len();
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
        for plane in planes {
            data.extend_from_slice(plane);
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

    fn overwrite_classic_inline_long_tag(data: &mut [u8], tag: u16, value: u32) {
        let ifd_offset = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let entry_count = u16::from_le_bytes(data[ifd_offset..ifd_offset + 2].try_into().unwrap());
        for entry_index in 0..usize::from(entry_count) {
            let entry = ifd_offset + 2 + entry_index * 12;
            let entry_tag = u16::from_le_bytes(data[entry..entry + 2].try_into().unwrap());
            if entry_tag == tag {
                data[entry + 8..entry + 12].copy_from_slice(&le_u32(value));
                return;
            }
        }
        panic!("tag {tag} not found");
    }

    /// Zero one element of a classic LONG-array tag (inline or deferred).
    fn zero_classic_long_array_element(data: &mut [u8], tag: u16, element_index: usize) {
        let ifd_offset = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let entry_count = u16::from_le_bytes(data[ifd_offset..ifd_offset + 2].try_into().unwrap());
        for entry_index in 0..usize::from(entry_count) {
            let entry = ifd_offset + 2 + entry_index * 12;
            let entry_tag = u16::from_le_bytes(data[entry..entry + 2].try_into().unwrap());
            if entry_tag != tag {
                continue;
            }
            let count = u32::from_le_bytes(data[entry + 4..entry + 8].try_into().unwrap()) as usize;
            assert!(element_index < count, "element index out of range");
            if count == 1 {
                data[entry + 8..entry + 12].copy_from_slice(&[0; 4]);
            } else {
                let value_offset =
                    u32::from_le_bytes(data[entry + 8..entry + 12].try_into().unwrap()) as usize;
                let element = value_offset + element_index * 4;
                data[element..element + 4].copy_from_slice(&[0; 4]);
            }
            return;
        }
        panic!("tag {tag} not found");
    }

    fn ghost_metadata(trailer: bool) -> GdalStructuralMetadata {
        GdalStructuralMetadata {
            block_leader_size_as_u32: true,
            block_trailer_repeats_last_4_bytes: trailer,
        }
    }

    fn read_ghost_payload(
        bytes: Vec<u8>,
        metadata: &GdalStructuralMetadata,
        offset: u64,
        byte_count: u64,
    ) -> crate::error::Result<Vec<u8>> {
        let source = BytesSource::new(bytes);
        super::read_gdal_block_payload(
            &source,
            metadata,
            crate::ByteOrder::LittleEndian,
            offset,
            byte_count,
            1024,
            0,
        )
    }

    #[test]
    fn ghost_block_prefers_wrapped_copy_with_matching_length() {
        // leader(4) | payload(4) | trailer(4, repeats last 4 payload bytes)
        let payload = [10u8, 20, 30, 40];
        let mut bytes = 4u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&payload);

        let result = read_ghost_payload(bytes, &ghost_metadata(true), 4, 4).unwrap();
        assert_eq!(result, payload);
    }

    #[test]
    fn ghost_block_falls_back_to_direct_read_without_leader() {
        // No leader bytes in the file even though the metadata declares one:
        // the direct payload-range read must win.
        let bytes = vec![9u8, 9, 9, 9, 1, 2, 3, 4];
        let result = read_ghost_payload(bytes, &ghost_metadata(false), 4, 4).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn ghost_block_falls_back_when_wrapped_read_is_out_of_bounds() {
        // Wrapped read (offset-4 .. +leader+trailer) would run past EOF; the
        // direct read still succeeds.
        let mut bytes = vec![0u8; 4];
        bytes.extend_from_slice(&[5, 6, 7, 8]);
        let result = read_ghost_payload(bytes, &ghost_metadata(true), 4, 4).unwrap();
        assert_eq!(result, vec![5, 6, 7, 8]);
    }

    #[test]
    fn ghost_block_skips_wrapped_read_for_small_offsets() {
        let bytes = vec![11u8, 12, 13, 14];
        let result = read_ghost_payload(bytes, &ghost_metadata(true), 0, 4).unwrap();
        assert_eq!(result, vec![11, 12, 13, 14]);
    }

    #[test]
    fn ghost_block_errors_when_both_reads_fail() {
        let bytes = vec![0u8; 4];
        let error = read_ghost_payload(bytes, &ghost_metadata(false), 8, 4).unwrap_err();
        assert!(matches!(error, Error::OffsetOutOfBounds { .. }), "{error}");
    }

    #[test]
    fn sparse_strip_reads_as_zero_fill() {
        let rows: [&[u8]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
        let mut data = build_multi_strip_tiff(4, &rows);
        zero_classic_long_array_element(&mut data, 273, 1); // StripOffsets[1]
        zero_classic_long_array_element(&mut data, 279, 1); // StripByteCounts[1]

        let file = TiffFile::from_bytes(data).unwrap();
        assert_eq!(
            file.read_image_bytes(0).unwrap(),
            vec![1, 2, 3, 4, 0, 0, 0, 0]
        );
    }

    #[test]
    fn zero_byte_count_block_reads_as_zero_fill() {
        let rows: [&[u8]; 2] = [&[1, 2], &[3, 4]];
        let mut data = build_multi_strip_tiff(2, &rows);
        zero_classic_long_array_element(&mut data, 279, 0); // StripByteCounts[0]

        let file = TiffFile::from_bytes(data).unwrap();
        assert_eq!(file.read_image_bytes(0).unwrap(), vec![0, 0, 3, 4]);
    }

    #[test]
    fn sparse_tile_reads_as_zero_fill() {
        let tile0 = vec![9u8; 256];
        let tile1 = vec![7u8; 256];
        let mut data = build_tiled_tiff(32, 16, 16, 16, &[&tile0, &tile1]);
        zero_classic_long_array_element(&mut data, 324, 1); // TileOffsets[1]
        zero_classic_long_array_element(&mut data, 325, 1); // TileByteCounts[1]

        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image_bytes(0).unwrap();
        assert_eq!(image.len(), 32 * 16);
        assert!(image.chunks(32).all(|row| {
            row[..16].iter().all(|&value| value == 9) && row[16..].iter().all(|&value| value == 0)
        }));
    }

    #[test]
    fn open_uses_safe_file_source_without_raw_slice() {
        let path = temp_tiff_path("open_uses_safe_file_source_without_raw_slice");
        fs::write(&path, build_stripped_tiff(1, 1, &[7], &[])).unwrap();

        let file = TiffFile::open(&path).unwrap();
        assert!(file.raw_bytes().is_none());
        assert_eq!(file.read_image_bytes(0).unwrap(), vec![7]);

        drop(file);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn open_mmap_exposes_raw_slice() {
        let bytes = build_stripped_tiff(1, 1, &[7], &[]);
        let path = temp_tiff_path("open_mmap_exposes_raw_slice");
        fs::write(&path, &bytes).unwrap();

        let file = unsafe { TiffFile::open_mmap(&path).unwrap() };
        assert_eq!(file.raw_bytes(), Some(bytes.as_slice()));
        assert_eq!(file.read_image_bytes(0).unwrap(), vec![7]);

        drop(file);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn decode_output_budget_rejects_large_storage_window_before_allocation() {
        let file = TiffFile::from_bytes_with_options(
            build_stripped_tiff(4, 4, &[0], &[]),
            OpenOptions {
                decode_output_bytes: 8,
                ..OpenOptions::default()
            },
        )
        .unwrap();

        let err = file.read_image_bytes(0).unwrap_err();
        assert!(matches!(
            err,
            Error::DecodeOutputTooLarge {
                requested: 16,
                limit: 8
            }
        ));
    }

    #[test]
    fn decode_output_budget_also_bounds_intersecting_storage_blocks() {
        let tile = vec![7u8; 16 * 16];
        let file = TiffFile::from_bytes_with_options(
            build_tiled_tiff(16, 16, 16, 16, &[&tile]),
            OpenOptions {
                decode_output_bytes: 16,
                ..OpenOptions::default()
            },
        )
        .unwrap();

        // The requested window is one byte, but servicing it requires
        // decoding the intersecting 256-byte tile.
        let err = file.read_window_bytes(0, 0, 0, 1, 1).unwrap_err();
        assert!(matches!(
            err,
            Error::DecodeOutputTooLarge {
                requested: 256,
                limit: 16
            }
        ));
    }

    #[test]
    fn decode_output_budget_rejects_large_color_decoded_output() {
        let mut color_map = Vec::new();
        color_map.extend((0u16..16).map(|value| value * 17 * 257));
        color_map.extend((0u16..16).map(|value| (15 - value) * 17 * 257));
        color_map.extend((0u16..16).map(|value| value * 8 * 257));
        let file = TiffFile::from_bytes_with_options(
            build_stripped_tiff(
                1,
                1,
                &[0x00],
                &[
                    (258, 3, 1, inline_short(4)),
                    (262, 3, 1, inline_short(3)),
                    (
                        320,
                        3,
                        color_map.len() as u32,
                        color_map.iter().flat_map(|value| le_u16(*value)).collect(),
                    ),
                ],
            ),
            OpenOptions {
                decode_output_bytes: 2,
                ..OpenOptions::default()
            },
        )
        .unwrap();

        let err = file.read_decoded_image_bytes(0).unwrap_err();
        assert!(matches!(
            err,
            Error::DecodeOutputTooLarge {
                requested: 3,
                limit: 2
            }
        ));
    }

    #[test]
    fn bigtiff_ifd_entry_count_respects_parse_budget_before_body_read() {
        let mut data = bigtiff_header(16);
        data.extend_from_slice(&le_u64(2));

        let err = match TiffFile::from_bytes_with_options(
            data,
            OpenOptions {
                parse_budgets: ParseBudgets {
                    max_ifd_entries: 1,
                    ..ParseBudgets::default()
                },
                ..OpenOptions::default()
            },
        ) {
            Ok(_) => panic!("expected parse budget error"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::InvalidImageLayout(message) if message.contains("entry count"))
        );
    }

    #[test]
    fn bigtiff_tag_value_bytes_respect_parse_budget_before_value_read() {
        let mut data = bigtiff_header(16);
        data.extend_from_slice(&le_u64(1));
        data.extend_from_slice(&le_u16(256));
        data.extend_from_slice(&le_u16(1));
        data.extend_from_slice(&le_u64(9));
        data.extend_from_slice(&le_u64(1024));
        data.extend_from_slice(&le_u64(0));

        let err = match TiffFile::from_bytes_with_options(
            data,
            OpenOptions {
                parse_budgets: ParseBudgets {
                    max_tag_value_bytes: 8,
                    ..ParseBudgets::default()
                },
                ..OpenOptions::default()
            },
        ) {
            Ok(_) => panic!("expected parse budget error"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::InvalidTagValue { tag: 256, reason } if reason.contains("parse budget"))
        );
    }

    #[test]
    fn bigtiff_tag_value_bytes_respect_aggregate_parse_budget() {
        let mut data = bigtiff_header(16);
        data.extend_from_slice(&le_u64(2));
        data.extend_from_slice(&le_u16(65000));
        data.extend_from_slice(&le_u16(1));
        data.extend_from_slice(&le_u64(8));
        data.extend_from_slice(&[0x11; 8]);
        data.extend_from_slice(&le_u16(65001));
        data.extend_from_slice(&le_u16(1));
        data.extend_from_slice(&le_u64(8));
        data.extend_from_slice(&[0x22; 8]);
        data.extend_from_slice(&le_u64(0));

        let err = match TiffFile::from_bytes_with_options(
            data,
            OpenOptions {
                parse_budgets: ParseBudgets {
                    max_tag_value_bytes: 8,
                    max_metadata_value_bytes: 8,
                    ..ParseBudgets::default()
                },
                ..OpenOptions::default()
            },
        ) {
            Ok(_) => panic!("expected aggregate parse budget error"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::InvalidTagValue { tag: 65001, reason } if reason.contains("aggregate metadata"))
        );
    }

    #[test]
    fn bigtiff_ifd_chain_respects_parse_budget() {
        let mut data = bigtiff_header(16);
        data.extend_from_slice(&le_u64(0));
        data.extend_from_slice(&le_u64(32));
        data.extend_from_slice(&le_u64(0));
        data.extend_from_slice(&le_u64(0));

        let err = match TiffFile::from_bytes_with_options(
            data,
            OpenOptions {
                parse_budgets: ParseBudgets {
                    max_ifds: 1,
                    ..ParseBudgets::default()
                },
                ..OpenOptions::default()
            },
        ) {
            Ok(_) => panic!("expected parse budget error"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Other(message) if message.contains("parse budget")));
    }

    #[test]
    fn rejects_bigtiff_long8_dimension_that_exceeds_u32() {
        let mut data = bigtiff_header(16);
        data.extend_from_slice(&le_u64(2));
        data.extend_from_slice(&le_u16(256));
        data.extend_from_slice(&le_u16(16));
        data.extend_from_slice(&le_u64(1));
        data.extend_from_slice(&le_u64(u64::from(u32::MAX) + 2));
        data.extend_from_slice(&le_u16(257));
        data.extend_from_slice(&le_u16(16));
        data.extend_from_slice(&le_u64(1));
        data.extend_from_slice(&le_u64(1));
        data.extend_from_slice(&le_u64(0));

        let file = TiffFile::from_bytes(data).unwrap();
        let err = file.ifd(0).unwrap().raster_layout().unwrap_err();
        assert!(
            matches!(err, Error::InvalidImageLayout(message) if message.contains("dimensions"))
        );
    }

    #[test]
    fn oversized_strip_byte_count_is_rejected_before_payload_read() {
        let data = build_stripped_tiff(
            2,
            2,
            &[1, 2, 3, 4],
            &[(279, 4, 1, le_u32(u32::MAX).to_vec())],
        );
        let source = Arc::new(CountingSource::new(data));
        let file = TiffFile::from_source(source.clone()).unwrap();
        source.reset_reads();

        let err = file.read_image_bytes(0).unwrap_err();
        assert!(err.to_string().contains("block read budget"));
        assert_eq!(source.reads(), 0);
    }

    #[test]
    fn oversized_tile_byte_count_is_rejected_before_payload_read() {
        let mut data = build_tiled_tiff(2, 2, 2, 2, &[&[1, 2, 3, 4]]);
        overwrite_classic_inline_long_tag(&mut data, 325, u32::MAX);
        let source = Arc::new(CountingSource::new(data));
        let file = TiffFile::from_source(source.clone()).unwrap();
        source.reset_reads();

        let err = file.read_image_bytes(0).unwrap_err();
        assert!(err.to_string().contains("block read budget"));
        assert_eq!(source.reads(), 0);
    }

    #[test]
    fn huge_planar_tile_count_overflow_is_rejected_without_panicking() {
        let data = build_tiled_tiff_with_overrides(
            u32::MAX,
            u32::MAX,
            1,
            1,
            &[&[0]],
            &[(277, 3, 1, inline_short(2)), (284, 3, 1, inline_short(2))],
        );
        let file = TiffFile::from_bytes(data).unwrap();

        let err = file.read_window_bytes(0, 0, 0, 1, 1).unwrap_err();
        assert!(
            matches!(err, Error::InvalidImageLayout(message) if message.contains("tile count"))
        );
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
    fn reads_single_chunky_band_and_window() {
        let data = build_stripped_tiff(
            2,
            2,
            &[
                1, 10, 100, //
                2, 20, 110, //
                3, 30, 120, //
                4, 40, 130,
            ],
            &[
                (262, 3, 1, inline_short(2)),
                (277, 3, 1, inline_short(3)),
                (279, 4, 1, le_u32(12).to_vec()),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();

        let green = file.read_band::<u8>(0, 1).unwrap();
        assert_eq!(green.shape(), &[2, 2]);
        let (green_values, offset) = green.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(green_values, vec![10, 20, 30, 40]);

        let blue_window = file.read_band_window::<u8>(0, 2, 0, 1, 2, 1).unwrap();
        assert_eq!(blue_window.shape(), &[2, 1]);
        let (blue_values, offset) = blue_window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(blue_values, vec![110, 130]);

        let err = file.read_band::<u8>(0, 3).unwrap_err();
        assert!(matches!(
            err,
            Error::BandIndexOutOfBounds {
                index: 3,
                band_count: 3
            }
        ));
    }

    #[test]
    fn planar_band_reads_only_requested_plane() {
        let data = build_planar_stripped_tiff(
            2,
            2,
            &[&[1, 2, 3, 4], &[10, 20, 30, 40], &[100, 110, 120, 130]],
        );
        let source = Arc::new(CountingSource::new(data));
        let file = TiffFile::from_source(source.clone()).unwrap();
        source.reset_reads();

        let blue = file.read_band::<u8>(0, 2).unwrap();
        assert_eq!(blue.shape(), &[2, 2]);
        let (values, offset) = blue.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![100, 110, 120, 130]);
        assert_eq!(source.reads(), 1);
    }

    #[test]
    fn keeps_subbyte_palette_reads_raw_and_offers_explicit_decoded_pixels() {
        let mut color_map = Vec::new();
        color_map.extend((0u16..16).map(|value| value * 17 * 257));
        color_map.extend((0u16..16).map(|value| (15 - value) * 17 * 257));
        color_map.extend((0u16..16).map(|value| value * 8 * 257));
        let data = build_stripped_tiff(
            4,
            1,
            &[0x01, 0x23],
            &[
                (258, 3, 1, inline_short(4)),
                (262, 3, 1, inline_short(3)),
                (
                    320,
                    3,
                    color_map.len() as u32,
                    color_map.iter().flat_map(|value| le_u16(*value)).collect(),
                ),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();

        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[1, 4]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![0, 1, 2, 3]);

        let image = file.read_decoded_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[1, 4, 3]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(
            values,
            vec![
                0, 255, 0, //
                17, 238, 8, //
                34, 221, 16, //
                51, 204, 24
            ]
        );

        let sample_bytes = file.read_image_bytes(0).unwrap();
        assert_eq!(sample_bytes, vec![0, 1, 2, 3]);
    }

    #[test]
    fn keeps_subsampled_ycbcr_reads_raw_and_offers_explicit_decoded_pixels() {
        let data = build_stripped_tiff(
            2,
            2,
            &[10u8, 20, 30, 40, 128, 128],
            &[
                (
                    258,
                    3,
                    3,
                    [8u16, 8, 8].into_iter().flat_map(le_u16).collect(),
                ),
                (262, 3, 1, inline_short(6)),
                (277, 3, 1, inline_short(3)),
                (530, 3, 2, [2u16, 2].into_iter().flat_map(le_u16).collect()),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();

        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[2, 2, 3]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(
            values,
            vec![
                10, 128, 128, //
                20, 128, 128, //
                30, 128, 128, //
                40, 128, 128
            ]
        );

        let image = file.read_decoded_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[2, 2, 3]);
        let (rgb, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(
            rgb,
            vec![
                10, 10, 10, //
                20, 20, 20, //
                30, 30, 30, //
                40, 40, 40
            ]
        );

        let samples = file.read_image::<u8>(0).unwrap();
        let (values, offset) = samples.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(
            values,
            vec![
                10, 128, 128, //
                20, 128, 128, //
                30, 128, 128, //
                40, 128, 128
            ]
        );
    }

    #[test]
    fn accepts_long_typed_bits_per_sample_and_sample_format() {
        // Nonconforming writers store BitsPerSample/SampleFormat as LONG.
        let data = build_stripped_tiff(
            1,
            1,
            &[0x34, 0x12],
            &[
                (258, 4, 1, le_u32(16).to_vec()),
                (339, 4, 1, le_u32(1).to_vec()),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u16>(0).unwrap();
        assert_eq!(image[[0, 0]], 0x1234);
    }

    #[test]
    fn rejects_unexpected_bits_per_sample_tag_type() {
        let data = build_stripped_tiff(1, 1, &[7], &[(258, 11, 1, 16f32.to_le_bytes().to_vec())]);
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image_bytes(0).unwrap_err();
        assert!(
            matches!(error, Error::UnexpectedTagType { tag: 258, .. }),
            "{error}"
        );
    }

    #[cfg(feature = "webp")]
    #[test]
    fn decodes_webp_compressed_rgb_tiles() {
        let (tile_w, tile_h) = (16usize, 16usize);
        let mut rgb = vec![0u8; tile_w * tile_h * 3];
        for row in 0..tile_h {
            for col in 0..tile_w {
                let base = (row * tile_w + col) * 3;
                rgb[base] = (row * 16) as u8;
                rgb[base + 1] = (col * 16) as u8;
                rgb[base + 2] = ((row + col) * 8) as u8;
            }
        }

        // Lossless WebP payload for the tile.
        let mut webp = Vec::new();
        image_webp::WebPEncoder::new(&mut webp)
            .encode(
                &rgb,
                tile_w as u32,
                tile_h as u32,
                image_webp::ColorType::Rgb8,
            )
            .unwrap();

        let data = build_tiled_tiff_with_overrides(
            16,
            16,
            16,
            16,
            &[&webp],
            &[
                (
                    258,
                    3,
                    3,
                    [8u16, 8, 8].into_iter().flat_map(le_u16).collect(),
                ),
                (259, 3, 1, inline_short(50001)),
                (262, 3, 1, inline_short(2)),
                (277, 3, 1, inline_short(3)),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[16, 16, 3]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, rgb);
    }

    #[cfg(feature = "webp")]
    #[test]
    fn rejects_webp_payload_dimensions_that_do_not_match_tile() {
        let rgb = vec![17u8; 8 * 32 * 3];
        let mut webp = Vec::new();
        image_webp::WebPEncoder::new(&mut webp)
            .encode(&rgb, 8, 32, image_webp::ColorType::Rgb8)
            .unwrap();

        let data = build_tiled_tiff_with_overrides(
            16,
            16,
            16,
            16,
            &[&webp],
            &[
                (
                    258,
                    3,
                    3,
                    [8u16, 8, 8].into_iter().flat_map(le_u16).collect(),
                ),
                (259, 3, 1, inline_short(50001)),
                (262, 3, 1, inline_short(2)),
                (277, 3, 1, inline_short(3)),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image::<u8>(0).unwrap_err();
        assert!(error.to_string().contains("dimensions 8x32"), "{error}");
    }

    #[test]
    fn fill_order_2_reverses_uncompressed_bytes() {
        let pixels = [0x01u8, 0x02, 0x80, 0xf0];
        let stored = pixels.map(u8::reverse_bits);
        let fill_order = (266, 3, 1, inline_short(2));

        let file = TiffFile::from_bytes(build_stripped_tiff(2, 2, &stored, &[fill_order])).unwrap();

        assert_eq!(file.ifd(0).unwrap().fill_order(), 2);
        assert_eq!(file.read_image_bytes(0).unwrap(), pixels);
    }

    #[test]
    fn fill_order_2_reverses_bytes_before_decompressing() {
        let pixels: Vec<u8> = (0..64).collect();
        let mut encoder = ZlibEncoder::new(Vec::new(), FlateCompression::default());
        encoder.write_all(&pixels).unwrap();
        let stored: Vec<u8> = encoder
            .finish()
            .unwrap()
            .into_iter()
            .map(u8::reverse_bits)
            .collect();
        let overrides = [(259, 3, 1, inline_short(8)), (266, 3, 1, inline_short(2))];

        let file = TiffFile::from_bytes(build_stripped_tiff(8, 8, &stored, &overrides)).unwrap();

        assert_eq!(file.read_image_bytes(0).unwrap(), pixels);
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn fill_order_2_does_not_apply_to_jpeg() {
        let pixels = vec![128u8; 64];
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 100)
            .encode(&pixels, 8, 8, jpeg_encoder::ColorType::Luma)
            .unwrap();
        let overrides = [
            (259, 3, 1, inline_short(7)),
            (262, 3, 1, inline_short(1)),
            (266, 3, 1, inline_short(2)),
        ];

        let file = TiffFile::from_bytes(build_stripped_tiff(8, 8, &jpeg, &overrides)).unwrap();

        assert_eq!(file.read_image_bytes(0).unwrap(), pixels);
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn jpeg_color_conversion_follows_the_photometric_tag() {
        // A JFIF stream with component IDs 1, 2, 3, which a standalone JPEG
        // decoder would always treat as YCbCr. Pure red is (76, 85, 255) there.
        let pixels = [255u8, 0, 0].repeat(64);
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 100)
            .encode(&pixels, 8, 8, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        let read_first_pixel = |photometric: u16| {
            let overrides = [
                (258, 3, 3, [8, 0, 8, 0, 8, 0].to_vec()),
                (259, 3, 1, inline_short(7)),
                (262, 3, 1, inline_short(photometric)),
                (277, 3, 1, inline_short(3)),
            ];
            let bytes = build_stripped_tiff(8, 8, &jpeg, &overrides);
            let pixels = TiffFile::from_bytes(bytes)
                .unwrap()
                .read_image_bytes(0)
                .unwrap();
            [pixels[0], pixels[1], pixels[2]]
        };
        let assert_near = |actual: [u8; 3], expected: [u8; 3]| {
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!(actual.abs_diff(expected) <= 2, "{actual} vs {expected}");
            }
        };

        const PHOTOMETRIC_RGB: u16 = 2;
        const PHOTOMETRIC_YCBCR: u16 = 6;
        assert_near(read_first_pixel(PHOTOMETRIC_YCBCR), [255, 0, 0]);
        // Tagged RGB means the components are already RGB, so they pass through.
        assert_near(read_first_pixel(PHOTOMETRIC_RGB), [76, 85, 255]);
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn rejects_jpeg_payload_dimensions_that_do_not_match_tile() {
        let pixels = vec![17u8; 8 * 32];
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 80)
            .encode(&pixels, 8, 32, jpeg_encoder::ColorType::Luma)
            .unwrap();

        // The pixel count matches 16x16, so a byte-length-only validation
        // would accept the block and silently reinterpret its row geometry.
        let data = build_tiled_tiff_with_overrides(
            16,
            16,
            16,
            16,
            &[&jpeg],
            &[(259, 3, 1, inline_short(7))],
        );
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image::<u8>(0).unwrap_err();
        assert!(error.to_string().contains("dimensions 8x32"), "{error}");
    }

    #[test]
    fn ycbcr_decode_honors_reference_black_white_ranges() {
        // BT.601 video-range references: luma 16..235, chroma 128 +/- 112.
        let reference: [u32; 12] = [16, 1, 235, 1, 128, 1, 240, 1, 128, 1, 240, 1];
        let data = build_stripped_tiff(
            1,
            1,
            &[126u8, 201, 190],
            &[
                (
                    258,
                    3,
                    3,
                    [8u16, 8, 8].into_iter().flat_map(le_u16).collect(),
                ),
                (262, 3, 1, inline_short(6)),
                (277, 3, 1, inline_short(3)),
                (532, 5, 6, reference.into_iter().flat_map(le_u32).collect()),
            ],
        );
        let file = TiffFile::from_bytes(data).unwrap();

        let image = file.read_decoded_image::<u8>(0).unwrap();
        let (rgb, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        // Expected values follow the TIFF 6.0 / libtiff formula, which scales
        // each chroma delta by 127/(ReferenceMax - ReferenceZero) for 8-bit
        // samples rather than by the full-scale denominator.
        assert_eq!(rgb, vec![227, 49, 255]);
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
    fn reads_lerc_f32_strip() {
        let mut blob = build_lerc2_header_v2(2, 2, 4, 6, 0.0, 1.0, 4.0, 1 + 16);
        blob.extend_from_slice(&0u32.to_le_bytes());
        blob.push(1);
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            blob.extend_from_slice(&value.to_le_bytes());
        }

        let data = build_lerc_tiff(2, 2, &blob, 32, 3, 1, None);
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<f32>(0).unwrap();
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn non_lerc_read_ignores_irrelevant_lerc_parameters() {
        let data = build_stripped_tiff(1, 1, &[9], &[(50674, 3, 1, inline_short(1))]);
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.into_raw_vec_and_offset().0, vec![9]);
    }

    #[test]
    fn reads_lerc_masked_f32_strip_as_nan() {
        let mask = [1u8, 0, 1, 1];
        let encoded_mask = encode_mask_rle(&mask);
        let mut blob =
            build_lerc2_header_v2(2, 2, 3, 6, 0.0, 1.0, 4.0, encoded_mask.len() + 1 + 12);
        blob.extend_from_slice(&(encoded_mask.len() as u32).to_le_bytes());
        blob.extend_from_slice(&encoded_mask);
        blob.push(1);
        for value in [1.0f32, 3.0, 4.0] {
            blob.extend_from_slice(&value.to_le_bytes());
        }

        let data = build_lerc_tiff(2, 2, &blob, 32, 3, 1, None);
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<f32>(0).unwrap();
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 4.0);
    }

    #[test]
    fn reads_lerc_chunky_rgb_band_set_strip() {
        let mut red = build_lerc2_header_v2(2, 1, 2, 1, 0.0, 1.0, 1.0, 0);
        red.extend_from_slice(&0u32.to_le_bytes());
        let mut green = build_lerc2_header_v2(2, 1, 2, 1, 0.0, 2.0, 2.0, 0);
        green.extend_from_slice(&0u32.to_le_bytes());
        let mut blue = build_lerc2_header_v2(2, 1, 2, 1, 0.0, 3.0, 3.0, 0);
        blue.extend_from_slice(&0u32.to_le_bytes());

        let mut blob = red;
        blob.extend_from_slice(&green);
        blob.extend_from_slice(&blue);

        let data = build_lerc_tiff(2, 1, &blob, 8, 1, 3, None);
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[1, 2, 3]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn reads_lerc_chunky_rgb_depth_blob_strip() {
        let mut blob = build_lerc2_header_v4(2, 1, 3, 2, 1, 0.0, 1.0, 6.0, 6 + 6 + 1 + 6);
        blob.extend_from_slice(&0u32.to_le_bytes());
        for value in [1u8, 2, 3] {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        for value in [4u8, 5, 6] {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        blob.push(1);
        blob.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let blob = finalize_lerc2_v4_with_checksum(blob);

        let data = build_lerc_tiff(2, 1, &blob, 8, 1, 3, Some([4, 0]));
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<u8>(0).unwrap();
        assert_eq!(image.shape(), &[1, 2, 3]);
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rejects_lerc2_blob_size_before_checksum_range_without_panicking() {
        let mut blob = build_lerc2_header_v4(1, 1, 1, 1, 1, 0.0, 1.0, 1.0, 0);
        blob[34..38].copy_from_slice(&8i32.to_le_bytes());

        let data = build_lerc_tiff(1, 1, &blob, 8, 1, 1, Some([4, 0]));
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image_bytes(0).unwrap_err();
        assert!(error.to_string().contains("invalid Lerc2 v4 blob size 8"));
    }

    #[test]
    fn rejects_lerc2_header_dimensions_before_allocating_mask() {
        let mut blob = build_lerc2_header_v2(u32::MAX, u32::MAX, 1, 1, 0.0, 0.0, 1.0, 4);
        blob.extend_from_slice(&4u32.to_le_bytes());
        blob.extend_from_slice(&[0, 0, 0, 0]);

        let data = build_lerc_tiff(1, 1, &blob, 8, 1, 1, None);
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image_bytes(0).unwrap_err();
        assert!(error.to_string().contains("LERC raster dimensions"));
    }

    #[test]
    fn rejects_truncated_lerc2_header_dimensions_before_decoder() {
        let blob = build_lerc2_header_v2(u32::MAX, u32::MAX, 1, 1, 0.0, 0.0, 1.0, 64);

        let data = build_lerc_tiff(1, 1, &blob, 8, 1, 1, None);
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image_bytes(0).unwrap_err();
        assert!(error.to_string().contains("LERC raster dimensions"));
    }

    #[test]
    fn reads_lerc_deflate_f32_strip() {
        let mut blob = build_lerc2_header_v2(2, 2, 4, 6, 0.0, 1.0, 4.0, 1 + 16);
        blob.extend_from_slice(&0u32.to_le_bytes());
        blob.push(1);
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            blob.extend_from_slice(&value.to_le_bytes());
        }

        let mut encoder = ZlibEncoder::new(Vec::new(), FlateCompression::default());
        std::io::Write::write_all(&mut encoder, &blob).unwrap();
        let compressed = encoder.finish().unwrap();

        let data = build_lerc_tiff(2, 2, &compressed, 32, 3, 1, Some([2, 1]));
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<f32>(0).unwrap();
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn reads_lerc_zstd_f32_strip() {
        let mut blob = build_lerc2_header_v2(2, 2, 4, 6, 0.0, 1.0, 4.0, 1 + 16);
        blob.extend_from_slice(&0u32.to_le_bytes());
        blob.push(1);
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            blob.extend_from_slice(&value.to_le_bytes());
        }

        let compressed = ruzstd::encoding::compress_to_vec(
            &blob[..],
            ruzstd::encoding::CompressionLevel::Fastest,
        );
        let data = build_lerc_tiff(2, 2, &compressed, 32, 3, 1, Some([2, 2]));
        let file = TiffFile::from_bytes(data).unwrap();
        let image = file.read_image::<f32>(0).unwrap();
        let (values, offset) = image.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
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

    #[test]
    fn rejects_zero_rows_per_strip_without_panicking() {
        let data = build_stripped_tiff(2, 2, &[1, 2, 3, 4], &[(278, 4, 1, le_u32(0).to_vec())]);
        let file = TiffFile::from_bytes(data).unwrap();
        let error = file.read_image_bytes(0).unwrap_err();
        assert!(error.to_string().contains("RowsPerStrip"));
    }
}
