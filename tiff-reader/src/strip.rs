//! Strip-based data access for TIFF images.

use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::cache::{BlockCache, BlockKey, BlockKind};
use crate::error::{Error, Result};
use crate::filters;
use crate::header::ByteOrder;
use crate::ifd::{Ifd, RasterLayout};
use crate::source::TiffSource;
use crate::GdalStructuralMetadata;

const TAG_JPEG_TABLES: u16 = 347;

pub(crate) fn read_image(
    source: &dyn TiffSource,
    ifd: &Ifd,
    byte_order: ByteOrder,
    cache: &BlockCache,
    gdal_structural_metadata: Option<&GdalStructuralMetadata>,
) -> Result<Vec<u8>> {
    let layout = ifd.raster_layout()?;
    let offsets = ifd
        .strip_offsets()
        .ok_or(Error::TagNotFound(crate::ifd::TAG_STRIP_OFFSETS))?;
    let counts = ifd
        .strip_byte_counts()
        .ok_or(Error::TagNotFound(crate::ifd::TAG_STRIP_BYTE_COUNTS))?;
    if offsets.len() != counts.len() {
        return Err(Error::InvalidImageLayout(format!(
            "StripOffsets has {} entries but StripByteCounts has {}",
            offsets.len(),
            counts.len()
        )));
    }

    let rows_per_strip = ifd.rows_per_strip().unwrap_or(ifd.height()) as usize;
    let strips_per_plane = layout.height.div_ceil(rows_per_strip);
    let expected = match layout.planar_configuration {
        1 => strips_per_plane,
        2 => strips_per_plane * layout.samples_per_pixel,
        planar => return Err(Error::UnsupportedPlanarConfiguration(planar)),
    };
    if offsets.len() != expected {
        return Err(Error::InvalidImageLayout(format!(
            "expected {expected} strips, found {}",
            offsets.len()
        )));
    }

    let output_len = layout
        .row_bytes()
        .checked_mul(layout.height)
        .ok_or_else(|| Error::InvalidImageLayout("decoded raster size overflows usize".into()))?;
    let mut output = vec![0u8; output_len];
    let specs: Vec<_> = (0..expected)
        .map(|strip_index| {
            let plane = if layout.planar_configuration == 1 {
                0
            } else {
                strip_index / strips_per_plane
            };
            let plane_strip_index = if layout.planar_configuration == 1 {
                strip_index
            } else {
                strip_index % strips_per_plane
            };
            let row_start = plane_strip_index * rows_per_strip;
            let rows_in_strip = rows_per_strip.min(layout.height.saturating_sub(row_start));
            StripBlockSpec {
                index: strip_index,
                plane,
                row_start,
                offset: offsets[strip_index],
                byte_count: counts[strip_index],
                rows_in_strip,
            }
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let decoded_blocks: Result<Vec<_>> = specs
        .iter()
        .map(|&spec| {
            read_strip_block(
                source,
                ifd,
                byte_order,
                cache,
                spec,
                &layout,
                gdal_structural_metadata,
            )
            .map(|block| (spec, block))
        })
        .collect();

    #[cfg(feature = "rayon")]
    let decoded_blocks: Result<Vec<_>> = specs
        .par_iter()
        .map(|&spec| {
            read_strip_block(
                source,
                ifd,
                byte_order,
                cache,
                spec,
                &layout,
                gdal_structural_metadata,
            )
            .map(|block| (spec, block))
        })
        .collect();

    for (spec, block) in decoded_blocks? {
        let block = &*block;

        if layout.planar_configuration == 1 {
            let dest_offset = spec
                .row_start
                .checked_mul(layout.row_bytes())
                .ok_or_else(|| Error::InvalidImageLayout("row offset overflows usize".into()))?;
            let block_len = spec
                .rows_in_strip
                .checked_mul(layout.row_bytes())
                .ok_or_else(|| {
                    Error::InvalidImageLayout("strip byte length overflows usize".into())
                })?;
            output[dest_offset..dest_offset + block_len].copy_from_slice(&block[..block_len]);
        } else {
            let src_row_bytes = layout.sample_plane_row_bytes();
            for row in 0..spec.rows_in_strip {
                let src_row = &block[row * src_row_bytes..(row + 1) * src_row_bytes];
                let dest_row_offset = (spec.row_start + row)
                    .checked_mul(layout.row_bytes())
                    .ok_or_else(|| {
                        Error::InvalidImageLayout("row offset overflows usize".into())
                    })?;
                let dest_row = &mut output[dest_row_offset..dest_row_offset + layout.row_bytes()];
                for col in 0..layout.width {
                    let src = &src_row
                        [col * layout.bytes_per_sample..(col + 1) * layout.bytes_per_sample];
                    let pixel_base =
                        col * layout.pixel_stride_bytes() + spec.plane * layout.bytes_per_sample;
                    dest_row[pixel_base..pixel_base + layout.bytes_per_sample].copy_from_slice(src);
                }
            }
        }
    }

    Ok(output)
}

#[derive(Clone, Copy)]
struct StripBlockSpec {
    index: usize,
    plane: usize,
    row_start: usize,
    offset: u64,
    byte_count: u64,
    rows_in_strip: usize,
}

fn read_strip_block(
    source: &dyn TiffSource,
    ifd: &Ifd,
    byte_order: ByteOrder,
    cache: &BlockCache,
    spec: StripBlockSpec,
    layout: &RasterLayout,
    gdal_structural_metadata: Option<&GdalStructuralMetadata>,
) -> Result<Arc<Vec<u8>>> {
    let cache_key = BlockKey {
        ifd_index: ifd.index,
        kind: BlockKind::Strip,
        block_index: spec.index,
    };
    if let Some(cached) = cache.get(&cache_key) {
        return Ok(cached);
    }

    let compressed = if let Some(bytes) = source.as_slice() {
        let start = usize::try_from(spec.offset).map_err(|_| Error::OffsetOutOfBounds {
            offset: spec.offset,
            length: spec.byte_count,
            data_len: bytes.len() as u64,
        })?;
        let len = usize::try_from(spec.byte_count).map_err(|_| Error::OffsetOutOfBounds {
            offset: spec.offset,
            length: spec.byte_count,
            data_len: bytes.len() as u64,
        })?;
        let end = start.checked_add(len).ok_or(Error::OffsetOutOfBounds {
            offset: spec.offset,
            length: spec.byte_count,
            data_len: bytes.len() as u64,
        })?;
        if end > bytes.len() {
            return Err(Error::OffsetOutOfBounds {
                offset: spec.offset,
                length: spec.byte_count,
                data_len: bytes.len() as u64,
            });
        }
        bytes[start..end].to_vec()
    } else {
        let len = usize::try_from(spec.byte_count).map_err(|_| Error::OffsetOutOfBounds {
            offset: spec.offset,
            length: spec.byte_count,
            data_len: source.len(),
        })?;
        source.read_exact_at(spec.offset, len)?
    };

    let compressed = match gdal_structural_metadata {
        Some(metadata) => metadata
            .unwrap_block(&compressed, byte_order, spec.offset)?
            .to_vec(),
        None => compressed,
    };

    let jpeg_tables = ifd
        .tag(TAG_JPEG_TABLES)
        .and_then(|tag| tag.value.as_bytes());
    let mut decoded = filters::decompress(ifd.compression(), &compressed, spec.index, jpeg_tables)?;
    let samples = if layout.planar_configuration == 1 {
        layout.samples_per_pixel
    } else {
        1
    };
    let expected_row_bytes = layout.width * samples * layout.bytes_per_sample;
    let expected_len = spec
        .rows_in_strip
        .checked_mul(expected_row_bytes)
        .ok_or_else(|| Error::InvalidImageLayout("strip size overflows usize".into()))?;
    if decoded.len() < expected_len {
        return Err(Error::DecompressionFailed {
            index: spec.index,
            reason: format!(
                "decoded strip is too small: expected at least {expected_len} bytes, found {}",
                decoded.len()
            ),
        });
    }
    if decoded.len() > expected_len {
        decoded.truncate(expected_len);
    }

    for row in decoded.chunks_exact_mut(expected_row_bytes) {
        filters::fix_endianness_and_predict(
            row,
            layout.bits_per_sample,
            samples as u16,
            byte_order,
            layout.predictor,
        )?;
    }

    Ok(cache.insert(cache_key, decoded))
}
