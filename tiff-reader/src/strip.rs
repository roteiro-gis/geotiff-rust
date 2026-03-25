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
use crate::{GdalStructuralMetadata, Window};

const TAG_JPEG_TABLES: u16 = 347;

pub(crate) fn read_window(
    source: &dyn TiffSource,
    ifd: &Ifd,
    byte_order: ByteOrder,
    cache: &BlockCache,
    window: Window,
    gdal_structural_metadata: Option<&GdalStructuralMetadata>,
) -> Result<Vec<u8>> {
    let layout = ifd.raster_layout()?;
    if window.is_empty() {
        return Ok(Vec::new());
    }

    let output_len = window.output_len(&layout)?;
    let mut output = vec![0u8; output_len];
    let window_row_end = window.row_end();
    let output_row_bytes = window.cols * layout.pixel_stride_bytes();

    let specs = collect_strip_specs(ifd, &layout)?;
    let relevant_specs: Vec<_> = specs
        .iter()
        .copied()
        .filter(|spec| {
            let spec_row_end = spec.row_start + spec.rows_in_strip;
            spec.row_start < window_row_end && spec_row_end > window.row_off
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let decoded_blocks: Result<Vec<_>> = relevant_specs
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
    let decoded_blocks: Result<Vec<_>> = relevant_specs
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
        let block_row_end = spec.row_start + spec.rows_in_strip;
        let copy_row_start = spec.row_start.max(window.row_off);
        let copy_row_end = block_row_end.min(window_row_end);

        if layout.planar_configuration == 1 {
            let src_row_bytes = layout.row_bytes();
            let copy_bytes_per_row = window.cols * layout.pixel_stride_bytes();
            for row in copy_row_start..copy_row_end {
                let src_row_index = row - spec.row_start;
                let dest_row_index = row - window.row_off;
                let src_offset =
                    src_row_index * src_row_bytes + window.col_off * layout.pixel_stride_bytes();
                let dest_offset = dest_row_index * output_row_bytes;
                output[dest_offset..dest_offset + copy_bytes_per_row]
                    .copy_from_slice(&block[src_offset..src_offset + copy_bytes_per_row]);
            }
        } else {
            let src_row_bytes = layout.sample_plane_row_bytes();
            for row in copy_row_start..copy_row_end {
                let src_row_index = row - spec.row_start;
                let dest_row_index = row - window.row_off;
                let src_row =
                    &block[src_row_index * src_row_bytes..(src_row_index + 1) * src_row_bytes];
                let dest_row = &mut output
                    [dest_row_index * output_row_bytes..(dest_row_index + 1) * output_row_bytes];
                for col in window.col_off..window.col_end() {
                    let src = &src_row
                        [col * layout.bytes_per_sample..(col + 1) * layout.bytes_per_sample];
                    let dest_col_index = col - window.col_off;
                    let pixel_base = dest_col_index * layout.pixel_stride_bytes()
                        + spec.plane * layout.bytes_per_sample;
                    dest_row[pixel_base..pixel_base + layout.bytes_per_sample].copy_from_slice(src);
                }
            }
        }
    }

    Ok(output)
}

fn collect_strip_specs(ifd: &Ifd, layout: &RasterLayout) -> Result<Vec<StripBlockSpec>> {
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

    let rows_per_strip = ifd.rows_per_strip().unwrap_or(ifd.height());
    if rows_per_strip == 0 {
        return Err(Error::InvalidImageLayout(
            "RowsPerStrip must be greater than zero".into(),
        ));
    }
    let rows_per_strip = rows_per_strip as usize;
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

    Ok((0..expected)
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
        .collect())
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
    let mut decoded = filters::decompress(
        ifd.compression(),
        &compressed,
        spec.index,
        jpeg_tables,
        expected_len,
    )?;
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
