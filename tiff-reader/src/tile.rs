//! Tile-based data access for TIFF images.

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
    let window_col_end = window.col_end();
    let output_row_bytes = window.cols * layout.pixel_stride_bytes();

    let specs = collect_tile_specs(ifd, &layout)?;
    let relevant_specs: Vec<_> = specs
        .iter()
        .copied()
        .filter(|spec| {
            let spec_row_end = spec.y + spec.rows_in_tile;
            let spec_col_end = spec.x + spec.cols_in_tile;
            spec.y < window_row_end
                && spec_row_end > window.row_off
                && spec.x < window_col_end
                && spec_col_end > window.col_off
        })
        .collect();

    #[cfg(feature = "rayon")]
    let decoded_blocks: Result<Vec<_>> = relevant_specs
        .par_iter()
        .map(|&spec| {
            read_tile_block(
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

    #[cfg(not(feature = "rayon"))]
    let decoded_blocks: Result<Vec<_>> = relevant_specs
        .iter()
        .map(|&spec| {
            read_tile_block(
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
        let copy_row_start = spec.y.max(window.row_off);
        let copy_row_end = (spec.y + spec.rows_in_tile).min(window_row_end);
        let copy_col_start = spec.x.max(window.col_off);
        let copy_col_end = (spec.x + spec.cols_in_tile).min(window_col_end);

        let src_row_bytes = spec.tile_width
            * if layout.planar_configuration == 1 {
                layout.pixel_stride_bytes()
            } else {
                layout.bytes_per_sample
            };

        if layout.planar_configuration == 1 {
            let copy_bytes_per_row = (copy_col_end - copy_col_start) * layout.pixel_stride_bytes();
            for row in copy_row_start..copy_row_end {
                let src_row_index = row - spec.y;
                let dest_row_index = row - window.row_off;
                let src_offset = src_row_index * src_row_bytes
                    + (copy_col_start - spec.x) * layout.pixel_stride_bytes();
                let dest_offset = dest_row_index * output_row_bytes
                    + (copy_col_start - window.col_off) * layout.pixel_stride_bytes();
                output[dest_offset..dest_offset + copy_bytes_per_row]
                    .copy_from_slice(&block[src_offset..src_offset + copy_bytes_per_row]);
            }
        } else {
            for row in copy_row_start..copy_row_end {
                let src_row_index = row - spec.y;
                let dest_row_index = row - window.row_off;
                let src_row =
                    &block[src_row_index * src_row_bytes..(src_row_index + 1) * src_row_bytes];
                let dest_row = &mut output
                    [dest_row_index * output_row_bytes..(dest_row_index + 1) * output_row_bytes];
                for col in copy_col_start..copy_col_end {
                    let src = &src_row[(col - spec.x) * layout.bytes_per_sample
                        ..(col - spec.x + 1) * layout.bytes_per_sample];
                    let pixel_base = (col - window.col_off) * layout.pixel_stride_bytes()
                        + spec.plane * layout.bytes_per_sample;
                    dest_row[pixel_base..pixel_base + layout.bytes_per_sample].copy_from_slice(src);
                }
            }
        }
    }

    Ok(output)
}

fn collect_tile_specs(ifd: &Ifd, layout: &RasterLayout) -> Result<Vec<TileBlockSpec>> {
    let tile_width = ifd
        .tile_width()
        .ok_or(Error::TagNotFound(crate::ifd::TAG_TILE_WIDTH))? as usize;
    let tile_height = ifd
        .tile_height()
        .ok_or(Error::TagNotFound(crate::ifd::TAG_TILE_LENGTH))? as usize;
    if tile_width == 0 || tile_height == 0 {
        return Err(Error::InvalidImageLayout(
            "tile width and height must be greater than zero".into(),
        ));
    }

    let offsets = ifd
        .tile_offsets()
        .ok_or(Error::TagNotFound(crate::ifd::TAG_TILE_OFFSETS))?;
    let counts = ifd
        .tile_byte_counts()
        .ok_or(Error::TagNotFound(crate::ifd::TAG_TILE_BYTE_COUNTS))?;
    if offsets.len() != counts.len() {
        return Err(Error::InvalidImageLayout(format!(
            "TileOffsets has {} entries but TileByteCounts has {}",
            offsets.len(),
            counts.len()
        )));
    }

    let tiles_across = layout.width.div_ceil(tile_width);
    let tiles_down = layout.height.div_ceil(tile_height);
    let tiles_per_plane = tiles_across * tiles_down;
    let expected = match layout.planar_configuration {
        1 => tiles_per_plane,
        2 => tiles_per_plane * layout.samples_per_pixel,
        planar => return Err(Error::UnsupportedPlanarConfiguration(planar)),
    };
    if offsets.len() != expected {
        return Err(Error::InvalidImageLayout(format!(
            "expected {expected} tiles, found {}",
            offsets.len()
        )));
    }

    Ok((0..expected)
        .map(|tile_index| {
            let plane = if layout.planar_configuration == 1 {
                0
            } else {
                tile_index / tiles_per_plane
            };
            let plane_tile_index = if layout.planar_configuration == 1 {
                tile_index
            } else {
                tile_index % tiles_per_plane
            };
            let tile_row = plane_tile_index / tiles_across;
            let tile_col = plane_tile_index % tiles_across;
            let x = tile_col * tile_width;
            let y = tile_row * tile_height;
            let cols_in_tile = tile_width.min(layout.width.saturating_sub(x));
            let rows_in_tile = tile_height.min(layout.height.saturating_sub(y));
            TileBlockSpec {
                index: tile_index,
                plane,
                x,
                y,
                cols_in_tile,
                rows_in_tile,
                offset: offsets[tile_index],
                byte_count: counts[tile_index],
                tile_width,
                tile_height,
            }
        })
        .collect())
}

#[derive(Clone, Copy)]
struct TileBlockSpec {
    index: usize,
    plane: usize,
    x: usize,
    y: usize,
    cols_in_tile: usize,
    rows_in_tile: usize,
    offset: u64,
    byte_count: u64,
    tile_width: usize,
    tile_height: usize,
}

fn read_tile_block(
    source: &dyn TiffSource,
    ifd: &Ifd,
    byte_order: ByteOrder,
    cache: &BlockCache,
    spec: TileBlockSpec,
    layout: &RasterLayout,
    gdal_structural_metadata: Option<&GdalStructuralMetadata>,
) -> Result<Arc<Vec<u8>>> {
    let cache_key = BlockKey {
        ifd_index: ifd.index,
        kind: BlockKind::Tile,
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
    let row_bytes = spec.tile_width * samples * layout.bytes_per_sample;
    let expected_len = spec
        .tile_height
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::InvalidImageLayout("tile size overflows usize".into()))?;
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
                "decoded tile is too small: expected at least {expected_len} bytes, found {}",
                decoded.len()
            ),
        });
    }
    if decoded.len() > expected_len {
        decoded.truncate(expected_len);
    }

    for row in decoded.chunks_exact_mut(row_bytes) {
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
