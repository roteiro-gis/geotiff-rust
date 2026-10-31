//! Tile-based data access for TIFF images.

use std::sync::Arc;

#[cfg(feature = "rayon")]
use parking_lot::Mutex;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::block_decode;
use crate::cache::{BlockCache, BlockKey, BlockKind};
use crate::error::{Error, Result};
use crate::header::ByteOrder;
use crate::ifd::{Ifd, RasterLayout};
use crate::source::TiffSource;
use crate::{
    allocate_decode_output, checked_layout_add, checked_layout_mul, copy_strided_samples,
    DecodeReadOptions, Window,
};

pub(crate) fn read_window(
    source: &dyn TiffSource,
    ifd: &Ifd,
    byte_order: ByteOrder,
    cache: &BlockCache,
    window: Window,
    options: DecodeReadOptions<'_>,
) -> Result<Vec<u8>> {
    let layout = ifd.raster_layout()?;
    if window.is_empty() {
        return Ok(Vec::new());
    }
    let ifd_offset = ifd.offset();
    let context = block_decode::BlockDecodeContext::new(ifd, layout, byte_order)?;

    let output_len = window.output_len(&layout)?;
    let mut output = allocate_decode_output(output_len, options.decode_output_bytes)?;

    let relevant_specs = collect_tile_specs_for_window(ifd, &layout, window, None)?;

    #[cfg(feature = "rayon")]
    {
        let output = Mutex::new(output.as_mut_slice());
        relevant_specs.par_iter().try_for_each(|&spec| {
            let block = read_tile_block(source, ifd_offset, cache, spec, &context, options)?;
            copy_tile_window_block(&mut output.lock(), block.as_slice(), spec, &layout, window)?;
            Ok::<(), Error>(())
        })?;
    }

    #[cfg(not(feature = "rayon"))]
    for spec in relevant_specs {
        let block = read_tile_block(source, ifd_offset, cache, spec, &context, options)?;
        copy_tile_window_block(&mut output, block.as_slice(), spec, &layout, window)?;
    }

    Ok(output)
}

pub(crate) fn read_window_band(
    source: &dyn TiffSource,
    ifd: &Ifd,
    byte_order: ByteOrder,
    cache: &BlockCache,
    window: Window,
    band_index: usize,
    options: DecodeReadOptions<'_>,
) -> Result<Vec<u8>> {
    let layout = ifd.raster_layout()?;
    if band_index >= layout.samples_per_pixel {
        return Err(Error::BandIndexOutOfBounds {
            index: band_index,
            band_count: layout.samples_per_pixel,
        });
    }
    if window.is_empty() {
        return Ok(Vec::new());
    }
    let ifd_offset = ifd.offset();
    let context = block_decode::BlockDecodeContext::new(ifd, layout, byte_order)?;

    let output_len = window.band_output_len(&layout)?;
    let mut output = allocate_decode_output(output_len, options.decode_output_bytes)?;

    let relevant_specs = collect_tile_specs_for_window(ifd, &layout, window, Some(band_index))?;

    #[cfg(feature = "rayon")]
    {
        let output = Mutex::new(output.as_mut_slice());
        relevant_specs.par_iter().try_for_each(|&spec| {
            let block = read_tile_block(source, ifd_offset, cache, spec, &context, options)?;
            copy_tile_band_window_block(
                &mut output.lock(),
                block.as_slice(),
                spec,
                &layout,
                window,
                band_index,
            )?;
            Ok::<(), Error>(())
        })?;
    }

    #[cfg(not(feature = "rayon"))]
    for spec in relevant_specs {
        let block = read_tile_block(source, ifd_offset, cache, spec, &context, options)?;
        copy_tile_band_window_block(
            &mut output,
            block.as_slice(),
            spec,
            &layout,
            window,
            band_index,
        )?;
    }

    Ok(output)
}

fn copy_tile_window_block(
    output: &mut [u8],
    block: &[u8],
    spec: TileBlockSpec,
    layout: &RasterLayout,
    window: Window,
) -> Result<()> {
    let pixel_stride = layout.checked_pixel_stride_bytes()?;
    let window_row_end = window.row_end();
    let window_col_end = window.col_end();
    let output_row_bytes = checked_layout_mul(window.cols, pixel_stride, "window row byte count")?;
    let copy_row_start = spec.y.max(window.row_off);
    let copy_row_end =
        checked_layout_add(spec.y, spec.rows_in_tile, "tile row range")?.min(window_row_end);
    let copy_col_start = spec.x.max(window.col_off);
    let copy_col_end =
        checked_layout_add(spec.x, spec.cols_in_tile, "tile column range")?.min(window_col_end);

    let src_row_bytes = checked_layout_mul(
        spec.tile_width,
        if layout.planar_configuration == 1 {
            pixel_stride
        } else {
            layout.bytes_per_sample
        },
        "tile source row byte count",
    )?;

    if layout.planar_configuration == 1 {
        let copy_bytes_per_row = checked_layout_mul(
            copy_col_end - copy_col_start,
            pixel_stride,
            "tile copy row byte count",
        )?;
        let src_col_offset = checked_layout_mul(
            copy_col_start - spec.x,
            pixel_stride,
            "tile source column offset",
        )?;
        let dest_col_offset = checked_layout_mul(
            copy_col_start - window.col_off,
            pixel_stride,
            "tile output column offset",
        )?;
        for row in copy_row_start..copy_row_end {
            let src_row_index = row - spec.y;
            let dest_row_index = row - window.row_off;
            let src_offset = checked_layout_add(
                checked_layout_mul(src_row_index, src_row_bytes, "tile source row offset")?,
                src_col_offset,
                "tile source offset",
            )?;
            let dest_offset = checked_layout_add(
                checked_layout_mul(dest_row_index, output_row_bytes, "tile output row offset")?,
                dest_col_offset,
                "tile output offset",
            )?;
            let src_end =
                checked_layout_add(src_offset, copy_bytes_per_row, "tile source copy range")?;
            let dest_end =
                checked_layout_add(dest_offset, copy_bytes_per_row, "tile output copy range")?;
            output[dest_offset..dest_end].copy_from_slice(&block[src_offset..src_end]);
        }
    } else {
        let plane_offset = checked_layout_mul(
            spec.plane,
            layout.bytes_per_sample,
            "tile plane byte offset",
        )?;
        let copy_cols = copy_col_end - copy_col_start;
        let src_col_offset = checked_layout_mul(
            copy_col_start - spec.x,
            layout.bytes_per_sample,
            "tile source column offset",
        )?;
        let dest_col_offset = checked_layout_add(
            checked_layout_mul(
                copy_col_start - window.col_off,
                pixel_stride,
                "tile output pixel offset",
            )?,
            plane_offset,
            "tile output sample offset",
        )?;
        for row in copy_row_start..copy_row_end {
            let src_row_index = row - spec.y;
            let dest_row_index = row - window.row_off;
            let src_row_offset =
                checked_layout_mul(src_row_index, src_row_bytes, "tile source row offset")?;
            let src_row_end =
                checked_layout_add(src_row_offset, src_row_bytes, "tile source row range")?;
            let dest_row_offset =
                checked_layout_mul(dest_row_index, output_row_bytes, "tile output row offset")?;
            let dest_row_end =
                checked_layout_add(dest_row_offset, output_row_bytes, "tile output row range")?;
            copy_strided_samples(
                &mut output[dest_row_offset..dest_row_end][dest_col_offset..],
                pixel_stride,
                &block[src_row_offset..src_row_end][src_col_offset..],
                layout.bytes_per_sample,
                layout.bytes_per_sample,
                copy_cols,
            );
        }
    }
    Ok(())
}

fn copy_tile_band_window_block(
    output: &mut [u8],
    block: &[u8],
    spec: TileBlockSpec,
    layout: &RasterLayout,
    window: Window,
    band_index: usize,
) -> Result<()> {
    let pixel_stride = layout.checked_pixel_stride_bytes()?;
    let window_row_end = window.row_end();
    let window_col_end = window.col_end();
    let output_row_bytes = checked_layout_mul(
        window.cols,
        layout.bytes_per_sample,
        "window band row byte count",
    )?;
    let copy_row_start = spec.y.max(window.row_off);
    let copy_row_end =
        checked_layout_add(spec.y, spec.rows_in_tile, "tile row range")?.min(window_row_end);
    let copy_col_start = spec.x.max(window.col_off);
    let copy_col_end =
        checked_layout_add(spec.x, spec.cols_in_tile, "tile column range")?.min(window_col_end);

    let src_row_bytes = checked_layout_mul(
        spec.tile_width,
        if layout.planar_configuration == 1 {
            pixel_stride
        } else {
            layout.bytes_per_sample
        },
        "tile source row byte count",
    )?;

    if layout.planar_configuration == 1 {
        let band_offset =
            checked_layout_mul(band_index, layout.bytes_per_sample, "band byte offset")?;
        let copy_cols = copy_col_end - copy_col_start;
        let src_col_offset = checked_layout_add(
            checked_layout_mul(
                copy_col_start - spec.x,
                pixel_stride,
                "tile source column offset",
            )?,
            band_offset,
            "tile source band offset",
        )?;
        let dest_col_offset = checked_layout_mul(
            copy_col_start - window.col_off,
            layout.bytes_per_sample,
            "tile output sample offset",
        )?;
        for row in copy_row_start..copy_row_end {
            let src_row_index = row - spec.y;
            let dest_row_index = row - window.row_off;
            let src_row_offset =
                checked_layout_mul(src_row_index, src_row_bytes, "tile source row offset")?;
            let src_row_end =
                checked_layout_add(src_row_offset, src_row_bytes, "tile source row range")?;
            let dest_row_offset =
                checked_layout_mul(dest_row_index, output_row_bytes, "tile output row offset")?;
            let dest_row_end =
                checked_layout_add(dest_row_offset, output_row_bytes, "tile output row range")?;
            copy_strided_samples(
                &mut output[dest_row_offset..dest_row_end][dest_col_offset..],
                layout.bytes_per_sample,
                &block[src_row_offset..src_row_end][src_col_offset..],
                pixel_stride,
                layout.bytes_per_sample,
                copy_cols,
            );
        }
    } else {
        let copy_bytes_per_row = checked_layout_mul(
            copy_col_end - copy_col_start,
            layout.bytes_per_sample,
            "tile copy row byte count",
        )?;
        let src_col_offset = checked_layout_mul(
            copy_col_start - spec.x,
            layout.bytes_per_sample,
            "tile source column offset",
        )?;
        let dest_col_offset = checked_layout_mul(
            copy_col_start - window.col_off,
            layout.bytes_per_sample,
            "tile output column offset",
        )?;
        for row in copy_row_start..copy_row_end {
            let src_row_index = row - spec.y;
            let dest_row_index = row - window.row_off;
            let src_offset = checked_layout_add(
                checked_layout_mul(src_row_index, src_row_bytes, "tile source row offset")?,
                src_col_offset,
                "tile source offset",
            )?;
            let dest_offset = checked_layout_add(
                checked_layout_mul(dest_row_index, output_row_bytes, "tile output row offset")?,
                dest_col_offset,
                "tile output offset",
            )?;
            let src_end =
                checked_layout_add(src_offset, copy_bytes_per_row, "tile source copy range")?;
            let dest_end =
                checked_layout_add(dest_offset, copy_bytes_per_row, "tile output copy range")?;
            output[dest_offset..dest_end].copy_from_slice(&block[src_offset..src_end]);
        }
    }
    Ok(())
}

fn collect_tile_specs_for_window(
    ifd: &Ifd,
    layout: &RasterLayout,
    window: Window,
    band_index: Option<usize>,
) -> Result<Vec<TileBlockSpec>> {
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
    let tiles_per_plane = tiles_across
        .checked_mul(tiles_down)
        .ok_or_else(tile_count_overflow)?;
    let expected = match layout.planar_configuration {
        1 => tiles_per_plane,
        2 => tiles_per_plane
            .checked_mul(layout.samples_per_pixel)
            .ok_or_else(tile_count_overflow)?,
        planar => return Err(Error::UnsupportedPlanarConfiguration(planar)),
    };
    if offsets.len() != expected {
        return Err(Error::InvalidImageLayout(format!(
            "expected {expected} tiles, found {}",
            offsets.len()
        )));
    }

    let first_tile_row = window.row_off / tile_height;
    let last_tile_row = window.row_end().div_ceil(tile_height).min(tiles_down);
    let first_tile_col = window.col_off / tile_width;
    let last_tile_col = window.col_end().div_ceil(tile_width).min(tiles_across);
    let plane_range = if layout.planar_configuration == 1 {
        0..1
    } else if let Some(band_index) = band_index {
        band_index..band_index + 1
    } else {
        0..layout.samples_per_pixel
    };
    let spec_count = (last_tile_row - first_tile_row)
        .saturating_mul(last_tile_col - first_tile_col)
        .saturating_mul(plane_range.end - plane_range.start);
    let mut specs = Vec::with_capacity(spec_count);

    for plane in plane_range {
        for tile_row in first_tile_row..last_tile_row {
            for tile_col in first_tile_col..last_tile_col {
                let plane_tile_index = tile_row
                    .checked_mul(tiles_across)
                    .and_then(|base| base.checked_add(tile_col))
                    .ok_or_else(tile_count_overflow)?;
                let tile_index = if layout.planar_configuration == 1 {
                    plane_tile_index
                } else {
                    plane
                        .checked_mul(tiles_per_plane)
                        .and_then(|base| base.checked_add(plane_tile_index))
                        .ok_or_else(tile_count_overflow)?
                };
                let x = tile_col * tile_width;
                let y = tile_row * tile_height;
                let cols_in_tile = tile_width.min(layout.width.saturating_sub(x));
                let rows_in_tile = tile_height.min(layout.height.saturating_sub(y));
                specs.push(TileBlockSpec {
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
                });
            }
        }
    }

    Ok(specs)
}

fn tile_count_overflow() -> Error {
    Error::InvalidImageLayout("tile count overflows usize".into())
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
    ifd_offset: u64,
    cache: &BlockCache,
    spec: TileBlockSpec,
    context: &block_decode::BlockDecodeContext<'_>,
    options: DecodeReadOptions<'_>,
) -> Result<Arc<Vec<u8>>> {
    block_decode::read_block(
        source,
        cache,
        block_decode::BlockSpec {
            key: BlockKey {
                ifd_offset,
                kind: BlockKind::Tile,
                block_index: spec.index,
            },
            index: spec.index,
            offset: spec.offset,
            byte_count: spec.byte_count,
            block_width: spec.tile_width,
            block_height: spec.tile_height,
        },
        context,
        options,
    )
}
