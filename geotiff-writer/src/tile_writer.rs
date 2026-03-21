//! Streaming tile-by-tile GeoTIFF writer.

use std::io::{Seek, Write};
use std::marker::PhantomData;

use ndarray::ArrayView2;
use tiff_writer::{ImageHandle, TiffWriter, WriteOptions};

use crate::builder::GeoTiffBuilder;
use crate::error::{Error, Result};
use crate::sample::WriteSample;

/// Streaming tile-by-tile GeoTIFF writer.
///
/// Tiles can be written in any order. Edge tiles are automatically padded.
/// Missing tiles are filled with the configured NoData value (or zero) on `finish()`.
pub struct StreamingTileWriter<T: WriteSample, W: Write + Seek> {
    writer: TiffWriter<W>,
    handle: ImageHandle,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    width: u32,
    height: u32,
    bands: u32,
    fill_value: T,
    written: Vec<bool>,
    _phantom: PhantomData<T>,
}

/// Parse a nodata string into a fill value for the given sample type.
fn parse_nodata_fill<T: WriteSample>(nodata: &Option<String>) -> T {
    let zero_bytes = vec![0u8; T::BYTES_PER_SAMPLE];
    let zero = T::decode_many(&zero_bytes)[0];
    let Some(nd) = nodata else { return zero };

    // Try to parse as f64, then convert to the target type via the cog helper
    let Ok(val) = nd.trim().parse::<f64>() else {
        return zero;
    };

    // Convert f64 → target type bytes → decode
    let bytes = match (T::SAMPLE_FORMAT, T::BITS_PER_SAMPLE) {
        (1, 8) => vec![val as u8],
        (2, 8) => vec![val as i8 as u8],
        (1, 16) => (val as u16).to_ne_bytes().to_vec(),
        (2, 16) => (val as i16).to_ne_bytes().to_vec(),
        (1, 32) => (val as u32).to_ne_bytes().to_vec(),
        (2, 32) => (val as i32).to_ne_bytes().to_vec(),
        (3, 32) => (val as f32).to_ne_bytes().to_vec(),
        (1, 64) => (val as u64).to_ne_bytes().to_vec(),
        (2, 64) => (val as i64).to_ne_bytes().to_vec(),
        (3, 64) => val.to_ne_bytes().to_vec(),
        _ => return zero,
    };
    T::decode_many(&bytes)[0]
}

impl<T: WriteSample, W: Write + Seek> StreamingTileWriter<T, W> {
    pub(crate) fn new(builder: GeoTiffBuilder, sink: W) -> Result<Self> {
        let tw = builder.tile_width.unwrap_or(256);
        let th = builder.tile_height.unwrap_or(256);

        let builder = if builder.tile_width.is_none() {
            builder.tile_size(tw, th)
        } else {
            builder
        };

        let fill_value = parse_nodata_fill::<T>(&builder.nodata);

        let ib = builder.to_image_builder::<T>();
        let num_blocks = ib.block_count();
        let tiles_across = (builder.width as usize).div_ceil(tw as usize);
        let tiles_down = (builder.height as usize).div_ceil(th as usize);

        let mut writer = TiffWriter::new(sink, WriteOptions::default())?;
        let handle = writer.add_image(ib)?;

        Ok(Self {
            writer,
            handle,
            tile_width: tw,
            tile_height: th,
            tiles_across: tiles_across as u32,
            tiles_down: tiles_down as u32,
            width: builder.width,
            height: builder.height,
            bands: builder.bands,
            fill_value,
            written: vec![false; num_blocks],
            _phantom: PhantomData,
        })
    }

    /// Write a tile at pixel offset (x_off, y_off).
    ///
    /// The data shape should match the actual tile size (may be smaller at edges).
    /// The tile is automatically padded to the full tile dimensions with the NoData fill value.
    pub fn write_tile(&mut self, x_off: usize, y_off: usize, data: &ArrayView2<T>) -> Result<()> {
        let tile_col = x_off / self.tile_width as usize;
        let tile_row = y_off / self.tile_height as usize;

        if tile_col >= self.tiles_across as usize || tile_row >= self.tiles_down as usize {
            return Err(Error::TileOutOfBounds {
                x_off,
                y_off,
                width: self.width,
                height: self.height,
            });
        }

        let tile_index = tile_row * self.tiles_across as usize + tile_col;
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let (data_h, data_w) = data.dim();
        let spp = self.bands as usize;

        let mut padded = vec![self.fill_value; tw * th * spp];
        for row in 0..data_h.min(th) {
            for col in 0..data_w.min(tw) {
                padded[row * tw * spp + col * spp] = data[[row, col]];
            }
        }

        self.writer.write_block(&self.handle, tile_index, &padded)?;
        self.written[tile_index] = true;
        Ok(())
    }

    /// Write a multi-band tile at pixel offset (x_off, y_off).
    /// Data shape: (tile_height, tile_width, bands) — interleaved.
    pub fn write_tile_3d(
        &mut self,
        x_off: usize,
        y_off: usize,
        data: &ndarray::ArrayView3<T>,
    ) -> Result<()> {
        let tile_col = x_off / self.tile_width as usize;
        let tile_row = y_off / self.tile_height as usize;

        if tile_col >= self.tiles_across as usize || tile_row >= self.tiles_down as usize {
            return Err(Error::TileOutOfBounds {
                x_off,
                y_off,
                width: self.width,
                height: self.height,
            });
        }

        let tile_index = tile_row * self.tiles_across as usize + tile_col;
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let (data_h, data_w, data_b) = data.dim();
        let spp = self.bands as usize;

        let mut padded = vec![self.fill_value; tw * th * spp];
        for row in 0..data_h.min(th) {
            for col in 0..data_w.min(tw) {
                for band in 0..data_b.min(spp) {
                    padded[(row * tw + col) * spp + band] = data[[row, col, band]];
                }
            }
        }

        self.writer.write_block(&self.handle, tile_index, &padded)?;
        self.written[tile_index] = true;
        Ok(())
    }

    /// Finish writing: fill missing tiles with the NoData value, finalize the file.
    pub fn finish(mut self) -> Result<W> {
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let spp = self.bands as usize;
        let empty_tile = vec![self.fill_value; tw * th * spp];

        for (i, written) in self.written.iter().enumerate() {
            if !*written {
                self.writer.write_block(&self.handle, i, &empty_tile)?;
            }
        }

        Ok(self.writer.finish()?)
    }
}
