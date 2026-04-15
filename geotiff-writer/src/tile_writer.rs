//! Streaming tile-by-tile GeoTIFF writer.

use std::io::{Seek, Write};
use std::marker::PhantomData;

use ndarray::ArrayView2;
use tiff_core::PlanarConfiguration;
use tiff_writer::{ImageHandle, TiffWriter, WriteOptions};

use crate::builder::GeoTiffBuilder;
use crate::error::{Error, Result};
use crate::sample::{nodata_fill_or_zero, NumericSample, WriteSample};

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
    planar_configuration: PlanarConfiguration,
    fill_value: T,
    written: Vec<bool>,
    _phantom: PhantomData<T>,
}

impl<T: NumericSample, W: Write + Seek> StreamingTileWriter<T, W> {
    pub(crate) fn new(builder: GeoTiffBuilder, sink: W) -> Result<Self> {
        let tw = builder.tile_width.unwrap_or(256);
        let th = builder.tile_height.unwrap_or(256);

        let builder = if builder.tile_width.is_none() {
            builder.tile_size(tw, th)
        } else {
            builder
        };

        let fill_value = nodata_fill_or_zero::<T>(&builder.nodata);

        let ib = builder.to_image_builder::<T>();
        let num_blocks = ib.block_count();
        let tiles_across = (builder.width as usize).div_ceil(tw as usize);
        let tiles_down = (builder.height as usize).div_ceil(th as usize);

        let mut writer = TiffWriter::new(
            sink,
            WriteOptions {
                byte_order: tiff_core::ByteOrder::LittleEndian,
                variant: builder.tiff_variant,
            },
        )?;
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
            planar_configuration: builder.planar_configuration,
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
        if self.bands != 1 {
            return Err(Error::Other(
                "write_tile only supports single-band output; use write_tile_3d for multi-band tiles".into(),
            ));
        }
        if x_off % self.tile_width as usize != 0 || y_off % self.tile_height as usize != 0 {
            return Err(Error::Other(format!(
                "tile offsets must align to tile boundaries of {}x{}, got ({x_off},{y_off})",
                self.tile_width, self.tile_height
            )));
        }

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
        let expected_h = (self.height as usize).saturating_sub(y_off).min(th);
        let expected_w = (self.width as usize).saturating_sub(x_off).min(tw);
        if data_h > expected_h || data_w > expected_w {
            return Err(Error::Other(format!(
                "tile data shape {}x{} exceeds raster bounds for tile starting at ({x_off},{y_off}); expected at most {}x{}",
                data_h, data_w, expected_h, expected_w
            )));
        }
        let mut padded = vec![self.fill_value; tw * th];
        for row in 0..data_h {
            for col in 0..data_w {
                padded[row * tw + col] = data[[row, col]];
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
        if x_off % self.tile_width as usize != 0 || y_off % self.tile_height as usize != 0 {
            return Err(Error::Other(format!(
                "tile offsets must align to tile boundaries of {}x{}, got ({x_off},{y_off})",
                self.tile_width, self.tile_height
            )));
        }

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
        let tiles_per_plane = self.tiles_across as usize * self.tiles_down as usize;
        let (data_h, data_w, data_b) = data.dim();
        let spp = self.bands as usize;
        let expected_h = (self.height as usize).saturating_sub(y_off).min(th);
        let expected_w = (self.width as usize).saturating_sub(x_off).min(tw);
        if data_h > expected_h || data_w > expected_w {
            return Err(Error::Other(format!(
                "tile data shape {}x{} exceeds raster bounds for tile starting at ({x_off},{y_off}); expected at most {}x{}",
                data_h, data_w, expected_h, expected_w
            )));
        }
        if data_b != spp {
            return Err(Error::DataSizeMismatch {
                expected: data_h * data_w * spp,
                actual: data_h * data_w * data_b,
            });
        }
        if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
            for band in 0..spp {
                let mut padded = vec![self.fill_value; tw * th];
                for row in 0..data_h {
                    for col in 0..data_w {
                        padded[row * tw + col] = data[[row, col, band]];
                    }
                }
                let block_index = band * tiles_per_plane + tile_index;
                self.writer
                    .write_block(&self.handle, block_index, &padded)?;
                self.written[block_index] = true;
            }
        } else {
            let mut padded = vec![self.fill_value; tw * th * spp];
            for row in 0..data_h {
                for col in 0..data_w {
                    for band in 0..spp {
                        padded[(row * tw + col) * spp + band] = data[[row, col, band]];
                    }
                }
            }

            self.writer.write_block(&self.handle, tile_index, &padded)?;
            self.written[tile_index] = true;
        }
        Ok(())
    }

    /// Finish writing: fill missing tiles with the NoData value, finalize the file.
    pub fn finish(mut self) -> Result<W> {
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let spp = if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
            1
        } else {
            self.bands as usize
        };
        let empty_tile = vec![self.fill_value; tw * th * spp];

        for (i, written) in self.written.iter().enumerate() {
            if !*written {
                self.writer.write_block(&self.handle, i, &empty_tile)?;
            }
        }

        Ok(self.writer.finish()?)
    }
}
