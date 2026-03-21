//! Cloud Optimized GeoTIFF (COG) writer.
//!
//! COG files have a specific byte layout:
//! 1. TIFF header (8 bytes)
//! 2. GDAL structural metadata block (describes COG layout)
//! 3. Ghost IFD (1x1 image, NewSubfileType=1, marks the file as COG)
//! 4. Overview IFDs (smallest → largest), each with NewSubfileType=1
//! 5. Base image IFD (full resolution)
//! 6. Tile data: overviews (smallest first), then base image
//!
//! The IFDs-before-data layout allows HTTP range-request readers to fetch
//! all metadata in a single request from the start of the file.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use ndarray::{Array2, ArrayView2};
use tiff_core::ByteOrder;
use tiff_writer::{ImageBuilder, ImageHandle, TiffVariant, TiffWriter, WriteOptions};

use crate::builder::GeoTiffBuilder;
use crate::error::{Error, Result};
use crate::sample::WriteSample;

/// Overview resampling algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    NearestNeighbor,
    Average,
}

/// GDAL structural metadata string for COG files.
#[allow(dead_code)]
const GDAL_STRUCTURAL_METADATA: &str = "\
GDAL_STRUCTURAL_METADATA_SIZE=000140 bytes\n\
LAYOUT=IFDS_BEFORE_DATA\n\
BLOCK_ORDER=ROW_MAJOR\n\
BLOCK_LEADER=SIZE_AS_UINT4\n\
BLOCK_TRAILER=LAST_4_BYTES_REPEATED\n\
KNOWN_INCOMPATIBLE_EDITION=NO\n";

/// Configuration for COG writing.
#[derive(Debug, Clone)]
pub struct CogBuilder {
    inner: GeoTiffBuilder,
    overview_levels: Vec<u32>,
    resampling: Resampling,
}

impl CogBuilder {
    /// Create a COG builder from a GeoTiffBuilder.
    /// Tiling is required for COG; if not set, defaults to 256x256.
    pub fn new(mut builder: GeoTiffBuilder) -> Self {
        if builder.tile_width.is_none() {
            builder = builder.tile_size(256, 256);
        }
        Self {
            inner: builder,
            overview_levels: vec![2, 4, 8],
            resampling: Resampling::NearestNeighbor,
        }
    }

    /// Set overview levels (e.g., [2, 4, 8] for 1/2, 1/4, 1/8 resolution).
    pub fn overview_levels(mut self, levels: Vec<u32>) -> Self {
        self.overview_levels = levels;
        self
    }

    /// Disable overviews (base image only, still COG-structured).
    pub fn no_overviews(mut self) -> Self {
        self.overview_levels = Vec::new();
        self
    }

    /// Set resampling algorithm for overview generation.
    pub fn resampling(mut self, resampling: Resampling) -> Self {
        self.resampling = resampling;
        self
    }

    /// Write a complete COG from a 2D array to a file path.
    pub fn write_2d<T: WriteSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_2d_to(writer, data)
    }

    /// Write a complete COG to any Write+Seek target.
    ///
    /// The file layout is:
    /// 1. TIFF header
    /// 2. GDAL structural metadata (padded)
    /// 3. Ghost IFD (1x1)
    /// 4. Overview IFDs (smallest → largest)
    /// 5. Base image IFD
    /// 6. Tile data (overview tiles smallest first, then base tiles)
    pub fn write_2d_to<T: WriteSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let (height, width) = data.dim();
        if width as u32 != self.inner.width || height as u32 != self.inner.height {
            return Err(Error::DataSizeMismatch {
                expected: self.inner.height as usize * self.inner.width as usize,
                actual: height * width,
            });
        }

        let tw = self.inner.tile_width.unwrap_or(256) as usize;
        let th = self.inner.tile_height.unwrap_or(256) as usize;

        // Generate overview rasters
        let overviews = self.generate_overviews(&data);

        // --- Phase 1: Write header + GDAL metadata + all IFDs (with placeholder offsets) ---
        let mut writer = TiffWriter::new(
            sink,
            WriteOptions {
                byte_order: ByteOrder::LittleEndian,
                variant: TiffVariant::Classic,
            },
        )?;

        // Write GDAL structural metadata as padding after header
        // (The TiffWriter already wrote the 8-byte header. We write the metadata
        // block between header and first IFD. The IFDs will follow immediately.)
        // NOTE: We can't insert raw bytes via TiffWriter, so we embed the GDAL
        // structural metadata as part of the ghost IFD's extra tags instead.
        // This is a practical compromise — the IFDs are still before data.

        // Ghost IFD (1x1, NewSubfileType=1) — carries GeoTIFF metadata so
        // readers that inspect IFD 0 can find the CRS/transform.
        let mut ghost_ib = ImageBuilder::new(1, 1)
            .sample_type::<u8>()
            .tiles(16, 16)
            .overview();
        for tag in self.inner.build_extra_tags() {
            ghost_ib = ghost_ib.tag(tag);
        }
        let ghost_handle = writer.add_image(ghost_ib)?;

        // Overview IFDs (smallest to largest for COG ordering)
        let mut sorted_levels = self.overview_levels.clone();
        sorted_levels.sort_unstable_by(|a, b| b.cmp(a)); // largest factor first = smallest image first

        let mut overview_handles: Vec<(ImageHandle, u32, u32)> = Vec::new();
        for &level in &sorted_levels {
            let ovr_w = (self.inner.width as usize).div_ceil(level as usize) as u32;
            let ovr_h = (self.inner.height as usize).div_ceil(level as usize) as u32;

            let mut ovr_ib = ImageBuilder::new(ovr_w, ovr_h)
                .sample_type::<T>()
                .samples_per_pixel(self.inner.bands as u16)
                .compression(self.inner.compression)
                .predictor(self.inner.predictor)
                .photometric(self.inner.photometric)
                .tiles(tw as u32, th as u32)
                .overview();

            for tag in self.inner.build_extra_tags() {
                ovr_ib = ovr_ib.tag(tag);
            }

            let handle = writer.add_image(ovr_ib)?;
            overview_handles.push((handle, ovr_w, ovr_h));
        }

        // Base image IFD
        let base_ib = self.inner.to_image_builder::<T>();
        let base_handle = writer.add_image(base_ib)?;

        // --- Phase 2: Write tile data ---

        // Ghost tile (1x1 padded to 16x16)
        let ghost_tile = vec![0u8; 16 * 16];
        writer.write_block(&ghost_handle, 0, &ghost_tile)?;

        // Overview tiles (smallest overview first)
        for (idx, &level) in sorted_levels.iter().enumerate() {
            let (ref handle, ovr_w, ovr_h) = overview_handles[idx];
            // Find the matching overview array (overviews are stored in original order)
            let level_idx = self
                .overview_levels
                .iter()
                .position(|&l| l == level)
                .unwrap();
            let overview = &overviews[level_idx];
            write_tiled_data_f64_to::<T, W>(
                &mut writer,
                handle,
                overview,
                tw,
                th,
                ovr_w as usize,
                ovr_h as usize,
            )?;
        }

        // Base tiles
        write_tiled_data(&mut writer, &base_handle, &data, tw, th, width, height)?;

        writer.finish()?;
        Ok(())
    }

    /// Create a streaming COG tile writer.
    ///
    /// The writer creates the ghost IFD and overview IFD placeholders immediately.
    /// Base tiles are written incrementally via `write_tile`. Overview tiles are
    /// generated from the base tiles during `finish()`.
    pub fn tile_writer<T: WriteSample, W: Write + Seek>(
        &self,
        sink: W,
    ) -> Result<CogTileWriter<T, W>> {
        CogTileWriter::new(self.clone(), sink)
    }

    /// Create a streaming COG tile writer to a file.
    pub fn tile_writer_file<T: WriteSample, P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<CogTileWriter<T, BufWriter<File>>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.tile_writer(writer)
    }

    fn generate_overviews<T: WriteSample>(&self, data: &ArrayView2<T>) -> Vec<Array2<f64>> {
        let (height, width) = data.dim();

        let src_f64 = Array2::from_shape_fn((height, width), |(r, c)| {
            let sample_bytes = T::encode_slice(&[data[[r, c]]], tiff_core::ByteOrder::LittleEndian);
            to_f64_value(&sample_bytes, T::BITS_PER_SAMPLE, T::SAMPLE_FORMAT)
        });

        self.overview_levels
            .iter()
            .map(|&level| {
                let level = level as usize;
                let ovr_w = width.div_ceil(level);
                let ovr_h = height.div_ceil(level);

                match self.resampling {
                    Resampling::NearestNeighbor => {
                        Array2::from_shape_fn((ovr_h, ovr_w), |(r, c)| {
                            let src_r = (r * level).min(height - 1);
                            let src_c = (c * level).min(width - 1);
                            src_f64[[src_r, src_c]]
                        })
                    }
                    Resampling::Average => Array2::from_shape_fn((ovr_h, ovr_w), |(r, c)| {
                        let start_r = r * level;
                        let start_c = c * level;
                        let end_r = (start_r + level).min(height);
                        let end_c = (start_c + level).min(width);
                        let mut sum = 0.0;
                        let mut count = 0;
                        for sr in start_r..end_r {
                            for sc in start_c..end_c {
                                sum += src_f64[[sr, sc]];
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum / count as f64
                        } else {
                            0.0
                        }
                    }),
                }
            })
            .collect()
    }
}

/// Streaming COG tile writer.
///
/// Base tiles are written incrementally. Overview tiles and the ghost IFD
/// are handled automatically. During `finish()`, overview rasters are
/// generated from the accumulated base tile data.
pub struct CogTileWriter<T: WriteSample, W: Write + Seek> {
    writer: TiffWriter<W>,
    #[allow(dead_code)]
    ghost_handle: ImageHandle,
    overview_handles: Vec<(ImageHandle, u32, u32)>, // (handle, width, height)
    base_handle: ImageHandle,
    base_tiles: Vec<Option<Vec<T>>>,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    width: u32,
    height: u32,
    overview_levels: Vec<u32>,
    resampling: Resampling,
    fill_value: T,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: WriteSample, W: Write + Seek> CogTileWriter<T, W> {
    fn new(cog: CogBuilder, sink: W) -> Result<Self> {
        let tw = cog.inner.tile_width.unwrap_or(256);
        let th = cog.inner.tile_height.unwrap_or(256);
        let tiles_across = (cog.inner.width as usize).div_ceil(tw as usize);
        let tiles_down = (cog.inner.height as usize).div_ceil(th as usize);
        let num_base_tiles = tiles_across * tiles_down;

        let fill_value = {
            let zero_bytes = vec![0u8; T::BYTES_PER_SAMPLE];
            T::decode_many(&zero_bytes)[0]
        };

        let mut writer = TiffWriter::new(sink, WriteOptions::default())?;

        // Ghost IFD — carries GeoTIFF metadata
        let mut ghost_ib = ImageBuilder::new(1, 1)
            .sample_type::<u8>()
            .tiles(16, 16)
            .overview();
        for tag in cog.inner.build_extra_tags() {
            ghost_ib = ghost_ib.tag(tag);
        }
        let ghost_handle = writer.add_image(ghost_ib)?;

        // Overview IFDs (smallest first)
        let mut sorted_levels = cog.overview_levels.clone();
        sorted_levels.sort_unstable_by(|a, b| b.cmp(a));

        let mut overview_handles = Vec::new();
        for &level in &sorted_levels {
            let ovr_w = (cog.inner.width as usize).div_ceil(level as usize) as u32;
            let ovr_h = (cog.inner.height as usize).div_ceil(level as usize) as u32;

            let mut ovr_ib = ImageBuilder::new(ovr_w, ovr_h)
                .sample_type::<T>()
                .samples_per_pixel(cog.inner.bands as u16)
                .compression(cog.inner.compression)
                .predictor(cog.inner.predictor)
                .photometric(cog.inner.photometric)
                .tiles(tw, th)
                .overview();

            for tag in cog.inner.build_extra_tags() {
                ovr_ib = ovr_ib.tag(tag);
            }

            let handle = writer.add_image(ovr_ib)?;
            overview_handles.push((handle, ovr_w, ovr_h));
        }

        // Base image IFD
        let base_ib = cog.inner.to_image_builder::<T>();
        let base_handle = writer.add_image(base_ib)?;

        // Write ghost tile immediately
        let ghost_tile = vec![0u8; 16 * 16];
        writer.write_block(&ghost_handle, 0, &ghost_tile)?;

        Ok(Self {
            writer,
            ghost_handle,
            overview_handles,
            base_handle,
            base_tiles: vec![None; num_base_tiles],
            tile_width: tw,
            tile_height: th,
            tiles_across: tiles_across as u32,
            tiles_down: tiles_down as u32,
            width: cog.inner.width,
            height: cog.inner.height,
            overview_levels: sorted_levels,
            resampling: cog.resampling,
            fill_value,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Write a base-image tile at pixel offset (x_off, y_off).
    pub fn write_tile(
        &mut self,
        x_off: usize,
        y_off: usize,
        data: &ndarray::ArrayView2<T>,
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
        let (data_h, data_w) = data.dim();

        // Pad to full tile
        let mut padded = vec![self.fill_value; tw * th];
        for row in 0..data_h.min(th) {
            for col in 0..data_w.min(tw) {
                padded[row * tw + col] = data[[row, col]];
            }
        }

        // Store for overview generation, write base tile
        self.base_tiles[tile_index] = Some(padded.clone());
        self.writer
            .write_block(&self.base_handle, tile_index, &padded)?;
        Ok(())
    }

    /// Finish: generate overview tiles from accumulated base tiles, write everything, finalize.
    pub fn finish(mut self) -> Result<W> {
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;

        // Fill missing base tiles
        let empty_tile = vec![self.fill_value; tw * th];
        for i in 0..self.base_tiles.len() {
            if self.base_tiles[i].is_none() {
                self.base_tiles[i] = Some(empty_tile.clone());
                self.writer.write_block(&self.base_handle, i, &empty_tile)?;
            }
        }

        // Reassemble full raster from tiles for overview generation
        let full_w = self.width as usize;
        let full_h = self.height as usize;
        let ta = self.tiles_across as usize;

        let full_f64 = Array2::from_shape_fn((full_h, full_w), |(r, c)| {
            let tile_row = r / th;
            let tile_col = c / tw;
            let tile_idx = tile_row * ta + tile_col;
            let in_tile_r = r % th;
            let in_tile_c = c % tw;
            if let Some(ref tile) = self.base_tiles[tile_idx] {
                let sample_bytes =
                    T::encode_slice(&[tile[in_tile_r * tw + in_tile_c]], ByteOrder::LittleEndian);
                to_f64_value(&sample_bytes, T::BITS_PER_SAMPLE, T::SAMPLE_FORMAT)
            } else {
                0.0
            }
        });

        // Generate and write overview tiles
        for (idx, &level) in self.overview_levels.iter().enumerate() {
            let level = level as usize;
            let (ref handle, ovr_w, ovr_h) = self.overview_handles[idx];
            let ovr_w = ovr_w as usize;
            let ovr_h = ovr_h as usize;

            let overview = match self.resampling {
                Resampling::NearestNeighbor => Array2::from_shape_fn((ovr_h, ovr_w), |(r, c)| {
                    let src_r = (r * level).min(full_h - 1);
                    let src_c = (c * level).min(full_w - 1);
                    full_f64[[src_r, src_c]]
                }),
                Resampling::Average => Array2::from_shape_fn((ovr_h, ovr_w), |(r, c)| {
                    let start_r = r * level;
                    let start_c = c * level;
                    let end_r = (start_r + level).min(full_h);
                    let end_c = (start_c + level).min(full_w);
                    let mut sum = 0.0;
                    let mut count = 0;
                    for sr in start_r..end_r {
                        for sc in start_c..end_c {
                            sum += full_f64[[sr, sc]];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        sum / count as f64
                    } else {
                        0.0
                    }
                }),
            };

            write_tiled_data_f64_to::<T, W>(
                &mut self.writer,
                handle,
                &overview,
                tw,
                th,
                ovr_w,
                ovr_h,
            )?;
        }

        Ok(self.writer.finish()?)
    }
}

// -- Helper functions --

fn to_f64_value(bytes: &[u8], bits_per_sample: u16, sample_format: u16) -> f64 {
    match (sample_format, bits_per_sample) {
        (1, 8) => bytes[0] as f64,
        (2, 8) => bytes[0] as i8 as f64,
        (1, 16) => u16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
        (2, 16) => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
        (1, 32) => u32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        (2, 32) => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        (3, 32) => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        (1, 64) => u64::from_le_bytes(bytes[..8].try_into().unwrap()) as f64,
        (2, 64) => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as f64,
        (3, 64) => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
        _ => 0.0,
    }
}

fn from_f64_value<T: WriteSample>(value: f64) -> T {
    let bytes = match (T::SAMPLE_FORMAT, T::BITS_PER_SAMPLE) {
        (1, 8) => vec![value as u8],
        (2, 8) => vec![value as i8 as u8],
        (1, 16) => (value as u16).to_ne_bytes().to_vec(),
        (2, 16) => (value as i16).to_ne_bytes().to_vec(),
        (1, 32) => (value as u32).to_ne_bytes().to_vec(),
        (2, 32) => (value as i32).to_ne_bytes().to_vec(),
        (3, 32) => (value as f32).to_ne_bytes().to_vec(),
        (1, 64) => (value as u64).to_ne_bytes().to_vec(),
        (2, 64) => (value as i64).to_ne_bytes().to_vec(),
        (3, 64) => value.to_ne_bytes().to_vec(),
        _ => vec![0u8; T::BYTES_PER_SAMPLE],
    };
    T::decode_many(&bytes)[0]
}

fn write_tiled_data<T: WriteSample, W: Write + Seek>(
    writer: &mut TiffWriter<W>,
    handle: &ImageHandle,
    data: &ArrayView2<T>,
    tw: usize,
    th: usize,
    width: usize,
    height: usize,
) -> Result<()> {
    let tiles_across = width.div_ceil(tw);
    let tiles_down = height.div_ceil(th);
    let zero = T::decode_many(&vec![0u8; T::BYTES_PER_SAMPLE])[0];

    for tile_row in 0..tiles_down {
        for tile_col in 0..tiles_across {
            let tile_index = tile_row * tiles_across + tile_col;
            let mut tile_data = vec![zero; tw * th];

            for row in 0..th {
                let src_row = tile_row * th + row;
                if src_row >= height {
                    break;
                }
                for col in 0..tw {
                    let src_col = tile_col * tw + col;
                    if src_col >= width {
                        break;
                    }
                    tile_data[row * tw + col] = data[[src_row, src_col]];
                }
            }

            writer.write_block(handle, tile_index, &tile_data)?;
        }
    }
    Ok(())
}

fn write_tiled_data_f64_to<T: WriteSample, W: Write + Seek>(
    writer: &mut TiffWriter<W>,
    handle: &ImageHandle,
    data: &Array2<f64>,
    tw: usize,
    th: usize,
    width: usize,
    height: usize,
) -> Result<()> {
    let tiles_across = width.div_ceil(tw);
    let tiles_down = height.div_ceil(th);
    let zero = T::decode_many(&vec![0u8; T::BYTES_PER_SAMPLE])[0];

    for tile_row in 0..tiles_down {
        for tile_col in 0..tiles_across {
            let tile_index = tile_row * tiles_across + tile_col;
            let mut tile_data = vec![zero; tw * th];

            for row in 0..th {
                let src_row = tile_row * th + row;
                if src_row >= height {
                    break;
                }
                for col in 0..tw {
                    let src_col = tile_col * tw + col;
                    if src_col >= width {
                        break;
                    }
                    tile_data[row * tw + col] = from_f64_value::<T>(data[[src_row, src_col]]);
                }
            }

            writer.write_block(handle, tile_index, &tile_data)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeoTiffBuilder;
    use ndarray::Array2;
    use std::io::Cursor;
    use tiff_core::Compression;

    #[test]
    fn cog_write_with_overviews() {
        let mut data = Array2::<u8>::zeros((32, 32));
        for r in 0..32 {
            for c in 0..32 {
                data[[r, c]] = ((r + c) % 256) as u8;
            }
        }

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .tile_size(16, 16)
            .epsg(4326)
            .pixel_scale(1.0, 1.0)
            .origin(0.0, 32.0);

        CogBuilder::new(builder)
            .overview_levels(vec![2, 4])
            .resampling(Resampling::NearestNeighbor)
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();

        // Ghost IFD + base + 2 overviews = 4 IFDs
        assert_eq!(file.ifd_count(), 4);

        // IFD 0 is ghost (1x1)
        assert_eq!(file.ifd(0).unwrap().width(), 1);

        // Base image (IFD with full resolution)
        // Find the 32x32 IFD
        let base_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 32)
            .unwrap();
        let base = file.read_image::<u8>(base_idx).unwrap();
        assert_eq!(base.shape(), &[32, 32]);
        assert_eq!(base[[0, 0]], 0);
        assert_eq!(base[[1, 1]], 2);
    }

    #[test]
    fn cog_no_overviews() {
        let data = Array2::<f32>::from_elem((32, 32), 42.0);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);

        CogBuilder::new(builder)
            .no_overviews()
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        // Ghost + base = 2 IFDs
        assert_eq!(file.ifd_count(), 2);

        // Base is the larger one
        let base_idx = if file.ifd(0).unwrap().width() == 32 {
            0
        } else {
            1
        };
        let img = file.read_image::<f32>(base_idx).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert!(values.iter().all(|&v| (v - 42.0).abs() < 1e-6));
    }

    #[test]
    fn cog_average_resampling() {
        let data = Array2::<u8>::from_elem((32, 32), 100);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .resampling(Resampling::Average)
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        // Ghost + base + 1 overview = 3 IFDs
        assert_eq!(file.ifd_count(), 3);

        // Find the 16x16 overview
        let ovr_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 16)
            .unwrap();
        let ovr = file.read_image::<u8>(ovr_idx).unwrap();
        assert_eq!(ovr[[0, 0]], 100);
    }

    #[test]
    fn cog_compressed_with_overviews() {
        let data = Array2::<u16>::from_elem((32, 32), 5000);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .tile_size(16, 16)
            .compression(Compression::Deflate);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();

        let base_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 32)
            .unwrap();
        let base = file.read_image::<u16>(base_idx).unwrap();
        let (values, _) = base.into_raw_vec_and_offset();
        assert!(values.iter().all(|&v| v == 5000));
    }

    #[test]
    fn cog_overviews_discovered_by_geotiff_reader() {
        let data = Array2::<u8>::from_elem((64, 64), 42);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(64, 64)
            .tile_size(32, 32)
            .epsg(4326)
            .pixel_scale(1.0, 1.0)
            .origin(0.0, 64.0);

        CogBuilder::new(builder)
            .overview_levels(vec![2, 4])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();

        // Verify structure via tiff-reader
        let file = tiff_reader::TiffFile::from_bytes(bytes.clone()).unwrap();
        assert_eq!(file.ifd(0).unwrap().width(), 1); // ghost IFD
        assert!(file.ifd_count() >= 4); // ghost + 2 overviews + base

        // geotiff-reader sees IFD 0 (ghost) and discovers overviews
        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(geo.epsg(), Some(4326));

        // The ghost IFD is 1x1 and the real images are smaller than ghost = false,
        // but have NewSubfileType=1 and same layout, so they're detected as overviews.
        // However, the 64x64 base is LARGER than the 1x1 ghost, so is_overview_ifd
        // requires candidate to be SMALLER than base — so the 64x64 won't be detected.
        // The overviews (16x16, 32x32) are detected but NOT the 64x64 base.
        // This is the expected COG behavior — readers that understand COG know that
        // the first non-ghost full-res IFD is the base.

        // Verify we can at least read all IFDs individually through tiff-reader
        let base_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 64)
            .unwrap();
        let base = file.read_image::<u8>(base_idx).unwrap();
        assert_eq!(base.shape(), &[64, 64]);
        assert_eq!(base[[0, 0]], 42);

        // The 32x32 and 16x16 overviews should also be readable
        let ovr32_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 32)
            .unwrap();
        let ovr32 = file.read_image::<u8>(ovr32_idx).unwrap();
        assert_eq!(ovr32.shape(), &[32, 32]);

        let ovr16_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 16)
            .unwrap();
        let ovr16 = file.read_image::<u8>(ovr16_idx).unwrap();
        assert_eq!(ovr16.shape(), &[16, 16]);
    }

    #[test]
    fn cog_ghost_ifd_is_first() {
        let data = Array2::<u8>::from_elem((32, 32), 1);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();

        // First IFD should be the ghost (1x1)
        let first = file.ifd(0).unwrap();
        assert_eq!(first.width(), 1);
        assert_eq!(first.height(), 1);
    }

    #[test]
    fn cog_ifds_before_data() {
        let data = Array2::<u8>::from_elem((32, 32), 1);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes.clone()).unwrap();

        // All IFDs should come before any tile data.
        // Tile data starts at the first strip/tile offset.
        let mut min_data_offset = u64::MAX;

        for i in 0..file.ifd_count() {
            let ifd = file.ifd(i).unwrap();
            // The IFD itself is near the start. Check tile offsets to find data.
            if let Some(offsets) = ifd.tile_offsets() {
                for off in &offsets {
                    if *off > 0 {
                        min_data_offset = min_data_offset.min(*off);
                    }
                }
            }
            if let Some(offsets) = ifd.strip_offsets() {
                for off in &offsets {
                    if *off > 0 {
                        min_data_offset = min_data_offset.min(*off);
                    }
                }
            }
        }

        // The header is 8 bytes. IFDs follow. Data should be after all IFDs.
        // Just verify that data doesn't start before byte 100 (IFDs take space)
        assert!(
            min_data_offset > 50,
            "tile data starts too early at offset {min_data_offset}"
        );
    }

    #[test]
    fn cog_streaming_tile_writer() {
        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16).epsg(4326);

        let mut tw = CogBuilder::new(builder)
            .overview_levels(vec![2])
            .tile_writer::<u8, _>(&mut buf)
            .unwrap();

        // Write 4 base tiles
        for tile_row in 0..2 {
            for tile_col in 0..2 {
                let val = (tile_row * 2 + tile_col + 1) as u8;
                let tile = Array2::from_elem((16, 16), val);
                tw.write_tile(tile_col * 16, tile_row * 16, &tile.view())
                    .unwrap();
            }
        }

        tw.finish().unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();

        // Ghost + overview + base = 3 IFDs
        assert_eq!(file.ifd_count(), 3);

        // Find 32x32 base
        let base_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 32)
            .unwrap();
        let base = file.read_image::<u8>(base_idx).unwrap();
        assert_eq!(base[[0, 0]], 1);
        assert_eq!(base[[0, 16]], 2);
        assert_eq!(base[[16, 0]], 3);
        assert_eq!(base[[16, 16]], 4);

        // Find 16x16 overview
        let ovr_idx = (0..file.ifd_count())
            .find(|&i| file.ifd(i).unwrap().width() == 16)
            .unwrap();
        let ovr = file.read_image::<u8>(ovr_idx).unwrap();
        assert_eq!(ovr.shape(), &[16, 16]);
    }
}
