//! Cloud Optimized GeoTIFF (COG) writer.
//!
//! COG files have a specific byte layout:
//! 1. TIFF header (8 bytes)
//! 2. GDAL structural metadata block (the COG "ghost area")
//! 3. Base image IFD (full resolution)
//! 4. Overview IFDs (largest → smallest)
//! 5. Tile offset/byte-count arrays
//! 6. Tile data: overviews (smallest first), then base image
//!
//! The IFDs-before-data layout allows HTTP range-request readers to fetch
//! all metadata in a single request from the start of the file.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use ndarray::{Array3, ArrayView2, ArrayView3, Axis};
use tiff_core::{ByteOrder, Compression, Predictor};
use tiff_writer::{ImageBuilder, ImageHandle, TiffVariant, TiffWriter, WriteOptions};

use crate::builder::GeoTiffBuilder;
use crate::error::{Error, Result};
use crate::sample::NumericSample;

/// Overview resampling algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    NearestNeighbor,
    Average,
}

fn gdal_structural_metadata_bytes(planar_configuration: tiff_core::PlanarConfiguration) -> Vec<u8> {
    let mut payload = String::from(
        "LAYOUT=IFDS_BEFORE_DATA\n\
BLOCK_ORDER=ROW_MAJOR\n\
BLOCK_LEADER=SIZE_AS_UINT4\n\
BLOCK_TRAILER=LAST_4_BYTES_REPEATED\n\
KNOWN_INCOMPATIBLE_EDITION=NO\n",
    );
    if matches!(planar_configuration, tiff_core::PlanarConfiguration::Planar) {
        payload.push_str("INTERLEAVE=BAND\n");
    }
    payload.push(' ');
    format!(
        "GDAL_STRUCTURAL_METADATA_SIZE={:06} bytes\n{}",
        payload.len(),
        payload
    )
    .into_bytes()
}

#[derive(Debug, Clone, Copy)]
struct CogBlockEncoding {
    compression: Compression,
    predictor: Predictor,
    samples_per_pixel: u16,
    row_width_pixels: usize,
}

#[derive(Debug, Clone, Copy)]
struct TileWritePlan {
    tile_width: usize,
    tile_height: usize,
    planar_configuration: tiff_core::PlanarConfiguration,
    compression: Compression,
    predictor: Predictor,
}

/// Configuration for COG writing.
#[derive(Debug, Clone)]
pub struct CogBuilder {
    inner: GeoTiffBuilder,
    overview_levels: Vec<u32>,
    resampling: Resampling,
}

fn gdal_block_leader(payload_len: usize, byte_order: ByteOrder) -> Vec<u8> {
    let mut leader = Vec::with_capacity(4);
    let block_len = u32::try_from(payload_len).expect("COG block payload exceeds u32::MAX");
    leader.extend_from_slice(&byte_order.write_u32(block_len));
    leader
}

fn gdal_block_trailer(bytes: &[u8]) -> Vec<u8> {
    if bytes.len() >= 4 {
        bytes[bytes.len() - 4..].to_vec()
    } else {
        bytes.to_vec()
    }
}

fn write_cog_block<T: NumericSample, W: Write + Seek>(
    writer: &mut TiffWriter<W>,
    handle: &ImageHandle,
    block_index: usize,
    samples: &[T],
    encoding: CogBlockEncoding,
) -> Result<()> {
    let compressed = tiff_writer::compress::compress_block(
        samples,
        ByteOrder::LittleEndian,
        encoding.compression,
        encoding.predictor,
        encoding.samples_per_pixel,
        encoding.row_width_pixels,
        block_index,
    )?;
    let leader = gdal_block_leader(compressed.len(), ByteOrder::LittleEndian);
    let trailer = gdal_block_trailer(&compressed);
    writer.write_block_raw_segmented(handle, block_index, &leader, &compressed, &trailer)?;
    Ok(())
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
    pub fn write_2d<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_2d_to(writer, data)
    }

    /// Write a complete multi-band COG from a 3D array to a file path.
    pub fn write_3d<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView3<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_3d_to(writer, data)
    }

    /// Write a complete COG to any Write+Seek target.
    ///
    /// The file layout is:
    /// 1. TIFF header
    /// 2. GDAL structural metadata ghost area
    /// 3. Base image IFD
    /// 4. Overview IFDs (largest → smallest)
    /// 5. Tile offset/byte-count arrays
    /// 6. Tile data (overview tiles smallest first, then base tiles)
    pub fn write_2d_to<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView2<T>,
    ) -> Result<()> {
        if self.inner.bands != 1 {
            return Err(Error::InvalidConfig(
                "write_2d_to requires a single-band builder; use write_3d_to for multi-band COGs"
                    .into(),
            ));
        }

        self.write_array_to(sink, data.insert_axis(Axis(2)))
    }

    /// Write a complete multi-band COG to any Write+Seek target.
    pub fn write_3d_to<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView3<T>,
    ) -> Result<()> {
        self.write_array_to(sink, data)
    }

    fn write_array_to<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView3<T>,
    ) -> Result<()> {
        let (height, width, bands) = data.dim();
        if width as u32 != self.inner.width
            || height as u32 != self.inner.height
            || bands as u32 != self.inner.bands
        {
            return Err(Error::DataSizeMismatch {
                expected: self.inner.height as usize
                    * self.inner.width as usize
                    * self.inner.bands as usize,
                actual: height * width * bands,
            });
        }

        let tw = self.inner.tile_width.unwrap_or(256) as usize;
        let th = self.inner.tile_height.unwrap_or(256) as usize;

        // Generate overview rasters
        let overviews = self.generate_overviews(data);

        // --- Phase 1: Write header + GDAL metadata + all IFDs (with placeholder offsets) ---
        let mut writer = TiffWriter::new(
            sink,
            WriteOptions {
                byte_order: ByteOrder::LittleEndian,
                variant: TiffVariant::Classic,
            },
        )?;
        writer.write_header_prefix(&gdal_structural_metadata_bytes(
            self.inner.planar_configuration,
        ))?;

        // Base image IFD first, per GDAL COG layout.
        let base_ib = self.inner.to_image_builder::<T>();
        let base_handle = writer.add_image(base_ib)?;

        // Overview IFDs follow from largest to smallest in dimensions.
        let mut sorted_levels = self.overview_levels.clone();
        sorted_levels.sort_unstable();

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
                .planar_configuration(self.inner.planar_configuration)
                .tiles(tw as u32, th as u32)
                .overview();

            for tag in self.inner.build_extra_tags() {
                ovr_ib = ovr_ib.tag(tag);
            }

            let handle = writer.add_image(ovr_ib)?;
            overview_handles.push((handle, ovr_w, ovr_h));
        }

        // --- Phase 2: Write tile data ---

        // Overview tiles (smallest overview first)
        for &level in sorted_levels.iter().rev() {
            let idx = sorted_levels
                .iter()
                .position(|&candidate| candidate == level)
                .unwrap();
            let (ref handle, _, _) = overview_handles[idx];
            // Find the matching overview array (overviews are stored in original order)
            let level_idx = self
                .overview_levels
                .iter()
                .position(|&l| l == level)
                .unwrap();
            let overview = &overviews[level_idx];
            write_tiled_data_3d::<T, W>(
                &mut writer,
                handle,
                overview.view(),
                TileWritePlan {
                    tile_width: tw,
                    tile_height: th,
                    planar_configuration: self.inner.planar_configuration,
                    compression: self.inner.compression,
                    predictor: self.inner.predictor,
                },
            )?;
        }

        // Base tiles
        write_tiled_data_3d::<T, W>(
            &mut writer,
            &base_handle,
            data,
            TileWritePlan {
                tile_width: tw,
                tile_height: th,
                planar_configuration: self.inner.planar_configuration,
                compression: self.inner.compression,
                predictor: self.inner.predictor,
            },
        )?;

        writer.finish()?;
        Ok(())
    }

    /// Create a streaming COG tile writer.
    ///
    /// The writer creates the base IFD and overview IFD placeholders immediately.
    /// Base tiles are written incrementally via `write_tile`. Overview tiles are
    /// generated from the base tiles during `finish()`.
    pub fn tile_writer<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
    ) -> Result<CogTileWriter<T, W>> {
        CogTileWriter::new(self.clone(), sink)
    }

    /// Create a streaming COG tile writer to a file.
    pub fn tile_writer_file<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<CogTileWriter<T, BufWriter<File>>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.tile_writer(writer)
    }

    fn generate_overviews<T: NumericSample>(&self, data: ArrayView3<T>) -> Vec<Array3<T>> {
        let (height, width, bands) = data.dim();
        self.overview_levels
            .iter()
            .map(|&level| {
                let level = level as usize;
                let ovr_w = width.div_ceil(level);
                let ovr_h = height.div_ceil(level);

                Array3::from_shape_fn((ovr_h, ovr_w, bands), |(r, c, band)| {
                    match self.resampling {
                        Resampling::NearestNeighbor => {
                            let src_r = (r * level).min(height - 1);
                            let src_c = (c * level).min(width - 1);
                            data[[src_r, src_c, band]]
                        }
                        Resampling::Average => {
                            let start_r = r * level;
                            let start_c = c * level;
                            let end_r = (start_r + level).min(height);
                            let end_c = (start_c + level).min(width);
                            let mut sum = 0.0;
                            let mut count = 0usize;
                            for sr in start_r..end_r {
                                for sc in start_c..end_c {
                                    sum += data[[sr, sc, band]].to_f64();
                                    count += 1;
                                }
                            }
                            if count == 0 {
                                T::zero()
                            } else {
                                T::from_f64(sum / count as f64)
                            }
                        }
                    }
                })
            })
            .collect()
    }
}

/// Streaming COG tile writer.
///
/// Base tiles are accumulated in memory. Overview and base blocks are emitted
/// during `finish()` in canonical COG order, and the file-level COG ghost area
/// is handled automatically.
pub struct CogTileWriter<T: NumericSample, W: Write + Seek> {
    writer: TiffWriter<W>,
    overview_handles: Vec<(ImageHandle, u32, u32)>, // (handle, width, height)
    base_handle: ImageHandle,
    base_pixels: Vec<T>,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    width: u32,
    height: u32,
    bands: u32,
    planar_configuration: tiff_core::PlanarConfiguration,
    compression: Compression,
    predictor: Predictor,
    overview_levels: Vec<u32>,
    resampling: Resampling,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: NumericSample, W: Write + Seek> CogTileWriter<T, W> {
    fn new(cog: CogBuilder, sink: W) -> Result<Self> {
        let tw = cog.inner.tile_width.unwrap_or(256);
        let th = cog.inner.tile_height.unwrap_or(256);
        let tiles_across = (cog.inner.width as usize).div_ceil(tw as usize);
        let tiles_down = (cog.inner.height as usize).div_ceil(th as usize);
        let fill_value = T::zero();

        let mut writer = TiffWriter::new(sink, WriteOptions::default())?;
        writer.write_header_prefix(&gdal_structural_metadata_bytes(
            cog.inner.planar_configuration,
        ))?;

        // Base image IFD first, followed by overview IFDs from largest to smallest.
        let base_ib = cog.inner.to_image_builder::<T>();
        let base_handle = writer.add_image(base_ib)?;

        let mut sorted_levels = cog.overview_levels.clone();
        sorted_levels.sort_unstable();

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
                .planar_configuration(cog.inner.planar_configuration)
                .tiles(tw, th)
                .overview();

            for tag in cog.inner.build_extra_tags() {
                ovr_ib = ovr_ib.tag(tag);
            }

            let handle = writer.add_image(ovr_ib)?;
            overview_handles.push((handle, ovr_w, ovr_h));
        }
        Ok(Self {
            writer,
            overview_handles,
            base_handle,
            base_pixels: vec![
                fill_value;
                cog.inner.width as usize
                    * cog.inner.height as usize
                    * cog.inner.bands as usize
            ],
            tile_width: tw,
            tile_height: th,
            tiles_across: tiles_across as u32,
            tiles_down: tiles_down as u32,
            width: cog.inner.width,
            height: cog.inner.height,
            bands: cog.inner.bands,
            planar_configuration: cog.inner.planar_configuration,
            compression: cog.inner.compression,
            predictor: cog.inner.predictor,
            overview_levels: sorted_levels,
            resampling: cog.resampling,
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
        if self.bands != 1 {
            return Err(Error::Other(
                "write_tile only supports single-band COG output; use write_tile_3d for multi-band tiles".into(),
            ));
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

        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let (data_h, data_w) = data.dim();

        for row in 0..data_h.min(th) {
            for col in 0..data_w.min(tw) {
                let value = data[[row, col]];
                let dst_row = y_off + row;
                let dst_col = x_off + col;
                if dst_row < self.height as usize && dst_col < self.width as usize {
                    let pixel_index = dst_row * self.width as usize + dst_col;
                    self.base_pixels[pixel_index] = value;
                }
            }
        }

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

        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let (data_h, data_w, data_b) = data.dim();
        let bands = self.bands as usize;
        if data_b != bands {
            return Err(Error::DataSizeMismatch {
                expected: data_h * data_w * bands,
                actual: data_h * data_w * data_b,
            });
        }

        for row in 0..data_h.min(th) {
            for col in 0..data_w.min(tw) {
                let dst_row = y_off + row;
                let dst_col = x_off + col;
                if dst_row >= self.height as usize || dst_col >= self.width as usize {
                    continue;
                }
                let pixel_index = (dst_row * self.width as usize + dst_col) * bands;
                for band in 0..bands {
                    self.base_pixels[pixel_index + band] = data[[row, col, band]];
                }
            }
        }

        Ok(())
    }

    /// Finish: generate overview tiles from accumulated base tiles, write everything, finalize.
    pub fn finish(mut self) -> Result<W> {
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let full_w = self.width as usize;
        let full_h = self.height as usize;
        let bands = self.bands as usize;

        let full = Array3::from_shape_vec((full_h, full_w, bands), self.base_pixels)
            .map_err(|err| Error::Other(format!("invalid streaming COG raster shape: {err}")))?;

        // Generate and write overview tiles
        for &level in self.overview_levels.iter().rev() {
            let idx = self
                .overview_levels
                .iter()
                .position(|&candidate| candidate == level)
                .unwrap();
            let (ref handle, _, _) = self.overview_handles[idx];

            let overview = generate_overview_3d(full.view(), level as usize, self.resampling);

            write_tiled_data_3d::<T, W>(
                &mut self.writer,
                handle,
                overview.view(),
                TileWritePlan {
                    tile_width: tw,
                    tile_height: th,
                    planar_configuration: self.planar_configuration,
                    compression: self.compression,
                    predictor: self.predictor,
                },
            )?;
        }

        write_tiled_data_3d::<T, W>(
            &mut self.writer,
            &self.base_handle,
            full.view(),
            TileWritePlan {
                tile_width: tw,
                tile_height: th,
                planar_configuration: self.planar_configuration,
                compression: self.compression,
                predictor: self.predictor,
            },
        )?;

        Ok(self.writer.finish()?)
    }
}

// -- Helper functions --

fn generate_overview_3d<T: NumericSample>(
    data: ArrayView3<T>,
    level: usize,
    resampling: Resampling,
) -> Array3<T> {
    let (height, width, bands) = data.dim();
    let ovr_w = width.div_ceil(level);
    let ovr_h = height.div_ceil(level);

    Array3::from_shape_fn((ovr_h, ovr_w, bands), |(r, c, band)| match resampling {
        Resampling::NearestNeighbor => {
            let src_r = (r * level).min(height - 1);
            let src_c = (c * level).min(width - 1);
            data[[src_r, src_c, band]]
        }
        Resampling::Average => {
            let start_r = r * level;
            let start_c = c * level;
            let end_r = (start_r + level).min(height);
            let end_c = (start_c + level).min(width);
            let mut sum = 0.0;
            let mut count = 0usize;
            for sr in start_r..end_r {
                for sc in start_c..end_c {
                    sum += data[[sr, sc, band]].to_f64();
                    count += 1;
                }
            }
            if count == 0 {
                T::zero()
            } else {
                T::from_f64(sum / count as f64)
            }
        }
    })
}

fn write_tiled_data_3d<T: NumericSample, W: Write + Seek>(
    writer: &mut TiffWriter<W>,
    handle: &ImageHandle,
    data: ArrayView3<T>,
    plan: TileWritePlan,
) -> Result<()> {
    let (height, width, bands) = data.dim();
    let tw = plan.tile_width;
    let th = plan.tile_height;
    let tiles_across = width.div_ceil(tw);
    let tiles_down = height.div_ceil(th);

    if matches!(
        plan.planar_configuration,
        tiff_core::PlanarConfiguration::Planar
    ) {
        let tiles_per_plane = tiles_across * tiles_down;
        for band in 0..bands {
            for tile_row in 0..tiles_down {
                for tile_col in 0..tiles_across {
                    let tile_index = tile_row * tiles_across + tile_col;
                    let mut tile_data = vec![T::zero(); tw * th];
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
                            tile_data[row * tw + col] = data[[src_row, src_col, band]];
                        }
                    }
                    let block_index = band * tiles_per_plane + tile_index;
                    write_cog_block(
                        writer,
                        handle,
                        block_index,
                        &tile_data,
                        CogBlockEncoding {
                            compression: plan.compression,
                            predictor: plan.predictor,
                            samples_per_pixel: 1,
                            row_width_pixels: tw,
                        },
                    )?;
                }
            }
        }
    } else {
        for tile_row in 0..tiles_down {
            for tile_col in 0..tiles_across {
                let tile_index = tile_row * tiles_across + tile_col;
                let mut tile_data = vec![T::zero(); tw * th * bands];
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
                        for band in 0..bands {
                            tile_data[(row * tw + col) * bands + band] =
                                data[[src_row, src_col, band]];
                        }
                    }
                }
                write_cog_block(
                    writer,
                    handle,
                    tile_index,
                    &tile_data,
                    CogBlockEncoding {
                        compression: plan.compression,
                        predictor: plan.predictor,
                        samples_per_pixel: bands as u16,
                        row_width_pixels: tw,
                    },
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeoTiffBuilder;
    use ndarray::{Array2, Array3};
    use std::io::Cursor;
    use tiff_core::{Compression, PhotometricInterpretation, PlanarConfiguration};

    fn assert_strictly_increasing_offsets(offsets: &[u64], context: &str) {
        for window in offsets.windows(2) {
            assert!(
                window[0] < window[1],
                "{context}: offsets are not strictly increasing: {offsets:?}"
            );
        }
    }

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

        // Base + 2 overviews = 3 IFDs
        assert_eq!(file.ifd_count(), 3);

        // IFD 0 is the base image.
        assert_eq!(file.ifd(0).unwrap().width(), 32);
        let base = file.read_image::<u8>(0).unwrap();
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
        assert_eq!(file.ifd_count(), 1);
        let img = file.read_image::<f32>(0).unwrap();
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
        assert_eq!(file.ifd_count(), 2);

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
        assert_eq!(file.ifd(0).unwrap().width(), 64); // base IFD
        assert!(file.ifd_count() >= 3); // base + 2 overviews

        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(geo.epsg(), Some(4326));
        assert_eq!(geo.base_ifd_index(), 0);
        assert_eq!(geo.width(), 64);
        assert_eq!(geo.height(), 64);
        assert_eq!(geo.overview_count(), 2);

        let base = geo.read_raster::<u8>().unwrap();
        assert_eq!(base.shape(), &[64, 64]);
        assert_eq!(base[[0, 0]], 42);

        let ovr32 = geo.read_overview::<u8>(0).unwrap();
        assert_eq!(ovr32.shape(), &[32, 32]);

        let ovr16 = geo.read_overview::<u8>(1).unwrap();
        assert_eq!(ovr16.shape(), &[16, 16]);
    }

    #[test]
    fn cog_base_ifd_is_first() {
        let data = Array2::<u8>::from_elem((32, 32), 1);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();

        // First IFD should be the base image
        let first = file.ifd(0).unwrap();
        assert_eq!(first.width(), 32);
        assert_eq!(first.height(), 32);
    }

    #[test]
    fn cog_writes_gdal_structural_metadata_after_header() {
        let data = Array2::<u8>::from_elem((32, 32), 1);

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let prefix = gdal_structural_metadata_bytes(tiff_core::PlanarConfiguration::Chunky);
        assert_eq!(&bytes[8..8 + prefix.len()], prefix.as_slice());
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

        // Base + overview = 2 IFDs
        assert_eq!(file.ifd_count(), 2);
        let base = file.read_image::<u8>(0).unwrap();
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

    #[test]
    fn cog_write_and_read_multiband_rgb_chunky() {
        let mut data = Array3::<u8>::zeros((32, 32, 3));
        for row in 0..32 {
            for col in 0..32 {
                data[[row, col, 0]] = ((row * 7 + col * 3) % 251) as u8;
                data[[row, col, 1]] = ((row * 5 + col * 11) % 251) as u8;
                data[[row, col, 2]] = ((row * 13 + col * 17) % 251) as u8;
            }
        }

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .tile_size(16, 16)
            .epsg(4326)
            .pixel_scale(1.0, 1.0)
            .origin(0.0, 32.0);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .resampling(Resampling::NearestNeighbor)
            .write_3d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes.clone()).unwrap();
        assert_eq!(geo.base_ifd_index(), 0);
        assert_eq!(geo.width(), 32);
        assert_eq!(geo.height(), 32);
        assert_eq!(geo.overview_count(), 1);

        let base = geo.read_raster::<u8>().unwrap();
        assert_eq!(base.shape(), &[32, 32, 3]);
        assert_eq!(base[[5, 9, 0]], data[[5, 9, 0]]);
        assert_eq!(base[[5, 9, 1]], data[[5, 9, 1]]);
        assert_eq!(base[[5, 9, 2]], data[[5, 9, 2]]);

        let overview = geo.read_overview::<u8>(0).unwrap();
        assert_eq!(overview.shape(), &[16, 16, 3]);
        assert_eq!(overview[[4, 6, 0]], data[[8, 12, 0]]);
        assert_eq!(overview[[4, 6, 1]], data[[8, 12, 1]]);
        assert_eq!(overview[[4, 6, 2]], data[[8, 12, 2]]);
    }

    #[test]
    fn cog_write_and_read_multiband_rgb_planar() {
        let mut data = Array3::<u8>::zeros((32, 32, 3));
        for row in 0..32 {
            for col in 0..32 {
                data[[row, col, 0]] = ((row * 3 + col * 5) % 251) as u8;
                data[[row, col, 1]] = ((row * 11 + col * 7) % 251) as u8;
                data[[row, col, 2]] = ((row * 17 + col * 13) % 251) as u8;
            }
        }

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .planar_configuration(PlanarConfiguration::Planar)
            .tile_size(16, 16)
            .epsg(4326);

        CogBuilder::new(builder)
            .overview_levels(vec![2])
            .write_3d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let tiff = tiff_reader::TiffFile::from_bytes(bytes.clone()).unwrap();
        let base_idx = (0..tiff.ifd_count())
            .find(|&i| tiff.ifd(i).unwrap().width() == 32)
            .unwrap();
        let overview_idx = (0..tiff.ifd_count())
            .find(|&i| tiff.ifd(i).unwrap().width() == 16)
            .unwrap();
        assert_eq!(tiff.ifd(base_idx).unwrap().planar_configuration(), 2);
        assert_eq!(tiff.ifd(overview_idx).unwrap().planar_configuration(), 2);
        assert_strictly_increasing_offsets(
            &tiff.ifd(base_idx).unwrap().tile_offsets().unwrap(),
            "planar COG base image",
        );
        assert_strictly_increasing_offsets(
            &tiff.ifd(overview_idx).unwrap().tile_offsets().unwrap(),
            "planar COG overview",
        );

        let base = tiff.read_image::<u8>(base_idx).unwrap();
        assert_eq!(base.shape(), &[32, 32, 3]);
        assert_eq!(base[[7, 10, 0]], data[[7, 10, 0]]);
        assert_eq!(base[[7, 10, 1]], data[[7, 10, 1]]);
        assert_eq!(base[[7, 10, 2]], data[[7, 10, 2]]);

        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        let raster = geo.read_raster::<u8>().unwrap();
        assert_eq!(raster.shape(), &[32, 32, 3]);
    }

    #[test]
    fn cog_streaming_multiband_planar_matches_oneshot() {
        let mut data = Array3::<u8>::zeros((32, 32, 3));
        for row in 0..32 {
            for col in 0..32 {
                data[[row, col, 0]] = ((row * 19 + col * 3) % 251) as u8;
                data[[row, col, 1]] = ((row * 7 + col * 23) % 251) as u8;
                data[[row, col, 2]] = ((row * 13 + col * 29) % 251) as u8;
            }
        }

        let builder = GeoTiffBuilder::new(32, 32)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .planar_configuration(PlanarConfiguration::Planar)
            .tile_size(16, 16)
            .compression(Compression::Deflate)
            .epsg(4326);

        let mut oneshot_buf = Cursor::new(Vec::new());
        CogBuilder::new(builder.clone())
            .overview_levels(vec![2])
            .write_3d_to(&mut oneshot_buf, data.view())
            .unwrap();

        let mut streaming_buf = Cursor::new(Vec::new());
        let mut writer = CogBuilder::new(builder)
            .overview_levels(vec![2])
            .tile_writer::<u8, _>(&mut streaming_buf)
            .unwrap();
        for tile_row in 0..2usize {
            for tile_col in 0..2usize {
                let y_off = tile_row * 16;
                let x_off = tile_col * 16;
                let tile = data
                    .slice(ndarray::s![y_off..y_off + 16, x_off..x_off + 16, ..])
                    .to_owned();
                writer.write_tile_3d(x_off, y_off, &tile.view()).unwrap();
            }
        }
        writer.finish().unwrap();

        let oneshot = geotiff_reader::GeoTiffFile::from_bytes(oneshot_buf.into_inner()).unwrap();
        let streaming =
            geotiff_reader::GeoTiffFile::from_bytes(streaming_buf.into_inner()).unwrap();

        assert_strictly_increasing_offsets(
            &oneshot.tiff().ifd(0).unwrap().tile_offsets().unwrap(),
            "oneshot planar COG base image",
        );
        assert_strictly_increasing_offsets(
            &streaming.tiff().ifd(0).unwrap().tile_offsets().unwrap(),
            "streaming planar COG base image",
        );

        let oneshot_base = oneshot.read_raster::<u8>().unwrap();
        let streaming_base = streaming.read_raster::<u8>().unwrap();
        assert_eq!(oneshot_base.shape(), streaming_base.shape());
        let (oneshot_values, _) = oneshot_base.into_raw_vec_and_offset();
        let (streaming_values, _) = streaming_base.into_raw_vec_and_offset();
        assert_eq!(oneshot_values, streaming_values);

        let oneshot_overview = oneshot.read_overview::<u8>(0).unwrap();
        let streaming_overview = streaming.read_overview::<u8>(0).unwrap();
        let (oneshot_values, _) = oneshot_overview.into_raw_vec_and_offset();
        let (streaming_values, _) = streaming_overview.into_raw_vec_and_offset();
        assert_eq!(oneshot_values, streaming_values);
    }
}
