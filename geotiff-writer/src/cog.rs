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
use tiff_writer::LercOptions;
use tiff_writer::{ImageBuilder, ImageHandle, TiffWriter, WriteOptions};

use crate::builder::GeoTiffBuilder;
use crate::error::{Error, Result};
use crate::sample::{parse_nodata_value, NumericSample};

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
    block_height: u32,
    lerc_options: Option<LercOptions>,
}

#[derive(Debug, Clone, Copy)]
struct TileWritePlan {
    tile_width: usize,
    tile_height: usize,
    planar_configuration: tiff_core::PlanarConfiguration,
    compression: Compression,
    predictor: Predictor,
    lerc_options: Option<LercOptions>,
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
    let compressed = if matches!(encoding.compression, Compression::Lerc) {
        let opts = encoding.lerc_options.unwrap_or_default();
        tiff_writer::compress::compress_block_lerc(
            samples,
            encoding.row_width_pixels as u32,
            encoding.block_height,
            encoding.samples_per_pixel as u32,
            &opts,
            block_index,
        )?
    } else {
        tiff_writer::compress::compress_block(
            samples,
            ByteOrder::LittleEndian,
            encoding.compression,
            encoding.predictor,
            encoding.samples_per_pixel,
            encoding.row_width_pixels,
            block_index,
        )?
    };
    let leader = gdal_block_leader(compressed.len(), ByteOrder::LittleEndian);
    let trailer = gdal_block_trailer(&compressed);
    writer.write_block_raw_segmented(handle, block_index, &leader, &compressed, &trailer)?;
    Ok(())
}

fn validate_overview_levels(levels: &[u32]) -> Result<Vec<u32>> {
    if let Some(invalid) = levels.iter().copied().find(|&level| level <= 1) {
        return Err(Error::InvalidConfig(format!(
            "overview levels must be greater than 1, got {invalid}"
        )));
    }

    let mut normalized = levels.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
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

    fn normalized_overview_levels(&self) -> Result<Vec<u32>> {
        validate_overview_levels(&self.overview_levels)
    }

    fn overview_image_builder<T: NumericSample>(
        &self,
        level: u32,
        tile_width: u32,
        tile_height: u32,
    ) -> ImageBuilder {
        let ovr_w = (self.inner.width as usize).div_ceil(level as usize) as u32;
        let ovr_h = (self.inner.height as usize).div_ceil(level as usize) as u32;

        let mut builder = ImageBuilder::new(ovr_w, ovr_h)
            .sample_type::<T>()
            .samples_per_pixel(self.inner.bands as u16)
            .compression(self.inner.compression)
            .predictor(self.inner.predictor)
            .photometric(self.inner.photometric)
            .planar_configuration(self.inner.planar_configuration)
            .tiles(tile_width, tile_height)
            .overview();

        if let Some(opts) = self.inner.lerc_options {
            builder = builder.lerc_options(opts);
        }

        for tag in self.inner.build_extra_tags() {
            builder = builder.tag(tag);
        }

        builder
    }

    fn estimated_output_bytes<T: NumericSample>(&self, overview_levels: &[u32]) -> u64 {
        let metadata_len =
            gdal_structural_metadata_bytes(self.inner.planar_configuration).len() as u64;
        let base = self.inner.to_image_builder::<T>();
        let base_blocks = u64::try_from(base.block_count()).unwrap_or(u64::MAX);
        let mut estimated = metadata_len
            .saturating_add(base.estimated_uncompressed_bytes())
            .saturating_add(base_blocks.saturating_mul(8));

        let tw = self.inner.tile_width.unwrap_or(256);
        let th = self.inner.tile_height.unwrap_or(256);
        for &level in overview_levels {
            let overview = self.overview_image_builder::<T>(level, tw, th);
            let block_count = u64::try_from(overview.block_count()).unwrap_or(u64::MAX);
            estimated = estimated
                .saturating_add(overview.estimated_uncompressed_bytes())
                .saturating_add(block_count.saturating_mul(8));
        }

        estimated
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
        let overview_levels = self.normalized_overview_levels()?;
        let nodata = parse_nodata_value::<T>(&self.inner.nodata);

        // Generate overview rasters
        let overviews = self.generate_overviews(data, nodata, &overview_levels);

        // --- Phase 1: Write header + GDAL metadata + all IFDs (with placeholder offsets) ---
        let mut writer = TiffWriter::new(
            sink,
            WriteOptions::auto(self.estimated_output_bytes::<T>(&overview_levels)),
        )?;
        writer.write_header_prefix(&gdal_structural_metadata_bytes(
            self.inner.planar_configuration,
        ))?;

        // Base image IFD first, per GDAL COG layout.
        let base_ib = self.inner.to_image_builder::<T>();
        let base_handle = writer.add_image(base_ib)?;

        // Overview IFDs follow from largest to smallest in dimensions.
        let mut overview_handles: Vec<(ImageHandle, u32, u32)> = Vec::new();
        for &level in &overview_levels {
            let ovr_w = (self.inner.width as usize).div_ceil(level as usize) as u32;
            let ovr_h = (self.inner.height as usize).div_ceil(level as usize) as u32;

            let ovr_ib = self.overview_image_builder::<T>(level, tw as u32, th as u32);
            let handle = writer.add_image(ovr_ib)?;
            overview_handles.push((handle, ovr_w, ovr_h));
        }

        // --- Phase 2: Write tile data ---

        // Overview tiles (smallest overview first)
        for idx in (0..overview_levels.len()).rev() {
            let (ref handle, _, _) = overview_handles[idx];
            let overview = &overviews[idx];
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
                    lerc_options: self.inner.lerc_options,
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
                lerc_options: self.inner.lerc_options,
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

    fn generate_overviews<T: NumericSample>(
        &self,
        data: ArrayView3<T>,
        nodata: Option<T>,
        overview_levels: &[u32],
    ) -> Vec<Array3<T>> {
        let (height, width, bands) = data.dim();
        overview_levels
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
                                    let value = data[[sr, sc, band]];
                                    if nodata.is_some_and(|nodata_value| value == nodata_value) {
                                        continue;
                                    }
                                    sum += value.to_f64();
                                    count += 1;
                                }
                            }
                            if count == 0 {
                                nodata.unwrap_or_else(T::zero)
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
    lerc_options: Option<LercOptions>,
    overview_levels: Vec<u32>,
    resampling: Resampling,
    nodata_value: Option<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: NumericSample, W: Write + Seek> CogTileWriter<T, W> {
    fn new(cog: CogBuilder, sink: W) -> Result<Self> {
        let tw = cog.inner.tile_width.unwrap_or(256);
        let th = cog.inner.tile_height.unwrap_or(256);
        let tiles_across = (cog.inner.width as usize).div_ceil(tw as usize);
        let tiles_down = (cog.inner.height as usize).div_ceil(th as usize);
        let overview_levels = cog.normalized_overview_levels()?;
        let nodata_value = parse_nodata_value::<T>(&cog.inner.nodata);
        let fill_value = nodata_value.unwrap_or_else(T::zero);

        let mut writer = TiffWriter::new(
            sink,
            WriteOptions::auto(cog.estimated_output_bytes::<T>(&overview_levels)),
        )?;
        writer.write_header_prefix(&gdal_structural_metadata_bytes(
            cog.inner.planar_configuration,
        ))?;

        // Base image IFD first, followed by overview IFDs from largest to smallest.
        let base_ib = cog.inner.to_image_builder::<T>();
        let base_handle = writer.add_image(base_ib)?;

        let mut overview_handles = Vec::new();
        for &level in &overview_levels {
            let ovr_w = (cog.inner.width as usize).div_ceil(level as usize) as u32;
            let ovr_h = (cog.inner.height as usize).div_ceil(level as usize) as u32;

            let ovr_ib = cog.overview_image_builder::<T>(level, tw, th);
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
            lerc_options: cog.inner.lerc_options,
            overview_levels,
            resampling: cog.resampling,
            nodata_value,
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
        for idx in (0..self.overview_levels.len()).rev() {
            let level = self.overview_levels[idx] as usize;
            let (ref handle, _, _) = self.overview_handles[idx];

            let overview =
                generate_overview_3d(full.view(), level, self.resampling, self.nodata_value);

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
                    lerc_options: self.lerc_options,
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
                lerc_options: self.lerc_options,
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
    nodata: Option<T>,
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
                    let value = data[[sr, sc, band]];
                    if nodata.is_some_and(|nodata_value| value == nodata_value) {
                        continue;
                    }
                    sum += value.to_f64();
                    count += 1;
                }
            }
            if count == 0 {
                nodata.unwrap_or_else(T::zero)
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
                            block_height: th as u32,
                            lerc_options: plan.lerc_options,
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
                        block_height: th as u32,
                        lerc_options: plan.lerc_options,
                    },
                )?;
            }
        }
    }
    Ok(())
}
