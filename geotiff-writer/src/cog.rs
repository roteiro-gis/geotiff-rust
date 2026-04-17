//! Cloud Optimized GeoTIFF (COG) writer.
//!
//! COG files have a specific byte layout:
//! 1. TIFF header
//! 2. GDAL structural metadata block (the COG "ghost area")
//! 3. Base image IFD (full resolution)
//! 4. Overview IFDs (largest → smallest)
//! 5. Tile offset/byte-count arrays
//! 6. Tile data: overviews (smallest first), then base image
//!
//! The IFDs-before-data layout allows HTTP range-request readers to fetch
//! all metadata in a single request from the start of the file.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use ndarray::{Array3, ArrayView2, ArrayView3, Axis};
use tempfile::tempfile;
use tiff_core::{ByteOrder, Compression, Predictor, Tag};
use tiff_writer::{encoder, ImageBuilder, TiffVariant};
use tiff_writer::{JpegOptions, LercOptions};

use crate::builder::GeoTiffBuilder;
use crate::error::{Error, Result};
use crate::sample::{parse_nodata_value, NumericSample};

/// Overview resampling algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    NearestNeighbor,
    Average,
}

fn checked_len_u64(len: usize, context: &str) -> Result<u64> {
    u64::try_from(len).map_err(|_| Error::Other(format!("{context} length exceeds u64::MAX")))
}

fn checked_add_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::Other(format!("{context} overflow")))
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
    jpeg_options: Option<JpegOptions>,
}

#[derive(Debug, Clone, Copy)]
struct TileWritePlan {
    tile_width: usize,
    tile_height: usize,
    planar_configuration: tiff_core::PlanarConfiguration,
    compression: Compression,
    predictor: Predictor,
    lerc_options: Option<LercOptions>,
    jpeg_options: Option<JpegOptions>,
}

#[derive(Debug, Clone, Copy)]
struct CogBlockRecord {
    spool_offset: u64,
    logical_offset_delta: u64,
    logical_byte_count: u64,
}

struct CogImage {
    builder: ImageBuilder,
    blocks: Vec<CogBlockRecord>,
}

struct PlannedCogImage {
    tags: Vec<Tag>,
    block_offsets: Vec<u64>,
    block_byte_counts: Vec<u64>,
}

struct CogLayout {
    base_offset: u64,
    is_bigtiff: bool,
    images: Vec<PlannedCogImage>,
}

struct BlockSpool {
    file: File,
    len: u64,
}

impl BlockSpool {
    fn new() -> Result<Self> {
        Ok(Self {
            file: tempfile()?,
            len: 0,
        })
    }

    fn append_segmented(
        &mut self,
        prefix: &[u8],
        payload: &[u8],
        suffix: &[u8],
    ) -> Result<CogBlockRecord> {
        let spool_offset = self.len;
        let prefix_len = checked_len_u64(prefix.len(), "COG block prefix")?;
        let payload_len = checked_len_u64(payload.len(), "COG block payload")?;
        let suffix_len = checked_len_u64(suffix.len(), "COG block suffix")?;
        let physical_len = checked_add_u64(
            checked_add_u64(prefix_len, payload_len, "COG block size")?,
            suffix_len,
            "COG block size",
        )?;

        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(prefix)?;
        self.file.write_all(payload)?;
        self.file.write_all(suffix)?;
        self.len = checked_add_u64(self.len, physical_len, "COG spool length")?;

        Ok(CogBlockRecord {
            spool_offset,
            logical_offset_delta: prefix_len,
            logical_byte_count: payload_len,
        })
    }

    fn copy_into<W: Write + Seek>(&mut self, sink: &mut W) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        sink.seek(SeekFrom::End(0))?;
        io::copy(&mut self.file, sink)?;
        Ok(())
    }
}

/// Configuration for COG writing.
#[derive(Debug, Clone)]
pub struct CogBuilder {
    inner: GeoTiffBuilder,
    overview_levels: Vec<u32>,
    resampling: Resampling,
}

fn gdal_block_leader(payload_len: usize, byte_order: ByteOrder) -> Result<Vec<u8>> {
    let block_len = u32::try_from(payload_len)
        .map_err(|_| Error::Other("COG block payload exceeds u32::MAX".into()))?;
    let mut leader = Vec::with_capacity(4);
    leader.extend_from_slice(&byte_order.write_u32(block_len));
    Ok(leader)
}

fn gdal_block_trailer(bytes: &[u8]) -> Vec<u8> {
    if bytes.len() >= 4 {
        bytes[bytes.len() - 4..].to_vec()
    } else {
        bytes.to_vec()
    }
}

fn compress_cog_block<T: NumericSample>(
    samples: &[T],
    block_index: usize,
    encoding: CogBlockEncoding,
) -> Result<Vec<u8>> {
    if matches!(encoding.compression, Compression::Lerc) {
        let opts = encoding.lerc_options.unwrap_or_default();
        tiff_writer::compress::compress_block_lerc(
            samples,
            encoding.row_width_pixels as u32,
            encoding.block_height,
            encoding.samples_per_pixel as u32,
            &opts,
            block_index,
        )
        .map_err(Into::into)
    } else {
        tiff_writer::compress::compress_block(
            samples,
            tiff_writer::compress::BlockEncodingOptions {
                byte_order: ByteOrder::LittleEndian,
                compression: encoding.compression,
                predictor: encoding.predictor,
                samples_per_pixel: encoding.samples_per_pixel,
                row_width_pixels: encoding.row_width_pixels,
                jpeg_options: encoding.jpeg_options.as_ref(),
            },
            block_index,
        )
        .map_err(Into::into)
    }
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

fn plan_cog_layout_for_variant(
    base_offset: u64,
    prefix_len: u64,
    images: &[CogImage],
    is_bigtiff: bool,
) -> Result<CogLayout> {
    let mut image_plans = Vec::with_capacity(images.len());
    let mut current = checked_add_u64(
        checked_add_u64(
            base_offset,
            encoder::header_len(is_bigtiff),
            "COG header size",
        )?,
        prefix_len,
        "COG prefix size",
    )?;

    for image in images {
        let expected_blocks = image.builder.block_count();
        if image.blocks.len() != expected_blocks {
            return Err(Error::Other(format!(
                "COG image is missing block records: expected {expected_blocks}, got {}",
                image.blocks.len()
            )));
        }
        let tags = image.builder.build_tags(is_bigtiff);
        current = checked_add_u64(
            current,
            encoder::estimate_ifd_size(ByteOrder::LittleEndian, is_bigtiff, &tags),
            "COG IFD layout",
        )?;
        if !is_bigtiff {
            u32::try_from(current).map_err(|_| {
                Error::Tiff(tiff_writer::Error::ClassicOffsetOverflow { offset: current })
            })?;
        }
        image_plans.push(PlannedCogImage {
            tags,
            block_offsets: Vec::with_capacity(image.blocks.len()),
            block_byte_counts: Vec::with_capacity(image.blocks.len()),
        });
    }

    let data_start = current;
    for (image, planned) in images.iter().zip(image_plans.iter_mut()) {
        for block in &image.blocks {
            let physical_start =
                checked_add_u64(data_start, block.spool_offset, "COG block physical offset")?;
            let logical_offset = checked_add_u64(
                physical_start,
                block.logical_offset_delta,
                "COG block logical offset",
            )?;
            if !is_bigtiff {
                u32::try_from(logical_offset).map_err(|_| {
                    Error::Tiff(tiff_writer::Error::ClassicOffsetOverflow {
                        offset: logical_offset,
                    })
                })?;
                u32::try_from(block.logical_byte_count).map_err(|_| {
                    Error::Tiff(tiff_writer::Error::ClassicByteCountOverflow {
                        byte_count: block.logical_byte_count,
                    })
                })?;
            }
            planned.block_offsets.push(logical_offset);
            planned.block_byte_counts.push(block.logical_byte_count);
        }
    }

    Ok(CogLayout {
        base_offset,
        is_bigtiff,
        images: image_plans,
    })
}

fn plan_cog_layout(
    base_offset: u64,
    prefix_len: u64,
    variant: TiffVariant,
    images: &[CogImage],
) -> Result<CogLayout> {
    match variant {
        TiffVariant::Classic => plan_cog_layout_for_variant(base_offset, prefix_len, images, false),
        TiffVariant::BigTiff => plan_cog_layout_for_variant(base_offset, prefix_len, images, true),
        TiffVariant::Auto => {
            match plan_cog_layout_for_variant(base_offset, prefix_len, images, false) {
                Ok(layout) => Ok(layout),
                Err(Error::Tiff(
                    tiff_writer::Error::ClassicOffsetOverflow { .. }
                    | tiff_writer::Error::ClassicByteCountOverflow { .. },
                )) => plan_cog_layout_for_variant(base_offset, prefix_len, images, true),
                Err(err) => Err(err),
            }
        }
    }
}

fn emit_cog<W: Write + Seek>(
    sink: &mut W,
    prefix: &[u8],
    images: &[CogImage],
    layout: &CogLayout,
    spool: &mut BlockSpool,
) -> Result<()> {
    sink.seek(SeekFrom::Start(layout.base_offset))?;
    encoder::write_header(sink, ByteOrder::LittleEndian, layout.is_bigtiff)?;
    sink.write_all(prefix)?;

    let mut ifd_results = Vec::with_capacity(images.len());
    for (image, planned) in images.iter().zip(&layout.images) {
        let (offsets_tag_code, byte_counts_tag_code) = image.builder.offset_tag_codes();
        let ifd_result = encoder::write_ifd(
            sink,
            ByteOrder::LittleEndian,
            layout.is_bigtiff,
            &planned.tags,
            offsets_tag_code,
            byte_counts_tag_code,
            image.builder.block_count(),
        )?;
        ifd_results.push(ifd_result);
    }

    for (index, image) in images.iter().enumerate() {
        let planned = &layout.images[index];
        let ifd_result = &ifd_results[index];
        let (offsets_tag_code, byte_counts_tag_code) = image.builder.offset_tag_codes();

        if image.blocks.len() == 1 {
            if let Some(off) = encoder::find_inline_tag_value_offset(
                ifd_result.ifd_offset,
                layout.is_bigtiff,
                &planned.tags,
                offsets_tag_code,
            ) {
                sink.seek(SeekFrom::Start(off))?;
                if layout.is_bigtiff {
                    sink.write_all(&ByteOrder::LittleEndian.write_u64(planned.block_offsets[0]))?;
                } else {
                    sink.write_all(&ByteOrder::LittleEndian.write_u32(
                        u32::try_from(planned.block_offsets[0]).map_err(|_| {
                            Error::Tiff(tiff_writer::Error::ClassicOffsetOverflow {
                                offset: planned.block_offsets[0],
                            })
                        })?,
                    ))?;
                }
            }
            if let Some(off) = encoder::find_inline_tag_value_offset(
                ifd_result.ifd_offset,
                layout.is_bigtiff,
                &planned.tags,
                byte_counts_tag_code,
            ) {
                sink.seek(SeekFrom::Start(off))?;
                if layout.is_bigtiff {
                    sink.write_all(
                        &ByteOrder::LittleEndian.write_u64(planned.block_byte_counts[0]),
                    )?;
                } else {
                    sink.write_all(&ByteOrder::LittleEndian.write_u32(
                        u32::try_from(planned.block_byte_counts[0]).map_err(|_| {
                            Error::Tiff(tiff_writer::Error::ClassicByteCountOverflow {
                                byte_count: planned.block_byte_counts[0],
                            })
                        })?,
                    ))?;
                }
            }
        } else {
            if let Some(off) = ifd_result.offsets_tag_data_offset {
                encoder::patch_block_offsets(
                    sink,
                    ByteOrder::LittleEndian,
                    layout.is_bigtiff,
                    off,
                    &planned.block_offsets,
                )?;
            }
            if let Some(off) = ifd_result.byte_counts_tag_data_offset {
                encoder::patch_block_byte_counts(
                    sink,
                    ByteOrder::LittleEndian,
                    layout.is_bigtiff,
                    off,
                    &planned.block_byte_counts,
                )?;
            }
        }

        if index == 0 {
            encoder::patch_first_ifd(
                sink,
                layout.base_offset,
                ByteOrder::LittleEndian,
                layout.is_bigtiff,
                ifd_result.ifd_offset,
            )?;
        } else {
            encoder::patch_next_ifd(
                sink,
                ByteOrder::LittleEndian,
                layout.is_bigtiff,
                ifd_results[index - 1].next_ifd_pointer_offset,
                ifd_result.ifd_offset,
            )?;
        }
    }

    sink.seek(SeekFrom::End(0))?;
    spool.copy_into(sink)?;
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

        let mut builder = self
            .inner
            .to_sized_image_builder::<T>(ovr_w, ovr_h)
            .tiles(tile_width, tile_height)
            .overview();

        if let Some(opts) = self.inner.lerc_options {
            builder = builder.lerc_options(opts);
        }
        if let Some(opts) = self.inner.jpeg_options {
            builder = builder.jpeg_options(opts);
        }

        for tag in self.inner.build_extra_tags() {
            builder = builder.tag(tag);
        }

        builder
    }

    fn validate_images<T: NumericSample>(
        &self,
        overview_levels: &[u32],
        tile_width: u32,
        tile_height: u32,
    ) -> Result<()> {
        self.inner.to_image_builder::<T>().validate()?;
        for &level in overview_levels {
            self.overview_image_builder::<T>(level, tile_width, tile_height)
                .validate()?;
        }
        Ok(())
    }

    fn build_images<T: NumericSample>(
        &self,
        overview_levels: &[u32],
        tile_width: u32,
        tile_height: u32,
    ) -> Vec<CogImage> {
        let mut images = Vec::with_capacity(1 + overview_levels.len());
        images.push(CogImage {
            builder: self.inner.to_image_builder::<T>(),
            blocks: Vec::new(),
        });
        for &level in overview_levels {
            images.push(CogImage {
                builder: self.overview_image_builder::<T>(level, tile_width, tile_height),
                blocks: Vec::new(),
            });
        }
        images
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
        mut sink: W,
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
        self.validate_images::<T>(&overview_levels, tw as u32, th as u32)?;
        let nodata = parse_nodata_value::<T>(&self.inner.nodata);
        let prefix = gdal_structural_metadata_bytes(self.inner.planar_configuration);
        let mut spool = BlockSpool::new()?;
        let mut images = self.build_images::<T>(&overview_levels, tw as u32, th as u32);

        for idx in (0..overview_levels.len()).rev() {
            let overview =
                generate_overview_3d(data, overview_levels[idx] as usize, self.resampling, nodata);
            images[1 + idx].blocks = spool_tiled_data_3d(
                &mut spool,
                overview.view(),
                TileWritePlan {
                    tile_width: tw,
                    tile_height: th,
                    planar_configuration: self.inner.planar_configuration,
                    compression: self.inner.compression,
                    predictor: self.inner.predictor,
                    lerc_options: self.inner.lerc_options,
                    jpeg_options: self.inner.jpeg_options,
                },
            )?;
        }

        images[0].blocks = spool_tiled_data_3d(
            &mut spool,
            data,
            TileWritePlan {
                tile_width: tw,
                tile_height: th,
                planar_configuration: self.inner.planar_configuration,
                compression: self.inner.compression,
                predictor: self.inner.predictor,
                lerc_options: self.inner.lerc_options,
                jpeg_options: self.inner.jpeg_options,
            },
        )?;

        let base_offset = sink.stream_position()?;
        let layout = plan_cog_layout(
            base_offset,
            checked_len_u64(prefix.len(), "COG prefix")?,
            self.inner.tiff_variant,
            &images,
        )?;
        emit_cog(&mut sink, &prefix, &images, &layout, &mut spool)?;
        Ok(())
    }

    /// Create a buffered COG tile writer.
    pub fn tile_writer<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
    ) -> Result<CogTileWriter<T, W>> {
        CogTileWriter::new(self.clone(), sink)
    }

    /// Create a buffered COG tile writer to a file.
    pub fn tile_writer_file<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<CogTileWriter<T, BufWriter<File>>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.tile_writer(writer)
    }
}

/// Buffered COG tile writer.
///
/// Tiles are written incrementally into an in-memory full-resolution raster,
/// and the final COG layout is emitted on `finish()`.
pub struct CogTileWriter<T: NumericSample, W: Write + Seek> {
    sink: W,
    cog: CogBuilder,
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
    jpeg_options: Option<JpegOptions>,
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
        cog.validate_images::<T>(&overview_levels, tw, th)?;
        let nodata_value = parse_nodata_value::<T>(&cog.inner.nodata);
        let fill_value = nodata_value.unwrap_or_else(T::zero);

        Ok(Self {
            sink,
            cog: cog.clone(),
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
            jpeg_options: cog.inner.jpeg_options,
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

        for row in 0..data_h {
            for col in 0..data_w {
                let pixel_index = (y_off + row) * self.width as usize + (x_off + col);
                self.base_pixels[pixel_index] = data[[row, col]];
            }
        }

        Ok(())
    }

    /// Write a multi-band tile at pixel offset (x_off, y_off).
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

        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let (data_h, data_w, data_b) = data.dim();
        let bands = self.bands as usize;
        let expected_h = (self.height as usize).saturating_sub(y_off).min(th);
        let expected_w = (self.width as usize).saturating_sub(x_off).min(tw);
        if data_h > expected_h || data_w > expected_w {
            return Err(Error::Other(format!(
                "tile data shape {}x{} exceeds raster bounds for tile starting at ({x_off},{y_off}); expected at most {}x{}",
                data_h, data_w, expected_h, expected_w
            )));
        }
        if data_b != bands {
            return Err(Error::DataSizeMismatch {
                expected: data_h * data_w * bands,
                actual: data_h * data_w * data_b,
            });
        }

        for row in 0..data_h {
            for col in 0..data_w {
                let pixel_index = ((y_off + row) * self.width as usize + (x_off + col)) * bands;
                for band in 0..bands {
                    self.base_pixels[pixel_index + band] = data[[row, col, band]];
                }
            }
        }

        Ok(())
    }

    /// Finish: generate overview tiles, emit the COG layout, and return the sink.
    pub fn finish(mut self) -> Result<W> {
        let tw = self.tile_width as usize;
        let th = self.tile_height as usize;
        let full_w = self.width as usize;
        let full_h = self.height as usize;
        let bands = self.bands as usize;

        let full = Array3::from_shape_vec((full_h, full_w, bands), self.base_pixels)
            .map_err(|err| Error::Other(format!("invalid streaming COG raster shape: {err}")))?;

        let prefix = gdal_structural_metadata_bytes(self.planar_configuration);
        let mut spool = BlockSpool::new()?;
        let mut images =
            self.cog
                .build_images::<T>(&self.overview_levels, self.tile_width, self.tile_height);

        for idx in (0..self.overview_levels.len()).rev() {
            let overview = generate_overview_3d(
                full.view(),
                self.overview_levels[idx] as usize,
                self.resampling,
                self.nodata_value,
            );
            images[1 + idx].blocks = spool_tiled_data_3d(
                &mut spool,
                overview.view(),
                TileWritePlan {
                    tile_width: tw,
                    tile_height: th,
                    planar_configuration: self.planar_configuration,
                    compression: self.compression,
                    predictor: self.predictor,
                    lerc_options: self.lerc_options,
                    jpeg_options: self.jpeg_options,
                },
            )?;
        }

        images[0].blocks = spool_tiled_data_3d(
            &mut spool,
            full.view(),
            TileWritePlan {
                tile_width: tw,
                tile_height: th,
                planar_configuration: self.planar_configuration,
                compression: self.compression,
                predictor: self.predictor,
                lerc_options: self.lerc_options,
                jpeg_options: self.jpeg_options,
            },
        )?;

        let base_offset = self.sink.stream_position()?;
        let layout = plan_cog_layout(
            base_offset,
            checked_len_u64(prefix.len(), "COG prefix")?,
            self.cog.inner.tiff_variant,
            &images,
        )?;
        emit_cog(&mut self.sink, &prefix, &images, &layout, &mut spool)?;
        Ok(self.sink)
    }
}

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

fn spool_tiled_data_3d<T: NumericSample>(
    spool: &mut BlockSpool,
    data: ArrayView3<T>,
    plan: TileWritePlan,
) -> Result<Vec<CogBlockRecord>> {
    let (height, width, bands) = data.dim();
    let tw = plan.tile_width;
    let th = plan.tile_height;
    let tiles_across = width.div_ceil(tw);
    let tiles_down = height.div_ceil(th);
    let total_blocks = if matches!(
        plan.planar_configuration,
        tiff_core::PlanarConfiguration::Planar
    ) {
        tiles_across * tiles_down * bands
    } else {
        tiles_across * tiles_down
    };
    let mut blocks = vec![
        CogBlockRecord {
            spool_offset: 0,
            logical_offset_delta: 0,
            logical_byte_count: 0,
        };
        total_blocks
    ];

    if matches!(
        plan.planar_configuration,
        tiff_core::PlanarConfiguration::Planar
    ) {
        let tiles_per_plane = tiles_across * tiles_down;
        for band in 0..bands {
            for tile_row in 0..tiles_down {
                for tile_col in 0..tiles_across {
                    let tile_index = tile_row * tiles_across + tile_col;
                    let block_index = band * tiles_per_plane + tile_index;
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
                    blocks[block_index] = spool_cog_block(
                        spool,
                        &tile_data,
                        block_index,
                        CogBlockEncoding {
                            compression: plan.compression,
                            predictor: plan.predictor,
                            samples_per_pixel: 1,
                            row_width_pixels: tw,
                            block_height: th as u32,
                            lerc_options: plan.lerc_options,
                            jpeg_options: plan.jpeg_options,
                        },
                    )?;
                }
            }
        }
    } else {
        for tile_row in 0..tiles_down {
            for tile_col in 0..tiles_across {
                let block_index = tile_row * tiles_across + tile_col;
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
                blocks[block_index] = spool_cog_block(
                    spool,
                    &tile_data,
                    block_index,
                    CogBlockEncoding {
                        compression: plan.compression,
                        predictor: plan.predictor,
                        samples_per_pixel: bands as u16,
                        row_width_pixels: tw,
                        block_height: th as u32,
                        lerc_options: plan.lerc_options,
                        jpeg_options: plan.jpeg_options,
                    },
                )?;
            }
        }
    }

    Ok(blocks)
}

fn spool_cog_block<T: NumericSample>(
    spool: &mut BlockSpool,
    samples: &[T],
    block_index: usize,
    encoding: CogBlockEncoding,
) -> Result<CogBlockRecord> {
    let compressed = compress_cog_block(samples, block_index, encoding)?;
    let leader = gdal_block_leader(compressed.len(), ByteOrder::LittleEndian)?;
    let trailer = gdal_block_trailer(&compressed);
    spool.append_segmented(&leader, &compressed, &trailer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_promotes_cog_layout_to_bigtiff_when_classic_offsets_overflow() {
        let prefix = gdal_structural_metadata_bytes(tiff_core::PlanarConfiguration::Chunky);
        let images = vec![CogImage {
            builder: ImageBuilder::new(1, 1).sample_type::<u8>().tiles(16, 16),
            blocks: vec![CogBlockRecord {
                spool_offset: u32::MAX as u64,
                logical_offset_delta: 4,
                logical_byte_count: 1,
            }],
        }];

        let layout = plan_cog_layout(
            0,
            checked_len_u64(prefix.len(), "COG prefix").unwrap(),
            TiffVariant::Auto,
            &images,
        )
        .unwrap();

        assert!(layout.is_bigtiff);
    }
}
