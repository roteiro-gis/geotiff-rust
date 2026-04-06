//! Image builder for configuring a single TIFF IFD.

use tiff_core::*;

use crate::sample::TiffWriteSample;

/// LERC encoding options for the TIFF writer.
///
/// Controls the LERC2 error tolerance and optional additional compression
/// applied to the encoded LERC blob before storage in the TIFF block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LercOptions {
    /// Maximum encoding error per sample value. Set to `0.0` for lossless.
    pub max_z_error: f64,
    /// Optional additional compression applied to the LERC blob.
    pub additional_compression: LercAdditionalCompression,
}

impl Default for LercOptions {
    fn default() -> Self {
        Self {
            max_z_error: 0.0,
            additional_compression: LercAdditionalCompression::None,
        }
    }
}

/// Describes how image data is organized: strips or tiles.
#[derive(Debug, Clone, Copy)]
pub enum DataLayout {
    /// Strip-based: each strip contains `rows_per_strip` rows.
    Strips { rows_per_strip: u32 },
    /// Tile-based: each tile is `width x height` pixels.
    Tiles { width: u32, height: u32 },
}

/// Builder for configuring a single image (IFD) within a TIFF file.
#[derive(Debug, Clone)]
pub struct ImageBuilder {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples_per_pixel: u16,
    pub(crate) bits_per_sample: u16,
    pub(crate) sample_format: SampleFormat,
    pub(crate) compression: Compression,
    pub(crate) predictor: Predictor,
    pub(crate) photometric: PhotometricInterpretation,
    pub(crate) planar_configuration: PlanarConfiguration,
    pub(crate) layout: DataLayout,
    pub(crate) extra_tags: Vec<Tag>,
    pub(crate) subfile_type: u32,
    pub(crate) lerc_options: Option<LercOptions>,
}

impl ImageBuilder {
    /// Create a new image builder with required dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            samples_per_pixel: 1,
            bits_per_sample: 8,
            sample_format: SampleFormat::Uint,
            compression: Compression::None,
            predictor: Predictor::None,
            photometric: PhotometricInterpretation::MinIsBlack,
            planar_configuration: PlanarConfiguration::Chunky,
            layout: DataLayout::Strips {
                rows_per_strip: height.min(256),
            },
            extra_tags: Vec::new(),
            subfile_type: 0,
            lerc_options: None,
        }
    }

    pub fn samples_per_pixel(mut self, spp: u16) -> Self {
        self.samples_per_pixel = spp;
        self
    }

    pub fn bits_per_sample(mut self, bps: u16) -> Self {
        self.bits_per_sample = bps;
        self
    }

    pub fn sample_format(mut self, fmt: SampleFormat) -> Self {
        self.sample_format = fmt;
        self
    }

    /// Configure from a TiffWriteSample type. Sets bits_per_sample and sample_format.
    pub fn sample_type<T: TiffWriteSample>(mut self) -> Self {
        self.bits_per_sample = T::BITS_PER_SAMPLE;
        self.sample_format =
            SampleFormat::from_code(T::SAMPLE_FORMAT).unwrap_or(SampleFormat::Uint);
        self
    }

    pub fn compression(mut self, c: Compression) -> Self {
        self.compression = c;
        if !matches!(c, Compression::Lerc) {
            self.lerc_options = None;
        }
        self
    }

    pub fn predictor(mut self, p: Predictor) -> Self {
        // LERC does not use TIFF predictors; ignore the request.
        if !matches!(self.compression, Compression::Lerc) {
            self.predictor = p;
        }
        self
    }

    pub fn photometric(mut self, p: PhotometricInterpretation) -> Self {
        self.photometric = p;
        self
    }

    /// Set chunky (interleaved) or separate planar sample layout for multi-band images.
    pub fn planar_configuration(mut self, p: PlanarConfiguration) -> Self {
        self.planar_configuration = p;
        self
    }

    /// Configure strip-based layout.
    pub fn strips(mut self, rows_per_strip: u32) -> Self {
        self.layout = DataLayout::Strips { rows_per_strip };
        self
    }

    /// Configure tile-based layout.
    pub fn tiles(mut self, tile_width: u32, tile_height: u32) -> Self {
        self.layout = DataLayout::Tiles {
            width: tile_width,
            height: tile_height,
        };
        self
    }

    /// Add an arbitrary extra tag to the IFD.
    pub fn tag(mut self, tag: Tag) -> Self {
        self.extra_tags.push(tag);
        self
    }

    /// Mark this IFD as a reduced-resolution overview.
    pub fn overview(mut self) -> Self {
        self.subfile_type = 1;
        self
    }

    /// Set LERC compression with the given options.
    ///
    /// This sets `compression = Lerc` and `predictor = None` (LERC performs
    /// its own quantization and does not use TIFF predictors).
    pub fn lerc_options(mut self, options: LercOptions) -> Self {
        self.compression = Compression::Lerc;
        self.predictor = Predictor::None;
        self.lerc_options = Some(options);
        self
    }

    /// Total number of blocks (strips or tiles) for this image configuration.
    pub fn block_count(&self) -> usize {
        let blocks_per_plane = match self.layout {
            DataLayout::Strips { rows_per_strip } => {
                let rps = rows_per_strip.max(1) as usize;
                (self.height as usize).div_ceil(rps)
            }
            DataLayout::Tiles { width, height } => {
                let tw = width.max(1) as usize;
                let th = height.max(1) as usize;
                let tiles_across = (self.width as usize).div_ceil(tw);
                let tiles_down = (self.height as usize).div_ceil(th);
                tiles_across * tiles_down
            }
        };
        if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
            blocks_per_plane * self.samples_per_pixel as usize
        } else {
            blocks_per_plane
        }
    }

    /// Expected number of samples for the block at `index`.
    pub fn block_sample_count(&self, index: usize) -> usize {
        let samples_per_pixel = self.block_samples_per_pixel() as usize;
        let plane_block_index = self.block_plane_index(index);
        match self.layout {
            DataLayout::Strips { rows_per_strip } => {
                let rps = rows_per_strip.max(1) as usize;
                let start_row = plane_block_index * rps;
                let end_row = ((plane_block_index + 1) * rps).min(self.height as usize);
                let rows = end_row.saturating_sub(start_row);
                rows * self.width as usize * samples_per_pixel
            }
            DataLayout::Tiles { width, height } => {
                // Tiles are always full-sized (padded at edges)
                width as usize * height as usize * samples_per_pixel
            }
        }
    }

    /// Estimated uncompressed image bytes.
    pub fn estimated_uncompressed_bytes(&self) -> u64 {
        let bps = (self.bits_per_sample / 8).max(1) as u64;
        self.width as u64 * self.height as u64 * self.samples_per_pixel as u64 * bps
    }

    /// The TIFF tag codes for offset and bytecount arrays.
    pub fn offset_tag_codes(&self) -> (u16, u16) {
        match self.layout {
            DataLayout::Strips { .. } => (TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS),
            DataLayout::Tiles { .. } => (TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS),
        }
    }

    /// Build the layout-specific tags (RowsPerStrip or TileWidth/TileLength).
    pub fn layout_tags(&self) -> Vec<Tag> {
        match self.layout {
            DataLayout::Strips { rows_per_strip } => {
                vec![Tag::new(
                    TAG_ROWS_PER_STRIP,
                    TagValue::Long(vec![rows_per_strip]),
                )]
            }
            DataLayout::Tiles { width, height } => {
                vec![
                    Tag::new(TAG_TILE_WIDTH, TagValue::Long(vec![width])),
                    Tag::new(TAG_TILE_LENGTH, TagValue::Long(vec![height])),
                ]
            }
        }
    }

    /// Row width in pixels for compression pipeline (tile_width or image_width).
    pub fn block_row_width(&self) -> usize {
        match self.layout {
            DataLayout::Strips { .. } => self.width as usize,
            DataLayout::Tiles { width, .. } => width as usize,
        }
    }

    /// Samples per pixel represented in a single block.
    pub fn block_samples_per_pixel(&self) -> u16 {
        if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
            1
        } else {
            self.samples_per_pixel
        }
    }

    fn block_plane_index(&self, index: usize) -> usize {
        if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
            index % self.blocks_per_plane()
        } else {
            index
        }
    }

    fn blocks_per_plane(&self) -> usize {
        match self.layout {
            DataLayout::Strips { rows_per_strip } => {
                let rps = rows_per_strip.max(1) as usize;
                (self.height as usize).div_ceil(rps)
            }
            DataLayout::Tiles { width, height } => {
                let tw = width.max(1) as usize;
                let th = height.max(1) as usize;
                let tiles_across = (self.width as usize).div_ceil(tw);
                let tiles_down = (self.height as usize).div_ceil(th);
                tiles_across * tiles_down
            }
        }
    }

    /// Height of the block at `index` in pixels.
    ///
    /// Tiles are always full-sized (padded at edges). Strips may be shorter
    /// for the final strip.
    pub fn block_height(&self, index: usize) -> u32 {
        match self.layout {
            DataLayout::Tiles { height, .. } => height,
            DataLayout::Strips { rows_per_strip } => {
                let plane_index = self.block_plane_index(index);
                let rps = rows_per_strip.max(1) as usize;
                let start_row = plane_index * rps;
                let remaining = (self.height as usize).saturating_sub(start_row);
                remaining.min(rps) as u32
            }
        }
    }

    /// Build the `TAG_LERC_PARAMETERS` tag if LERC compression is configured.
    pub fn lerc_parameters_tag(&self) -> Option<Tag> {
        if !matches!(self.compression, Compression::Lerc) {
            return None;
        }
        let opts = self.lerc_options.unwrap_or_default();
        Some(Tag::new(
            TAG_LERC_PARAMETERS,
            TagValue::Long(vec![2, opts.additional_compression.to_code()]),
        ))
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(crate::error::Error::InvalidConfig(
                "image dimensions must be positive".into(),
            ));
        }
        if !matches!(self.bits_per_sample, 8 | 16 | 32 | 64) {
            return Err(crate::error::Error::InvalidConfig(format!(
                "bits_per_sample must be 8, 16, 32, or 64, got {}",
                self.bits_per_sample
            )));
        }
        if let DataLayout::Tiles { width, height } = self.layout {
            if width % 16 != 0 || height % 16 != 0 {
                return Err(crate::error::Error::InvalidConfig(format!(
                    "tile dimensions must be multiples of 16, got {}x{}",
                    width, height
                )));
            }
        }
        if matches!(self.compression, Compression::Lerc)
            && !matches!(self.predictor, Predictor::None)
        {
            return Err(crate::error::Error::InvalidConfig(
                "LERC compression does not support TIFF predictors".into(),
            ));
        }
        Ok(())
    }
}
