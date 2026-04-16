//! Image builder for configuring a single TIFF IFD.

use tiff_core::*;

use crate::encoder;
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

/// JPEG encoding options for the TIFF writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegOptions {
    /// Quality in the range 1..=100.
    pub quality: u8,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self { quality: 75 }
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
    pub(crate) extra_samples: Vec<ExtraSample>,
    pub(crate) color_map: Option<ColorMap>,
    pub(crate) ink_set: Option<InkSet>,
    pub(crate) ycbcr_subsampling: Option<[u16; 2]>,
    pub(crate) ycbcr_positioning: Option<YCbCrPositioning>,
    pub(crate) planar_configuration: PlanarConfiguration,
    pub(crate) layout: DataLayout,
    pub(crate) extra_tags: Vec<Tag>,
    pub(crate) subfile_type: u32,
    pub(crate) lerc_options: Option<LercOptions>,
    pub(crate) jpeg_options: Option<JpegOptions>,
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
            extra_samples: Vec::new(),
            color_map: None,
            ink_set: None,
            ycbcr_subsampling: None,
            ycbcr_positioning: None,
            planar_configuration: PlanarConfiguration::Chunky,
            layout: DataLayout::Strips {
                rows_per_strip: height.min(256),
            },
            extra_tags: Vec::new(),
            subfile_type: 0,
            lerc_options: None,
            jpeg_options: None,
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
        if !matches!(c, Compression::Jpeg) {
            self.jpeg_options = None;
        }
        if matches!(c, Compression::Lerc | Compression::Jpeg) {
            self.predictor = Predictor::None;
        }
        self
    }

    pub fn predictor(mut self, p: Predictor) -> Self {
        // LERC and JPEG do not use TIFF predictors; ignore the request.
        if !matches!(self.compression, Compression::Lerc | Compression::Jpeg) {
            self.predictor = p;
        }
        self
    }

    pub fn photometric(mut self, p: PhotometricInterpretation) -> Self {
        self.photometric = p;
        self
    }

    /// Set TIFF ExtraSamples semantics for channels beyond the base color model.
    pub fn extra_samples(mut self, extra_samples: Vec<ExtraSample>) -> Self {
        self.extra_samples = extra_samples;
        self
    }

    /// Set a palette ColorMap for `PhotometricInterpretation::Palette`.
    pub fn color_map(mut self, color_map: ColorMap) -> Self {
        self.color_map = Some(color_map);
        self
    }

    /// Set the InkSet tag for separated photometric data.
    pub fn ink_set(mut self, ink_set: InkSet) -> Self {
        self.ink_set = Some(ink_set);
        self
    }

    /// Set TIFF YCbCr chroma subsampling factors.
    pub fn ycbcr_subsampling(mut self, subsampling: [u16; 2]) -> Self {
        self.ycbcr_subsampling = Some(subsampling);
        self
    }

    /// Set TIFF YCbCr sample positioning.
    pub fn ycbcr_positioning(mut self, positioning: YCbCrPositioning) -> Self {
        self.ycbcr_positioning = Some(positioning);
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
        self.jpeg_options = None;
        self
    }

    /// Set JPEG compression with the given options.
    ///
    /// This sets `compression = Jpeg` and `predictor = None` (JPEG uses its
    /// own transform and entropy coding pipeline rather than TIFF predictors).
    ///
    /// Multi-band JPEG requires `planar_configuration(Planar)` so each encoded
    /// strip/tile is a single grayscale component.
    pub fn jpeg_options(mut self, options: JpegOptions) -> Self {
        self.compression = Compression::Jpeg;
        self.predictor = Predictor::None;
        self.jpeg_options = Some(options);
        self.lerc_options = None;
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

    /// Build the serialized TIFF tags for this image definition.
    pub fn build_tags(&self, is_bigtiff: bool) -> Vec<Tag> {
        let mut extra_tags = self.extra_tags.clone();
        if let Some(lerc_tag) = self.lerc_parameters_tag() {
            extra_tags.push(lerc_tag);
        }
        let extra_samples = self
            .effective_extra_samples()
            .expect("ImageBuilder::build_tags requires a validated color model");
        if !extra_samples.is_empty() {
            extra_tags.push(Tag::new(
                TAG_EXTRA_SAMPLES,
                TagValue::Short(
                    extra_samples
                        .iter()
                        .copied()
                        .map(ExtraSample::to_code)
                        .collect(),
                ),
            ));
        }
        if let Some(color_map) = &self.color_map {
            extra_tags.push(Tag::new(
                TAG_COLOR_MAP,
                TagValue::Short(color_map.encode_tag_values()),
            ));
        }
        if let Some(ink_set) = self.ink_set {
            extra_tags.push(Tag::new(
                TAG_INK_SET,
                TagValue::Short(vec![ink_set.to_code()]),
            ));
        }
        if let Some([h, v]) = self.ycbcr_subsampling {
            extra_tags.push(Tag::new(TAG_YCBCR_SUBSAMPLING, TagValue::Short(vec![h, v])));
        }
        if let Some(positioning) = self.ycbcr_positioning {
            extra_tags.push(Tag::new(
                TAG_YCBCR_POSITIONING,
                TagValue::Short(vec![positioning.to_code()]),
            ));
        }

        let (offsets_tag_code, byte_counts_tag_code) = self.offset_tag_codes();
        let layout_tags = self.layout_tags();

        encoder::build_image_tags(&encoder::ImageTagParams {
            width: self.width,
            height: self.height,
            samples_per_pixel: self.samples_per_pixel,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.sample_format.to_code(),
            compression: self.compression.to_code(),
            photometric: self.photometric.to_code(),
            predictor: self.predictor.to_code(),
            planar_configuration: self.planar_configuration.to_code(),
            subfile_type: self.subfile_type,
            extra_tags: &extra_tags,
            offsets_tag_code,
            byte_counts_tag_code,
            num_blocks: self.block_count(),
            layout_tags: &layout_tags,
            is_bigtiff,
        })
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
        if self.samples_per_pixel == 0 {
            return Err(crate::error::Error::InvalidConfig(
                "samples_per_pixel must be greater than zero".into(),
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
        if matches!(self.compression, Compression::OldJpeg) {
            return Err(crate::error::Error::InvalidConfig(
                "Old-style JPEG compression is not supported for writing; use Compression::Jpeg"
                    .into(),
            ));
        }
        self.validate_color_model()?;
        if matches!(self.compression, Compression::Jpeg) {
            self.validate_jpeg_config()?;
        }
        Ok(())
    }

    fn validate_color_model(&self) -> crate::error::Result<()> {
        if !matches!(self.photometric, PhotometricInterpretation::Palette)
            && self.color_map.is_some()
        {
            return Err(crate::error::Error::InvalidConfig(
                "ColorMap is only valid with palette photometric interpretation".into(),
            ));
        }

        if !matches!(self.photometric, PhotometricInterpretation::Separated)
            && self.ink_set.is_some()
        {
            return Err(crate::error::Error::InvalidConfig(
                "InkSet is only valid with separated photometric interpretation".into(),
            ));
        }

        let base_samples: u16 = match self.photometric {
            PhotometricInterpretation::MinIsWhite | PhotometricInterpretation::MinIsBlack => 1,
            PhotometricInterpretation::Rgb => 3,
            PhotometricInterpretation::Palette => {
                let color_map =
                    self.color_map
                        .as_ref()
                        .ok_or(crate::error::Error::InvalidConfig(
                            "palette photometric interpretation requires a ColorMap".into(),
                        ))?;
                let expected_entries =
                    1usize
                        .checked_shl(self.bits_per_sample as u32)
                        .ok_or_else(|| {
                            crate::error::Error::InvalidConfig(format!(
                                "palette BitsPerSample {} exceeds usize shift width",
                                self.bits_per_sample
                            ))
                        })?;
                if color_map.len() != expected_entries {
                    return Err(crate::error::Error::InvalidConfig(format!(
                        "palette ColorMap has {} entries but BitsPerSample={} requires {}",
                        color_map.len(),
                        self.bits_per_sample,
                        expected_entries
                    )));
                }
                1
            }
            PhotometricInterpretation::Mask => 1,
            PhotometricInterpretation::Separated => match self.ink_set.unwrap_or(InkSet::Cmyk) {
                InkSet::Cmyk => 4,
                InkSet::NotCmyk | InkSet::Unknown(_) => {
                    return Err(crate::error::Error::InvalidConfig(
                        "separated photometric interpretation currently requires InkSet::Cmyk"
                            .into(),
                    ))
                }
            },
            PhotometricInterpretation::YCbCr => 3,
            PhotometricInterpretation::CieLab => 3,
        };

        let _ = self.effective_extra_samples_for_base(base_samples)?;

        if matches!(self.photometric, PhotometricInterpretation::YCbCr) {
            if !matches!(self.sample_format, SampleFormat::Uint) || self.bits_per_sample != 8 {
                return Err(crate::error::Error::InvalidConfig(
                    "YCbCr photometric interpretation requires 8-bit unsigned samples".into(),
                ));
            }
            if let Some(subsampling) = self.ycbcr_subsampling {
                if subsampling != [1, 1] {
                    return Err(crate::error::Error::InvalidConfig(format!(
                        "YCbCr subsampling {:?} is not supported by the current writer",
                        subsampling
                    )));
                }
            }
        } else if self.ycbcr_subsampling.is_some() || self.ycbcr_positioning.is_some() {
            return Err(crate::error::Error::InvalidConfig(
                "YCbCr-specific tags require YCbCr photometric interpretation".into(),
            ));
        }

        Ok(())
    }

    fn effective_extra_samples(&self) -> crate::error::Result<Vec<ExtraSample>> {
        let base_samples = match self.photometric {
            PhotometricInterpretation::MinIsWhite | PhotometricInterpretation::MinIsBlack => 1,
            PhotometricInterpretation::Rgb => 3,
            PhotometricInterpretation::Palette => 1,
            PhotometricInterpretation::Mask => 1,
            PhotometricInterpretation::Separated => 4,
            PhotometricInterpretation::YCbCr => 3,
            PhotometricInterpretation::CieLab => 3,
        };
        self.effective_extra_samples_for_base(base_samples)
    }

    fn effective_extra_samples_for_base(
        &self,
        base_samples: u16,
    ) -> crate::error::Result<Vec<ExtraSample>> {
        let implied_extra_samples = self
            .samples_per_pixel
            .checked_sub(base_samples)
            .ok_or_else(|| {
                crate::error::Error::InvalidConfig(format!(
                    "{} photometric interpretation requires at least {} samples, got {}",
                    photometric_name(self.photometric),
                    base_samples,
                    self.samples_per_pixel
                ))
            })?;
        if self.extra_samples.len() > implied_extra_samples as usize {
            return Err(crate::error::Error::InvalidConfig(format!(
                "{} photometric interpretation has {} total channels but {} ExtraSamples",
                photometric_name(self.photometric),
                self.samples_per_pixel,
                self.extra_samples.len()
            )));
        }

        let mut extra_samples = self.extra_samples.clone();
        extra_samples.resize(implied_extra_samples as usize, ExtraSample::Unspecified);
        Ok(extra_samples)
    }

    fn validate_jpeg_config(&self) -> crate::error::Result<()> {
        let options = self.jpeg_options.unwrap_or_default();
        if !(1..=100).contains(&options.quality) {
            return Err(crate::error::Error::InvalidConfig(format!(
                "JPEG quality must be in the range 1..=100, got {}",
                options.quality
            )));
        }
        if self.bits_per_sample != 8 {
            return Err(crate::error::Error::InvalidConfig(format!(
                "JPEG compression requires 8-bit samples, got {} bits",
                self.bits_per_sample
            )));
        }
        if !matches!(self.sample_format, SampleFormat::Uint) {
            return Err(crate::error::Error::InvalidConfig(format!(
                "JPEG compression requires unsigned integer samples, got {:?}",
                self.sample_format
            )));
        }
        if !matches!(self.predictor, Predictor::None) {
            return Err(crate::error::Error::InvalidConfig(
                "JPEG compression does not support TIFF predictors".into(),
            ));
        }

        let block_width = self.block_row_width();
        if block_width > u16::MAX as usize {
            return Err(crate::error::Error::InvalidConfig(format!(
                "JPEG block width must be <= {}, got {}",
                u16::MAX,
                block_width
            )));
        }
        let max_block_height = match self.layout {
            DataLayout::Strips { rows_per_strip } => rows_per_strip.max(1),
            DataLayout::Tiles { height, .. } => height,
        };
        if max_block_height > u16::MAX as u32 {
            return Err(crate::error::Error::InvalidConfig(format!(
                "JPEG block height must be <= {}, got {}",
                u16::MAX,
                max_block_height
            )));
        }

        let block_samples_per_pixel = self.block_samples_per_pixel();
        if block_samples_per_pixel != 1 {
            return Err(crate::error::Error::InvalidConfig(format!(
                "JPEG write currently supports one sample per encoded block, got {}; use planar configuration for multi-band JPEG",
                block_samples_per_pixel
            )));
        }

        if matches!(
            self.photometric,
            PhotometricInterpretation::Palette | PhotometricInterpretation::Mask
        ) {
            return Err(crate::error::Error::InvalidConfig(format!(
                "{:?} photometric interpretation is not supported with JPEG compression",
                self.photometric
            )));
        }

        Ok(())
    }
}

fn photometric_name(photometric: PhotometricInterpretation) -> &'static str {
    match photometric {
        PhotometricInterpretation::MinIsWhite => "MinIsWhite",
        PhotometricInterpretation::MinIsBlack => "MinIsBlack",
        PhotometricInterpretation::Rgb => "RGB",
        PhotometricInterpretation::Palette => "Palette",
        PhotometricInterpretation::Mask => "TransparencyMask",
        PhotometricInterpretation::Separated => "Separated",
        PhotometricInterpretation::YCbCr => "YCbCr",
        PhotometricInterpretation::CieLab => "CIELab",
    }
}
