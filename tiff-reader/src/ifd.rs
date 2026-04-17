use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::header::{ByteOrder, TiffHeader};
use crate::io::Cursor;
use crate::source::TiffSource;
use crate::tag::{parse_tag_bigtiff, parse_tag_classic, Tag};

pub use tiff_core::constants::{
    TAG_BITS_PER_SAMPLE, TAG_COLOR_MAP, TAG_COMPRESSION, TAG_EXTRA_SAMPLES, TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH, TAG_INK_SET, TAG_LERC_PARAMETERS, TAG_PHOTOMETRIC_INTERPRETATION,
    TAG_PLANAR_CONFIGURATION, TAG_PREDICTOR, TAG_REFERENCE_BLACK_WHITE, TAG_ROWS_PER_STRIP,
    TAG_SAMPLES_PER_PIXEL, TAG_SAMPLE_FORMAT, TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS,
    TAG_SUB_IFDS, TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH, TAG_TILE_OFFSETS, TAG_TILE_WIDTH,
    TAG_YCBCR_POSITIONING, TAG_YCBCR_SUBSAMPLING,
};
pub use tiff_core::RasterLayout;

pub use tiff_core::{
    ColorMap, ColorModel, ExtraSample, InkSet, LercAdditionalCompression,
    PhotometricInterpretation, YCbCrPositioning,
};

/// Parsed TIFF `LercParameters` tag payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LercParameters {
    pub version: u32,
    pub additional_compression: LercAdditionalCompression,
}

/// A parsed Image File Directory (IFD).
#[derive(Debug, Clone)]
pub struct Ifd {
    /// Tags in this IFD, sorted by tag code.
    tags: Vec<Tag>,
    /// Index of this IFD in the chain (0-based).
    pub index: usize,
}

impl Ifd {
    /// Look up a tag by its code.
    pub fn tag(&self, code: u16) -> Option<&Tag> {
        self.tags
            .binary_search_by_key(&code, |tag| tag.code)
            .ok()
            .map(|index| &self.tags[index])
    }

    /// Returns all tags in this IFD.
    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.tag_u32(TAG_IMAGE_WIDTH).unwrap_or(0)
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.tag_u32(TAG_IMAGE_LENGTH).unwrap_or(0)
    }

    /// Bits per sample for each channel.
    pub fn bits_per_sample(&self) -> Vec<u16> {
        self.tag(TAG_BITS_PER_SAMPLE)
            .and_then(|tag| tag.value.as_u16_slice().map(|values| values.to_vec()))
            .unwrap_or_else(|| vec![1])
    }

    /// Compression scheme (1 = none, 5 = LZW, 8 = Deflate, ...).
    pub fn compression(&self) -> u16 {
        self.tag_u16(TAG_COMPRESSION).unwrap_or(1)
    }

    /// Photometric interpretation.
    pub fn photometric_interpretation(&self) -> Option<u16> {
        self.tag_u16(TAG_PHOTOMETRIC_INTERPRETATION)
    }

    /// Typed photometric interpretation, defaulting to `MinIsBlack` when the
    /// TIFF tag is omitted.
    pub fn photometric_interpretation_enum(&self) -> Option<PhotometricInterpretation> {
        PhotometricInterpretation::from_code(self.photometric_interpretation().unwrap_or(1))
    }

    /// Number of samples (bands) per pixel.
    pub fn samples_per_pixel(&self) -> u16 {
        self.tag_u16(TAG_SAMPLES_PER_PIXEL).unwrap_or(1)
    }

    /// Returns `true` if this IFD uses tiled layout.
    pub fn is_tiled(&self) -> bool {
        self.tag(TAG_TILE_WIDTH).is_some() && self.tag(TAG_TILE_LENGTH).is_some()
    }

    /// Tile width (only for tiled IFDs).
    pub fn tile_width(&self) -> Option<u32> {
        self.tag_u32(TAG_TILE_WIDTH)
    }

    /// Tile height (only for tiled IFDs).
    pub fn tile_height(&self) -> Option<u32> {
        self.tag_u32(TAG_TILE_LENGTH)
    }

    /// Rows per strip. Defaults to the image height when not present.
    pub fn rows_per_strip(&self) -> Option<u32> {
        Some(
            self.tag_u32(TAG_ROWS_PER_STRIP)
                .unwrap_or_else(|| self.height()),
        )
    }

    /// Sample format for each channel.
    pub fn sample_format(&self) -> Vec<u16> {
        self.tag(TAG_SAMPLE_FORMAT)
            .and_then(|tag| tag.value.as_u16_slice().map(|values| values.to_vec()))
            .unwrap_or_else(|| vec![1])
    }

    /// Planar configuration. Defaults to chunky (1).
    pub fn planar_configuration(&self) -> u16 {
        self.tag_u16(TAG_PLANAR_CONFIGURATION).unwrap_or(1)
    }

    /// Predictor. Defaults to no predictor (1).
    pub fn predictor(&self) -> u16 {
        self.tag_u16(TAG_PREDICTOR).unwrap_or(1)
    }

    /// TIFF-side LERC parameters, when present.
    pub fn lerc_parameters(&self) -> Result<Option<LercParameters>> {
        let Some(tag) = self.tag(TAG_LERC_PARAMETERS) else {
            return Ok(None);
        };
        let values = tag.value.as_u32_slice().ok_or(Error::UnexpectedTagType {
            tag: TAG_LERC_PARAMETERS,
            expected: "LONG",
            actual: tag.tag_type.to_code(),
        })?;
        if values.len() < 2 {
            return Err(Error::InvalidTagValue {
                tag: TAG_LERC_PARAMETERS,
                reason: "LercParameters must contain at least version and additional compression"
                    .into(),
            });
        }
        let additional_compression =
            LercAdditionalCompression::from_code(values[1]).ok_or(Error::InvalidTagValue {
                tag: TAG_LERC_PARAMETERS,
                reason: format!("unsupported LERC additional compression code {}", values[1]),
            })?;
        Ok(Some(LercParameters {
            version: values[0],
            additional_compression,
        }))
    }

    /// TIFF ExtraSamples semantics.
    pub fn extra_samples(&self) -> Result<Vec<ExtraSample>> {
        let Some(tag) = self.tag(TAG_EXTRA_SAMPLES) else {
            return Ok(Vec::new());
        };
        let values = tag.value.as_u16_slice().ok_or(Error::UnexpectedTagType {
            tag: TAG_EXTRA_SAMPLES,
            expected: "SHORT",
            actual: tag.tag_type.to_code(),
        })?;
        Ok(values.iter().copied().map(ExtraSample::from_code).collect())
    }

    /// TIFF ColorMap values for palette images.
    pub fn color_map(&self) -> Result<Option<ColorMap>> {
        let Some(tag) = self.tag(TAG_COLOR_MAP) else {
            return Ok(None);
        };
        let values = tag.value.as_u16_slice().ok_or(Error::UnexpectedTagType {
            tag: TAG_COLOR_MAP,
            expected: "SHORT",
            actual: tag.tag_type.to_code(),
        })?;
        ColorMap::from_tag_values(values)
            .map(Some)
            .map_err(|reason| Error::InvalidTagValue {
                tag: TAG_COLOR_MAP,
                reason,
            })
    }

    /// TIFF InkSet semantics for separated photometric data.
    pub fn ink_set(&self) -> Result<Option<InkSet>> {
        let Some(tag) = self.tag(TAG_INK_SET) else {
            return Ok(None);
        };
        let value = tag.value.as_u16().ok_or(Error::UnexpectedTagType {
            tag: TAG_INK_SET,
            expected: "SHORT",
            actual: tag.tag_type.to_code(),
        })?;
        Ok(Some(InkSet::from_code(value)))
    }

    /// TIFF YCbCr chroma subsampling factors.
    pub fn ycbcr_subsampling(&self) -> Result<Option<[u16; 2]>> {
        let Some(tag) = self.tag(TAG_YCBCR_SUBSAMPLING) else {
            return Ok(None);
        };
        let values = tag.value.as_u16_slice().ok_or(Error::UnexpectedTagType {
            tag: TAG_YCBCR_SUBSAMPLING,
            expected: "SHORT",
            actual: tag.tag_type.to_code(),
        })?;
        match values {
            [h, v] => Ok(Some([*h, *v])),
            _ => Err(Error::InvalidTagValue {
                tag: TAG_YCBCR_SUBSAMPLING,
                reason: format!("expected 2 SHORT values, found {}", values.len()),
            }),
        }
    }

    /// TIFF YCbCr sample positioning.
    pub fn ycbcr_positioning(&self) -> Result<Option<YCbCrPositioning>> {
        let Some(tag) = self.tag(TAG_YCBCR_POSITIONING) else {
            return Ok(None);
        };
        let value = tag.value.as_u16().ok_or(Error::UnexpectedTagType {
            tag: TAG_YCBCR_POSITIONING,
            expected: "SHORT",
            actual: tag.tag_type.to_code(),
        })?;
        Ok(Some(YCbCrPositioning::from_code(value)))
    }

    /// TIFF ReferenceBlackWhite values normalized to `f64`.
    pub fn reference_black_white(&self) -> Result<Option<[f64; 6]>> {
        let Some(tag) = self.tag(TAG_REFERENCE_BLACK_WHITE) else {
            return Ok(None);
        };
        let values = tag.value.as_f64_vec().ok_or(Error::UnexpectedTagType {
            tag: TAG_REFERENCE_BLACK_WHITE,
            expected: "RATIONAL or DOUBLE",
            actual: tag.tag_type.to_code(),
        })?;
        match values.as_slice() {
            [a, b, c, d, e, f] => Ok(Some([*a, *b, *c, *d, *e, *f])),
            _ => Err(Error::InvalidTagValue {
                tag: TAG_REFERENCE_BLACK_WHITE,
                reason: format!("expected 6 values, found {}", values.len()),
            }),
        }
    }

    /// Structured color-model metadata synthesized from TIFF photometric and
    /// ancillary color tags.
    pub fn color_model(&self) -> Result<ColorModel> {
        let photometric = self
            .photometric_interpretation_enum()
            .ok_or(Error::InvalidTagValue {
                tag: TAG_PHOTOMETRIC_INTERPRETATION,
                reason: format!(
                    "unsupported photometric interpretation {}",
                    self.photometric_interpretation().unwrap_or(1)
                ),
            })?;
        let samples_per_pixel = self.samples_per_pixel();
        let extra_samples = self.extra_samples()?;

        match photometric {
            PhotometricInterpretation::MinIsWhite => Ok(ColorModel::Grayscale {
                white_is_zero: true,
                extra_samples: resolve_fixed_model_extra_samples(
                    photometric,
                    samples_per_pixel,
                    1,
                    extra_samples,
                )?,
            }),
            PhotometricInterpretation::MinIsBlack => Ok(ColorModel::Grayscale {
                white_is_zero: false,
                extra_samples: resolve_fixed_model_extra_samples(
                    photometric,
                    samples_per_pixel,
                    1,
                    extra_samples,
                )?,
            }),
            PhotometricInterpretation::Rgb => Ok(ColorModel::Rgb {
                extra_samples: resolve_fixed_model_extra_samples(
                    photometric,
                    samples_per_pixel,
                    3,
                    extra_samples,
                )?,
            }),
            PhotometricInterpretation::Palette => {
                let color_map = self.color_map()?.ok_or(Error::InvalidImageLayout(
                    "palette TIFF is missing ColorMap".into(),
                ))?;
                Ok(ColorModel::Palette {
                    color_map,
                    extra_samples: resolve_fixed_model_extra_samples(
                        photometric,
                        samples_per_pixel,
                        1,
                        extra_samples,
                    )?,
                })
            }
            PhotometricInterpretation::Mask => Ok(ColorModel::TransparencyMask),
            PhotometricInterpretation::Separated => {
                let ink_set = self.ink_set()?.unwrap_or(InkSet::Cmyk);
                if ink_set == InkSet::Cmyk {
                    let extra_samples = resolve_fixed_model_extra_samples(
                        photometric,
                        samples_per_pixel,
                        4,
                        extra_samples,
                    )?;
                    Ok(ColorModel::Cmyk { extra_samples })
                } else {
                    let color_channels = samples_per_pixel
                        .checked_sub(extra_samples.len() as u16)
                        .ok_or_else(|| {
                            Error::InvalidImageLayout(format!(
                                "{} photometric interpretation defines more ExtraSamples than total channels",
                                photometric_name(photometric)
                            ))
                        })?;
                    Ok(ColorModel::Separated {
                        ink_set,
                        color_channels,
                        extra_samples,
                    })
                }
            }
            PhotometricInterpretation::YCbCr => Ok(ColorModel::YCbCr {
                subsampling: self.ycbcr_subsampling()?.unwrap_or([1, 1]),
                positioning: self
                    .ycbcr_positioning()?
                    .unwrap_or(YCbCrPositioning::Centered),
                extra_samples: resolve_fixed_model_extra_samples(
                    photometric,
                    samples_per_pixel,
                    3,
                    extra_samples,
                )?,
            }),
            PhotometricInterpretation::CieLab => Ok(ColorModel::CieLab {
                extra_samples: resolve_fixed_model_extra_samples(
                    photometric,
                    samples_per_pixel,
                    3,
                    extra_samples,
                )?,
            }),
        }
    }

    /// Strip offsets as normalized `u64`s.
    pub fn strip_offsets(&self) -> Option<Vec<u64>> {
        self.tag_u64_list(TAG_STRIP_OFFSETS)
    }

    /// Strip byte counts as normalized `u64`s.
    pub fn strip_byte_counts(&self) -> Option<Vec<u64>> {
        self.tag_u64_list(TAG_STRIP_BYTE_COUNTS)
    }

    /// Tile offsets as normalized `u64`s.
    pub fn tile_offsets(&self) -> Option<Vec<u64>> {
        self.tag_u64_list(TAG_TILE_OFFSETS)
    }

    /// Tile byte counts as normalized `u64`s.
    pub fn tile_byte_counts(&self) -> Option<Vec<u64>> {
        self.tag_u64_list(TAG_TILE_BYTE_COUNTS)
    }

    /// SubIFD offsets as normalized `u64`s.
    pub fn sub_ifd_offsets(&self) -> Option<Vec<u64>> {
        self.tag_u64_list(TAG_SUB_IFDS)
    }

    /// Normalize and validate the raster layout for typed reads.
    pub fn raster_layout(&self) -> Result<RasterLayout> {
        let width = self.width();
        let height = self.height();
        if width == 0 || height == 0 {
            return Err(Error::InvalidImageLayout(format!(
                "image dimensions must be positive, got {}x{}",
                width, height
            )));
        }

        let samples_per_pixel = self.samples_per_pixel();
        if samples_per_pixel == 0 {
            return Err(Error::InvalidImageLayout(
                "SamplesPerPixel must be greater than zero".into(),
            ));
        }
        let samples_per_pixel = samples_per_pixel as usize;

        let bits = normalize_u16_values(
            TAG_BITS_PER_SAMPLE,
            self.bits_per_sample(),
            samples_per_pixel,
            1,
        )?;
        let formats = normalize_u16_values(
            TAG_SAMPLE_FORMAT,
            self.sample_format(),
            samples_per_pixel,
            1,
        )?;

        let first_bits = bits[0];
        let first_format = formats[0];
        if !bits.iter().all(|&value| value == first_bits) {
            return Err(Error::InvalidImageLayout(
                "mixed BitsPerSample values are not supported".into(),
            ));
        }
        if !formats.iter().all(|&value| value == first_format) {
            return Err(Error::InvalidImageLayout(
                "mixed SampleFormat values are not supported".into(),
            ));
        }
        if !matches!(first_format, 1..=3) {
            return Err(Error::UnsupportedSampleFormat(first_format));
        }
        validate_sample_encoding(first_format, first_bits)?;

        let planar_configuration = self.planar_configuration();
        if !matches!(planar_configuration, 1 | 2) {
            return Err(Error::UnsupportedPlanarConfiguration(planar_configuration));
        }

        let predictor = self.predictor();
        if !matches!(predictor, 1..=3) {
            return Err(Error::UnsupportedPredictor(predictor));
        }
        if first_bits < 8 && predictor != 1 {
            return Err(Error::InvalidImageLayout(
                "predictors are not supported for sub-byte sample encodings".into(),
            ));
        }

        validate_color_model(self, samples_per_pixel as u16, first_bits)?;

        Ok(RasterLayout {
            width: width as usize,
            height: height as usize,
            samples_per_pixel,
            bits_per_sample: first_bits,
            bytes_per_sample: usize::from(first_bits.div_ceil(8)),
            sample_format: first_format,
            planar_configuration,
            predictor,
        })
    }

    /// Normalize the raster layout produced by high-level pixel reads.
    ///
    /// This layout reflects color-decoded output rather than the on-disk sample
    /// organization. For example, palette and YCbCr rasters decode to RGB
    /// pixels, and sub-byte integer rasters expand to 8-bit samples.
    pub fn decoded_raster_layout(&self) -> Result<RasterLayout> {
        let storage = self.raster_layout()?;
        let color_model = self.color_model()?;
        let decoded_samples = match &color_model {
            ColorModel::Palette { extra_samples, .. } => 3 + extra_samples.len(),
            ColorModel::Cmyk { extra_samples } => 3 + extra_samples.len(),
            ColorModel::YCbCr { extra_samples, .. } => 3 + extra_samples.len(),
            ColorModel::Grayscale { extra_samples, .. } => 1 + extra_samples.len(),
            ColorModel::Rgb { extra_samples } => 3 + extra_samples.len(),
            ColorModel::Separated {
                color_channels,
                extra_samples,
                ..
            } => *color_channels as usize + extra_samples.len(),
            ColorModel::CieLab { extra_samples } => 3 + extra_samples.len(),
            ColorModel::TransparencyMask => 1,
        };
        let (sample_format, bits_per_sample) = match &color_model {
            ColorModel::Palette { color_map, .. } => {
                if color_map_is_u8_equivalent(color_map) {
                    (1, 8)
                } else {
                    (1, 16)
                }
            }
            ColorModel::YCbCr { .. } | ColorModel::Cmyk { .. } => {
                if storage.sample_format != 1 {
                    return Err(Error::InvalidImageLayout(
                        "decoded YCbCr/CMYK reads require unsigned integer source samples".into(),
                    ));
                }
                (1, decoded_uint_bits(storage.bits_per_sample))
            }
            _ => (
                storage.sample_format,
                decoded_bits(storage.sample_format, storage.bits_per_sample)?,
            ),
        };

        Ok(RasterLayout {
            width: storage.width,
            height: storage.height,
            samples_per_pixel: decoded_samples,
            bits_per_sample,
            bytes_per_sample: usize::from(bits_per_sample.div_ceil(8)),
            sample_format,
            planar_configuration: 1,
            predictor: 1,
        })
    }

    fn tag_u16(&self, code: u16) -> Option<u16> {
        self.tag(code).and_then(|tag| tag.value.as_u16())
    }

    fn tag_u32(&self, code: u16) -> Option<u32> {
        self.tag(code).and_then(|tag| tag.value.as_u32())
    }

    fn tag_u64_list(&self, code: u16) -> Option<Vec<u64>> {
        self.tag(code).and_then(|tag| tag.value.as_u64_vec())
    }
}

/// Parse the chain of IFDs starting from the header's first IFD offset.
pub fn parse_ifd_chain(source: &dyn TiffSource, header: &TiffHeader) -> Result<Vec<Ifd>> {
    let mut ifds = Vec::new();
    let mut offset = header.first_ifd_offset;
    let mut index = 0usize;
    let mut seen_offsets = HashSet::new();

    while offset != 0 {
        if !seen_offsets.insert(offset) {
            return Err(Error::InvalidImageLayout(format!(
                "IFD chain contains a loop at offset {offset}"
            )));
        }
        if offset >= source.len() {
            return Err(Error::Truncated {
                offset,
                needed: 2,
                available: source.len().saturating_sub(offset),
            });
        }

        let (tags, next_offset) = read_ifd(source, header, offset)?;

        ifds.push(Ifd { tags, index });
        offset = next_offset;
        index += 1;

        if index > 10_000 {
            return Err(Error::Other("IFD chain exceeds 10,000 entries".into()));
        }
    }

    Ok(ifds)
}

/// Parse a single IFD at the given file offset.
pub fn parse_ifd_at(source: &dyn TiffSource, header: &TiffHeader, offset: u64) -> Result<Ifd> {
    let (tags, _) = read_ifd(source, header, offset)?;
    Ok(Ifd {
        tags,
        index: usize::try_from(offset).unwrap_or(usize::MAX),
    })
}

fn read_ifd(source: &dyn TiffSource, header: &TiffHeader, offset: u64) -> Result<(Vec<Tag>, u64)> {
    let entry_count_size = if header.is_bigtiff() { 8usize } else { 2usize };
    let entry_size = if header.is_bigtiff() {
        20usize
    } else {
        12usize
    };
    let next_offset_size = if header.is_bigtiff() { 8usize } else { 4usize };

    let count_bytes = source.read_exact_at(offset, entry_count_size)?;
    let mut count_cursor = Cursor::new(&count_bytes, header.byte_order);
    let count = if header.is_bigtiff() {
        usize::try_from(count_cursor.read_u64()?).map_err(|_| {
            Error::InvalidImageLayout("BigTIFF entry count does not fit in usize".into())
        })?
    } else {
        count_cursor.read_u16()? as usize
    };

    let entries_len = count
        .checked_mul(entry_size)
        .and_then(|v| v.checked_add(next_offset_size))
        .ok_or_else(|| Error::InvalidImageLayout("IFD byte length overflows usize".into()))?;
    let body = source.read_exact_at(offset + entry_count_size as u64, entries_len)?;
    let mut cursor = Cursor::new(&body, header.byte_order);

    if header.is_bigtiff() {
        let tags = parse_tags_bigtiff(&mut cursor, count, source, header.byte_order)?;
        let next = cursor.read_u64()?;
        Ok((tags, next))
    } else {
        let tags = parse_tags_classic(&mut cursor, count, source, header.byte_order)?;
        let next = cursor.read_u32()? as u64;
        Ok((tags, next))
    }
}

fn normalize_u16_values(
    tag: u16,
    values: Vec<u16>,
    expected_len: usize,
    default_value: u16,
) -> Result<Vec<u16>> {
    match values.len() {
        0 => Ok(vec![default_value; expected_len]),
        1 if expected_len > 1 => Ok(vec![values[0]; expected_len]),
        len if len == expected_len => Ok(values),
        len => Err(Error::InvalidTagValue {
            tag,
            reason: format!("expected 1 or {expected_len} values, found {len}"),
        }),
    }
}

fn resolve_fixed_model_extra_samples(
    photometric: PhotometricInterpretation,
    samples_per_pixel: u16,
    base_samples: u16,
    mut extra_samples: Vec<ExtraSample>,
) -> Result<Vec<ExtraSample>> {
    let implied_extra_samples = samples_per_pixel.checked_sub(base_samples).ok_or_else(|| {
        Error::InvalidImageLayout(format!(
            "{} photometric interpretation requires at least {base_samples} samples, got {samples_per_pixel}",
            photometric_name(photometric)
        ))
    })?;
    if extra_samples.len() > implied_extra_samples as usize {
        return Err(Error::InvalidImageLayout(format!(
            "{} photometric interpretation has {} total channels but {} ExtraSamples",
            photometric_name(photometric),
            samples_per_pixel,
            extra_samples.len()
        )));
    }
    extra_samples.resize(implied_extra_samples as usize, ExtraSample::Unspecified);
    Ok(extra_samples)
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

fn validate_sample_encoding(sample_format: u16, bits_per_sample: u16) -> Result<()> {
    let supported = match sample_format {
        1 => matches!(bits_per_sample, 1 | 2 | 4 | 8 | 16 | 32 | 64),
        2 => matches!(bits_per_sample, 8 | 16 | 32 | 64),
        3 => matches!(bits_per_sample, 32 | 64),
        _ => false,
    };
    if !supported {
        return Err(Error::UnsupportedBitsPerSample(bits_per_sample));
    }
    Ok(())
}

fn decoded_uint_bits(bits_per_sample: u16) -> u16 {
    bits_per_sample.max(8)
}

fn decoded_bits(sample_format: u16, bits_per_sample: u16) -> Result<u16> {
    if sample_format == 1 {
        Ok(decoded_uint_bits(bits_per_sample))
    } else {
        validate_sample_encoding(sample_format, bits_per_sample)?;
        Ok(bits_per_sample)
    }
}

fn color_map_is_u8_equivalent(color_map: &ColorMap) -> bool {
    color_map
        .red()
        .iter()
        .chain(color_map.green().iter())
        .chain(color_map.blue().iter())
        .all(|&value| value % 257 == 0)
}

fn validate_color_model(ifd: &Ifd, samples_per_pixel: u16, bits_per_sample: u16) -> Result<()> {
    let color_model = ifd.color_model()?;

    match &color_model {
        ColorModel::Grayscale { extra_samples, .. } => {
            validate_expected_samples(samples_per_pixel, 1, extra_samples.len())?;
        }
        ColorModel::Palette {
            color_map,
            extra_samples,
        } => {
            let expected_entries = 1usize.checked_shl(bits_per_sample as u32).ok_or_else(|| {
                Error::InvalidImageLayout(format!(
                    "palette BitsPerSample {bits_per_sample} exceeds usize shift width"
                ))
            })?;
            if color_map.len() != expected_entries {
                return Err(Error::InvalidImageLayout(format!(
                    "palette ColorMap has {} entries but BitsPerSample={} requires {}",
                    color_map.len(),
                    bits_per_sample,
                    expected_entries
                )));
            }
            validate_expected_samples(samples_per_pixel, 1, extra_samples.len())?;
        }
        ColorModel::Rgb { extra_samples } => {
            validate_expected_samples(samples_per_pixel, 3, extra_samples.len())?;
        }
        ColorModel::TransparencyMask => {
            validate_expected_samples(samples_per_pixel, 1, 0)?;
        }
        ColorModel::Cmyk { extra_samples } => {
            validate_expected_samples(samples_per_pixel, 4, extra_samples.len())?;
        }
        ColorModel::Separated {
            color_channels,
            extra_samples,
            ..
        } => {
            if *color_channels == 0 {
                return Err(Error::InvalidImageLayout(
                    "separated photometric interpretation must have at least one base ink channel"
                        .into(),
                ));
            }
            validate_expected_samples(samples_per_pixel, *color_channels, extra_samples.len())?;
        }
        ColorModel::YCbCr {
            subsampling,
            extra_samples,
            ..
        } => {
            if subsampling.contains(&0) {
                return Err(Error::InvalidImageLayout(format!(
                    "YCbCr subsampling {:?} must be positive",
                    subsampling
                )));
            }
            if *subsampling != [1, 1] && !extra_samples.is_empty() {
                return Err(Error::InvalidImageLayout(
                    "subsampled YCbCr with ExtraSamples is not supported".into(),
                ));
            }
            if *subsampling != [1, 1] && ifd.predictor() != 1 {
                return Err(Error::InvalidImageLayout(
                    "subsampled YCbCr does not support TIFF predictors".into(),
                ));
            }
            validate_expected_samples(samples_per_pixel, 3, extra_samples.len())?;
        }
        ColorModel::CieLab { extra_samples } => {
            validate_expected_samples(samples_per_pixel, 3, extra_samples.len())?;
        }
    }

    Ok(())
}

fn validate_expected_samples(
    samples_per_pixel: u16,
    base_samples: u16,
    extra_sample_count: usize,
) -> Result<()> {
    let expected_samples = base_samples
        .checked_add(extra_sample_count as u16)
        .ok_or_else(|| Error::InvalidImageLayout("samples per pixel overflow".into()))?;
    if samples_per_pixel != expected_samples {
        return Err(Error::InvalidImageLayout(format!(
            "SamplesPerPixel={samples_per_pixel} does not match color model base channels {base_samples} plus {extra_sample_count} ExtraSamples"
        )));
    }
    Ok(())
}

/// Parse classic TIFF IFD entries (12 bytes each).
fn parse_tags_classic(
    cursor: &mut Cursor<'_>,
    count: usize,
    source: &dyn TiffSource,
    byte_order: ByteOrder,
) -> Result<Vec<Tag>> {
    let mut tags = Vec::with_capacity(count);
    for _ in 0..count {
        let code = cursor.read_u16()?;
        let type_code = cursor.read_u16()?;
        let value_count = cursor.read_u32()? as u64;
        let value_offset_bytes = cursor.read_bytes(4)?;
        let tag = parse_tag_classic(
            code,
            type_code,
            value_count,
            value_offset_bytes,
            source,
            byte_order,
        )?;
        tags.push(tag);
    }
    tags.sort_by_key(|tag| tag.code);
    Ok(tags)
}

/// Parse BigTIFF IFD entries (20 bytes each).
fn parse_tags_bigtiff(
    cursor: &mut Cursor<'_>,
    count: usize,
    source: &dyn TiffSource,
    byte_order: ByteOrder,
) -> Result<Vec<Tag>> {
    let mut tags = Vec::with_capacity(count);
    for _ in 0..count {
        let code = cursor.read_u16()?;
        let type_code = cursor.read_u16()?;
        let value_count = cursor.read_u64()?;
        let value_offset_bytes = cursor.read_bytes(8)?;
        let tag = parse_tag_bigtiff(
            code,
            type_code,
            value_count,
            value_offset_bytes,
            source,
            byte_order,
        )?;
        tags.push(tag);
    }
    tags.sort_by_key(|tag| tag.code);
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::{
        ColorModel, ExtraSample, Ifd, InkSet, LercAdditionalCompression, RasterLayout,
        TAG_BITS_PER_SAMPLE, TAG_COLOR_MAP, TAG_EXTRA_SAMPLES, TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH,
        TAG_INK_SET, TAG_LERC_PARAMETERS, TAG_PHOTOMETRIC_INTERPRETATION, TAG_SAMPLES_PER_PIXEL,
        TAG_SAMPLE_FORMAT, TAG_YCBCR_SUBSAMPLING,
    };
    use crate::tag::{Tag, TagType, TagValue};

    fn make_ifd(tags: Vec<Tag>) -> Ifd {
        let mut tags = tags;
        tags.sort_by_key(|tag| tag.code);
        Ifd { tags, index: 0 }
    }

    #[test]
    fn normalizes_single_value_sample_tags() {
        let ifd = make_ifd(vec![
            Tag {
                code: TAG_IMAGE_WIDTH,
                tag_type: TagType::Long,
                count: 1,
                value: TagValue::Long(vec![10]),
            },
            Tag {
                code: TAG_IMAGE_LENGTH,
                tag_type: TagType::Long,
                count: 1,
                value: TagValue::Long(vec![5]),
            },
            Tag {
                code: TAG_SAMPLES_PER_PIXEL,
                tag_type: TagType::Short,
                count: 1,
                value: TagValue::Short(vec![3]),
            },
            Tag {
                code: TAG_BITS_PER_SAMPLE,
                tag_type: TagType::Short,
                count: 1,
                value: TagValue::Short(vec![16]),
            },
            Tag {
                code: TAG_SAMPLE_FORMAT,
                tag_type: TagType::Short,
                count: 1,
                value: TagValue::Short(vec![1]),
            },
        ]);

        let layout = ifd.raster_layout().unwrap();
        assert_eq!(layout.width, 10);
        assert_eq!(layout.height, 5);
        assert_eq!(layout.samples_per_pixel, 3);
        assert_eq!(layout.bytes_per_sample, 2);
    }

    #[test]
    fn rejects_mixed_sample_formats() {
        let ifd = make_ifd(vec![
            Tag {
                code: TAG_IMAGE_WIDTH,
                tag_type: TagType::Long,
                count: 1,
                value: TagValue::Long(vec![1]),
            },
            Tag {
                code: TAG_IMAGE_LENGTH,
                tag_type: TagType::Long,
                count: 1,
                value: TagValue::Long(vec![1]),
            },
            Tag {
                code: TAG_SAMPLES_PER_PIXEL,
                tag_type: TagType::Short,
                count: 1,
                value: TagValue::Short(vec![2]),
            },
            Tag {
                code: TAG_BITS_PER_SAMPLE,
                tag_type: TagType::Short,
                count: 2,
                value: TagValue::Short(vec![16, 16]),
            },
            Tag {
                code: TAG_SAMPLE_FORMAT,
                tag_type: TagType::Short,
                count: 2,
                value: TagValue::Short(vec![1, 3]),
            },
        ]);

        assert!(ifd.raster_layout().is_err());
    }

    #[test]
    fn raster_layout_helpers_match_expected_strides() {
        let layout = RasterLayout {
            width: 4,
            height: 3,
            samples_per_pixel: 2,
            bits_per_sample: 16,
            bytes_per_sample: 2,
            sample_format: 1,
            planar_configuration: 1,
            predictor: 1,
        };
        assert_eq!(layout.pixel_stride_bytes(), 4);
        assert_eq!(layout.row_bytes(), 16);
        assert_eq!(layout.sample_plane_row_bytes(), 8);
    }

    #[test]
    fn parses_lerc_parameters() {
        let ifd = make_ifd(vec![Tag {
            code: TAG_LERC_PARAMETERS,
            tag_type: TagType::Long,
            count: 2,
            value: TagValue::Long(vec![4, 2]),
        }]);

        let params = ifd.lerc_parameters().unwrap().unwrap();
        assert_eq!(params.version, 4);
        assert_eq!(
            params.additional_compression,
            LercAdditionalCompression::Zstd
        );
    }

    #[test]
    fn parses_palette_color_model_and_extra_alpha() {
        let ifd = make_ifd(vec![
            Tag::new(TAG_IMAGE_WIDTH, TagValue::Long(vec![2])),
            Tag::new(TAG_IMAGE_LENGTH, TagValue::Long(vec![2])),
            Tag::new(TAG_SAMPLES_PER_PIXEL, TagValue::Short(vec![2])),
            Tag::new(TAG_BITS_PER_SAMPLE, TagValue::Short(vec![8, 8])),
            Tag::new(TAG_SAMPLE_FORMAT, TagValue::Short(vec![1, 1])),
            Tag::new(TAG_PHOTOMETRIC_INTERPRETATION, TagValue::Short(vec![3])),
            Tag::new(TAG_EXTRA_SAMPLES, TagValue::Short(vec![2])),
            Tag::new(
                TAG_COLOR_MAP,
                TagValue::Short(
                    (0u16..256)
                        .chain((0u16..256).map(|value| value.saturating_mul(2)))
                        .chain((0u16..256).map(|value| value.saturating_mul(3)))
                        .collect(),
                ),
            ),
        ]);

        let model = ifd.color_model().unwrap();
        match model {
            ColorModel::Palette {
                color_map,
                extra_samples,
            } => {
                assert_eq!(color_map.len(), 256);
                assert_eq!(extra_samples, vec![ExtraSample::UnassociatedAlpha]);
            }
            other => panic!("unexpected color model: {other:?}"),
        }

        let layout = ifd.raster_layout().unwrap();
        assert_eq!(layout.samples_per_pixel, 2);
    }

    #[test]
    fn parses_cmyk_color_model() {
        let ifd = make_ifd(vec![
            Tag::new(TAG_IMAGE_WIDTH, TagValue::Long(vec![1])),
            Tag::new(TAG_IMAGE_LENGTH, TagValue::Long(vec![1])),
            Tag::new(TAG_SAMPLES_PER_PIXEL, TagValue::Short(vec![4])),
            Tag::new(TAG_BITS_PER_SAMPLE, TagValue::Short(vec![8, 8, 8, 8])),
            Tag::new(TAG_SAMPLE_FORMAT, TagValue::Short(vec![1, 1, 1, 1])),
            Tag::new(TAG_PHOTOMETRIC_INTERPRETATION, TagValue::Short(vec![5])),
            Tag::new(TAG_INK_SET, TagValue::Short(vec![1])),
        ]);

        assert!(matches!(
            ifd.color_model().unwrap(),
            ColorModel::Cmyk { .. }
        ));
        assert_eq!(ifd.ink_set().unwrap(), Some(InkSet::Cmyk));
        assert_eq!(ifd.raster_layout().unwrap().samples_per_pixel, 4);
    }

    #[test]
    fn rejects_palette_without_colormap() {
        let ifd = make_ifd(vec![
            Tag::new(TAG_IMAGE_WIDTH, TagValue::Long(vec![1])),
            Tag::new(TAG_IMAGE_LENGTH, TagValue::Long(vec![1])),
            Tag::new(TAG_SAMPLES_PER_PIXEL, TagValue::Short(vec![1])),
            Tag::new(TAG_BITS_PER_SAMPLE, TagValue::Short(vec![8])),
            Tag::new(TAG_SAMPLE_FORMAT, TagValue::Short(vec![1])),
            Tag::new(TAG_PHOTOMETRIC_INTERPRETATION, TagValue::Short(vec![3])),
        ]);

        let error = ifd.raster_layout().unwrap_err();
        assert!(
            matches!(error, crate::error::Error::InvalidImageLayout(message) if message.contains("ColorMap"))
        );
    }

    #[test]
    fn accepts_subsampled_ycbcr_storage_layouts() {
        let ifd = make_ifd(vec![
            Tag::new(TAG_IMAGE_WIDTH, TagValue::Long(vec![2])),
            Tag::new(TAG_IMAGE_LENGTH, TagValue::Long(vec![2])),
            Tag::new(TAG_SAMPLES_PER_PIXEL, TagValue::Short(vec![3])),
            Tag::new(TAG_BITS_PER_SAMPLE, TagValue::Short(vec![8, 8, 8])),
            Tag::new(TAG_SAMPLE_FORMAT, TagValue::Short(vec![1, 1, 1])),
            Tag::new(TAG_PHOTOMETRIC_INTERPRETATION, TagValue::Short(vec![6])),
            Tag::new(TAG_YCBCR_SUBSAMPLING, TagValue::Short(vec![2, 2])),
        ]);

        let layout = ifd.raster_layout().unwrap();
        assert_eq!(layout.samples_per_pixel, 3);
        assert_eq!(ifd.decoded_raster_layout().unwrap().samples_per_pixel, 3);
    }
}
