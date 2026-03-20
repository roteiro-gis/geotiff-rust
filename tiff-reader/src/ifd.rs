use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::header::{ByteOrder, TiffHeader};
use crate::io::Cursor;
use crate::source::TiffSource;
use crate::tag::Tag;

/// A parsed Image File Directory (IFD).
#[derive(Debug, Clone)]
pub struct Ifd {
    /// Tags in this IFD, sorted by tag code.
    tags: Vec<Tag>,
    /// Index of this IFD in the chain (0-based).
    pub index: usize,
}

/// Raster layout information normalized from TIFF tags.
#[derive(Debug, Clone, Copy)]
pub struct RasterLayout {
    pub width: usize,
    pub height: usize,
    pub samples_per_pixel: usize,
    pub bits_per_sample: u16,
    pub bytes_per_sample: usize,
    pub sample_format: u16,
    pub planar_configuration: u16,
    pub predictor: u16,
}

impl RasterLayout {
    pub fn pixel_stride_bytes(&self) -> usize {
        self.samples_per_pixel * self.bytes_per_sample
    }

    pub fn row_bytes(&self) -> usize {
        self.width * self.pixel_stride_bytes()
    }

    pub fn sample_plane_row_bytes(&self) -> usize {
        self.width * self.bytes_per_sample
    }
}

// Well-known TIFF tag codes.
pub const TAG_IMAGE_WIDTH: u16 = 256;
pub const TAG_IMAGE_LENGTH: u16 = 257;
pub const TAG_BITS_PER_SAMPLE: u16 = 258;
pub const TAG_COMPRESSION: u16 = 259;
pub const TAG_PHOTOMETRIC_INTERPRETATION: u16 = 262;
pub const TAG_STRIP_OFFSETS: u16 = 273;
pub const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub const TAG_ROWS_PER_STRIP: u16 = 278;
pub const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub const TAG_PLANAR_CONFIGURATION: u16 = 284;
pub const TAG_PREDICTOR: u16 = 317;
pub const TAG_TILE_WIDTH: u16 = 322;
pub const TAG_TILE_LENGTH: u16 = 323;
pub const TAG_TILE_OFFSETS: u16 = 324;
pub const TAG_TILE_BYTE_COUNTS: u16 = 325;
pub const TAG_SAMPLE_FORMAT: u16 = 339;

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
        Some(self.tag_u32(TAG_ROWS_PER_STRIP).unwrap_or_else(|| self.height()))
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
        if !matches!(first_bits, 8 | 16 | 32 | 64) {
            return Err(Error::UnsupportedBitsPerSample(first_bits));
        }
        if !matches!(first_format, 1..=3) {
            return Err(Error::UnsupportedSampleFormat(first_format));
        }

        let planar_configuration = self.planar_configuration();
        if !matches!(planar_configuration, 1 | 2) {
            return Err(Error::UnsupportedPlanarConfiguration(planar_configuration));
        }

        let predictor = self.predictor();
        if !matches!(predictor, 1..=3) {
            return Err(Error::UnsupportedPredictor(predictor));
        }

        Ok(RasterLayout {
            width: width as usize,
            height: height as usize,
            samples_per_pixel,
            bits_per_sample: first_bits,
            bytes_per_sample: (first_bits / 8) as usize,
            sample_format: first_format,
            planar_configuration,
            predictor,
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

fn read_ifd(source: &dyn TiffSource, header: &TiffHeader, offset: u64) -> Result<(Vec<Tag>, u64)> {
    let entry_count_size = if header.is_bigtiff() { 8usize } else { 2usize };
    let entry_size = if header.is_bigtiff() { 20usize } else { 12usize };
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
        let tag = Tag::parse_classic(code, type_code, value_count, value_offset_bytes, source, byte_order)?;
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
        let tag = Tag::parse_bigtiff(code, type_code, value_count, value_offset_bytes, source, byte_order)?;
        tags.push(tag);
    }
    tags.sort_by_key(|tag| tag.code);
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::{Ifd, RasterLayout, TAG_BITS_PER_SAMPLE, TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH, TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL};
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
}
