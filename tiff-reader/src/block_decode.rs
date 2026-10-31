use crate::error::{Error, Result};
use crate::filters;
use crate::header::ByteOrder;
use crate::ifd::{Ifd, LercAdditionalCompression};
use lerc_core::{DataType, PixelData};
use lerc_reader::DecodedBandSet;
use tiff_core::{ColorModel, Compression, RasterLayout, SampleFormat};

const LERC1_MAGIC_PREFIX: &[u8; 9] = b"CntZImage";
const LERC2_MAGIC: &[u8; 6] = b"Lerc2 ";
const COMPRESSED_BLOCK_INPUT_RATIO: usize = 4;
const COMPRESSED_BLOCK_INPUT_SLACK: usize = 4096;
const FILL_ORDER_LSB_FIRST: u16 = 2;

/// Per-IFD metadata resolved once per read so per-block decoding avoids
/// re-deriving (and re-allocating) color-model and codec parameters.
pub(crate) struct BlockDecodeContext<'a> {
    pub layout: RasterLayout,
    pub byte_order: ByteOrder,
    pub compression_code: u16,
    pub compression: Option<Compression>,
    pub color_model: ColorModel,
    pub lerc_additional_compression: LercAdditionalCompression,
    pub jpeg_tables: Option<&'a [u8]>,
    pub bits_reversed: bool,
}

impl<'a> BlockDecodeContext<'a> {
    pub fn new(ifd: &'a Ifd, layout: RasterLayout, byte_order: ByteOrder) -> Result<Self> {
        let compression_code = ifd.compression();
        let compression = Compression::from_code(compression_code);
        Ok(Self {
            layout,
            byte_order,
            compression_code,
            compression,
            color_model: ifd.color_model()?,
            lerc_additional_compression: if compression == Some(Compression::Lerc) {
                ifd.lerc_parameters()?
                    .map(|params| params.additional_compression)
                    .unwrap_or(LercAdditionalCompression::None)
            } else {
                LercAdditionalCompression::None
            },
            jpeg_tables: ifd
                .tag(tiff_core::TAG_JPEG_TABLES)
                .and_then(|tag| tag.value.as_bytes()),
            // libtiff reverses stored bytes before every codec except JPEG,
            // whose bitstream has a fixed bit order of its own.
            bits_reversed: ifd.fill_order() == FILL_ORDER_LSB_FIRST
                && compression != Some(Compression::Jpeg),
        })
    }

    fn block_samples(&self) -> usize {
        if self.layout.planar_configuration == 1 {
            self.layout.samples_per_pixel
        } else {
            1
        }
    }

    fn is_subsampled_ycbcr_non_jpeg(&self) -> bool {
        matches!(
            &self.color_model,
            ColorModel::YCbCr {
                subsampling,
                extra_samples,
                ..
            } if *subsampling != [1, 1]
                && extra_samples.is_empty()
                && self.compression != Some(Compression::Jpeg)
        )
    }
}

pub(crate) struct BlockDecodeRequest<'a> {
    pub context: &'a BlockDecodeContext<'a>,
    pub compressed: &'a [u8],
    pub index: usize,
    pub block_width: usize,
    pub block_height: usize,
}

#[derive(Clone, Copy)]
struct SerializationPlan {
    pixel_count: usize,
    band_count: usize,
    depth: usize,
    layout: RasterLayout,
    index: usize,
}

#[derive(Clone, Copy)]
struct Lerc2Header {
    width: u32,
    height: u32,
    depth: u32,
    data_type: DataType,
    blob_size: usize,
}

pub(crate) fn decode_compressed_block(request: BlockDecodeRequest<'_>) -> Result<Vec<u8>> {
    if request.context.bits_reversed {
        let reversed: Vec<u8> = request
            .compressed
            .iter()
            .map(|byte| byte.reverse_bits())
            .collect();
        return decode_block_payload(BlockDecodeRequest {
            compressed: &reversed,
            ..request
        });
    }
    decode_block_payload(request)
}

fn decode_block_payload(request: BlockDecodeRequest<'_>) -> Result<Vec<u8>> {
    let samples = request.context.block_samples();
    let expected_len = expected_encoded_block_len(&request, samples)?;

    if request.context.compression != Some(Compression::Lerc) {
        let image_layout = matches!(
            request.context.compression,
            Some(Compression::Jpeg | Compression::WebP)
        )
        .then_some(filters::ImageDecodeLayout {
            width: request.block_width,
            height: request.block_height,
            samples_per_pixel: samples,
        });
        let mut decoded = filters::decompress_with_layout(
            request.context.compression_code,
            request.compressed,
            request.index,
            request.context.jpeg_tables,
            expected_len,
            image_layout,
        )?;
        if decoded.len() < expected_len {
            return Err(Error::DecompressionFailed {
                index: request.index,
                reason: format!(
                    "decoded block is too small: expected at least {expected_len} bytes, found {}",
                    decoded.len()
                ),
            });
        }
        if decoded.len() > expected_len {
            decoded.truncate(expected_len);
        }
        let is_subsampled_ycbcr = request.context.is_subsampled_ycbcr_non_jpeg();
        let row_bytes = if is_subsampled_ycbcr {
            decoded.len()
        } else if request.context.layout.bits_per_sample < 8 {
            if request.context.layout.planar_configuration == 1 {
                request
                    .context
                    .layout
                    .checked_packed_row_bytes_for_width(request.block_width)?
            } else {
                request
                    .context
                    .layout
                    .checked_packed_sample_plane_row_bytes_for_width(request.block_width)?
            }
        } else {
            request
                .block_width
                .checked_mul(samples)
                .and_then(|value| value.checked_mul(request.context.layout.bytes_per_sample))
                .ok_or_else(|| Error::InvalidImageLayout("block row size overflows usize".into()))?
        };
        let mut predictor_scratch = Vec::new();
        for row in decoded.chunks_exact_mut(row_bytes) {
            filters::fix_endianness_and_predict_with_scratch(
                row,
                request.context.layout.bits_per_sample,
                samples as u16,
                request.context.byte_order,
                request.context.layout.predictor,
                &mut predictor_scratch,
            )?;
        }
        if is_subsampled_ycbcr {
            let ColorModel::YCbCr { subsampling, .. } = &request.context.color_model else {
                unreachable!();
            };
            decoded = expand_subsampled_ycbcr(
                &decoded,
                request.context.layout.bytes_per_sample,
                request.block_width,
                request.block_height,
                *subsampling,
            )?;
        } else if request.context.layout.bits_per_sample < 8 {
            decoded = unpack_subbyte_block(
                &decoded,
                request.context.layout.bits_per_sample,
                samples,
                request.block_width,
                request.block_height,
                request.index,
            )?;
        }
        return Ok(decoded);
    }

    decode_lerc_block(request, expected_len)
}

/// Final decoded byte length produced by `decode_compressed_block`.
///
/// This differs from `expected_encoded_block_len` for layouts whose decode
/// step changes the sample representation: sub-byte samples unpack to one
/// byte per sample and subsampled YCbCr expands to full-resolution chroma.
pub(crate) fn decoded_block_len(request: &BlockDecodeRequest<'_>) -> Result<usize> {
    let samples = request.context.block_samples();
    if request.context.is_subsampled_ycbcr_non_jpeg() {
        return request
            .block_width
            .checked_mul(request.block_height)
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|values| values.checked_mul(request.context.layout.bytes_per_sample))
            .ok_or_else(|| {
                Error::InvalidImageLayout("expanded YCbCr block overflows usize".into())
            });
    }
    if request.context.layout.bits_per_sample < 8 {
        return request
            .block_width
            .checked_mul(samples)
            .and_then(|row_samples| row_samples.checked_mul(request.block_height))
            .ok_or_else(|| {
                Error::InvalidImageLayout("sub-byte block size overflows usize".into())
            });
    }
    expected_encoded_block_len(request, samples)
}

pub(crate) fn compressed_block_byte_count_limit(request: &BlockDecodeRequest<'_>) -> Result<usize> {
    let samples = request.context.block_samples();
    let decoded_len_limit = expected_encoded_block_len(request, samples)?;

    match request.context.compression {
        Some(Compression::None) => Ok(decoded_len_limit),
        Some(Compression::Lerc) => {
            let lerc_payload_len_limit =
                expected_lerc_payload_len_limit(request, decoded_len_limit)?;
            match request.context.lerc_additional_compression {
                LercAdditionalCompression::None => Ok(lerc_payload_len_limit),
                LercAdditionalCompression::Deflate | LercAdditionalCompression::Zstd => {
                    compressed_input_len_limit(lerc_payload_len_limit)
                }
            }
        }
        _ => compressed_input_len_limit(decoded_len_limit),
    }
}

fn expected_encoded_block_len(request: &BlockDecodeRequest<'_>, samples: usize) -> Result<usize> {
    if request.context.is_subsampled_ycbcr_non_jpeg() {
        let ColorModel::YCbCr { subsampling, .. } = &request.context.color_model else {
            unreachable!();
        };
        let units_across = request.block_width.div_ceil(subsampling[0] as usize);
        let units_down = request.block_height.div_ceil(subsampling[1] as usize);
        let samples_per_unit = usize::from(subsampling[0])
            .checked_mul(usize::from(subsampling[1]))
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| Error::InvalidImageLayout("YCbCr unit size overflows usize".into()))?;
        return units_across
            .checked_mul(units_down)
            .and_then(|units| units.checked_mul(samples_per_unit))
            .and_then(|values| values.checked_mul(request.context.layout.bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout("YCbCr block size overflows usize".into()));
    }

    let row_bytes = if request.context.layout.bits_per_sample < 8 {
        if request.context.layout.planar_configuration == 1 {
            request
                .context
                .layout
                .checked_packed_row_bytes_for_width(request.block_width)?
        } else {
            request
                .context
                .layout
                .checked_packed_sample_plane_row_bytes_for_width(request.block_width)?
        }
    } else {
        request
            .block_width
            .checked_mul(samples)
            .and_then(|value| value.checked_mul(request.context.layout.bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout("block row size overflows usize".into()))?
    };
    request
        .block_height
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::InvalidImageLayout("block size overflows usize".into()))
}

fn unpack_subbyte_block(
    packed: &[u8],
    bits_per_sample: u16,
    samples_per_pixel: usize,
    block_width: usize,
    block_height: usize,
    index: usize,
) -> Result<Vec<u8>> {
    debug_assert!(matches!(bits_per_sample, 1 | 2 | 4));
    let row_samples = block_width
        .checked_mul(samples_per_pixel)
        .ok_or_else(|| Error::InvalidImageLayout("sub-byte row samples overflow usize".into()))?;
    let row_bytes = (row_samples * bits_per_sample as usize).div_ceil(8);
    let expected_len = row_bytes
        .checked_mul(block_height)
        .ok_or_else(|| Error::InvalidImageLayout("sub-byte block size overflows usize".into()))?;
    if packed.len() != expected_len {
        return Err(Error::DecompressionFailed {
            index,
            reason: format!(
                "sub-byte decoded block length {} does not match expected {expected_len}",
                packed.len()
            ),
        });
    }

    let mut unpacked = Vec::with_capacity(row_samples * block_height);
    let mask = ((1u16 << bits_per_sample) - 1) as u8;
    let samples_per_byte = 8 / bits_per_sample as usize;
    for row in packed.chunks_exact(row_bytes) {
        for sample_index in 0..row_samples {
            let byte = row[sample_index / samples_per_byte];
            let shift = 8 - bits_per_sample as usize * ((sample_index % samples_per_byte) + 1);
            unpacked.push((byte >> shift) & mask);
        }
    }
    Ok(unpacked)
}

fn expand_subsampled_ycbcr(
    packed: &[u8],
    bytes_per_sample: usize,
    block_width: usize,
    block_height: usize,
    subsampling: [u16; 2],
) -> Result<Vec<u8>> {
    let h = usize::from(subsampling[0]);
    let v = usize::from(subsampling[1]);
    let units_across = block_width.div_ceil(h);
    let units_down = block_height.div_ceil(v);
    let samples_per_unit = h
        .checked_mul(v)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| Error::InvalidImageLayout("YCbCr unit size overflows usize".into()))?;
    let unit_bytes = samples_per_unit
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| Error::InvalidImageLayout("YCbCr unit byte size overflows usize".into()))?;
    let expected_len = units_across
        .checked_mul(units_down)
        .and_then(|units| units.checked_mul(unit_bytes))
        .ok_or_else(|| Error::InvalidImageLayout("YCbCr block size overflows usize".into()))?;
    if packed.len() != expected_len {
        return Err(Error::InvalidImageLayout(format!(
            "YCbCr block length {} does not match expected {expected_len}",
            packed.len()
        )));
    }

    let mut expanded = vec![
        0u8;
        block_width
            .checked_mul(block_height)
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout(
                "expanded YCbCr block overflows usize".into()
            ))?
    ];

    let mut offset = 0usize;
    for unit_row in 0..units_down {
        for unit_col in 0..units_across {
            let y_values = &packed[offset..offset + h * v * bytes_per_sample];
            offset += h * v * bytes_per_sample;
            let cb = &packed[offset..offset + bytes_per_sample];
            offset += bytes_per_sample;
            let cr = &packed[offset..offset + bytes_per_sample];
            offset += bytes_per_sample;

            for dy in 0..v {
                let row = unit_row * v + dy;
                if row >= block_height {
                    break;
                }
                for dx in 0..h {
                    let col = unit_col * h + dx;
                    if col >= block_width {
                        break;
                    }
                    let pixel_index = row
                        .checked_mul(block_width)
                        .and_then(|value| value.checked_add(col))
                        .ok_or_else(|| {
                            Error::InvalidImageLayout(
                                "expanded YCbCr pixel index overflows usize".into(),
                            )
                        })?;
                    let dest = pixel_index
                        .checked_mul(3 * bytes_per_sample)
                        .ok_or_else(|| {
                            Error::InvalidImageLayout(
                                "expanded YCbCr output index overflows usize".into(),
                            )
                        })?;
                    let y_offset = (dy * h + dx) * bytes_per_sample;
                    expanded[dest..dest + bytes_per_sample]
                        .copy_from_slice(&y_values[y_offset..y_offset + bytes_per_sample]);
                    expanded[dest + bytes_per_sample..dest + 2 * bytes_per_sample]
                        .copy_from_slice(cb);
                    expanded[dest + 2 * bytes_per_sample..dest + 3 * bytes_per_sample]
                        .copy_from_slice(cr);
                }
            }
        }
    }

    Ok(expanded)
}

fn decode_lerc_block(request: BlockDecodeRequest<'_>, expected_len: usize) -> Result<Vec<u8>> {
    let lerc_payload_len_limit = expected_lerc_payload_len_limit(&request, expected_len)?;
    let payload = match request.context.lerc_additional_compression {
        LercAdditionalCompression::None => {
            if request.compressed.len() > lerc_payload_len_limit {
                return Err(Error::DecompressionFailed {
                    index: request.index,
                    reason: format!(
                        "LERC: payload byte count {} exceeds TIFF block budget {lerc_payload_len_limit}",
                        request.compressed.len()
                    ),
                });
            }
            request.compressed.to_vec()
        }
        LercAdditionalCompression::Deflate => filters::decompress(
            Compression::Deflate.to_code(),
            request.compressed,
            request.index,
            None,
            lerc_payload_len_limit,
        )?,
        LercAdditionalCompression::Zstd => filters::decompress(
            Compression::Zstd.to_code(),
            request.compressed,
            request.index,
            None,
            lerc_payload_len_limit,
        )?,
    };

    validate_lerc_payload_before_decode(
        &payload,
        request.context.layout,
        request.block_width,
        request.block_height,
        request.index,
    )?;
    let decoded =
        lerc_reader::decode_band_set(&payload).map_err(|error| Error::DecompressionFailed {
            index: request.index,
            reason: format!("LERC: {error}"),
        })?;
    validate_lerc_layout(
        &decoded,
        request.context.layout,
        request.block_width,
        request.block_height,
        request.index,
    )?;
    serialize_lerc_band_set(
        &decoded,
        request.context.layout,
        expected_len,
        request.index,
    )
}

fn expected_lerc_payload_len_limit(
    request: &BlockDecodeRequest<'_>,
    expected_len: usize,
) -> Result<usize> {
    let samples = request.context.block_samples();
    let pixel_count = request
        .block_width
        .checked_mul(request.block_height)
        .ok_or_else(|| {
            Error::InvalidImageLayout("LERC block pixel count overflows usize".into())
        })?;
    let sample_count = pixel_count
        .checked_mul(samples)
        .ok_or_else(|| Error::InvalidImageLayout("LERC sample count overflows usize".into()))?;

    // LERC additional compression inflates to the LERC container, not directly
    // to pixels. Allow headers, masks, and block metadata while keeping a finite
    // decompression ceiling before LERC header validation runs.
    let raw_payload_budget = expected_len
        .checked_mul(4)
        .and_then(|value| value.checked_add(sample_count))
        .and_then(|value| value.checked_add(samples.saturating_mul(1024)))
        .and_then(|value| value.checked_add(4096))
        .ok_or_else(|| Error::InvalidImageLayout("LERC payload budget overflows usize".into()))?;
    Ok(raw_payload_budget.max(expected_len))
}

fn compressed_input_len_limit(decoded_len_limit: usize) -> Result<usize> {
    decoded_len_limit
        .checked_mul(COMPRESSED_BLOCK_INPUT_RATIO)
        .and_then(|value| value.checked_add(COMPRESSED_BLOCK_INPUT_SLACK))
        .ok_or_else(|| {
            Error::InvalidImageLayout("compressed block input budget overflows usize".into())
        })
}

fn validate_lerc_payload_before_decode(
    payload: &[u8],
    layout: RasterLayout,
    block_width: usize,
    block_height: usize,
    index: usize,
) -> Result<()> {
    let mut offset = 0usize;
    let expected_type = expected_lerc_data_type(layout)?;
    let expected_samples = if layout.planar_configuration == 1 {
        layout.samples_per_pixel
    } else {
        1
    };
    let mut band_count = 0usize;
    let mut shared_depth = None;

    while offset < payload.len() {
        let slice = &payload[offset..];
        if slice.starts_with(LERC1_MAGIC_PREFIX) || !slice.starts_with(LERC2_MAGIC) {
            // Lerc1 does not carry a cheap top-level blob size. Let the reader
            // produce the canonical error for Lerc1 and invalid-magic payloads.
            return Ok(());
        }

        let Some(header) = parse_lerc2_header(slice, index)? else {
            return Ok(());
        };
        if header.width as usize != block_width || header.height as usize != block_height {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC raster dimensions {}x{} do not match TIFF block {}x{}",
                    header.width, header.height, block_width, block_height
                ),
            });
        }
        if header.data_type != expected_type {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC data type {} does not match TIFF sample layout (sample_format={} bits_per_sample={})",
                    header.data_type.name(),
                    layout.sample_format,
                    layout.bits_per_sample
                ),
            });
        }
        if header.depth == 0 {
            return Err(Error::DecompressionFailed {
                index,
                reason: "LERC depth must be greater than zero".into(),
            });
        }
        match shared_depth {
            Some(depth) if depth != header.depth => {
                return Err(Error::DecompressionFailed {
                    index,
                    reason: "LERC band set contains mismatched depth values".into(),
                });
            }
            Some(_) => {}
            None => shared_depth = Some(header.depth),
        }
        if header.blob_size > slice.len() {
            return Ok(());
        }

        band_count += 1;
        offset = offset
            .checked_add(header.blob_size)
            .ok_or_else(|| Error::InvalidImageLayout("LERC band offset overflows usize".into()))?;
    }

    if let Some(depth) = shared_depth {
        let depth = depth as usize;
        if !((band_count == 1 && depth == expected_samples)
            || (depth == 1 && band_count == expected_samples))
        {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC band/depth layout band_count={band_count} depth={depth} does not match TIFF samples_per_pixel={expected_samples}"
                ),
            });
        }
    }

    Ok(())
}

fn parse_lerc2_header(slice: &[u8], index: usize) -> Result<Option<Lerc2Header>> {
    let Some(version) = read_i32_le_at(slice, 6) else {
        return Ok(None);
    };
    let Some(blob_size_offset) = lerc2_blob_size_offset(version) else {
        return Ok(None);
    };
    let Some(blob_size) = read_i32_le_at(slice, blob_size_offset) else {
        return Ok(None);
    };
    let Some(min_blob_size) = lerc2_min_blob_size(version) else {
        return Ok(None);
    };
    if blob_size <= 0 || (blob_size as usize) < min_blob_size {
        return Err(Error::DecompressionFailed {
            index,
            reason: format!(
                "LERC: invalid Lerc2 v{version} blob size {blob_size}; expected at least {min_blob_size} bytes"
            ),
        });
    }

    let Some(height_offset) = lerc2_height_offset(version) else {
        return Ok(None);
    };
    let Some(height) = read_u32_le_at(slice, height_offset) else {
        return Ok(None);
    };
    let Some(width) = read_u32_le_at(slice, height_offset + 4) else {
        return Ok(None);
    };
    let depth = if version >= 4 {
        let Some(depth) = read_u32_le_at(slice, height_offset + 8) else {
            return Ok(None);
        };
        depth
    } else {
        1
    };
    let Some(data_type_code) = read_i32_le_at(slice, blob_size_offset + 4) else {
        return Ok(None);
    };
    let data_type =
        DataType::from_code(data_type_code).map_err(|error| Error::DecompressionFailed {
            index,
            reason: format!("LERC: {error}"),
        })?;

    Ok(Some(Lerc2Header {
        width,
        height,
        depth,
        data_type,
        blob_size: blob_size as usize,
    }))
}

fn lerc2_height_offset(version: i32) -> Option<usize> {
    match version {
        1 | 2 => Some(10),
        3..=6 => Some(14),
        _ => None,
    }
}

fn lerc2_blob_size_offset(version: i32) -> Option<usize> {
    match version {
        1 | 2 => Some(26),
        3 => Some(30),
        4..=6 => Some(34),
        _ => None,
    }
}

fn lerc2_min_blob_size(version: i32) -> Option<usize> {
    match version {
        1 | 2 => Some(62),
        3 => Some(66),
        4 | 5 => Some(70),
        6 => Some(94),
        _ => None,
    }
}

fn read_i32_le_at(bytes: &[u8], offset: usize) -> Option<i32> {
    let end = offset.checked_add(4)?;
    Some(i32::from_le_bytes(bytes.get(offset..end)?.try_into().ok()?))
}

fn read_u32_le_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    Some(u32::from_le_bytes(bytes.get(offset..end)?.try_into().ok()?))
}

fn validate_lerc_layout(
    decoded: &DecodedBandSet,
    layout: RasterLayout,
    block_width: usize,
    block_height: usize,
    index: usize,
) -> Result<()> {
    let expected_type = expected_lerc_data_type(layout)?;
    let band_count = decoded.info().band_count();
    if band_count == 0 {
        return Err(Error::DecompressionFailed {
            index,
            reason: "LERC band set must contain at least one band".into(),
        });
    }
    if decoded.bands().len() != band_count {
        return Err(Error::DecompressionFailed {
            index,
            reason: format!(
                "LERC decoded band payload count {} does not match metadata band count {band_count}",
                decoded.bands().len()
            ),
        });
    }
    for band in decoded.info().bands() {
        if band.width as usize != block_width || band.height as usize != block_height {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC raster dimensions {}x{} do not match TIFF block {}x{}",
                    band.width, band.height, block_width, block_height
                ),
            });
        }
        if band.data_type != expected_type {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC data type {} does not match TIFF sample layout (sample_format={} bits_per_sample={})",
                    band.data_type.name(),
                    layout.sample_format,
                    layout.bits_per_sample
                ),
            });
        }
    }

    let expected_samples = if layout.planar_configuration == 1 {
        layout.samples_per_pixel
    } else {
        1
    };
    let depth = decoded.info().depth().max(1) as usize;
    if !((band_count == 1 && depth == expected_samples)
        || (depth == 1 && band_count == expected_samples))
    {
        return Err(Error::DecompressionFailed {
            index,
            reason: format!(
                "LERC band/depth layout band_count={band_count} depth={depth} does not match TIFF samples_per_pixel={expected_samples}"
            ),
        });
    }

    Ok(())
}

fn expected_lerc_data_type(layout: RasterLayout) -> Result<DataType> {
    match (
        SampleFormat::from_code(layout.sample_format),
        layout.bits_per_sample,
    ) {
        (Some(SampleFormat::Uint), 8) => Ok(DataType::U8),
        (Some(SampleFormat::Uint), 16) => Ok(DataType::U16),
        (Some(SampleFormat::Uint), 32) => Ok(DataType::U32),
        (Some(SampleFormat::Int), 8) => Ok(DataType::I8),
        (Some(SampleFormat::Int), 16) => Ok(DataType::I16),
        (Some(SampleFormat::Int), 32) => Ok(DataType::I32),
        (Some(SampleFormat::Float), 32) => Ok(DataType::F32),
        (Some(SampleFormat::Float), 64) => Ok(DataType::F64),
        _ => Err(Error::InvalidImageLayout(format!(
            "LERC does not support sample_format={} bits_per_sample={}",
            layout.sample_format, layout.bits_per_sample
        ))),
    }
}

fn serialize_lerc_band_set(
    decoded: &DecodedBandSet,
    layout: RasterLayout,
    expected_len: usize,
    index: usize,
) -> Result<Vec<u8>> {
    let pixel_count = decoded.info().bands()[0].pixel_count().map_err(|error| {
        Error::InvalidImageLayout(format!("LERC pixel count overflow: {error}"))
    })?;
    let mut out = Vec::with_capacity(expected_len);
    let plan = SerializationPlan {
        pixel_count,
        band_count: decoded.info().band_count(),
        depth: decoded.info().depth().max(1) as usize,
        layout,
        index,
    };

    match &decoded.bands()[0] {
        PixelData::I8(_) => {
            serialize_typed::<i8, _>(decoded, plan, 0, &mut out, |band| match band {
                PixelData::I8(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::U8(_) => {
            serialize_typed::<u8, _>(decoded, plan, 0, &mut out, |band| match band {
                PixelData::U8(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::I16(_) => {
            serialize_typed::<i16, _>(decoded, plan, 0, &mut out, |band| match band {
                PixelData::I16(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::U16(_) => {
            serialize_typed::<u16, _>(decoded, plan, 0, &mut out, |band| match band {
                PixelData::U16(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::I32(_) => {
            serialize_typed::<i32, _>(decoded, plan, 0, &mut out, |band| match band {
                PixelData::I32(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::U32(_) => {
            serialize_typed::<u32, _>(decoded, plan, 0, &mut out, |band| match band {
                PixelData::U32(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::F32(_) => {
            serialize_typed::<f32, _>(decoded, plan, f32::NAN, &mut out, |band| match band {
                PixelData::F32(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
        PixelData::F64(_) => {
            serialize_typed::<f64, _>(decoded, plan, f64::NAN, &mut out, |band| match band {
                PixelData::F64(values) => Some(values.as_slice()),
                _ => None,
            })?
        }
    }

    if out.len() != expected_len {
        return Err(Error::DecompressionFailed {
            index: plan.index,
            reason: format!(
                "decoded LERC block length {} does not match expected TIFF block length {expected_len}",
                out.len()
            ),
        });
    }

    Ok(out)
}

fn serialize_typed<'a, T, F>(
    decoded: &'a DecodedBandSet,
    plan: SerializationPlan,
    invalid_fill: T,
    out: &mut Vec<u8>,
    slice_for: F,
) -> Result<()>
where
    T: NativeEndianBytes + 'a,
    F: Fn(&'a PixelData) -> Option<&'a [T]>,
{
    let expected_samples = if plan.layout.planar_configuration == 1 {
        plan.layout.samples_per_pixel
    } else {
        1
    };
    let band_slices = decoded
        .bands()
        .iter()
        .map(|band| {
            slice_for(band).ok_or_else(|| {
                Error::InvalidImageLayout("LERC bands use mixed sample types".into())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if plan.band_count == 1 {
        let values = band_slices[0];
        let expected_values = plan
            .pixel_count
            .checked_mul(plan.depth)
            .ok_or_else(|| Error::InvalidImageLayout("LERC sample count overflows usize".into()))?;
        if values.len() != expected_values || plan.depth != expected_samples {
            return Err(Error::DecompressionFailed {
                index: plan.index,
                reason: format!(
                    "LERC single-band depth layout produced {} values with depth {} for {} pixels and TIFF samples_per_pixel={expected_samples}",
                    values.len(),
                    plan.depth,
                    plan.pixel_count
                ),
            });
        }
        let mask = decoded_band_mask(decoded, 0, plan.pixel_count, plan.index)?;
        for pixel in 0..plan.pixel_count {
            let valid = mask.map(|mask| mask[pixel] != 0).unwrap_or(true);
            let base = pixel * plan.depth;
            for sample in &values[base..base + plan.depth] {
                if valid {
                    sample.write_ne(out);
                } else {
                    invalid_fill.write_ne(out);
                }
            }
        }
        return Ok(());
    }

    if plan.depth != 1 || plan.band_count != expected_samples {
        return Err(Error::DecompressionFailed {
            index: plan.index,
            reason: format!(
                "LERC band-set layout band_count={} depth={} does not match TIFF samples_per_pixel={expected_samples}",
                plan.band_count, plan.depth
            ),
        });
    }

    for values in &band_slices {
        if values.len() != plan.pixel_count {
            return Err(Error::DecompressionFailed {
                index: plan.index,
                reason: format!(
                    "LERC band length {} does not match block pixel count {}",
                    values.len(),
                    plan.pixel_count
                ),
            });
        }
    }
    let band_masks = (0..band_slices.len())
        .map(|band_index| decoded_band_mask(decoded, band_index, plan.pixel_count, plan.index))
        .collect::<Result<Vec<_>>>()?;

    for pixel in 0..plan.pixel_count {
        for (band_index, values) in band_slices.iter().enumerate() {
            let valid = band_masks[band_index]
                .map(|mask| mask[pixel] != 0)
                .unwrap_or(true);
            if valid {
                values[pixel].write_ne(out);
            } else {
                invalid_fill.write_ne(out);
            }
        }
    }

    Ok(())
}

fn decoded_band_mask(
    decoded: &DecodedBandSet,
    band_index: usize,
    pixel_count: usize,
    index: usize,
) -> Result<Option<&[u8]>> {
    let Some(mask) = decoded
        .band_masks()
        .get(band_index)
        .and_then(|mask| mask.as_deref())
    else {
        return Ok(None);
    };
    if mask.len() != pixel_count {
        return Err(Error::DecompressionFailed {
            index,
            reason: format!(
                "LERC mask length {} does not match block pixel count {pixel_count}",
                mask.len()
            ),
        });
    }
    Ok(Some(mask))
}

trait NativeEndianBytes: Copy {
    fn write_ne(self, out: &mut Vec<u8>);
}

impl NativeEndianBytes for i8 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.push(self as u8);
    }
}

impl NativeEndianBytes for u8 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

impl NativeEndianBytes for i16 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl NativeEndianBytes for u16 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl NativeEndianBytes for i32 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl NativeEndianBytes for u32 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl NativeEndianBytes for f32 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl NativeEndianBytes for f64 {
    fn write_ne(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}
