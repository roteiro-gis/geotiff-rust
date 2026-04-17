use crate::error::{Error, Result};
use crate::filters;
use crate::header::ByteOrder;
use crate::ifd::{Ifd, LercAdditionalCompression};
use lerc_core::{DataType, PixelData};
use lerc_reader::DecodedBandSet;
use tiff_core::{ColorModel, Compression, RasterLayout, SampleFormat};

pub(crate) struct BlockDecodeRequest<'a> {
    pub ifd: &'a Ifd,
    pub layout: RasterLayout,
    pub byte_order: ByteOrder,
    pub compressed: &'a [u8],
    pub index: usize,
    pub jpeg_tables: Option<&'a [u8]>,
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

pub(crate) fn decode_compressed_block(request: BlockDecodeRequest<'_>) -> Result<Vec<u8>> {
    let samples = if request.layout.planar_configuration == 1 {
        request.layout.samples_per_pixel
    } else {
        1
    };
    let expected_len = expected_encoded_block_len(&request, samples)?;

    if Compression::from_code(request.ifd.compression()) != Some(Compression::Lerc) {
        let mut decoded = filters::decompress(
            request.ifd.compression(),
            request.compressed,
            request.index,
            request.jpeg_tables,
            expected_len,
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
        let color_model = request.ifd.color_model()?;
        let is_subsampled_ycbcr = is_subsampled_ycbcr_non_jpeg(request.ifd, &color_model);
        let row_bytes = if is_subsampled_ycbcr {
            decoded.len()
        } else if request.layout.bits_per_sample < 8 {
            if request.layout.planar_configuration == 1 {
                request
                    .layout
                    .packed_row_bytes_for_width(request.block_width)
            } else {
                request
                    .layout
                    .packed_sample_plane_row_bytes_for_width(request.block_width)
            }
        } else {
            request
                .block_width
                .checked_mul(samples)
                .and_then(|value| value.checked_mul(request.layout.bytes_per_sample))
                .ok_or_else(|| Error::InvalidImageLayout("block row size overflows usize".into()))?
        };
        for row in decoded.chunks_exact_mut(row_bytes) {
            filters::fix_endianness_and_predict(
                row,
                request.layout.bits_per_sample,
                samples as u16,
                request.byte_order,
                request.layout.predictor,
            )?;
        }
        if is_subsampled_ycbcr {
            let ColorModel::YCbCr { subsampling, .. } = color_model else {
                unreachable!();
            };
            decoded = expand_subsampled_ycbcr(
                &decoded,
                request.layout.bytes_per_sample,
                request.block_width,
                request.block_height,
                subsampling,
            )?;
        } else if request.layout.bits_per_sample < 8 {
            decoded = unpack_subbyte_block(
                &decoded,
                request.layout.bits_per_sample,
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

fn expected_encoded_block_len(request: &BlockDecodeRequest<'_>, samples: usize) -> Result<usize> {
    let color_model = request.ifd.color_model()?;
    if is_subsampled_ycbcr_non_jpeg(request.ifd, &color_model) {
        let ColorModel::YCbCr { subsampling, .. } = color_model else {
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
            .and_then(|values| values.checked_mul(request.layout.bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout("YCbCr block size overflows usize".into()));
    }

    let row_bytes = if request.layout.bits_per_sample < 8 {
        if request.layout.planar_configuration == 1 {
            request
                .layout
                .packed_row_bytes_for_width(request.block_width)
        } else {
            request
                .layout
                .packed_sample_plane_row_bytes_for_width(request.block_width)
        }
    } else {
        request
            .block_width
            .checked_mul(samples)
            .and_then(|value| value.checked_mul(request.layout.bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout("block row size overflows usize".into()))?
    };
    request
        .block_height
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::InvalidImageLayout("block size overflows usize".into()))
}

fn is_subsampled_ycbcr_non_jpeg(ifd: &Ifd, color_model: &ColorModel) -> bool {
    matches!(
        color_model,
        ColorModel::YCbCr {
            subsampling,
            extra_samples,
            ..
        } if *subsampling != [1, 1]
            && extra_samples.is_empty()
            && Compression::from_code(ifd.compression()) != Some(Compression::Jpeg)
    )
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
    let payload = match request
        .ifd
        .lerc_parameters()?
        .map(|params| params.additional_compression)
        .unwrap_or(LercAdditionalCompression::None)
    {
        LercAdditionalCompression::None => request.compressed.to_vec(),
        LercAdditionalCompression::Deflate => filters::decompress(
            Compression::Deflate.to_code(),
            request.compressed,
            request.index,
            None,
            expected_len,
        )?,
        LercAdditionalCompression::Zstd => filters::decompress(
            Compression::Zstd.to_code(),
            request.compressed,
            request.index,
            None,
            expected_len,
        )?,
    };

    let decoded =
        lerc_reader::decode_band_set(&payload).map_err(|error| Error::DecompressionFailed {
            index: request.index,
            reason: format!("LERC: {error}"),
        })?;
    validate_lerc_layout(
        &decoded,
        request.layout,
        request.block_width,
        request.block_height,
        request.index,
    )?;
    serialize_lerc_band_set(&decoded, request.layout, expected_len, request.index)
}

fn validate_lerc_layout(
    decoded: &DecodedBandSet,
    layout: RasterLayout,
    block_width: usize,
    block_height: usize,
    index: usize,
) -> Result<()> {
    let expected_type = expected_lerc_data_type(layout)?;
    for band in &decoded.info.bands {
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
    let band_count = decoded.info.band_count();
    let depth = decoded.info.depth().max(1) as usize;
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
    let pixel_count = decoded.info.bands[0].pixel_count().map_err(|error| {
        Error::InvalidImageLayout(format!("LERC pixel count overflow: {error}"))
    })?;
    let mut out = Vec::with_capacity(expected_len);
    let plan = SerializationPlan {
        pixel_count,
        band_count: decoded.info.band_count(),
        depth: decoded.info.depth().max(1) as usize,
        layout,
        index,
    };

    match &decoded.bands[0] {
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
        .bands
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
        let mask = decoded.band_masks.first().and_then(|mask| mask.as_deref());
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

    for pixel in 0..plan.pixel_count {
        for (band_index, values) in band_slices.iter().enumerate() {
            let valid = decoded.band_masks[band_index]
                .as_deref()
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
