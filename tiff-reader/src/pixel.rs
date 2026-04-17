use crate::error::{Error, Result};
use crate::ifd::Ifd;
use tiff_core::{ColorMap, ColorModel, Compression, RasterLayout};

pub(crate) fn decode_pixels(
    ifd: &Ifd,
    storage_layout: &RasterLayout,
    width: usize,
    height: usize,
    sample_bytes: &[u8],
) -> Result<(RasterLayout, Vec<u8>)> {
    let decoded_layout = ifd.decoded_raster_layout()?;
    let decoded_layout = RasterLayout {
        width,
        height,
        ..decoded_layout
    };
    let color_model = ifd.color_model()?;
    if can_passthrough(ifd, &color_model, storage_layout, &decoded_layout) {
        return Ok((decoded_layout, sample_bytes.to_vec()));
    }

    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidImageLayout("decoded pixel count overflows usize".into()))?;
    let input_channels = storage_layout.samples_per_pixel;
    let expected_input_len = pixel_count
        .checked_mul(input_channels)
        .and_then(|samples| samples.checked_mul(storage_layout.bytes_per_sample))
        .ok_or_else(|| Error::InvalidImageLayout("decoded sample buffer length overflows usize".into()))?;
    if sample_bytes.len() != expected_input_len {
        return Err(Error::InvalidImageLayout(format!(
            "decoded sample buffer length {} does not match expected {expected_input_len}",
            sample_bytes.len()
        )));
    }

    let mut out = Vec::with_capacity(
        pixel_count
            .checked_mul(decoded_layout.samples_per_pixel)
            .and_then(|samples| samples.checked_mul(decoded_layout.bytes_per_sample))
            .ok_or_else(|| Error::InvalidImageLayout("decoded output buffer length overflows usize".into()))?,
    );

    match color_model {
        ColorModel::Grayscale {
            white_is_zero,
            extra_samples,
        } => {
            if storage_layout.sample_format != 1 {
                return Err(Error::InvalidImageLayout(
                    "grayscale color decoding requires unsigned integer source samples".into(),
                ));
            }
            for pixel in 0..pixel_count {
                let mut value = read_uint_sample(sample_bytes, storage_layout, pixel, 0);
                if white_is_zero {
                    value = invert_uint_bits(value, storage_layout.bits_per_sample);
                }
                write_uint_sample(
                    &mut out,
                    scale_uint_bits(
                        value,
                        storage_layout.bits_per_sample,
                        decoded_layout.bits_per_sample,
                    ),
                    decoded_layout.bits_per_sample,
                );
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    1,
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::Palette {
            color_map,
            extra_samples,
        } => {
            for pixel in 0..pixel_count {
                let index = read_uint_sample(sample_bytes, storage_layout, pixel, 0) as usize;
                if index >= color_map.len() {
                    return Err(Error::InvalidImageLayout(format!(
                        "palette index {index} exceeds ColorMap length {}",
                        color_map.len()
                    )));
                }
                write_uint_sample(
                    &mut out,
                    palette_channel_value(&color_map, index, 0, decoded_layout.bits_per_sample),
                    decoded_layout.bits_per_sample,
                );
                write_uint_sample(
                    &mut out,
                    palette_channel_value(&color_map, index, 1, decoded_layout.bits_per_sample),
                    decoded_layout.bits_per_sample,
                );
                write_uint_sample(
                    &mut out,
                    palette_channel_value(&color_map, index, 2, decoded_layout.bits_per_sample),
                    decoded_layout.bits_per_sample,
                );
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    1,
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::Rgb { extra_samples } => {
            for pixel in 0..pixel_count {
                for channel in 0..3 {
                    let value = read_uint_sample(sample_bytes, storage_layout, pixel, channel);
                    write_uint_sample(
                        &mut out,
                        scale_uint_bits(
                            value,
                            storage_layout.bits_per_sample,
                            decoded_layout.bits_per_sample,
                        ),
                        decoded_layout.bits_per_sample,
                    );
                }
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    3,
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::TransparencyMask => {
            for pixel in 0..pixel_count {
                let value = read_uint_sample(sample_bytes, storage_layout, pixel, 0);
                write_uint_sample(
                    &mut out,
                    scale_uint_bits(
                        value,
                        storage_layout.bits_per_sample,
                        decoded_layout.bits_per_sample,
                    ),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::Cmyk { extra_samples } => {
            for pixel in 0..pixel_count {
                let c = normalize_uint_sample(
                    read_uint_sample(sample_bytes, storage_layout, pixel, 0),
                    storage_layout.bits_per_sample,
                );
                let m = normalize_uint_sample(
                    read_uint_sample(sample_bytes, storage_layout, pixel, 1),
                    storage_layout.bits_per_sample,
                );
                let y = normalize_uint_sample(
                    read_uint_sample(sample_bytes, storage_layout, pixel, 2),
                    storage_layout.bits_per_sample,
                );
                let k = normalize_uint_sample(
                    read_uint_sample(sample_bytes, storage_layout, pixel, 3),
                    storage_layout.bits_per_sample,
                );
                for value in [(1.0 - c) * (1.0 - k), (1.0 - m) * (1.0 - k), (1.0 - y) * (1.0 - k)]
                {
                    write_uint_sample(
                        &mut out,
                        denormalize_uint_sample(value, decoded_layout.bits_per_sample),
                        decoded_layout.bits_per_sample,
                    );
                }
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    4,
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::YCbCr {
            extra_samples, ..
        } => {
            if Compression::from_code(ifd.compression()) == Some(Compression::Jpeg) {
                return Ok((decoded_layout, sample_bytes.to_vec()));
            }
            let reference_black_white =
                ifd.reference_black_white()?
                    .unwrap_or_else(|| default_reference_black_white(storage_layout.bits_per_sample));
            let chroma_denominator = bit_max(storage_layout.bits_per_sample) as f64;
            for pixel in 0..pixel_count {
                let y = read_uint_sample(sample_bytes, storage_layout, pixel, 0) as f64;
                let cb = read_uint_sample(sample_bytes, storage_layout, pixel, 1) as f64;
                let cr = read_uint_sample(sample_bytes, storage_layout, pixel, 2) as f64;
                let y_norm =
                    normalize_reference_sample(y, reference_black_white[0], reference_black_white[1]);
                let cb_delta = (cb - reference_black_white[2]) / chroma_denominator;
                let cr_delta = (cr - reference_black_white[4]) / chroma_denominator;
                let r = clamp01(y_norm + 1.402 * cr_delta);
                let g = clamp01(y_norm - 0.344_136_286 * cb_delta - 0.714_136_286 * cr_delta);
                let b = clamp01(y_norm + 1.772 * cb_delta);
                for value in [r, g, b] {
                    write_uint_sample(
                        &mut out,
                        denormalize_uint_sample(value, decoded_layout.bits_per_sample),
                        decoded_layout.bits_per_sample,
                    );
                }
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    3,
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::Separated {
            color_channels,
            extra_samples,
            ..
        } => {
            for pixel in 0..pixel_count {
                for channel in 0..usize::from(color_channels) {
                    let value = read_uint_sample(sample_bytes, storage_layout, pixel, channel);
                    write_uint_sample(
                        &mut out,
                        scale_uint_bits(
                            value,
                            storage_layout.bits_per_sample,
                            decoded_layout.bits_per_sample,
                        ),
                        decoded_layout.bits_per_sample,
                    );
                }
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    usize::from(color_channels),
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
        ColorModel::CieLab { extra_samples } => {
            for pixel in 0..pixel_count {
                for channel in 0..3 {
                    let value = read_uint_sample(sample_bytes, storage_layout, pixel, channel);
                    write_uint_sample(
                        &mut out,
                        scale_uint_bits(
                            value,
                            storage_layout.bits_per_sample,
                            decoded_layout.bits_per_sample,
                        ),
                        decoded_layout.bits_per_sample,
                    );
                }
                copy_extra_samples(
                    &mut out,
                    sample_bytes,
                    storage_layout,
                    pixel,
                    3,
                    extra_samples.len(),
                    decoded_layout.bits_per_sample,
                );
            }
        }
    }

    Ok((decoded_layout, out))
}

fn can_passthrough(
    ifd: &Ifd,
    color_model: &ColorModel,
    storage_layout: &RasterLayout,
    decoded_layout: &RasterLayout,
) -> bool {
    if storage_layout.bits_per_sample < 8 {
        return false;
    }
    if storage_layout.sample_format != decoded_layout.sample_format
        || storage_layout.bits_per_sample != decoded_layout.bits_per_sample
        || storage_layout.samples_per_pixel != decoded_layout.samples_per_pixel
    {
        return false;
    }
    match color_model {
        ColorModel::Grayscale {
            white_is_zero: false,
            ..
        }
        | ColorModel::Rgb { .. }
        | ColorModel::TransparencyMask
        | ColorModel::Separated { .. }
        | ColorModel::CieLab { .. } => true,
        ColorModel::YCbCr { .. } => {
            Compression::from_code(ifd.compression()) == Some(Compression::Jpeg)
        }
        _ => false,
    }
}

fn copy_extra_samples(
    out: &mut Vec<u8>,
    sample_bytes: &[u8],
    storage_layout: &RasterLayout,
    pixel_index: usize,
    first_extra_channel: usize,
    extra_count: usize,
    decoded_bits_per_sample: u16,
) {
    for channel in first_extra_channel..first_extra_channel + extra_count {
        let value = read_uint_sample(sample_bytes, storage_layout, pixel_index, channel);
        write_uint_sample(
            out,
            scale_uint_bits(
                value,
                storage_layout.bits_per_sample,
                decoded_bits_per_sample,
            ),
            decoded_bits_per_sample,
        );
    }
}

fn read_uint_sample(
    sample_bytes: &[u8],
    layout: &RasterLayout,
    pixel_index: usize,
    channel: usize,
) -> u64 {
    let offset = (pixel_index * layout.samples_per_pixel + channel) * layout.bytes_per_sample;
    match layout.bytes_per_sample {
        1 => u64::from(sample_bytes[offset]),
        2 => u64::from(u16::from_ne_bytes(
            sample_bytes[offset..offset + 2].try_into().unwrap(),
        )),
        4 => u64::from(u32::from_ne_bytes(
            sample_bytes[offset..offset + 4].try_into().unwrap(),
        )),
        8 => u64::from_ne_bytes(sample_bytes[offset..offset + 8].try_into().unwrap()),
        _ => unreachable!(),
    }
}

fn write_uint_sample(out: &mut Vec<u8>, value: u64, bits_per_sample: u16) {
    match bits_per_sample {
        8 => out.push(value as u8),
        16 => out.extend_from_slice(&(value as u16).to_ne_bytes()),
        32 => out.extend_from_slice(&(value as u32).to_ne_bytes()),
        64 => out.extend_from_slice(&value.to_ne_bytes()),
        other => unreachable!("unsupported decoded integer bit depth {other}"),
    }
}

fn scale_uint_bits(value: u64, from_bits: u16, to_bits: u16) -> u64 {
    if from_bits == to_bits {
        return value;
    }
    let from_max = bit_max(from_bits);
    let to_max = bit_max(to_bits);
    (((value as u128 * to_max) + from_max / 2) / from_max) as u64
}

fn invert_uint_bits(value: u64, bits_per_sample: u16) -> u64 {
    bit_max(bits_per_sample) as u64 - value
}

fn normalize_uint_sample(value: u64, bits_per_sample: u16) -> f64 {
    value as f64 / bit_max(bits_per_sample) as f64
}

fn denormalize_uint_sample(value: f64, bits_per_sample: u16) -> u64 {
    (clamp01(value) * bit_max(bits_per_sample) as f64).round() as u64
}

fn bit_max(bits_per_sample: u16) -> u128 {
    (1u128 << bits_per_sample) - 1
}

fn palette_channel_value(
    color_map: &ColorMap,
    index: usize,
    channel: usize,
    decoded_bits_per_sample: u16,
) -> u64 {
    let value = match channel {
        0 => color_map.red()[index],
        1 => color_map.green()[index],
        2 => color_map.blue()[index],
        _ => unreachable!(),
    };
    scale_uint_bits(u64::from(value), 16, decoded_bits_per_sample)
}

fn default_reference_black_white(bits_per_sample: u16) -> [f64; 6] {
    let max = bit_max(bits_per_sample) as f64;
    let chroma_zero = (1u128 << bits_per_sample.saturating_sub(1)) as f64;
    [0.0, max, chroma_zero, max, chroma_zero, max]
}

fn normalize_reference_sample(value: f64, black: f64, white: f64) -> f64 {
    if (white - black).abs() < f64::EPSILON {
        0.0
    } else {
        (value - black) / (white - black)
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
