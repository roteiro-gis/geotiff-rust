use crate::error::{Error, Result};
use crate::filters;
use crate::header::ByteOrder;
use crate::ifd::{Ifd, LercAdditionalCompression};
use lerc_core::{DataType, DecodedBandSet, PixelData};
use tiff_core::{Compression, RasterLayout, SampleFormat};

pub(crate) fn decode_compressed_block(
    ifd: &Ifd,
    layout: RasterLayout,
    byte_order: ByteOrder,
    compressed: &[u8],
    index: usize,
    jpeg_tables: Option<&[u8]>,
    block_width: usize,
    block_height: usize,
) -> Result<Vec<u8>> {
    let samples = if layout.planar_configuration == 1 {
        layout.samples_per_pixel
    } else {
        1
    };
    let row_bytes = block_width
        .checked_mul(samples)
        .and_then(|value| value.checked_mul(layout.bytes_per_sample))
        .ok_or_else(|| Error::InvalidImageLayout("block row size overflows usize".into()))?;
    let expected_len = block_height
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::InvalidImageLayout("block size overflows usize".into()))?;

    if Compression::from_code(ifd.compression()) != Some(Compression::Lerc) {
        let mut decoded = filters::decompress(
            ifd.compression(),
            compressed,
            index,
            jpeg_tables,
            expected_len,
        )?;
        if decoded.len() < expected_len {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "decoded block is too small: expected at least {expected_len} bytes, found {}",
                    decoded.len()
                ),
            });
        }
        if decoded.len() > expected_len {
            decoded.truncate(expected_len);
        }
        for row in decoded.chunks_exact_mut(row_bytes) {
            filters::fix_endianness_and_predict(
                row,
                layout.bits_per_sample,
                samples as u16,
                byte_order,
                layout.predictor,
            )?;
        }
        return Ok(decoded);
    }

    decode_lerc_block(
        ifd,
        layout,
        compressed,
        index,
        block_width,
        block_height,
        expected_len,
    )
}

fn decode_lerc_block(
    ifd: &Ifd,
    layout: RasterLayout,
    compressed: &[u8],
    index: usize,
    block_width: usize,
    block_height: usize,
    expected_len: usize,
) -> Result<Vec<u8>> {
    let payload = match ifd
        .lerc_parameters()?
        .map(|params| params.additional_compression)
        .unwrap_or(LercAdditionalCompression::None)
    {
        LercAdditionalCompression::None => compressed.to_vec(),
        LercAdditionalCompression::Deflate => filters::decompress(
            Compression::Deflate.to_code(),
            compressed,
            index,
            None,
            expected_len,
        )?,
        LercAdditionalCompression::Zstd => filters::decompress(
            Compression::Zstd.to_code(),
            compressed,
            index,
            None,
            expected_len,
        )?,
    };

    let decoded =
        lerc_reader::decode_band_set(&payload).map_err(|error| Error::DecompressionFailed {
            index,
            reason: format!("LERC: {error}"),
        })?;
    validate_lerc_layout(&decoded, layout, block_width, block_height, index)?;
    serialize_lerc_band_set(&decoded, layout, expected_len, index)
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
    let band_count = decoded.info.band_count();
    let depth = decoded.info.depth().max(1) as usize;
    let mut out = Vec::with_capacity(expected_len);

    match &decoded.bands[0] {
        PixelData::I8(_) => serialize_typed::<i8, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            0,
            &mut out,
            |band| match band {
                PixelData::I8(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::U8(_) => serialize_typed::<u8, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            0,
            &mut out,
            |band| match band {
                PixelData::U8(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::I16(_) => serialize_typed::<i16, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            0,
            &mut out,
            |band| match band {
                PixelData::I16(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::U16(_) => serialize_typed::<u16, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            0,
            &mut out,
            |band| match band {
                PixelData::U16(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::I32(_) => serialize_typed::<i32, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            0,
            &mut out,
            |band| match band {
                PixelData::I32(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::U32(_) => serialize_typed::<u32, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            0,
            &mut out,
            |band| match band {
                PixelData::U32(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::F32(_) => serialize_typed::<f32, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            f32::NAN,
            &mut out,
            |band| match band {
                PixelData::F32(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
        PixelData::F64(_) => serialize_typed::<f64, _>(
            decoded,
            pixel_count,
            band_count,
            depth,
            layout,
            index,
            f64::NAN,
            &mut out,
            |band| match band {
                PixelData::F64(values) => Some(values.as_slice()),
                _ => None,
            },
        )?,
    }

    if out.len() != expected_len {
        return Err(Error::DecompressionFailed {
            index,
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
    pixel_count: usize,
    band_count: usize,
    depth: usize,
    layout: RasterLayout,
    index: usize,
    invalid_fill: T,
    out: &mut Vec<u8>,
    slice_for: F,
) -> Result<()>
where
    T: NativeEndianBytes + 'a,
    F: Fn(&'a PixelData) -> Option<&'a [T]>,
{
    let expected_samples = if layout.planar_configuration == 1 {
        layout.samples_per_pixel
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

    if band_count == 1 {
        let values = band_slices[0];
        let expected_values = pixel_count
            .checked_mul(depth)
            .ok_or_else(|| Error::InvalidImageLayout("LERC sample count overflows usize".into()))?;
        if values.len() != expected_values || depth != expected_samples {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC single-band depth layout produced {} values with depth {} for {} pixels and TIFF samples_per_pixel={expected_samples}",
                    values.len(),
                    depth,
                    pixel_count
                ),
            });
        }
        let mask = decoded.band_masks.first().and_then(|mask| mask.as_deref());
        for pixel in 0..pixel_count {
            let valid = mask.map(|mask| mask[pixel] != 0).unwrap_or(true);
            let base = pixel * depth;
            for sample in &values[base..base + depth] {
                if valid {
                    sample.write_ne(out);
                } else {
                    invalid_fill.write_ne(out);
                }
            }
        }
        return Ok(());
    }

    if depth != 1 || band_count != expected_samples {
        return Err(Error::DecompressionFailed {
            index,
            reason: format!(
                "LERC band-set layout band_count={band_count} depth={depth} does not match TIFF samples_per_pixel={expected_samples}"
            ),
        });
    }

    for values in &band_slices {
        if values.len() != pixel_count {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "LERC band length {} does not match block pixel count {pixel_count}",
                    values.len()
                ),
            });
        }
    }

    for pixel in 0..pixel_count {
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
