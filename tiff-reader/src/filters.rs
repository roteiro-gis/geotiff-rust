//! Compression filter pipeline for TIFF strip/tile decompression.

#[cfg(any(feature = "jpeg", feature = "zstd", feature = "webp"))]
use std::io::Cursor;
use std::io::Read;
#[cfg(feature = "jpeg")]
use std::panic::{self, AssertUnwindSafe};

use crate::error::{Error, Result};
use crate::header::ByteOrder;
use tiff_core::{Compression, Predictor};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(any(feature = "jpeg", feature = "webp")), allow(dead_code))]
pub(crate) struct ImageDecodeLayout {
    pub width: usize,
    pub height: usize,
    pub samples_per_pixel: usize,
    /// Whether the TIFF's photometric interpretation says samples are YCbCr.
    pub ycbcr: bool,
}

/// Decompress a strip or tile according to the TIFF compression scheme.
pub fn decompress(
    compression: u16,
    data: &[u8],
    index: usize,
    _jpeg_tables: Option<&[u8]>,
    decoded_len_limit: usize,
) -> Result<Vec<u8>> {
    decompress_with_layout(
        compression,
        data,
        index,
        _jpeg_tables,
        decoded_len_limit,
        None,
    )
}

pub(crate) fn decompress_with_layout(
    compression: u16,
    data: &[u8],
    index: usize,
    _jpeg_tables: Option<&[u8]>,
    decoded_len_limit: usize,
    expected_layout: Option<ImageDecodeLayout>,
) -> Result<Vec<u8>> {
    // The value is consumed only by feature-gated image codecs.
    let _ = expected_layout;
    match Compression::from_code(compression) {
        Some(Compression::None) => {
            if data.len() > decoded_len_limit {
                return Err(decoded_block_too_large(
                    index,
                    "uncompressed",
                    decoded_len_limit,
                ));
            }
            Ok(data.to_vec())
        }
        Some(Compression::Deflate | Compression::DeflateOld) => {
            decompress_deflate(data, index, decoded_len_limit)
        }
        Some(Compression::Lzw) => decompress_lzw(data, index, decoded_len_limit),
        Some(Compression::PackBits) => decompress_packbits(data, index, decoded_len_limit),
        Some(Compression::Lerc) => Err(Error::UnsupportedCompression(compression)),
        #[cfg(feature = "jpeg")]
        Some(Compression::OldJpeg) => Err(Error::UnsupportedCompression(compression)),
        #[cfg(feature = "jpeg")]
        Some(Compression::Jpeg) => decompress_jpeg(
            data,
            index,
            _jpeg_tables,
            decoded_len_limit,
            expected_layout,
        ),
        #[cfg(not(feature = "jpeg"))]
        Some(Compression::OldJpeg | Compression::Jpeg) => {
            Err(Error::UnsupportedCompression(compression))
        }
        #[cfg(feature = "zstd")]
        Some(Compression::Zstd) => decompress_zstd(data, index, decoded_len_limit),
        #[cfg(not(feature = "zstd"))]
        Some(Compression::Zstd) => Err(Error::UnsupportedCompression(compression)),
        #[cfg(feature = "webp")]
        Some(Compression::WebP) => decompress_webp(data, index, decoded_len_limit, expected_layout),
        #[cfg(not(feature = "webp"))]
        Some(Compression::WebP) => Err(Error::UnsupportedCompression(compression)),
        None => Err(Error::UnsupportedCompression(compression)),
    }
}

/// Normalize row bytes into native-endian decoded samples and reverse any TIFF predictor.
pub fn fix_endianness_and_predict(
    row: &mut [u8],
    bit_depth: u16,
    samples: u16,
    byte_order: ByteOrder,
    predictor: u16,
) -> Result<()> {
    fix_endianness_and_predict_with_scratch(
        row,
        bit_depth,
        samples,
        byte_order,
        predictor,
        &mut Vec::new(),
    )
}

/// Scratch-buffer variant used by per-row decode loops so the floating-point
/// predictor's working copy is not reallocated for every row.
pub(crate) fn fix_endianness_and_predict_with_scratch(
    row: &mut [u8],
    bit_depth: u16,
    samples: u16,
    byte_order: ByteOrder,
    predictor: u16,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    match Predictor::from_code(predictor) {
        Some(Predictor::None) => {
            fix_endianness(row, byte_order, bit_depth);
            Ok(())
        }
        Some(Predictor::Horizontal) => {
            fix_endianness(row, byte_order, bit_depth);
            reverse_horizontal_predictor(row, bit_depth, samples);
            Ok(())
        }
        Some(Predictor::FloatingPoint) => match bit_depth {
            16 | 32 | 64 => {
                scratch.clear();
                scratch.extend_from_slice(row);
                match bit_depth {
                    16 => predict_f16(scratch, row, samples),
                    32 => predict_f32(scratch, row, samples),
                    _ => predict_f64(scratch, row, samples),
                }
                Ok(())
            }
            _ => Err(Error::UnsupportedPredictor(3)),
        },
        None => Err(Error::UnsupportedPredictor(predictor)),
    }
}

fn decompress_deflate(data: &[u8], index: usize, decoded_len_limit: usize) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;

    let decoder = ZlibDecoder::new(data);
    read_bounded_to_end(decoder, index, "deflate", decoded_len_limit)
}

fn decompress_lzw(data: &[u8], index: usize, decoded_len_limit: usize) -> Result<Vec<u8>> {
    use weezl::decode::Configuration;
    use weezl::{BitOrder, LzwStatus};

    let mut decoder = Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
        .with_yield_on_full_buffer(true)
        .build();
    let probe_limit = decoded_len_probe_limit(index, "LZW", decoded_len_limit)?;
    let mut out = Vec::with_capacity(decoded_len_limit.min(8192));
    let mut input_offset = 0usize;
    let mut scratch = [0u8; 8192];

    loop {
        let remaining = probe_limit.saturating_sub(out.len());
        if remaining == 0 {
            return Err(decoded_block_too_large(index, "LZW", decoded_len_limit));
        }

        let output_len = remaining.min(scratch.len());
        let result = decoder.decode_bytes(&data[input_offset..], &mut scratch[..output_len]);
        input_offset += result.consumed_in;
        out.extend_from_slice(&scratch[..result.consumed_out]);
        if out.len() > decoded_len_limit {
            return Err(decoded_block_too_large(index, "LZW", decoded_len_limit));
        }

        match result.status {
            Err(e) => {
                return Err(Error::DecompressionFailed {
                    index,
                    reason: format!("LZW: {e}"),
                })
            }
            Ok(LzwStatus::Done) => return Ok(out),
            Ok(LzwStatus::Ok) => {
                if result.consumed_in == 0 && result.consumed_out == 0 {
                    return Err(Error::DecompressionFailed {
                        index,
                        reason: "LZW: decoder made no progress".into(),
                    });
                }
            }
            Ok(LzwStatus::NoProgress) => {
                if result.consumed_out == output_len {
                    continue;
                }
                return Err(Error::DecompressionFailed {
                    index,
                    reason: "LZW: stream ended before end marker".into(),
                });
            }
        }
    }
}

fn decompress_packbits(data: &[u8], index: usize, decoded_len_limit: usize) -> Result<Vec<u8>> {
    let probe_limit = decoded_len_probe_limit(index, "PackBits", decoded_len_limit)?;
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while cursor < data.len() {
        let header = data[cursor] as i8;
        cursor += 1;

        if header >= 0 {
            let count = header as usize + 1;
            let end = cursor + count;
            if end > data.len() {
                return Err(Error::DecompressionFailed {
                    index,
                    reason: "PackBits literal run is truncated".into(),
                });
            }
            append_bounded_bytes(
                &mut out,
                &data[cursor..end],
                index,
                "PackBits",
                decoded_len_limit,
                probe_limit,
            )?;
            cursor = end;
        } else if header != -128 {
            if cursor >= data.len() {
                return Err(Error::DecompressionFailed {
                    index,
                    reason: "PackBits repeat run is truncated".into(),
                });
            }
            let count = (1i16 - header as i16) as usize;
            let byte = data[cursor];
            cursor += 1;
            append_bounded_repeat(
                &mut out,
                byte,
                count,
                index,
                "PackBits",
                decoded_len_limit,
                probe_limit,
            )?;
        }
    }

    Ok(out)
}

#[cfg(feature = "jpeg")]
fn decompress_jpeg(
    data: &[u8],
    index: usize,
    jpeg_tables: Option<&[u8]>,
    decoded_len_limit: usize,
    expected_layout: Option<ImageDecodeLayout>,
) -> Result<Vec<u8>> {
    let stream = merge_jpeg_stream(jpeg_tables, data);
    // Component IDs and Adobe markers are hints for standalone JPEGs. In a TIFF
    // the color space comes from the photometric tag alone, and streams from
    // encoders other than libjpeg often contradict it. The decoder has no raw
    // interleaved mode, so `RGB` is used to interleave three components
    // unchanged, and its `CMYK` inversion is undone below.
    let color_transform = expected_layout.and_then(|layout| match layout.samples_per_pixel {
        3 if layout.ycbcr => Some(jpeg_decoder::ColorTransform::YCbCr),
        3 => Some(jpeg_decoder::ColorTransform::RGB),
        4 => Some(jpeg_decoder::ColorTransform::CMYK),
        _ => None,
    });
    panic::catch_unwind(AssertUnwindSafe(|| {
        let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(stream));
        decoder.set_max_decoding_buffer_size(decoded_len_limit);
        if let Some(color_transform) = color_transform {
            decoder.set_color_transform(color_transform);
        }
        decoder.read_info()?;
        validate_jpeg_metadata(&decoder, decoded_len_limit, expected_layout)?;
        let mut decoded = decoder.decode()?;
        if matches!(color_transform, Some(jpeg_decoder::ColorTransform::CMYK)) {
            for value in &mut decoded {
                *value = !*value;
            }
        }
        Ok::<_, jpeg_decoder::Error>(decoded)
    }))
    .map_err(|payload| Error::DecompressionFailed {
        index,
        reason: format!(
            "JPEG decoder panicked: {}",
            panic_payload_message(payload.as_ref())
        ),
    })?
    .map_err(|e| Error::DecompressionFailed {
        index,
        reason: format!("JPEG: {e}"),
    })
}

#[cfg(feature = "jpeg")]
fn validate_jpeg_metadata<R: std::io::Read>(
    decoder: &jpeg_decoder::Decoder<R>,
    decoded_len_limit: usize,
    expected_layout: Option<ImageDecodeLayout>,
) -> std::result::Result<(), jpeg_decoder::Error> {
    let info = decoder.info().ok_or_else(|| {
        jpeg_decoder::Error::Format("JPEG metadata missing after read_info".into())
    })?;
    let decoded_len = usize::from(info.width)
        .checked_mul(usize::from(info.height))
        .and_then(|pixels| pixels.checked_mul(info.pixel_format.pixel_bytes()))
        .ok_or_else(|| jpeg_decoder::Error::Format("JPEG decoded size overflow".into()))?;
    if decoded_len > decoded_len_limit {
        return Err(jpeg_decoder::Error::Format(format!(
            "JPEG decoded size {decoded_len} exceeds TIFF block budget {decoded_len_limit}"
        )));
    }
    if let Some(expected) = expected_layout {
        if usize::from(info.width) != expected.width || usize::from(info.height) != expected.height
        {
            return Err(jpeg_decoder::Error::Format(format!(
                "JPEG dimensions {}x{} do not match TIFF block {}x{}",
                info.width, info.height, expected.width, expected.height
            )));
        }
        if info.pixel_format.pixel_bytes() != expected.samples_per_pixel {
            return Err(jpeg_decoder::Error::Format(format!(
                "JPEG channel count {} does not match TIFF block channel count {}",
                info.pixel_format.pixel_bytes(),
                expected.samples_per_pixel
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "zstd")]
fn decompress_zstd(data: &[u8], index: usize, decoded_len_limit: usize) -> Result<Vec<u8>> {
    let decoder = ruzstd::decoding::StreamingDecoder::new(Cursor::new(data)).map_err(|error| {
        Error::DecompressionFailed {
            index,
            reason: format!("ZSTD: {error}"),
        }
    })?;
    read_bounded_to_end(decoder, index, "ZSTD", decoded_len_limit)
}

#[cfg(feature = "webp")]
fn decompress_webp(
    data: &[u8],
    index: usize,
    decoded_len_limit: usize,
    expected_layout: Option<ImageDecodeLayout>,
) -> Result<Vec<u8>> {
    let mut decoder = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        image_webp::WebPDecoder::new(Cursor::new(data))
    }))
    .map_err(|payload| Error::DecompressionFailed {
        index,
        reason: format!(
            "WebP decoder panicked: {}",
            webp_panic_payload_message(payload.as_ref())
        ),
    })?
    .map_err(|error| Error::DecompressionFailed {
        index,
        reason: format!("WebP: {error}"),
    })?;

    if decoder.is_animated() {
        return Err(Error::DecompressionFailed {
            index,
            reason: "WebP: animated payloads are not valid TIFF blocks".into(),
        });
    }
    if let Some(expected) = expected_layout {
        let (actual_width, actual_height) = decoder.dimensions();
        if actual_width as usize != expected.width || actual_height as usize != expected.height {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "WebP dimensions {actual_width}x{actual_height} do not match TIFF block {}x{}",
                    expected.width, expected.height
                ),
            });
        }
        let actual_samples = if decoder.has_alpha() { 4 } else { 3 };
        if actual_samples != expected.samples_per_pixel {
            return Err(Error::DecompressionFailed {
                index,
                reason: format!(
                    "WebP channel count {actual_samples} does not match TIFF block channel count {}",
                    expected.samples_per_pixel
                ),
            });
        }
    }

    let output_len = decoder
        .output_buffer_size()
        .ok_or_else(|| Error::DecompressionFailed {
            index,
            reason: "WebP output size overflows usize".into(),
        })?;
    if output_len > decoded_len_limit {
        return Err(decoded_block_too_large(index, "WebP", decoded_len_limit));
    }

    let mut out = vec![0u8; output_len];
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decoder.read_image(&mut out)
    }))
    .map_err(|payload| Error::DecompressionFailed {
        index,
        reason: format!(
            "WebP decoder panicked: {}",
            webp_panic_payload_message(payload.as_ref())
        ),
    })?
    .map_err(|error| Error::DecompressionFailed {
        index,
        reason: format!("WebP: {error}"),
    })?;
    Ok(out)
}

#[cfg(feature = "webp")]
fn webp_panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".into()
    }
}

fn read_bounded_to_end<R: Read>(
    reader: R,
    index: usize,
    codec: &'static str,
    decoded_len_limit: usize,
) -> Result<Vec<u8>> {
    let probe_limit = decoded_len_probe_limit(index, codec, decoded_len_limit)?;
    let mut reader = reader.take(probe_limit as u64);
    let mut out = Vec::with_capacity(decoded_len_limit.min(64 * 1024));
    let mut scratch = vec![0u8; probe_limit.min(64 * 1024)];

    loop {
        let remaining = probe_limit.saturating_sub(out.len());
        if remaining == 0 {
            break;
        }
        let read_len = remaining.min(scratch.len());
        let bytes_read =
            reader
                .read(&mut scratch[..read_len])
                .map_err(|e| Error::DecompressionFailed {
                    index,
                    reason: format!("{codec}: {e}"),
                })?;
        if bytes_read == 0 {
            break;
        }
        out.extend_from_slice(&scratch[..bytes_read]);
        if out.len() > decoded_len_limit {
            return Err(decoded_block_too_large(index, codec, decoded_len_limit));
        }
    }

    Ok(out)
}

fn decoded_len_probe_limit(
    index: usize,
    codec: &'static str,
    decoded_len_limit: usize,
) -> Result<usize> {
    decoded_len_limit
        .checked_add(1)
        .ok_or_else(|| Error::DecompressionFailed {
            index,
            reason: format!("{codec}: TIFF block budget is too large to probe safely"),
        })
}

fn decoded_block_too_large(index: usize, codec: &'static str, decoded_len_limit: usize) -> Error {
    Error::DecompressionFailed {
        index,
        reason: format!("{codec}: decoded block exceeds TIFF block budget {decoded_len_limit}"),
    }
}

fn append_bounded_bytes(
    out: &mut Vec<u8>,
    bytes: &[u8],
    index: usize,
    codec: &'static str,
    decoded_len_limit: usize,
    probe_limit: usize,
) -> Result<()> {
    let remaining = probe_limit.saturating_sub(out.len());
    let copy_len = bytes.len().min(remaining);
    out.extend_from_slice(&bytes[..copy_len]);
    if copy_len < bytes.len() || out.len() > decoded_len_limit {
        return Err(decoded_block_too_large(index, codec, decoded_len_limit));
    }
    Ok(())
}

fn append_bounded_repeat(
    out: &mut Vec<u8>,
    byte: u8,
    count: usize,
    index: usize,
    codec: &'static str,
    decoded_len_limit: usize,
    probe_limit: usize,
) -> Result<()> {
    let remaining = probe_limit.saturating_sub(out.len());
    let copy_len = count.min(remaining);
    out.resize(out.len() + copy_len, byte);
    if copy_len < count || out.len() > decoded_len_limit {
        return Err(decoded_block_too_large(index, codec, decoded_len_limit));
    }
    Ok(())
}

#[cfg(feature = "jpeg")]
fn merge_jpeg_stream(jpeg_tables: Option<&[u8]>, scan_data: &[u8]) -> Vec<u8> {
    if jpeg_tables.is_none() {
        return scan_data.to_vec();
    }

    let tables = jpeg_tables.unwrap_or_default();
    let table_body = match tables.strip_suffix(&[0xff, 0xd9]) {
        Some(without_eoi) => without_eoi,
        None => tables,
    };
    let scan_body = match scan_data.strip_prefix(&[0xff, 0xd8]) {
        Some(without_soi) => without_soi,
        None => scan_data,
    };

    let mut merged = Vec::with_capacity(table_body.len() + scan_body.len() + 2);
    if table_body.starts_with(&[0xff, 0xd8]) {
        merged.extend_from_slice(table_body);
    } else {
        merged.extend_from_slice(&[0xff, 0xd8]);
        merged.extend_from_slice(table_body);
    }
    merged.extend_from_slice(scan_body);
    if !merged.ends_with(&[0xff, 0xd9]) {
        merged.extend_from_slice(&[0xff, 0xd9]);
    }
    merged
}

#[cfg(feature = "jpeg")]
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".into()
    }
}

fn fix_endianness(buf: &mut [u8], byte_order: ByteOrder, bit_depth: u16) {
    let host_is_little_endian = cfg!(target_endian = "little");
    let data_is_little_endian = matches!(byte_order, ByteOrder::LittleEndian);
    if host_is_little_endian == data_is_little_endian {
        return;
    }

    let chunk = match bit_depth {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    };
    if chunk == 1 {
        return;
    }

    for value in buf.chunks_exact_mut(chunk) {
        value.reverse();
    }
}

fn reverse_horizontal_predictor(buf: &mut [u8], bit_depth: u16, samples: u16) {
    let bytes_per_value = match bit_depth {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    };
    let lookback = usize::from(samples) * bytes_per_value;

    match bytes_per_value {
        1 => {
            for index in lookback..buf.len() {
                buf[index] = buf[index].wrapping_add(buf[index - lookback]);
            }
        }
        2 => {
            for index in (lookback..buf.len()).step_by(2) {
                let current = u16::from_ne_bytes(buf[index..index + 2].try_into().unwrap());
                let previous = u16::from_ne_bytes(
                    buf[index - lookback..index - lookback + 2]
                        .try_into()
                        .unwrap(),
                );
                buf[index..index + 2]
                    .copy_from_slice(&current.wrapping_add(previous).to_ne_bytes());
            }
        }
        4 => {
            for index in (lookback..buf.len()).step_by(4) {
                let current = u32::from_ne_bytes(buf[index..index + 4].try_into().unwrap());
                let previous = u32::from_ne_bytes(
                    buf[index - lookback..index - lookback + 4]
                        .try_into()
                        .unwrap(),
                );
                buf[index..index + 4]
                    .copy_from_slice(&current.wrapping_add(previous).to_ne_bytes());
            }
        }
        _ => {
            for index in (lookback..buf.len()).step_by(8) {
                let current = u64::from_ne_bytes(buf[index..index + 8].try_into().unwrap());
                let previous = u64::from_ne_bytes(
                    buf[index - lookback..index - lookback + 8]
                        .try_into()
                        .unwrap(),
                );
                buf[index..index + 8]
                    .copy_from_slice(&current.wrapping_add(previous).to_ne_bytes());
            }
        }
    }
}

fn predict_f16(input: &mut [u8], output: &mut [u8], samples: u16) {
    let samples = usize::from(samples);
    for i in samples..input.len() {
        input[i] = input[i].wrapping_add(input[i - samples]);
    }
    for (i, chunk) in output.chunks_mut(2).enumerate() {
        chunk.copy_from_slice(&u16::to_ne_bytes(u16::from_be_bytes([
            input[i],
            input[input.len() / 2 + i],
        ])));
    }
}

fn predict_f32(input: &mut [u8], output: &mut [u8], samples: u16) {
    let samples = usize::from(samples);
    for i in samples..input.len() {
        input[i] = input[i].wrapping_add(input[i - samples]);
    }
    for (i, chunk) in output.chunks_mut(4).enumerate() {
        chunk.copy_from_slice(&u32::to_ne_bytes(u32::from_be_bytes([
            input[i],
            input[input.len() / 4 + i],
            input[input.len() / 2 + i],
            input[input.len() / 4 * 3 + i],
        ])));
    }
}

fn predict_f64(input: &mut [u8], output: &mut [u8], samples: u16) {
    let samples = usize::from(samples);
    for i in samples..input.len() {
        input[i] = input[i].wrapping_add(input[i - samples]);
    }
    for (i, chunk) in output.chunks_mut(8).enumerate() {
        chunk.copy_from_slice(&u64::to_ne_bytes(u64::from_be_bytes([
            input[i],
            input[input.len() / 8 + i],
            input[input.len() / 8 * 2 + i],
            input[input.len() / 8 * 3 + i],
            input[input.len() / 8 * 4 + i],
            input[input.len() / 8 * 5 + i],
            input[input.len() / 8 * 6 + i],
            input[input.len() / 8 * 7 + i],
        ])));
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    #[cfg(not(feature = "jpeg"))]
    use super::decompress;
    #[cfg(feature = "jpeg")]
    use super::{decompress, merge_jpeg_stream};
    use super::{decompress_lzw, decompress_packbits, fix_endianness_and_predict};
    use crate::header::ByteOrder;
    use std::io::Write;
    use tiff_core::Compression;

    #[test]
    fn horizontal_predictor_restores_u16_rows() {
        let mut row = vec![1, 0, 1, 0, 2, 0];
        fix_endianness_and_predict(&mut row, 16, 1, ByteOrder::LittleEndian, 2).unwrap();
        assert_eq!(row, vec![1, 0, 2, 0, 4, 0]);
    }

    #[test]
    fn packbits_decoder_rejects_truncated_repeat_run() {
        let err = decompress_packbits(&[0xff], 0, 1).unwrap_err();
        assert!(err.to_string().contains("PackBits"));
    }

    #[test]
    fn deflate_decoder_rejects_blocks_that_exceed_budget() {
        let payload = [0x2a; 128];
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&payload).unwrap();
        let compressed = encoder.finish().unwrap();

        let err =
            decompress(Compression::Deflate.to_code(), &compressed, 0, None, 127).unwrap_err();
        assert!(err.to_string().contains("block budget"));
    }

    #[test]
    fn lzw_decoder_rejects_blocks_that_exceed_budget() {
        let payload = [0x2a; 128];
        let compressed = weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
            .encode(&payload)
            .unwrap();

        let err = decompress_lzw(&compressed, 0, 127).unwrap_err();
        assert!(err.to_string().contains("block budget"));
    }

    #[test]
    fn packbits_decoder_rejects_blocks_that_exceed_budget() {
        let err = decompress_packbits(&[0x81, 0x2a], 0, 127).unwrap_err();
        assert!(err.to_string().contains("block budget"));
    }

    #[test]
    fn uncompressed_decoder_rejects_blocks_that_exceed_budget() {
        let err =
            decompress(Compression::None.to_code(), &[1, 2, 3, 4, 5], 0, None, 4).unwrap_err();
        assert!(err.to_string().contains("block budget"));
    }

    #[test]
    fn lzw_real_cog_tile_requires_repeated_trailer_bytes() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../testdata/interoperability/gdal/gcore/data/cog/byte_little_endian_golden.tif");
        let bytes = std::fs::read(fixture).unwrap();

        let without_trailer = &bytes[570..570 + 1223];
        let with_trailer = &bytes[570..570 + 1227];

        assert!(decompress_lzw(without_trailer, 0, 1_000_000).is_ok());
        assert!(decompress_lzw(with_trailer, 0, 1_000_000).is_ok());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_decoder_rejects_blocks_that_exceed_budget() {
        let payload = [0x2a; 128];
        let compressed = ruzstd::encoding::compress_to_vec(
            &payload[..],
            ruzstd::encoding::CompressionLevel::Fastest,
        );

        let err = decompress(Compression::Zstd.to_code(), &compressed, 0, None, 127).unwrap_err();
        assert!(err.to_string().contains("block budget"));
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn merges_jpeg_tables_with_abbreviated_scan() {
        let merged = merge_jpeg_stream(
            Some(&[0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0xff, 0xd9]),
            &[0xff, 0xda, 0x00, 0x08, 0x00],
        );
        assert_eq!(&merged[..6], &[0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43]);
        assert!(merged.ends_with(&[0xff, 0xd9]));
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn jpeg_decoder_rejects_frame_sizes_that_exceed_tiff_budget() {
        let mut jpeg = vec![
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x14, 0x00, 0x14, 0x01, 0x01, 0x11,
            0x00, 0xff, 0xc4, 0x00, 0x17, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0x04, 0x06, 0xff, 0xc4,
            0x00, 0x2a, 0x10, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x05, 0x05, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x11, 0x03, 0x04, 0x00, 0x18, 0x31, 0x41,
            0x13, 0x21, 0x51, 0x71, 0x05, 0x22, 0x61, 0x91, 0xb1, 0x14, 0x42, 0x62, 0xc1, 0xf0,
            0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0x75, 0xc5, 0xb7, 0xd2,
            0x31, 0x4a, 0x75, 0x51, 0xe0, 0x65, 0xf2, 0x19, 0xd8, 0x8d, 0x7d, 0xfe, 0x71, 0x19,
            0x2b, 0x94, 0x54, 0x2c, 0x33, 0x38, 0x20, 0x2f, 0x7d, 0xf5, 0xd2, 0x40, 0x18, 0x6b,
            0xdc, 0x3d, 0xa0, 0x44, 0x15, 0xc9, 0x2c, 0xa1, 0xc8, 0x5c, 0xa4, 0x2c, 0xed, 0xcc,
            0x74, 0x83, 0xcb, 0xaf, 0x59, 0xc2, 0xaf, 0x0f, 0x02, 0xb3, 0x2e, 0x57, 0xfc, 0x79,
            0x15, 0x9f, 0x58, 0xee, 0x3f, 0x7b, 0xe0, 0x59, 0x95, 0x84, 0x26, 0x56, 0xac, 0xc2,
            0x62, 0xa0, 0x8c, 0xa4, 0x91, 0xc9, 0x44, 0xed, 0xa4, 0x9e, 0x9a, 0x08, 0xc1, 0x8a,
            0x54, 0x9d, 0x41, 0xe3, 0xa4, 0xe8, 0x65, 0x01, 0xe7, 0xdc, 0xff, 0x00, 0x6d, 0x8d,
            0x2f, 0x89, 0x5b, 0x50, 0xbe, 0xb9, 0x4a, 0x0d, 0x4c, 0x53, 0x51, 0x01, 0x8a, 0x31,
            0x9a, 0x92, 0x22, 0x5a, 0x49, 0xe7, 0xda, 0x37, 0xeb, 0x8c, 0xc5, 0xc7, 0x0a, 0xd5,
            0x87, 0x0a, 0x85, 0x30, 0xc7, 0xee, 0x69, 0x27, 0x40, 0x77, 0x3e, 0xbf, 0x18, 0x99,
            0xae, 0x1c, 0xb6, 0xc0, 0x0d, 0x02, 0xf9, 0x47, 0xb0, 0x81, 0x8f, 0xff, 0xd9,
        ];
        jpeg[7] = 0x9b;
        jpeg[8] = 0x43;
        jpeg[9] = 0xee;
        jpeg[10] = 0x23;

        let error = decompress(Compression::Jpeg.to_code(), &jpeg, 0, None, 20 * 20).unwrap_err();
        assert!(error.to_string().contains("block budget"));
    }
}
