//! Compression filter pipeline for TIFF strip/tile decompression.

use crate::error::{Error, Result};
use crate::header::ByteOrder;

/// Decompress a strip or tile according to the TIFF compression tag.
pub fn decompress(compression: u16, data: &[u8], index: usize) -> Result<Vec<u8>> {
    match compression {
        1 => Ok(data.to_vec()),
        8 | 32946 => decompress_deflate(data, index),
        5 => decompress_lzw(data, index),
        32773 => decompress_packbits(data, index),
        #[cfg(feature = "jpeg")]
        6 | 7 => decompress_jpeg(data, index),
        #[cfg(feature = "zstd")]
        50000 => decompress_zstd(data, index),
        _ => Err(Error::UnsupportedCompression(compression)),
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
    match predictor {
        1 => {
            fix_endianness(row, byte_order, bit_depth);
            Ok(())
        }
        2 => {
            fix_endianness(row, byte_order, bit_depth);
            reverse_horizontal_predictor(row, bit_depth, samples);
            Ok(())
        }
        3 => match bit_depth {
            16 => {
                let mut encoded = row.to_vec();
                predict_f16(&mut encoded, row, samples);
                Ok(())
            }
            32 => {
                let mut encoded = row.to_vec();
                predict_f32(&mut encoded, row, samples);
                Ok(())
            }
            64 => {
                let mut encoded = row.to_vec();
                predict_f64(&mut encoded, row, samples);
                Ok(())
            }
            _ => Err(Error::UnsupportedPredictor(3)),
        },
        _ => Err(Error::UnsupportedPredictor(predictor)),
    }
}

fn decompress_deflate(data: &[u8], index: usize) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::DecompressionFailed {
            index,
            reason: format!("deflate: {e}"),
        })?;
    Ok(out)
}

fn decompress_lzw(data: &[u8], index: usize) -> Result<Vec<u8>> {
    use weezl::decode::Decoder;
    use weezl::BitOrder;

    let mut decoder = Decoder::new(BitOrder::Msb, 8);
    decoder.decode(data).map_err(|e| Error::DecompressionFailed {
        index,
        reason: format!("LZW: {e}"),
    })
}

fn decompress_packbits(data: &[u8], index: usize) -> Result<Vec<u8>> {
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
            out.extend_from_slice(&data[cursor..end]);
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
            out.resize(out.len() + count, byte);
        }
    }

    Ok(out)
}

#[cfg(feature = "jpeg")]
fn decompress_jpeg(data: &[u8], index: usize) -> Result<Vec<u8>> {
    use jpeg_decoder::Decoder;

    let mut decoder = Decoder::new(data);
    decoder
        .decode()
        .map_err(|e| Error::DecompressionFailed {
            index,
            reason: format!("JPEG: {e}"),
        })
}

#[cfg(feature = "zstd")]
fn decompress_zstd(data: &[u8], index: usize) -> Result<Vec<u8>> {
    zstd::bulk::decompress(data, 64 * 1024 * 1024).map_err(|e| Error::DecompressionFailed {
        index,
        reason: format!("ZSTD: {e}"),
    })
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
                let previous = u16::from_ne_bytes(buf[index - lookback..index - lookback + 2].try_into().unwrap());
                buf[index..index + 2].copy_from_slice(&current.wrapping_add(previous).to_ne_bytes());
            }
        }
        4 => {
            for index in (lookback..buf.len()).step_by(4) {
                let current = u32::from_ne_bytes(buf[index..index + 4].try_into().unwrap());
                let previous = u32::from_ne_bytes(buf[index - lookback..index - lookback + 4].try_into().unwrap());
                buf[index..index + 4].copy_from_slice(&current.wrapping_add(previous).to_ne_bytes());
            }
        }
        _ => {
            for index in (lookback..buf.len()).step_by(8) {
                let current = u64::from_ne_bytes(buf[index..index + 8].try_into().unwrap());
                let previous = u64::from_ne_bytes(buf[index - lookback..index - lookback + 8].try_into().unwrap());
                buf[index..index + 8].copy_from_slice(&current.wrapping_add(previous).to_ne_bytes());
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
    use super::{decompress_packbits, fix_endianness_and_predict};
    use crate::header::ByteOrder;

    #[test]
    fn horizontal_predictor_restores_u16_rows() {
        let mut row = vec![1, 0, 1, 0, 2, 0];
        fix_endianness_and_predict(&mut row, 16, 1, ByteOrder::LittleEndian, 2).unwrap();
        assert_eq!(row, vec![1, 0, 2, 0, 4, 0]);
    }

    #[test]
    fn packbits_decoder_rejects_truncated_repeat_run() {
        let err = decompress_packbits(&[0xff], 0).unwrap_err();
        assert!(err.to_string().contains("PackBits"));
    }
}
