//! Compression pipeline: forward predictor + compress.
//!
//! Standard codecs (LZW, Deflate, Zstd) follow: encode bytes → predictor → compress.
//! LERC operates directly on typed samples via [`compress_block_lerc`], bypassing
//! the byte-order encoding and predictor stages.
//!
//! This is the inverse of `tiff-reader/src/filters.rs`.

use crate::builder::{JpegOptions, LercOptions};
use crate::error::{Error, Result};
use tiff_core::{ByteOrder, Compression, Predictor, LERC_VERSION_2_4};

use crate::sample::TiffWriteSample;

/// Encoding parameters for a single TIFF strip or tile block.
#[derive(Debug, Clone, Copy)]
pub struct BlockEncodingOptions<'a> {
    pub byte_order: ByteOrder,
    pub compression: Compression,
    pub predictor: Predictor,
    pub samples_per_pixel: u16,
    pub row_width_pixels: usize,
    pub jpeg_options: Option<&'a JpegOptions>,
    /// Chroma subsampling for interleaved YCbCr JPEG blocks.
    pub jpeg_sampling: Option<[u16; 2]>,
    /// Deflate level (0-9) for `Deflate`/`DeflateOld`; `None` uses the codec default.
    pub deflate_level: Option<u32>,
}

/// Full compression pipeline: native samples → file-order bytes → predictor → compress.
pub fn compress_block<T: TiffWriteSample>(
    samples: &[T],
    options: BlockEncodingOptions<'_>,
    index: usize,
) -> Result<Vec<u8>> {
    let BlockEncodingOptions {
        byte_order,
        compression,
        predictor,
        samples_per_pixel,
        row_width_pixels,
        jpeg_options,
        jpeg_sampling,
        deflate_level,
    } = options;

    validate_deflate_level(compression, deflate_level, index)?;
    if samples_per_pixel == 0 || row_width_pixels == 0 || T::BYTES_PER_SAMPLE == 0 {
        return Err(Error::CompressionFailed {
            index,
            reason:
                "block row width, samples per pixel, and sample byte width must be greater than zero"
                    .into(),
        });
    }
    match predictor {
        Predictor::Horizontal if T::SAMPLE_FORMAT == 3 => {
            return Err(Error::CompressionFailed {
                index,
                reason: "horizontal predictor requires an integer sample type".into(),
            });
        }
        Predictor::FloatingPoint
            if T::SAMPLE_FORMAT != 3 || !matches!(T::BITS_PER_SAMPLE, 16 | 32 | 64) =>
        {
            return Err(Error::CompressionFailed {
                index,
                reason: "floating-point predictor requires a 16-, 32-, or 64-bit float sample type"
                    .into(),
            });
        }
        _ => {}
    }

    if matches!(compression, Compression::Jpeg) {
        return compress_block_jpeg(
            samples,
            samples_per_pixel,
            row_width_pixels,
            jpeg_options.copied().unwrap_or_default(),
            jpeg_sampling,
            index,
        );
    }

    let mut encoded = T::encode_slice(samples, byte_order);
    let expected_encoded_len = samples
        .len()
        .checked_mul(T::BYTES_PER_SAMPLE)
        .ok_or_else(|| Error::CompressionFailed {
            index,
            reason: "encoded block byte length overflows usize".into(),
        })?;
    if encoded.len() != expected_encoded_len {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "sample encoder returned {} bytes, expected {expected_encoded_len}",
                encoded.len()
            ),
        });
    }
    let row_bytes = row_width_pixels
        .checked_mul(T::BYTES_PER_SAMPLE)
        .and_then(|bytes| bytes.checked_mul(usize::from(samples_per_pixel)))
        .ok_or_else(|| Error::CompressionFailed {
            index,
            reason: "block row byte length overflows usize".into(),
        })?;
    if encoded.len() % row_bytes != 0 {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "encoded block byte length {} is not divisible by row byte length {row_bytes}",
                encoded.len()
            ),
        });
    }
    for row in encoded.chunks_exact_mut(row_bytes) {
        apply_forward_predictor(
            row,
            predictor,
            T::BITS_PER_SAMPLE,
            samples_per_pixel,
            byte_order,
        )?;
    }
    compress_with_level(&encoded, compression, deflate_level, index)
}

#[cfg(feature = "jpeg")]
fn compress_block_jpeg<T: TiffWriteSample>(
    samples: &[T],
    samples_per_pixel: u16,
    row_width_pixels: usize,
    options: JpegOptions,
    sampling: Option<[u16; 2]>,
    index: usize,
) -> Result<Vec<u8>> {
    if T::BITS_PER_SAMPLE != 8 || T::SAMPLE_FORMAT != 1 {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "JPEG write requires 8-bit unsigned samples, got sample_format={} bits_per_sample={}",
                T::SAMPLE_FORMAT,
                T::BITS_PER_SAMPLE
            ),
        });
    }
    let samples_per_pixel = usize::from(samples_per_pixel);
    if !matches!(samples_per_pixel, 1 | 3) {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "JPEG write supports 1 or 3 samples per block, got {samples_per_pixel}"
            ),
        });
    }
    let pixels_per_row = row_width_pixels
        .checked_mul(samples_per_pixel)
        .ok_or_else(|| Error::CompressionFailed {
            index,
            reason: "JPEG row size overflows usize".into(),
        })?;
    if pixels_per_row == 0 {
        return Ok(Vec::new());
    }
    if !samples.len().is_multiple_of(pixels_per_row) {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "JPEG block sample count {} is not divisible by row size {}",
                samples.len(),
                pixels_per_row
            ),
        });
    }
    let height = samples.len() / pixels_per_row;
    let bytes = T::encode_slice(samples, ByteOrder::LittleEndian);
    compress_jpeg(
        &bytes,
        row_width_pixels,
        height,
        samples_per_pixel,
        options,
        sampling,
        index,
    )
}

#[cfg(not(feature = "jpeg"))]
fn compress_block_jpeg<T: TiffWriteSample>(
    _samples: &[T],
    _samples_per_pixel: u16,
    _row_width_pixels: usize,
    _options: JpegOptions,
    _sampling: Option<[u16; 2]>,
    index: usize,
) -> Result<Vec<u8>> {
    Err(Error::CompressionFailed {
        index,
        reason: "JPEG compression requires the 'jpeg' feature".into(),
    })
}

/// Compress raw bytes using the specified compression scheme.
///
/// LERC compression operates on typed samples, not raw bytes. Use
/// [`compress_block_lerc`] for LERC encoding.
pub fn compress(data: &[u8], compression: Compression, index: usize) -> Result<Vec<u8>> {
    compress_with_level(data, compression, None, index)
}

/// [`compress`] with an explicit Deflate level (0-9) for Deflate blocks.
pub fn compress_with_level(
    data: &[u8],
    compression: Compression,
    deflate_level: Option<u32>,
    index: usize,
) -> Result<Vec<u8>> {
    validate_deflate_level(compression, deflate_level, index)?;
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Lzw => compress_lzw(data, index),
        Compression::Deflate | Compression::DeflateOld => {
            compress_deflate_with_level(data, deflate_level, index)
        }
        #[cfg(feature = "jpeg")]
        Compression::Jpeg => Err(Error::CompressionFailed {
            index,
            reason: "JPEG operates on 8-bit sample blocks; use compress_block()".into(),
        }),
        #[cfg(not(feature = "jpeg"))]
        Compression::Jpeg => Err(Error::CompressionFailed {
            index,
            reason: "JPEG compression requires the 'jpeg' feature".into(),
        }),
        #[cfg(feature = "zstd")]
        Compression::Zstd => compress_zstd(data, index),
        Compression::Lerc => Err(Error::CompressionFailed {
            index,
            reason: "LERC operates on typed samples; use compress_block_lerc() instead".into(),
        }),
        other => Err(Error::CompressionFailed {
            index,
            reason: format!("compression {:?} is not supported for writing", other),
        }),
    }
}

fn validate_deflate_level(
    compression: Compression,
    deflate_level: Option<u32>,
    index: usize,
) -> Result<()> {
    let Some(level) = deflate_level else {
        return Ok(());
    };
    if level > 9 {
        return Err(Error::CompressionFailed {
            index,
            reason: format!("Deflate compression level must be 0-9, got {level}"),
        });
    }
    if !matches!(compression, Compression::Deflate | Compression::DeflateOld) {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "Deflate compression level requires Deflate compression, got {compression:?}"
            ),
        });
    }
    Ok(())
}

/// Full LERC compression pipeline: typed samples → LERC2 blob → optional additional compression.
///
/// This is the LERC counterpart of [`compress_block`]. LERC operates directly on
/// typed sample values (no byte-order encoding, no TIFF predictor).
pub fn compress_block_lerc<T: TiffWriteSample>(
    samples: &[T],
    block_width: u32,
    block_height: u32,
    depth: u32,
    options: &LercOptions,
    index: usize,
) -> Result<Vec<u8>> {
    let blob = T::lerc_encode_block(
        samples,
        block_width,
        block_height,
        depth,
        options.max_z_error,
        index,
    )?;
    validate_tiff_lerc_version(&blob, index)?;

    match options.additional_compression {
        tiff_core::LercAdditionalCompression::None => Ok(blob),
        tiff_core::LercAdditionalCompression::Deflate => compress_deflate(&blob, index),
        tiff_core::LercAdditionalCompression::Zstd => {
            #[cfg(feature = "zstd")]
            {
                compress_zstd(&blob, index)
            }
            #[cfg(not(feature = "zstd"))]
            {
                Err(Error::CompressionFailed {
                    index,
                    reason: "LERC+Zstd requires the 'zstd' feature".into(),
                })
            }
        }
    }
}

/// Low-level LERC2 encoding for a single typed raster block.
///
/// Called by `TiffWriteSample::lerc_encode_block` implementations for
/// LERC-compatible types (i8, u8, i16, u16, i32, u32, f32, f64).
pub(crate) fn lerc_encode<T: lerc_core::Sample>(
    samples: &[T],
    width: u32,
    height: u32,
    depth: u32,
    max_z_error: f64,
    index: usize,
) -> Result<Vec<u8>> {
    let raster = lerc_core::RasterView::new(width, height, depth, samples).map_err(|e| {
        Error::CompressionFailed {
            index,
            reason: format!("LERC raster view: {e}"),
        }
    })?;
    // TIFF LercParameters declares Lerc2 v4. The default compact profile may
    // emit v5 blobs for multi-band integer rasters, which that tag can't describe.
    let options = lerc_writer::EncodeOptions::new()
        .with_max_z_error(max_z_error)
        .with_profile(lerc_writer::EncoderProfile::Gdal);
    lerc_writer::encode(raster, None, options).map_err(|e| Error::CompressionFailed {
        index,
        reason: format!("LERC encode: {e}"),
    })
}

fn validate_tiff_lerc_version(blob: &[u8], index: usize) -> Result<()> {
    let Some(version_bytes) = blob.get(6..10) else {
        return Err(Error::CompressionFailed {
            index,
            reason: "LERC encode produced a blob without a Lerc2 version header".into(),
        });
    };
    let version = i32::from_le_bytes(
        version_bytes
            .try_into()
            .expect("slice length checked by blob.get(6..10)"),
    );
    if version != LERC_VERSION_2_4 as i32 {
        return Err(Error::CompressionFailed {
            index,
            reason: format!(
                "LERC2 version {version} is not supported by TIFF LERC_PARAMETERS; expected version {LERC_VERSION_2_4}"
            ),
        });
    }
    Ok(())
}

/// Apply forward predictor to a row of file-order bytes (in-place).
fn apply_forward_predictor(
    row: &mut [u8],
    predictor: Predictor,
    bits_per_sample: u16,
    samples_per_pixel: u16,
    byte_order: ByteOrder,
) -> Result<()> {
    match predictor {
        Predictor::None => Ok(()),
        Predictor::Horizontal => {
            forward_horizontal_differencing(row, bits_per_sample, samples_per_pixel, byte_order);
            Ok(())
        }
        Predictor::FloatingPoint => {
            forward_float_predictor(row, bits_per_sample, samples_per_pixel, byte_order);
            Ok(())
        }
    }
}

/// Forward horizontal differencing: each sample = sample - previous.
/// Operates on file-order bytes. This is the inverse of the reader's
/// `reverse_horizontal_predictor`.
///
/// Must iterate right-to-left so we don't clobber values we still need.
fn forward_horizontal_differencing(
    buf: &mut [u8],
    bit_depth: u16,
    samples: u16,
    byte_order: ByteOrder,
) {
    let bpv = match bit_depth {
        0..=8 => 1usize,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    };
    let n_values = buf.len() / bpv;
    let skip = usize::from(samples); // first `samples` values are kept as-is

    if skip >= n_values {
        return;
    }

    // Iterate value indices right-to-left
    for vi in (skip..n_values).rev() {
        let pos = vi * bpv;
        let prev = (vi - skip) * bpv;
        match bpv {
            1 => {
                buf[pos] = buf[pos].wrapping_sub(buf[prev]);
            }
            2 => {
                let cur = byte_order.read_u16([buf[pos], buf[pos + 1]]);
                let prv = byte_order.read_u16([buf[prev], buf[prev + 1]]);
                let d = byte_order.write_u16(cur.wrapping_sub(prv));
                buf[pos..pos + 2].copy_from_slice(&d);
            }
            4 => {
                let cur = byte_order.read_u32(buf[pos..pos + 4].try_into().unwrap());
                let prv = byte_order.read_u32(buf[prev..prev + 4].try_into().unwrap());
                let d = byte_order.write_u32(cur.wrapping_sub(prv));
                buf[pos..pos + 4].copy_from_slice(&d);
            }
            _ => {
                let cur = byte_order.read_u64(buf[pos..pos + 8].try_into().unwrap());
                let prv = byte_order.read_u64(buf[prev..prev + 8].try_into().unwrap());
                let d = byte_order.write_u64(cur.wrapping_sub(prv));
                buf[pos..pos + 8].copy_from_slice(&d);
            }
        }
    }
}

/// Forward floating-point predictor (TIFF predictor 3).
///
/// The TIFF float predictor always operates on big-endian byte planes,
/// regardless of the file's byte order. The process is:
/// 1. Convert each float value to big-endian bytes
/// 2. Interleave into byte planes (all byte[0]s, all byte[1]s, ...)
/// 3. Apply forward byte differencing (delta encoding)
///
/// The `byte_order` parameter indicates the current byte order of `buf`
/// (as written by encode_slice), so we can convert to BE properly.
fn forward_float_predictor(buf: &mut [u8], bit_depth: u16, samples: u16, byte_order: ByteOrder) {
    let bps = match bit_depth {
        16 => 2usize,
        32 => 4,
        64 => 8,
        _ => return,
    };
    let n_values = buf.len() / bps;
    if n_values == 0 {
        return;
    }

    // Step 1+2: Convert each value to BE and interleave into byte planes.
    let need_swap = matches!(byte_order, ByteOrder::LittleEndian);
    let mut tmp = vec![0u8; buf.len()];
    for i in 0..n_values {
        let base = i * bps;
        for b in 0..bps {
            // BE byte `b` is at reversed position for LE data
            let src_b = if need_swap { bps - 1 - b } else { b };
            tmp[b * n_values + i] = buf[base + src_b];
        }
    }

    // Step 3: Forward byte differencing with lookback = samples
    let samples = usize::from(samples);
    for i in (samples..tmp.len()).rev() {
        tmp[i] = tmp[i].wrapping_sub(tmp[i - samples]);
    }

    buf.copy_from_slice(&tmp);
}

fn compress_lzw(data: &[u8], index: usize) -> Result<Vec<u8>> {
    use weezl::encode::Encoder;
    use weezl::BitOrder;

    let mut encoder = Encoder::with_tiff_size_switch(BitOrder::Msb, 8);
    encoder.encode(data).map_err(|e| Error::CompressionFailed {
        index,
        reason: format!("LZW: {e}"),
    })
}

fn compress_deflate(data: &[u8], index: usize) -> Result<Vec<u8>> {
    compress_deflate_with_level(data, None, index)
}

fn compress_deflate_with_level(data: &[u8], level: Option<u32>, index: usize) -> Result<Vec<u8>> {
    use flate2::write::ZlibEncoder;
    use std::io::Write;

    let level = level.map_or_else(flate2::Compression::default, flate2::Compression::new);
    let mut encoder = ZlibEncoder::new(Vec::new(), level);
    encoder
        .write_all(data)
        .map_err(|e| Error::CompressionFailed {
            index,
            reason: format!("deflate write: {e}"),
        })?;
    encoder.finish().map_err(|e| Error::CompressionFailed {
        index,
        reason: format!("deflate finish: {e}"),
    })
}

#[cfg(feature = "jpeg")]
fn compress_jpeg(
    data: &[u8],
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    options: JpegOptions,
    sampling: Option<[u16; 2]>,
    index: usize,
) -> Result<Vec<u8>> {
    compress_jpeg_inner(
        data,
        width,
        height,
        samples_per_pixel,
        options,
        sampling,
        index,
    )
}

#[cfg(feature = "jpeg")]
#[allow(clippy::too_many_arguments)]
fn compress_jpeg_inner(
    data: &[u8],
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    options: JpegOptions,
    sampling: Option<[u16; 2]>,
    index: usize,
) -> Result<Vec<u8>> {
    let width = u16::try_from(width).map_err(|_| Error::CompressionFailed {
        index,
        reason: format!("JPEG block width {width} exceeds u16::MAX"),
    })?;
    let height = u16::try_from(height).map_err(|_| Error::CompressionFailed {
        index,
        reason: format!("JPEG block height {height} exceeds u16::MAX"),
    })?;
    let color_type = match samples_per_pixel {
        1 => jpeg_encoder::ColorType::Luma,
        3 => jpeg_encoder::ColorType::Rgb,
        other => {
            return Err(Error::CompressionFailed {
                index,
                reason: format!("JPEG write supports 1 or 3 samples per block, got {other}"),
            })
        }
    };

    let mut out = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut out, options.quality);
    if let Some([horizontal, vertical]) = sampling {
        let factor = u8::try_from(horizontal)
            .ok()
            .zip(u8::try_from(vertical).ok())
            .and_then(|(h, v)| jpeg_encoder::SamplingFactor::from_factors(h, v))
            .ok_or_else(|| Error::CompressionFailed {
                index,
                reason: format!("unsupported JPEG chroma subsampling {horizontal}x{vertical}"),
            })?;
        encoder.set_sampling_factor(factor);
    }
    encoder
        .encode(data, width, height, color_type)
        .map_err(|error| Error::CompressionFailed {
            index,
            reason: format!("JPEG: {error}"),
        })?;
    Ok(out)
}

#[cfg(feature = "zstd")]
fn compress_zstd(data: &[u8], _index: usize) -> Result<Vec<u8>> {
    Ok(ruzstd::encoding::compress_to_vec(
        std::io::Cursor::new(data),
        ruzstd::encoding::CompressionLevel::Fastest,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_no_compression() {
        let data = vec![1u8, 2, 3, 4, 5, 6];
        let compressed = compress(&data, Compression::None, 0).unwrap();
        assert_eq!(compressed, data);
    }

    #[test]
    fn roundtrip_lzw() {
        let data = vec![0u8; 256];
        let compressed = compress(&data, Compression::Lzw, 0).unwrap();
        assert!(compressed.len() < data.len());

        // Decompress with weezl to verify
        let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
        let decompressed = decoder.decode(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_deflate() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = compress(&data, Compression::Deflate, 0).unwrap();

        // Decompress with flate2 to verify
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut decoder = ZlibDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn explicit_deflate_level_rejects_invalid_configuration() {
        let error = compress_with_level(&[1, 2, 3], Compression::Deflate, Some(10), 7).unwrap_err();
        assert!(matches!(
            error,
            Error::CompressionFailed { index: 7, reason }
                if reason.contains("must be 0-9")
        ));

        let error = compress_with_level(&[1, 2, 3], Compression::Lzw, Some(6), 8).unwrap_err();
        assert!(matches!(
            error,
            Error::CompressionFailed { index: 8, reason }
                if reason.contains("requires Deflate compression")
        ));
    }

    #[test]
    fn block_compression_rejects_invalid_row_layouts_and_predictors() {
        let options = BlockEncodingOptions {
            byte_order: ByteOrder::LittleEndian,
            compression: Compression::Deflate,
            predictor: Predictor::None,
            samples_per_pixel: 1,
            row_width_pixels: 2,
            jpeg_options: None,
            jpeg_sampling: None,
            deflate_level: None,
        };
        let error = compress_block(&[1u8, 2, 3], options, 4).unwrap_err();
        assert!(
            matches!(error, Error::CompressionFailed { index: 4, reason } if reason.contains("not divisible"))
        );

        let error = compress_block(
            &[1u16],
            BlockEncodingOptions {
                row_width_pixels: usize::MAX,
                ..options
            },
            5,
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::CompressionFailed { index: 5, reason } if reason.contains("overflows"))
        );

        let error = compress_block(
            &[1.0f32],
            BlockEncodingOptions {
                predictor: Predictor::Horizontal,
                row_width_pixels: 1,
                ..options
            },
            6,
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::CompressionFailed { index: 6, reason } if reason.contains("integer"))
        );
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn roundtrip_zstd() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = compress(&data, Compression::Zstd, 0).unwrap();
        let mut decoder =
            ruzstd::decoding::StreamingDecoder::new(std::io::Cursor::new(&compressed)).unwrap();
        let mut decompressed = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut decompressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn forward_horizontal_u8() {
        // [1, 2, 4, 7] → differences → [1, 1, 2, 3]
        let mut buf = vec![1u8, 2, 4, 7];
        forward_horizontal_differencing(&mut buf, 8, 1, ByteOrder::LittleEndian);
        assert_eq!(buf, vec![1, 1, 2, 3]);
    }

    #[test]
    fn forward_horizontal_u16_le() {
        // [1, 2, 4] in u16 LE → differences → [1, 1, 2]
        let bo = ByteOrder::LittleEndian;
        let mut buf = Vec::new();
        buf.extend_from_slice(&bo.write_u16(1));
        buf.extend_from_slice(&bo.write_u16(2));
        buf.extend_from_slice(&bo.write_u16(4));

        forward_horizontal_differencing(&mut buf, 16, 1, bo);

        let v0 = bo.read_u16([buf[0], buf[1]]);
        let v1 = bo.read_u16([buf[2], buf[3]]);
        let v2 = bo.read_u16([buf[4], buf[5]]);
        assert_eq!((v0, v1, v2), (1, 1, 2));
    }
}
