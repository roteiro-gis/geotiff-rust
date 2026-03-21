//! Type-safe sample encoding for TIFF writes.

use tiff_core::ByteOrder;

/// Types that can be written as TIFF samples.
pub trait TiffWriteSample: tiff_core::TiffSample + Copy + Send + Sync {
    /// TIFF SampleFormat code (1=uint, 2=int, 3=float).
    const SAMPLE_FORMAT: u16;
    /// TIFF BitsPerSample value.
    const BITS_PER_SAMPLE: u16;
    /// Bytes per sample.
    const BYTES_PER_SAMPLE: usize;

    /// Encode a slice of samples into file-order bytes.
    fn encode_slice(samples: &[Self], byte_order: ByteOrder) -> Vec<u8>;
}

macro_rules! impl_write_sample_8 {
    ($ty:ty, $format:expr) => {
        impl TiffWriteSample for $ty {
            const SAMPLE_FORMAT: u16 = $format;
            const BITS_PER_SAMPLE: u16 = 8;
            const BYTES_PER_SAMPLE: usize = 1;

            fn encode_slice(samples: &[Self], _byte_order: ByteOrder) -> Vec<u8> {
                samples.iter().map(|&v| v as u8).collect()
            }
        }
    };
}

macro_rules! impl_write_sample {
    ($ty:ty, $format:expr, $bits:expr, $bytes:expr, $write_fn:ident) => {
        impl TiffWriteSample for $ty {
            const SAMPLE_FORMAT: u16 = $format;
            const BITS_PER_SAMPLE: u16 = $bits;
            const BYTES_PER_SAMPLE: usize = $bytes;

            fn encode_slice(samples: &[Self], byte_order: ByteOrder) -> Vec<u8> {
                let mut out = Vec::with_capacity(samples.len() * $bytes);
                for &v in samples {
                    out.extend_from_slice(&byte_order.$write_fn(v));
                }
                out
            }
        }
    };
}

impl_write_sample_8!(u8, 1);
impl_write_sample_8!(i8, 2);
impl_write_sample!(u16, 1, 16, 2, write_u16);
impl_write_sample!(i16, 2, 16, 2, write_i16);
impl_write_sample!(u32, 1, 32, 4, write_u32);
impl_write_sample!(i32, 2, 32, 4, write_i32);
impl_write_sample!(f32, 3, 32, 4, write_f32);
impl_write_sample!(u64, 1, 64, 8, write_u64);
impl_write_sample!(i64, 2, 64, 8, write_i64);
impl_write_sample!(f64, 3, 64, 8, write_f64);
