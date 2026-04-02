//! Re-export WriteSample from tiff-writer and add lightweight numeric helpers.

pub use tiff_writer::TiffWriteSample as WriteSample;

/// Numeric conversions used internally by overview generation and fill handling.
#[doc(hidden)]
pub trait NumericSample: WriteSample {
    fn zero() -> Self;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

macro_rules! impl_numeric_sample {
    ($ty:ty) => {
        impl NumericSample for $ty {
            fn zero() -> Self {
                0 as $ty
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    };
}

impl_numeric_sample!(u8);
impl_numeric_sample!(i8);
impl_numeric_sample!(u16);
impl_numeric_sample!(i16);
impl_numeric_sample!(u32);
impl_numeric_sample!(i32);
impl_numeric_sample!(f32);
impl_numeric_sample!(u64);
impl_numeric_sample!(i64);
impl_numeric_sample!(f64);
