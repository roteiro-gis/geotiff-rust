//! Re-export WriteSample from tiff-writer and add lightweight numeric helpers.

pub use tiff_writer::TiffWriteSample as WriteSample;

/// Numeric conversions used internally by overview generation and fill handling.
#[doc(hidden)]
pub trait NumericSample: WriteSample + PartialEq {
    fn zero() -> Self;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

pub(crate) fn parse_nodata_value<T: NumericSample>(nodata: &Option<String>) -> Option<T> {
    let nd = nodata.as_ref()?;
    let value = nd.trim().parse::<f64>().ok()?;
    Some(T::from_f64(value))
}

pub(crate) fn nodata_fill_or_zero<T: NumericSample>(nodata: &Option<String>) -> T {
    parse_nodata_value(nodata).unwrap_or_else(T::zero)
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
