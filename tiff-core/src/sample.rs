use crate::layout::RasterLayout;

/// Types that can be read directly from a decoded TIFF raster.
pub trait TiffSample: Clone + 'static {
    fn matches_layout(layout: &RasterLayout) -> bool;
    fn decode_many(bytes: &[u8]) -> Vec<Self>;
    fn type_name() -> &'static str;
}

macro_rules! impl_tiff_sample {
    ($ty:ty, $format:expr, $bits:expr, $chunk:expr, $from_ne:expr) => {
        impl TiffSample for $ty {
            fn matches_layout(layout: &RasterLayout) -> bool {
                layout.sample_format == $format && layout.bits_per_sample == $bits
            }

            fn decode_many(bytes: &[u8]) -> Vec<Self> {
                bytes.chunks_exact($chunk).map($from_ne).collect()
            }

            fn type_name() -> &'static str {
                stringify!($ty)
            }
        }
    };
}

impl_tiff_sample!(u8, 1, 8, 1, |chunk: &[u8]| chunk[0]);
impl_tiff_sample!(i8, 2, 8, 1, |chunk: &[u8]| chunk[0] as i8);
impl_tiff_sample!(u16, 1, 16, 2, |chunk: &[u8]| u16::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(i16, 2, 16, 2, |chunk: &[u8]| i16::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(u32, 1, 32, 4, |chunk: &[u8]| u32::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(i32, 2, 32, 4, |chunk: &[u8]| i32::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(f32, 3, 32, 4, |chunk: &[u8]| f32::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(u64, 1, 64, 8, |chunk: &[u8]| u64::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(i64, 2, 64, 8, |chunk: &[u8]| i64::from_ne_bytes(
    chunk.try_into().unwrap()
));
impl_tiff_sample!(f64, 3, 64, 8, |chunk: &[u8]| f64::from_ne_bytes(
    chunk.try_into().unwrap()
));
