/// Raster layout information normalized from TIFF tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterLayout {
    pub width: usize,
    pub height: usize,
    pub samples_per_pixel: usize,
    pub bits_per_sample: u16,
    pub bytes_per_sample: usize,
    pub sample_format: u16,
    pub planar_configuration: u16,
    pub predictor: u16,
}

impl RasterLayout {
    pub fn pixel_stride_bytes(&self) -> usize {
        self.samples_per_pixel * self.bytes_per_sample
    }

    pub fn packed_row_bytes_for_width(&self, width: usize) -> usize {
        width
            .checked_mul(self.samples_per_pixel)
            .and_then(|samples| samples.checked_mul(self.bits_per_sample as usize))
            .map(|bits| bits.div_ceil(8))
            .unwrap_or(usize::MAX)
    }

    pub fn row_bytes(&self) -> usize {
        self.width * self.pixel_stride_bytes()
    }

    pub fn packed_row_bytes(&self) -> usize {
        self.packed_row_bytes_for_width(self.width)
    }

    pub fn packed_sample_plane_row_bytes_for_width(&self, width: usize) -> usize {
        width
            .checked_mul(self.bits_per_sample as usize)
            .map(|bits| bits.div_ceil(8))
            .unwrap_or(usize::MAX)
    }

    pub fn sample_plane_row_bytes(&self) -> usize {
        self.width * self.bytes_per_sample
    }

    pub fn packed_sample_plane_row_bytes(&self) -> usize {
        self.packed_sample_plane_row_bytes_for_width(self.width)
    }
}
