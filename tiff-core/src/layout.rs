use thiserror::Error;

/// Errors produced while computing derived raster layout sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LayoutError {
    #[error("{0} overflows usize")]
    SizeOverflow(&'static str),
}

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
    /// Whether samples are stored as a bitstream rather than filling whole
    /// bytes. Such samples decode into the next larger byte-sized container.
    pub fn is_bit_packed(&self) -> bool {
        !matches!(self.bits_per_sample, 8 | 16 | 32 | 64)
    }

    pub fn checked_pixel_stride_bytes(&self) -> Result<usize, LayoutError> {
        self.samples_per_pixel
            .checked_mul(self.bytes_per_sample)
            .ok_or(LayoutError::SizeOverflow("pixel stride byte count"))
    }

    pub fn pixel_stride_bytes(&self) -> usize {
        self.checked_pixel_stride_bytes().unwrap_or(usize::MAX)
    }

    pub fn checked_row_bytes_for_width(&self, width: usize) -> Result<usize, LayoutError> {
        width
            .checked_mul(self.checked_pixel_stride_bytes()?)
            .ok_or(LayoutError::SizeOverflow("row byte count"))
    }

    pub fn checked_packed_row_bytes_for_width(&self, width: usize) -> Result<usize, LayoutError> {
        width
            .checked_mul(self.samples_per_pixel)
            .and_then(|samples| samples.checked_mul(self.bits_per_sample as usize))
            .map(|bits| bits.div_ceil(8))
            .ok_or(LayoutError::SizeOverflow("packed row byte count"))
    }

    pub fn packed_row_bytes_for_width(&self, width: usize) -> usize {
        self.checked_packed_row_bytes_for_width(width)
            .unwrap_or(usize::MAX)
    }

    pub fn checked_row_bytes(&self) -> Result<usize, LayoutError> {
        self.checked_row_bytes_for_width(self.width)
    }

    pub fn row_bytes(&self) -> usize {
        self.checked_row_bytes().unwrap_or(usize::MAX)
    }

    pub fn checked_packed_row_bytes(&self) -> Result<usize, LayoutError> {
        self.checked_packed_row_bytes_for_width(self.width)
    }

    pub fn packed_row_bytes(&self) -> usize {
        self.checked_packed_row_bytes().unwrap_or(usize::MAX)
    }

    pub fn checked_sample_plane_row_bytes_for_width(
        &self,
        width: usize,
    ) -> Result<usize, LayoutError> {
        width
            .checked_mul(self.bytes_per_sample)
            .ok_or(LayoutError::SizeOverflow("sample plane row byte count"))
    }

    pub fn checked_packed_sample_plane_row_bytes_for_width(
        &self,
        width: usize,
    ) -> Result<usize, LayoutError> {
        width
            .checked_mul(self.bits_per_sample as usize)
            .map(|bits| bits.div_ceil(8))
            .ok_or(LayoutError::SizeOverflow(
                "packed sample plane row byte count",
            ))
    }

    pub fn packed_sample_plane_row_bytes_for_width(&self, width: usize) -> usize {
        self.checked_packed_sample_plane_row_bytes_for_width(width)
            .unwrap_or(usize::MAX)
    }

    pub fn checked_sample_plane_row_bytes(&self) -> Result<usize, LayoutError> {
        self.checked_sample_plane_row_bytes_for_width(self.width)
    }

    pub fn sample_plane_row_bytes(&self) -> usize {
        self.checked_sample_plane_row_bytes().unwrap_or(usize::MAX)
    }

    pub fn checked_packed_sample_plane_row_bytes(&self) -> Result<usize, LayoutError> {
        self.checked_packed_sample_plane_row_bytes_for_width(self.width)
    }

    pub fn packed_sample_plane_row_bytes(&self) -> usize {
        self.checked_packed_sample_plane_row_bytes()
            .unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::{LayoutError, RasterLayout};

    fn layout(width: usize, samples_per_pixel: usize, bits_per_sample: u16) -> RasterLayout {
        RasterLayout {
            width,
            height: 1,
            samples_per_pixel,
            bits_per_sample,
            bytes_per_sample: usize::from(bits_per_sample.div_ceil(8)),
            sample_format: 1,
            planar_configuration: 1,
            predictor: 1,
        }
    }

    #[test]
    fn checked_layout_helpers_return_expected_byte_counts() {
        let layout = layout(4, 3, 16);

        assert_eq!(layout.checked_pixel_stride_bytes().unwrap(), 6);
        assert_eq!(layout.checked_row_bytes().unwrap(), 24);
        assert_eq!(layout.checked_sample_plane_row_bytes().unwrap(), 8);
        assert_eq!(layout.checked_packed_row_bytes().unwrap(), 24);
        assert_eq!(layout.checked_packed_sample_plane_row_bytes().unwrap(), 8);
    }

    #[test]
    fn checked_layout_helpers_reject_usize_overflow() {
        let layout = RasterLayout {
            width: usize::MAX,
            height: 1,
            samples_per_pixel: usize::MAX,
            bits_per_sample: 16,
            bytes_per_sample: 2,
            sample_format: 1,
            planar_configuration: 1,
            predictor: 1,
        };

        assert!(matches!(
            layout.checked_pixel_stride_bytes(),
            Err(LayoutError::SizeOverflow(_))
        ));
        assert!(matches!(
            layout.checked_row_bytes(),
            Err(LayoutError::SizeOverflow(_))
        ));
        assert!(matches!(
            layout.checked_sample_plane_row_bytes(),
            Err(LayoutError::SizeOverflow(_))
        ));
        assert!(matches!(
            layout.checked_packed_row_bytes(),
            Err(LayoutError::SizeOverflow(_))
        ));
    }
}
