//! Geo-transform: pixel coordinates to/from geographic coordinates.

use crate::crs::RasterType;

/// An affine geo-transform mapping pixel (col, row) to map (x, y).
///
/// Follows the GDAL convention:
/// ```text
/// x = origin_x + col * pixel_width + row * skew_x
/// y = origin_y + col * skew_y     + row * pixel_height
/// ```
///
/// For north-up images, `skew_x` and `skew_y` are 0 and `pixel_height` is negative.
#[derive(Debug, Clone, Copy)]
pub struct GeoTransform {
    pub origin_x: f64,
    pub pixel_width: f64,
    pub skew_x: f64,
    pub origin_y: f64,
    pub skew_y: f64,
    pub pixel_height: f64,
}

impl GeoTransform {
    /// Build from ModelTiepoint (tag 33922) and ModelPixelScale (tag 33550).
    pub fn from_tiepoint_and_scale(tiepoint: &[f64; 6], pixel_scale: &[f64; 3]) -> Self {
        Self::from_tiepoint_and_scale_with_raster_type(
            tiepoint,
            pixel_scale,
            RasterType::PixelIsArea,
        )
    }

    /// Build from ModelTiepoint and ModelPixelScale using the GeoTIFF raster type.
    ///
    /// The returned transform is normalized to a corner-based affine transform so
    /// bounds and pixel-space math stay consistent for both PixelIsArea and
    /// PixelIsPoint rasters.
    pub fn from_tiepoint_and_scale_with_raster_type(
        tiepoint: &[f64; 6],
        pixel_scale: &[f64; 3],
        raster_type: RasterType,
    ) -> Self {
        // tiepoint: [I, J, K, X, Y, Z]
        // pixel_scale: [ScaleX, ScaleY, ScaleZ]
        let pixel_offset = match raster_type {
            RasterType::PixelIsPoint => 0.5,
            RasterType::PixelIsArea | RasterType::Unknown(_) => 0.0,
        };
        Self {
            origin_x: tiepoint[3] - (tiepoint[0] + pixel_offset) * pixel_scale[0],
            pixel_width: pixel_scale[0],
            skew_x: 0.0,
            origin_y: tiepoint[4] + (tiepoint[1] + pixel_offset) * pixel_scale[1],
            skew_y: 0.0,
            pixel_height: -pixel_scale[1],
        }
    }

    /// Build from a 4x4 ModelTransformation matrix (tag 34264), row-major.
    pub fn from_transformation_matrix(matrix: &[f64; 16]) -> Self {
        Self {
            origin_x: matrix[3],
            pixel_width: matrix[0],
            skew_x: matrix[1],
            origin_y: matrix[7],
            skew_y: matrix[4],
            pixel_height: matrix[5],
        }
    }

    /// Create from origin + pixel size (north-up, no skew).
    pub fn from_origin_and_pixel_size(
        origin_x: f64,
        origin_y: f64,
        pixel_width: f64,
        pixel_height: f64,
    ) -> Self {
        Self {
            origin_x,
            pixel_width,
            skew_x: 0.0,
            origin_y,
            skew_y: 0.0,
            pixel_height,
        }
    }

    /// Convert pixel coordinates (col, row) to map coordinates (x, y).
    pub fn pixel_to_geo(&self, col: f64, row: f64) -> (f64, f64) {
        let x = self.origin_x + col * self.pixel_width + row * self.skew_x;
        let y = self.origin_y + col * self.skew_y + row * self.pixel_height;
        (x, y)
    }

    /// Convert map coordinates (x, y) to pixel coordinates (col, row).
    ///
    /// Returns `None` if the transform is degenerate (zero determinant).
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let det = self.pixel_width * self.pixel_height - self.skew_x * self.skew_y;
        if det.abs() < 1e-15 {
            return None;
        }
        let dx = x - self.origin_x;
        let dy = y - self.origin_y;
        let col = (self.pixel_height * dx - self.skew_x * dy) / det;
        let row = (-self.skew_y * dx + self.pixel_width * dy) / det;
        Some((col, row))
    }

    /// Returns the geographic bounds (min_x, min_y, max_x, max_y) for an image
    /// of the given width and height.
    pub fn bounds(&self, width: u32, height: u32) -> [f64; 4] {
        let corners = [
            self.pixel_to_geo(0.0, 0.0),
            self.pixel_to_geo(width as f64, 0.0),
            self.pixel_to_geo(0.0, height as f64),
            self.pixel_to_geo(width as f64, height as f64),
        ];
        let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|c| c.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|c| c.1)
            .fold(f64::NEG_INFINITY, f64::max);
        [min_x, min_y, max_x, max_y]
    }

    /// Serialize to a tiepoint + pixel_scale pair (for north-up, no-skew images).
    /// Returns `None` if there is skew (use `to_transformation_matrix` instead).
    pub fn to_tiepoint_and_scale(&self) -> Option<([f64; 6], [f64; 3])> {
        if self.skew_x.abs() > 1e-15 || self.skew_y.abs() > 1e-15 {
            return None;
        }
        let tiepoint = [0.0, 0.0, 0.0, self.origin_x, self.origin_y, 0.0];
        let scale = [self.pixel_width, -self.pixel_height, 0.0];
        Some((tiepoint, scale))
    }

    /// Serialize to a 4x4 transformation matrix (row-major).
    pub fn to_transformation_matrix(&self) -> [f64; 16] {
        [
            self.pixel_width,
            self.skew_x,
            0.0,
            self.origin_x,
            self.skew_y,
            self.pixel_height,
            0.0,
            self.origin_y,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crs::RasterType;

    #[test]
    fn tiepoint_and_scale_roundtrip() {
        let tp = [0.0, 0.0, 0.0, -180.0, 90.0, 0.0];
        let scale = [0.1, 0.1, 0.0];
        let gt = GeoTransform::from_tiepoint_and_scale(&tp, &scale);

        let (x, y) = gt.pixel_to_geo(0.0, 0.0);
        assert!((x - (-180.0)).abs() < 1e-10);
        assert!((y - 90.0).abs() < 1e-10);

        let (x2, y2) = gt.pixel_to_geo(10.0, 10.0);
        assert!((x2 - (-179.0)).abs() < 1e-10);
        assert!((y2 - 89.0).abs() < 1e-10);

        let (col, row) = gt.geo_to_pixel(x2, y2).unwrap();
        assert!((col - 10.0).abs() < 1e-10);
        assert!((row - 10.0).abs() < 1e-10);
    }

    #[test]
    fn bounds_calculation() {
        let tp = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        let scale = [1.0, 1.0, 0.0];
        let gt = GeoTransform::from_tiepoint_and_scale(&tp, &scale);
        let bounds = gt.bounds(10, 10);
        assert!((bounds[0] - 0.0).abs() < 1e-10);
        assert!((bounds[1] - 0.0).abs() < 1e-10);
        assert!((bounds[2] - 10.0).abs() < 1e-10);
        assert!((bounds[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn pixel_is_point_tiepoint_is_normalized_to_outer_bounds() {
        let tp = [0.0, 0.0, 0.0, 100.0, 200.0, 0.0];
        let scale = [2.0, 2.0, 0.0];
        let gt = GeoTransform::from_tiepoint_and_scale_with_raster_type(
            &tp,
            &scale,
            RasterType::PixelIsPoint,
        );

        let (min_x, max_y) = gt.pixel_to_geo(0.0, 0.0);
        assert!((min_x - 99.0).abs() < 1e-10);
        assert!((max_y - 201.0).abs() < 1e-10);

        let (center_x, center_y) = gt.pixel_to_geo(0.5, 0.5);
        assert!((center_x - 100.0).abs() < 1e-10);
        assert!((center_y - 200.0).abs() < 1e-10);
    }

    #[test]
    fn to_tiepoint_and_scale_roundtrips() {
        let gt = GeoTransform::from_origin_and_pixel_size(-180.0, 90.0, 0.1, -0.1);
        let (tp, scale) = gt.to_tiepoint_and_scale().unwrap();
        let gt2 = GeoTransform::from_tiepoint_and_scale(&tp, &scale);
        assert!((gt2.origin_x - gt.origin_x).abs() < 1e-10);
        assert!((gt2.origin_y - gt.origin_y).abs() < 1e-10);
        assert!((gt2.pixel_width - gt.pixel_width).abs() < 1e-10);
        assert!((gt2.pixel_height - gt.pixel_height).abs() < 1e-10);
    }

    #[test]
    fn skewed_transform_returns_none_for_tiepoint_scale() {
        let gt = GeoTransform {
            origin_x: 0.0,
            pixel_width: 1.0,
            skew_x: 0.5,
            origin_y: 0.0,
            skew_y: 0.0,
            pixel_height: -1.0,
        };
        assert!(gt.to_tiepoint_and_scale().is_none());
    }

    #[test]
    fn transformation_matrix_roundtrips() {
        let gt = GeoTransform::from_origin_and_pixel_size(100.0, 200.0, 0.5, -0.5);
        let matrix = gt.to_transformation_matrix();
        let gt2 = GeoTransform::from_transformation_matrix(&matrix);
        assert!((gt2.origin_x - gt.origin_x).abs() < 1e-10);
        assert!((gt2.origin_y - gt.origin_y).abs() < 1e-10);
        assert!((gt2.pixel_width - gt.pixel_width).abs() < 1e-10);
        assert!((gt2.pixel_height - gt.pixel_height).abs() < 1e-10);
    }
}
