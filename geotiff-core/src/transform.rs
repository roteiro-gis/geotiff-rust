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
        //
        // ScaleX keeps its sign because GDAL writes east-to-west rasters that
        // way. A negative ScaleY is read as north-up, also following GDAL.
        let pixel_offset = pixel_center_offset(raster_type);
        let scale_y = pixel_scale[1].abs();
        Self {
            origin_x: tiepoint[3] - (tiepoint[0] + pixel_offset) * pixel_scale[0],
            pixel_width: pixel_scale[0],
            skew_x: 0.0,
            origin_y: tiepoint[4] + (tiepoint[1] + pixel_offset) * scale_y,
            skew_y: 0.0,
            pixel_height: -scale_y,
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

    /// Build from a ModelTransformation matrix using the GeoTIFF raster type.
    ///
    /// PixelIsPoint matrices map pixel centers, so the result is shifted by
    /// half a pixel along both raster axes to give the same corner-based
    /// transform as [`Self::from_tiepoint_and_scale_with_raster_type`].
    pub fn from_transformation_matrix_with_raster_type(
        matrix: &[f64; 16],
        raster_type: RasterType,
    ) -> Self {
        let transform = Self::from_transformation_matrix(matrix);
        let pixel_offset = pixel_center_offset(raster_type);
        let (origin_x, origin_y) = transform.pixel_to_geo(-pixel_offset, -pixel_offset);
        Self {
            origin_x,
            origin_y,
            ..transform
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
    /// Returns `None` if the transform is degenerate or numerically singular.
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let det = self.pixel_width * self.pixel_height - self.skew_x * self.skew_y;
        let determinant_scale =
            (self.pixel_width * self.pixel_height).abs() + (self.skew_x * self.skew_y).abs();
        if !det.is_finite() || det == 0.0 || det.abs() <= f64::EPSILON * determinant_scale {
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
    ///
    /// This uses PixelIsArea semantics. Use
    /// [`Self::to_tiepoint_and_scale_with_raster_type`] when writing a GeoTIFF
    /// with an explicit raster type.
    pub fn to_tiepoint_and_scale(&self) -> Option<([f64; 6], [f64; 3])> {
        self.to_tiepoint_and_scale_with_raster_type(RasterType::PixelIsArea)
    }

    /// Serialize to a tiepoint + pixel_scale pair using the GeoTIFF raster type.
    ///
    /// The transform is stored internally as a corner-based affine transform.
    /// PixelIsPoint tiepoints, however, refer to pixel centers, so the emitted
    /// tiepoint is shifted by half a pixel to roundtrip through
    /// [`Self::from_tiepoint_and_scale_with_raster_type`] without changing the
    /// normalized transform.
    pub fn to_tiepoint_and_scale_with_raster_type(
        &self,
        raster_type: RasterType,
    ) -> Option<([f64; 6], [f64; 3])> {
        // ModelPixelScale values are positive by definition. Preserve flipped
        // axes and every non-zero skew exactly by using ModelTransformation
        // instead of silently approximating them as north-up.
        if self.skew_x != 0.0
            || self.skew_y != 0.0
            || !self.pixel_width.is_finite()
            || !self.pixel_height.is_finite()
            || self.pixel_width <= 0.0
            || self.pixel_height >= 0.0
        {
            return None;
        }
        let scale = [self.pixel_width, -self.pixel_height, 0.0];
        let pixel_offset = pixel_center_offset(raster_type);
        let tiepoint = [
            0.0,
            0.0,
            0.0,
            self.origin_x + pixel_offset * scale[0],
            self.origin_y - pixel_offset * scale[1],
            0.0,
        ];
        Some((tiepoint, scale))
    }

    /// Serialize to a 4x4 transformation matrix using the GeoTIFF raster type.
    ///
    /// The inverse of [`Self::from_transformation_matrix_with_raster_type`].
    pub fn to_transformation_matrix_with_raster_type(&self, raster_type: RasterType) -> [f64; 16] {
        let pixel_offset = pixel_center_offset(raster_type);
        let (origin_x, origin_y) = self.pixel_to_geo(pixel_offset, pixel_offset);
        Self {
            origin_x,
            origin_y,
            ..*self
        }
        .to_transformation_matrix()
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

/// Raster-space offset between a pixel's corner and the point it georeferences.
fn pixel_center_offset(raster_type: RasterType) -> f64 {
    match raster_type {
        RasterType::PixelIsPoint => 0.5,
        RasterType::PixelIsArea | RasterType::Unknown(_) => 0.0,
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
    fn pixel_is_point_tiepoint_and_scale_roundtrips_normalized_transform() {
        let gt = GeoTransform::from_origin_and_pixel_size(99.0, 201.0, 2.0, -2.0);
        let (tp, scale) = gt
            .to_tiepoint_and_scale_with_raster_type(RasterType::PixelIsPoint)
            .unwrap();
        assert!((tp[3] - 100.0).abs() < 1e-10);
        assert!((tp[4] - 200.0).abs() < 1e-10);

        let gt2 = GeoTransform::from_tiepoint_and_scale_with_raster_type(
            &tp,
            &scale,
            RasterType::PixelIsPoint,
        );
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
    fn tiny_but_invertible_transform_roundtrips() {
        let gt = GeoTransform::from_origin_and_pixel_size(10.0, 20.0, 1e-12, -1e-12);
        let (x, y) = gt.pixel_to_geo(3.0, 4.0);
        let (col, row) = gt.geo_to_pixel(x, y).unwrap();
        assert!((col - 3.0).abs() < 1e-3);
        assert!((row - 4.0).abs() < 1e-3);
    }

    #[test]
    fn flipped_or_tiny_skewed_transforms_require_a_matrix() {
        let flipped = GeoTransform::from_origin_and_pixel_size(0.0, 0.0, -1.0, -1.0);
        assert!(flipped.to_tiepoint_and_scale().is_none());

        let skewed = GeoTransform {
            origin_x: 0.0,
            pixel_width: 1.0,
            skew_x: 1e-16,
            origin_y: 0.0,
            skew_y: 0.0,
            pixel_height: -1.0,
        };
        assert!(skewed.to_tiepoint_and_scale().is_none());
    }

    #[test]
    fn signed_pixel_scales_match_gdal() {
        // Expected values come from GDAL 3.12 reading the same tags.
        let tiepoint = [3.0, 4.0, 0.0, 100.0, 200.0, 0.0];

        let east_to_west = GeoTransform::from_tiepoint_and_scale_with_raster_type(
            &tiepoint,
            &[-2.0, 2.0, 0.0],
            RasterType::PixelIsPoint,
        );
        assert_eq!(east_to_west.origin_x, 107.0);
        assert_eq!(east_to_west.origin_y, 209.0);
        assert_eq!(east_to_west.pixel_width, -2.0);
        assert_eq!(east_to_west.pixel_height, -2.0);

        let negative_y = GeoTransform::from_tiepoint_and_scale(&tiepoint, &[2.0, -2.0, 0.0]);
        assert_eq!(negative_y.origin_x, 94.0);
        assert_eq!(negative_y.origin_y, 208.0);
        assert_eq!(negative_y.pixel_height, -2.0);
    }

    #[test]
    fn pixel_is_point_matrix_is_shifted_to_the_corner_like_gdal() {
        let matrix = [
            2.0, 1.0, 0.0, 100.0, 0.5, -2.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let area = GeoTransform::from_transformation_matrix_with_raster_type(
            &matrix,
            RasterType::PixelIsArea,
        );
        assert_eq!((area.origin_x, area.origin_y), (100.0, 200.0));

        let point = GeoTransform::from_transformation_matrix_with_raster_type(
            &matrix,
            RasterType::PixelIsPoint,
        );
        assert_eq!((point.origin_x, point.origin_y), (98.5, 200.75));
        assert_eq!(
            point.to_transformation_matrix_with_raster_type(RasterType::PixelIsPoint),
            matrix
        );
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
