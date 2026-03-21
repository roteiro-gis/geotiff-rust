/// Parsed geospatial metadata from GeoKeys and model tags.
#[derive(Debug, Clone)]
pub struct GeoMetadata {
    /// EPSG code for the coordinate reference system, if present.
    pub epsg: Option<u32>,
    /// Model tiepoints: (I, J, K, X, Y, Z) tuples.
    pub tiepoints: Vec<[f64; 6]>,
    /// Pixel scale: (ScaleX, ScaleY, ScaleZ).
    pub pixel_scale: Option<[f64; 3]>,
    /// 4x4 model transformation matrix (row-major), if present.
    pub transformation: Option<[f64; 16]>,
    /// Nodata value as a string (parsed from GDAL_NODATA tag).
    pub nodata: Option<String>,
    /// Number of bands (samples per pixel).
    pub band_count: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Geographic bounds derived from the transform.
    pub geo_bounds: Option<[f64; 4]>,
}
