//! Coordinate Reference System extraction from GeoKeys.

use crate::geokeys::{self, GeoKeyDirectory};

/// GeoTIFF raster-space interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterType {
    PixelIsArea,
    PixelIsPoint,
    Unknown(u16),
}

impl RasterType {
    pub fn from_code(code: u16) -> Self {
        match code {
            1 => Self::PixelIsArea,
            2 => Self::PixelIsPoint,
            other => Self::Unknown(other),
        }
    }
}

/// Extracted CRS information from GeoKeys.
#[derive(Debug, Clone)]
pub struct CrsInfo {
    /// Model type: 1 = Projected, 2 = Geographic, 3 = Geocentric.
    pub model_type: u16,
    /// Raster type: 1 = PixelIsArea, 2 = PixelIsPoint.
    pub raster_type: u16,
    /// EPSG code for a projected CRS (from ProjectedCSTypeGeoKey).
    pub projected_epsg: Option<u16>,
    /// EPSG code for a geographic CRS (from GeographicTypeGeoKey).
    pub geographic_epsg: Option<u16>,
    /// Citation string for the projected CRS.
    pub projection_citation: Option<String>,
    /// Citation string for the geographic CRS.
    pub geographic_citation: Option<String>,
}

impl CrsInfo {
    /// Extract CRS information from a GeoKey directory.
    pub fn from_geokeys(geokeys: &GeoKeyDirectory) -> Self {
        Self {
            model_type: geokeys.get_short(geokeys::GT_MODEL_TYPE).unwrap_or(0),
            raster_type: geokeys.get_short(geokeys::GT_RASTER_TYPE).unwrap_or(1),
            projected_epsg: geokeys.get_short(geokeys::PROJECTED_CS_TYPE),
            geographic_epsg: geokeys.get_short(geokeys::GEOGRAPHIC_TYPE),
            projection_citation: geokeys.get_ascii(geokeys::PROJ_CITATION).map(String::from),
            geographic_citation: geokeys.get_ascii(geokeys::GEOG_CITATION).map(String::from),
        }
    }

    /// Returns the most specific EPSG code available.
    pub fn epsg(&self) -> Option<u32> {
        self.projected_epsg
            .or(self.geographic_epsg)
            .map(|e| e as u32)
    }

    /// Returns the GeoTIFF raster-space interpretation.
    pub fn raster_type_enum(&self) -> RasterType {
        RasterType::from_code(self.raster_type)
    }
}
