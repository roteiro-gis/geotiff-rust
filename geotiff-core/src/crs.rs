//! Coordinate Reference System extraction from GeoKeys.

use crate::geokeys::{self, GeoKeyDirectory, GeoKeyValue};

/// GeoTIFF model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Projected,
    Geographic,
    Geocentric,
    Unknown(u16),
}

impl ModelType {
    pub fn from_code(code: u16) -> Self {
        match code {
            1 => Self::Projected,
            2 => Self::Geographic,
            3 => Self::Geocentric,
            other => Self::Unknown(other),
        }
    }

    pub fn code(&self) -> u16 {
        match self {
            Self::Projected => 1,
            Self::Geographic => 2,
            Self::Geocentric => 3,
            Self::Unknown(v) => *v,
        }
    }
}

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

    pub fn code(&self) -> u16 {
        match self {
            Self::PixelIsArea => 1,
            Self::PixelIsPoint => 2,
            Self::Unknown(v) => *v,
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
    /// EPSG code for a geocentric CRS (stored in the geodetic GeoKey 2048).
    pub geocentric_epsg: Option<u16>,
    /// Citation string for the projected CRS.
    pub projection_citation: Option<String>,
    /// Citation string for the geographic CRS.
    pub geographic_citation: Option<String>,
}

impl CrsInfo {
    /// Extract CRS information from a GeoKey directory.
    pub fn from_geokeys(geokeys: &GeoKeyDirectory) -> Self {
        let model_type = geokeys.get_short(geokeys::GT_MODEL_TYPE).unwrap_or(0);
        let geodetic_epsg = geokeys.get_short(geokeys::GEOGRAPHIC_TYPE);
        Self {
            model_type,
            raster_type: geokeys.get_short(geokeys::GT_RASTER_TYPE).unwrap_or(1),
            projected_epsg: geokeys.get_short(geokeys::PROJECTED_CS_TYPE),
            geographic_epsg: (model_type != ModelType::Geocentric.code())
                .then_some(geodetic_epsg)
                .flatten(),
            geocentric_epsg: (model_type == ModelType::Geocentric.code())
                .then_some(geodetic_epsg)
                .flatten(),
            projection_citation: geokeys.get_ascii(geokeys::PROJ_CITATION).map(String::from),
            geographic_citation: geokeys.get_ascii(geokeys::GEOG_CITATION).map(String::from),
        }
    }

    /// Returns the most specific EPSG code available.
    pub fn epsg(&self) -> Option<u32> {
        self.projected_epsg
            .or(self.geographic_epsg)
            .or(self.geocentric_epsg)
            .map(|e| e as u32)
    }

    /// Returns the GeoTIFF raster-space interpretation.
    pub fn raster_type_enum(&self) -> RasterType {
        RasterType::from_code(self.raster_type)
    }

    /// Returns the GeoTIFF model type.
    pub fn model_type_enum(&self) -> ModelType {
        ModelType::from_code(self.model_type)
    }

    /// Populate a GeoKeyDirectory from this CRS info.
    pub fn apply_to_geokeys(&self, geokeys: &mut GeoKeyDirectory) {
        geokeys.set(geokeys::GT_MODEL_TYPE, GeoKeyValue::Short(self.model_type));
        geokeys.set(
            geokeys::GT_RASTER_TYPE,
            GeoKeyValue::Short(self.raster_type),
        );
        if let Some(epsg) = self.projected_epsg {
            geokeys.set(geokeys::PROJECTED_CS_TYPE, GeoKeyValue::Short(epsg));
        } else {
            geokeys.remove(geokeys::PROJECTED_CS_TYPE);
        }
        if let Some(epsg) = self.geographic_epsg.or(self.geocentric_epsg) {
            geokeys.set(geokeys::GEOGRAPHIC_TYPE, GeoKeyValue::Short(epsg));
        } else {
            geokeys.remove(geokeys::GEOGRAPHIC_TYPE);
        }
        if let Some(ref citation) = self.projection_citation {
            geokeys.set(geokeys::PROJ_CITATION, GeoKeyValue::Ascii(citation.clone()));
        }
        if let Some(ref citation) = self.geographic_citation {
            geokeys.set(geokeys::GEOG_CITATION, GeoKeyValue::Ascii(citation.clone()));
        }
    }
}
