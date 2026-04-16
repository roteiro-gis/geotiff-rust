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

/// Horizontal CRS component extracted from GeoKeys.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HorizontalCrs {
    /// EPSG code for a projected CRS (ProjectedCRSGeoKey / 3072).
    pub projected_epsg: Option<u16>,
    /// EPSG code for a geodetic CRS (GeodeticCRSGeoKey / 2048).
    pub geodetic_epsg: Option<u16>,
    /// Citation string for the projected CRS.
    pub projection_citation: Option<String>,
    /// Citation string for the geodetic CRS.
    pub geodetic_citation: Option<String>,
}

/// Vertical CRS component extracted from GeoKeys.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VerticalCrs {
    /// EPSG code for a vertical CRS (VerticalCSTypeGeoKey / 4096).
    pub epsg: Option<u16>,
    /// Vertical datum code (VerticalDatumGeoKey / 4098).
    pub datum: Option<u16>,
    /// Vertical units code (VerticalUnitsGeoKey / 4099).
    pub units: Option<u16>,
    /// Citation string for the vertical CRS.
    pub citation: Option<String>,
}

/// Structured CRS interpretation synthesized from GeoKeys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrsKind {
    /// No usable CRS keys were present.
    Unspecified,
    /// A horizontal-only CRS, parameterized by GeoTIFF model type.
    Horizontal {
        model_type: ModelType,
        horizontal: HorizontalCrs,
    },
    /// A vertical-only CRS.
    Vertical(VerticalCrs),
    /// A compound CRS composed of horizontal and vertical components.
    Compound {
        model_type: ModelType,
        horizontal: HorizontalCrs,
        vertical: VerticalCrs,
    },
}

/// Extracted CRS information from GeoKeys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrsInfo {
    /// Model type: 1 = Projected, 2 = Geographic, 3 = Geocentric.
    pub model_type: u16,
    /// Raster type: 1 = PixelIsArea, 2 = PixelIsPoint.
    pub raster_type: u16,
    /// Horizontal CRS component derived from projected/geodetic keys.
    pub horizontal: Option<HorizontalCrs>,
    /// Vertical CRS component derived from vertical GeoKeys.
    pub vertical: Option<VerticalCrs>,
}

impl CrsInfo {
    /// Extract CRS information from a GeoKey directory.
    pub fn from_geokeys(geokeys: &GeoKeyDirectory) -> Self {
        let model_type = geokeys.get_short(geokeys::GT_MODEL_TYPE).unwrap_or(0);
        let horizontal = HorizontalCrs {
            projected_epsg: geokeys.get_short(geokeys::PROJECTED_CRS_TYPE),
            geodetic_epsg: geokeys.get_short(geokeys::GEODETIC_CRS_TYPE),
            projection_citation: geokeys.get_ascii(geokeys::PROJ_CITATION).map(String::from),
            geodetic_citation: geokeys
                .get_ascii(geokeys::GEODETIC_CITATION)
                .map(String::from),
        };
        let vertical = VerticalCrs {
            epsg: geokeys.get_short(geokeys::VERTICAL_CS_TYPE),
            datum: geokeys.get_short(geokeys::VERTICAL_DATUM),
            units: geokeys.get_short(geokeys::VERTICAL_UNITS),
            citation: geokeys
                .get_ascii(geokeys::VERTICAL_CITATION)
                .map(String::from),
        };

        Self {
            model_type,
            raster_type: geokeys.get_short(geokeys::GT_RASTER_TYPE).unwrap_or(1),
            horizontal: horizontal.is_present(model_type).then_some(horizontal),
            vertical: vertical.is_present().then_some(vertical),
        }
    }

    /// Returns the most specific EPSG code available.
    pub fn epsg(&self) -> Option<u32> {
        self.primary_horizontal_epsg()
            .or(self.vertical_epsg())
            .map(|epsg| epsg as u32)
    }

    /// Returns the GeoTIFF raster-space interpretation.
    pub fn raster_type_enum(&self) -> RasterType {
        RasterType::from_code(self.raster_type)
    }

    /// Returns the GeoTIFF model type.
    pub fn model_type_enum(&self) -> ModelType {
        ModelType::from_code(self.model_type)
    }

    /// Returns the structured horizontal/vertical CRS interpretation.
    pub fn crs_kind(&self) -> CrsKind {
        match (&self.horizontal, &self.vertical) {
            (Some(horizontal), Some(vertical)) => CrsKind::Compound {
                model_type: self.model_type_enum(),
                horizontal: horizontal.clone(),
                vertical: vertical.clone(),
            },
            (Some(horizontal), None) => CrsKind::Horizontal {
                model_type: self.model_type_enum(),
                horizontal: horizontal.clone(),
            },
            (None, Some(vertical)) => CrsKind::Vertical(vertical.clone()),
            (None, None) => CrsKind::Unspecified,
        }
    }

    /// Returns the horizontal CRS component, if present.
    pub fn horizontal(&self) -> Option<&HorizontalCrs> {
        self.horizontal.as_ref()
    }

    /// Returns the vertical CRS component, if present.
    pub fn vertical(&self) -> Option<&VerticalCrs> {
        self.vertical.as_ref()
    }

    /// Returns the projected CRS EPSG code, if present.
    pub fn projected_epsg(&self) -> Option<u16> {
        self.horizontal
            .as_ref()
            .and_then(|horizontal| horizontal.projected_epsg)
    }

    /// Returns the geodetic CRS EPSG code from GeoKey 2048, if present.
    pub fn geodetic_epsg(&self) -> Option<u16> {
        self.horizontal
            .as_ref()
            .and_then(|horizontal| horizontal.geodetic_epsg)
    }

    /// Returns the geographic CRS EPSG code when the model type is geographic.
    pub fn geographic_epsg(&self) -> Option<u16> {
        matches!(self.model_type_enum(), ModelType::Geographic)
            .then(|| self.geodetic_epsg())
            .flatten()
    }

    /// Returns the geocentric CRS EPSG code when the model type is geocentric.
    pub fn geocentric_epsg(&self) -> Option<u16> {
        matches!(self.model_type_enum(), ModelType::Geocentric)
            .then(|| self.geodetic_epsg())
            .flatten()
    }

    /// Returns the vertical CRS EPSG code, if present.
    pub fn vertical_epsg(&self) -> Option<u16> {
        self.vertical.as_ref().and_then(|vertical| vertical.epsg)
    }

    /// Returns the projected CRS citation, if present.
    pub fn projection_citation(&self) -> Option<&str> {
        self.horizontal
            .as_ref()
            .and_then(|horizontal| horizontal.projection_citation.as_deref())
    }

    /// Returns the geodetic CRS citation, if present.
    pub fn geodetic_citation(&self) -> Option<&str> {
        self.horizontal
            .as_ref()
            .and_then(|horizontal| horizontal.geodetic_citation.as_deref())
    }

    /// Returns the vertical CRS citation, if present.
    pub fn vertical_citation(&self) -> Option<&str> {
        self.vertical
            .as_ref()
            .and_then(|vertical| vertical.citation.as_deref())
    }

    /// Returns the vertical datum code, if present.
    pub fn vertical_datum(&self) -> Option<u16> {
        self.vertical.as_ref().and_then(|vertical| vertical.datum)
    }

    /// Returns the vertical units code, if present.
    pub fn vertical_units(&self) -> Option<u16> {
        self.vertical.as_ref().and_then(|vertical| vertical.units)
    }

    /// Populate a GeoKeyDirectory from this CRS info.
    pub fn apply_to_geokeys(&self, geokeys: &mut GeoKeyDirectory) {
        set_optional_short(
            geokeys,
            geokeys::GT_MODEL_TYPE,
            (self.model_type != 0).then_some(self.model_type),
        );
        set_optional_short(
            geokeys,
            geokeys::GT_RASTER_TYPE,
            (self.raster_type != 0).then_some(self.raster_type),
        );

        if let Some(horizontal) = &self.horizontal {
            set_optional_short(
                geokeys,
                geokeys::PROJECTED_CRS_TYPE,
                horizontal.projected_epsg,
            );
            set_optional_short(
                geokeys,
                geokeys::GEODETIC_CRS_TYPE,
                horizontal.geodetic_epsg,
            );
            set_optional_ascii(
                geokeys,
                geokeys::PROJ_CITATION,
                horizontal.projection_citation.as_deref(),
            );
            set_optional_ascii(
                geokeys,
                geokeys::GEODETIC_CITATION,
                horizontal.geodetic_citation.as_deref(),
            );
        } else {
            clear_horizontal_geokeys(geokeys);
        }

        if let Some(vertical) = &self.vertical {
            set_optional_short(geokeys, geokeys::VERTICAL_CS_TYPE, vertical.epsg);
            set_optional_short(geokeys, geokeys::VERTICAL_DATUM, vertical.datum);
            set_optional_short(geokeys, geokeys::VERTICAL_UNITS, vertical.units);
            set_optional_ascii(
                geokeys,
                geokeys::VERTICAL_CITATION,
                vertical.citation.as_deref(),
            );
        } else {
            clear_vertical_geokeys(geokeys);
        }
    }

    fn primary_horizontal_epsg(&self) -> Option<u16> {
        let horizontal = self.horizontal.as_ref()?;
        match self.model_type_enum() {
            ModelType::Projected => horizontal.projected_epsg.or(horizontal.geodetic_epsg),
            ModelType::Geographic | ModelType::Geocentric | ModelType::Unknown(_) => {
                horizontal.geodetic_epsg.or(horizontal.projected_epsg)
            }
        }
    }
}

impl HorizontalCrs {
    fn is_present(&self, model_type: u16) -> bool {
        model_type != 0
            || self.projected_epsg.is_some()
            || self.geodetic_epsg.is_some()
            || self.projection_citation.is_some()
            || self.geodetic_citation.is_some()
    }
}

impl VerticalCrs {
    fn is_present(&self) -> bool {
        self.epsg.is_some()
            || self.datum.is_some()
            || self.units.is_some()
            || self.citation.is_some()
    }
}

fn set_optional_short(geokeys: &mut GeoKeyDirectory, id: u16, value: Option<u16>) {
    if let Some(value) = value {
        geokeys.set(id, GeoKeyValue::Short(value));
    } else {
        geokeys.remove(id);
    }
}

fn set_optional_ascii(geokeys: &mut GeoKeyDirectory, id: u16, value: Option<&str>) {
    if let Some(value) = value {
        geokeys.set(id, GeoKeyValue::Ascii(value.to_string()));
    } else {
        geokeys.remove(id);
    }
}

fn clear_horizontal_geokeys(geokeys: &mut GeoKeyDirectory) {
    geokeys.remove(geokeys::PROJECTED_CRS_TYPE);
    geokeys.remove(geokeys::GEODETIC_CRS_TYPE);
    geokeys.remove(geokeys::PROJ_CITATION);
    geokeys.remove(geokeys::GEODETIC_CITATION);
}

fn clear_vertical_geokeys(geokeys: &mut GeoKeyDirectory) {
    geokeys.remove(geokeys::VERTICAL_CS_TYPE);
    geokeys.remove(geokeys::VERTICAL_DATUM);
    geokeys.remove(geokeys::VERTICAL_UNITS);
    geokeys.remove(geokeys::VERTICAL_CITATION);
}

#[cfg(test)]
mod tests {
    use super::{CrsInfo, CrsKind, GeoKeyDirectory, GeoKeyValue, ModelType, RasterType};
    use crate::geokeys;

    #[test]
    fn parses_geocentric_horizontal_crs() {
        let mut geokeys = GeoKeyDirectory::new();
        geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Geocentric.code()),
        );
        geokeys.set(
            geokeys::GT_RASTER_TYPE,
            GeoKeyValue::Short(RasterType::PixelIsArea.code()),
        );
        geokeys.set(geokeys::GEODETIC_CRS_TYPE, GeoKeyValue::Short(4978));

        let crs = CrsInfo::from_geokeys(&geokeys);
        assert_eq!(crs.geocentric_epsg(), Some(4978));
        assert_eq!(crs.geographic_epsg(), None);
        assert!(matches!(
            crs.crs_kind(),
            CrsKind::Horizontal {
                model_type: ModelType::Geocentric,
                ..
            }
        ));
    }

    #[test]
    fn parses_compound_projected_vertical_crs() {
        let mut geokeys = GeoKeyDirectory::new();
        geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Projected.code()),
        );
        geokeys.set(geokeys::PROJECTED_CRS_TYPE, GeoKeyValue::Short(32616));
        geokeys.set(geokeys::VERTICAL_CS_TYPE, GeoKeyValue::Short(5703));
        geokeys.set(
            geokeys::VERTICAL_CITATION,
            GeoKeyValue::Ascii("NAVD88 height".into()),
        );
        geokeys.set(geokeys::VERTICAL_UNITS, GeoKeyValue::Short(9001));

        let crs = CrsInfo::from_geokeys(&geokeys);
        assert_eq!(crs.projected_epsg(), Some(32616));
        assert_eq!(crs.vertical_epsg(), Some(5703));
        assert_eq!(crs.vertical_units(), Some(9001));
        assert_eq!(crs.vertical_citation(), Some("NAVD88 height"));
        assert!(matches!(
            crs.crs_kind(),
            CrsKind::Compound {
                model_type: ModelType::Projected,
                ..
            }
        ));
    }

    #[test]
    fn apply_to_geokeys_roundtrips_vertical_and_horizontal_components() {
        let original = CrsInfo {
            model_type: ModelType::Projected.code(),
            raster_type: RasterType::PixelIsPoint.code(),
            horizontal: Some(super::HorizontalCrs {
                projected_epsg: Some(26916),
                geodetic_epsg: Some(4269),
                projection_citation: Some("NAD83 / UTM zone 16N".into()),
                geodetic_citation: Some("NAD83".into()),
            }),
            vertical: Some(super::VerticalCrs {
                epsg: Some(5703),
                datum: Some(5103),
                units: Some(9001),
                citation: Some("NAVD88 height".into()),
            }),
        };

        let mut geokeys = GeoKeyDirectory::new();
        original.apply_to_geokeys(&mut geokeys);
        let roundtrip = CrsInfo::from_geokeys(&geokeys);

        assert_eq!(roundtrip, original);
        assert_eq!(roundtrip.epsg(), Some(26916));
    }
}
