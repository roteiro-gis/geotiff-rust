//! Shared GeoTIFF types used by both `geotiff-reader` and `geotiff-writer`.
//!
//! Provides the foundational GeoTIFF types: GeoKey directory, CRS info,
//! affine geo-transforms, and well-known tag/key constants.

pub mod crs;
pub mod geokeys;
pub mod metadata;
pub mod tags;
pub mod transform;

pub use crs::{CrsInfo, ModelType, RasterType};
pub use geokeys::{GeoKey, GeoKeyDirectory, GeoKeyValue};
pub use metadata::GeoMetadata;
pub use transform::GeoTransform;
