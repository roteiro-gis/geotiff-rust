//! Pure-Rust GeoTIFF and COG writer with compression, tiling, and overview support.
//!
//! Compression: None, LZW, Deflate, JPEG (feature), LERC, LERC+Deflate, LERC+Zstd, ZSTD (feature).
//!
//! # Example
//!
//! ```no_run
//! use geotiff_writer::GeoTiffBuilder;
//! use ndarray::Array2;
//!
//! let data = Array2::<f32>::zeros((100, 100));
//! GeoTiffBuilder::new(100, 100)
//!     .epsg(4326)
//!     .pixel_scale(0.01, 0.01)
//!     .origin(-180.0, 90.0)
//!     .nodata("-9999")
//!     .write_2d("output.tif", data.view())
//!     .unwrap();
//! ```

pub mod builder;
pub mod cog;
pub mod error;
pub mod sample;
pub mod tile_writer;

pub use builder::GeoTiffBuilder;
pub use cog::{CogBuilder, CogTileWriter, Resampling};
pub use error::{Error, Result};
pub use sample::{NumericSample, WriteSample};
pub use tile_writer::StreamingTileWriter;

// Re-export core types for convenience
pub use geotiff_core::{
    CrsInfo, GeoKeyDirectory, GeoKeyValue, GeoTransform, ModelType, RasterType,
};
pub use tiff_core::{
    Compression, LercAdditionalCompression, PhotometricInterpretation, PlanarConfiguration,
    Predictor,
};
pub use tiff_writer::{JpegOptions, LercOptions, TiffVariant};
