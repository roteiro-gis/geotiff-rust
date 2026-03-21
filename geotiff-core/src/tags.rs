//! Well-known GeoTIFF TIFF tag codes.

/// ModelPixelScaleTag (33550) — pixel size in map units.
pub const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
/// ModelTiepointTag (33922) — raster-to-model tiepoint pairs.
pub const TAG_MODEL_TIEPOINT: u16 = 33922;
/// ModelTransformationTag (34264) — 4x4 transformation matrix.
pub const TAG_MODEL_TRANSFORMATION: u16 = 34264;
/// GeoKeyDirectoryTag (34735) — GeoKey directory as SHORT array.
pub const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
/// GeoDoubleParamsTag (34736) — double parameters for GeoKeys.
pub const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
/// GeoAsciiParamsTag (34737) — ASCII parameters for GeoKeys.
pub const TAG_GEO_ASCII_PARAMS: u16 = 34737;
/// GDAL NoData tag (42113).
pub const TAG_GDAL_NODATA: u16 = 42113;
/// NewSubfileType (254) — used for overview identification.
pub const TAG_NEW_SUBFILE_TYPE: u16 = 254;
/// SubfileType (255) — legacy subfile type.
pub const TAG_SUBFILE_TYPE: u16 = 255;
