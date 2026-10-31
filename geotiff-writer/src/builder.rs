//! GeoTiffBuilder: fluent API for constructing GeoTIFF files.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use geotiff_core::geokeys::{self, GeoKeyDirectory, GeoKeyValue};
use geotiff_core::tags;
use geotiff_core::transform::GeoTransform;
use geotiff_core::{CrsInfo, ModelType, RasterType};
use ndarray::{ArrayView2, ArrayView3};
use tiff_core::{
    ColorMap, Compression, ExtraSample, InkSet, PhotometricInterpretation, PlanarConfiguration,
    Predictor, Tag, TagValue, YCbCrPositioning,
};
use tiff_writer::{DataLayout, ImageBuilder, JpegOptions, TiffVariant, TiffWriter, WriteOptions};

use crate::error::{Error, Result};
use crate::sample::{nodata_fill_or_zero, NumericSample, WriteSample};
use crate::tile_writer::StreamingTileWriter;

pub(crate) fn checked_sample_count(dimensions: &[usize], context: &str) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(1usize, |sample_count, &dimension| {
            sample_count
                .checked_mul(dimension)
                .ok_or_else(|| Error::Other(format!("{context} sample count overflows usize")))
        })
}

fn checked_builder_dimension(dimension: u32, name: &str) -> Result<usize> {
    usize::try_from(dimension)
        .map_err(|_| Error::Other(format!("{name} dimension exceeds usize::MAX")))
}

fn dimension_matches(actual: usize, expected: u32) -> bool {
    matches!(u32::try_from(actual), Ok(actual) if actual == expected)
}

/// Builder for constructing GeoTIFF files with metadata.
#[derive(Debug, Clone)]
pub struct GeoTiffBuilder {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bands: u32,
    pub(crate) geokeys: GeoKeyDirectory,
    pub(crate) pixel_scale: Option<[f64; 3]>,
    pub(crate) tiepoint: Option<[f64; 6]>,
    pub(crate) tiepoint_is_origin: bool,
    pub(crate) affine_transform: Option<GeoTransform>,
    pub(crate) transformation_matrix: Option<[f64; 16]>,
    pub(crate) nodata: Option<String>,
    pub(crate) compression: Compression,
    pub(crate) predictor: Predictor,
    pub(crate) lerc_options: Option<tiff_writer::LercOptions>,
    pub(crate) jpeg_options: Option<JpegOptions>,
    pub(crate) deflate_level: Option<u32>,
    pub(crate) sparse: bool,
    pub(crate) extra_samples: Vec<ExtraSample>,
    pub(crate) color_map: Option<ColorMap>,
    pub(crate) ink_set: Option<InkSet>,
    pub(crate) ycbcr_subsampling: Option<[u16; 2]>,
    pub(crate) ycbcr_positioning: Option<YCbCrPositioning>,
    pub(crate) planar_configuration: PlanarConfiguration,
    pub(crate) tile_width: Option<u32>,
    pub(crate) tile_height: Option<u32>,
    pub(crate) photometric: PhotometricInterpretation,
    pub(crate) tiff_variant: TiffVariant,
}

impl GeoTiffBuilder {
    /// Create a new builder for a raster of the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bands: 1,
            geokeys: GeoKeyDirectory::new(),
            pixel_scale: None,
            tiepoint: None,
            tiepoint_is_origin: false,
            affine_transform: None,
            transformation_matrix: None,
            nodata: None,
            compression: Compression::None,
            predictor: Predictor::None,
            lerc_options: None,
            jpeg_options: None,
            deflate_level: None,
            sparse: false,
            extra_samples: Vec::new(),
            color_map: None,
            ink_set: None,
            ycbcr_subsampling: None,
            ycbcr_positioning: None,
            planar_configuration: PlanarConfiguration::Chunky,
            tile_width: None,
            tile_height: None,
            photometric: PhotometricInterpretation::MinIsBlack,
            tiff_variant: TiffVariant::Auto,
        }
    }

    /// Set the number of bands (samples per pixel). Default: 1.
    pub fn bands(mut self, bands: u32) -> Self {
        self.bands = bands;
        self
    }

    /// Set CRS by EPSG code.
    ///
    /// If a model type was already set explicitly, it is preserved. Otherwise
    /// this uses a projected-vs-geodetic heuristic and recognizes EPSG:4978
    /// as geocentric.
    pub fn epsg(mut self, code: u16) -> Self {
        let model_type = self
            .geokeys
            .get_short(geokeys::GT_MODEL_TYPE)
            .map(ModelType::from_code)
            .unwrap_or_else(|| {
                if code == 4978 {
                    ModelType::Geocentric
                } else if (4000..5000).contains(&code) {
                    ModelType::Geographic
                } else {
                    ModelType::Projected
                }
            });

        match model_type {
            ModelType::Projected => self = self.projected_epsg(code),
            ModelType::Geographic => self = self.geographic_epsg(code),
            ModelType::Geocentric | ModelType::Unknown(_) => self = self.geocentric_epsg(code),
        }
        self
    }

    /// Apply a structured CRS model directly.
    pub fn crs(mut self, crs: CrsInfo) -> Self {
        crs.apply_to_geokeys(&mut self.geokeys);
        self
    }

    /// Set a projected CRS by EPSG code.
    pub fn projected_epsg(mut self, code: u16) -> Self {
        self.geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Projected.code()),
        );
        self.geokeys
            .set(geokeys::PROJECTED_CRS_TYPE, GeoKeyValue::Short(code));
        self.geokeys.remove(geokeys::GEODETIC_CRS_TYPE);
        self
    }

    /// Set a geographic CRS by EPSG code.
    pub fn geographic_epsg(mut self, code: u16) -> Self {
        self.geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Geographic.code()),
        );
        self.geokeys.remove(geokeys::PROJECTED_CRS_TYPE);
        self.geokeys
            .set(geokeys::GEODETIC_CRS_TYPE, GeoKeyValue::Short(code));
        self
    }

    /// Set a geocentric CRS by EPSG code.
    pub fn geocentric_epsg(mut self, code: u16) -> Self {
        self.geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Geocentric.code()),
        );
        self.geokeys.remove(geokeys::PROJECTED_CRS_TYPE);
        self.geokeys
            .set(geokeys::GEODETIC_CRS_TYPE, GeoKeyValue::Short(code));
        self
    }

    /// Set a vertical CRS by EPSG code. When combined with a horizontal CRS
    /// this forms a compound CRS.
    pub fn vertical_epsg(mut self, code: u16) -> Self {
        self.geokeys
            .set(geokeys::VERTICAL_CS_TYPE, GeoKeyValue::Short(code));
        self
    }

    /// Set the vertical datum code.
    pub fn vertical_datum(mut self, code: u16) -> Self {
        self.geokeys
            .set(geokeys::VERTICAL_DATUM, GeoKeyValue::Short(code));
        self
    }

    /// Set the vertical units code.
    pub fn vertical_units(mut self, code: u16) -> Self {
        self.geokeys
            .set(geokeys::VERTICAL_UNITS, GeoKeyValue::Short(code));
        self
    }

    /// Set the vertical CRS citation string.
    pub fn vertical_citation(mut self, citation: &str) -> Self {
        self.geokeys.set(
            geokeys::VERTICAL_CITATION,
            GeoKeyValue::Ascii(citation.to_string()),
        );
        self
    }

    /// Set the model type explicitly.
    pub fn model_type(mut self, mt: ModelType) -> Self {
        self.geokeys
            .set(geokeys::GT_MODEL_TYPE, GeoKeyValue::Short(mt.code()));
        self
    }

    /// Set the raster type (PixelIsArea or PixelIsPoint).
    pub fn raster_type(mut self, rt: RasterType) -> Self {
        self.geokeys
            .set(geokeys::GT_RASTER_TYPE, GeoKeyValue::Short(rt.code()));
        self
    }

    /// Set an arbitrary GeoKey.
    pub fn geokey(mut self, id: u16, value: GeoKeyValue) -> Self {
        self.geokeys.set(id, value);
        self
    }

    /// Set pixel scale (X, Y).
    pub fn pixel_scale(mut self, scale_x: f64, scale_y: f64) -> Self {
        self.pixel_scale = Some([scale_x, scale_y, 0.0]);
        self.affine_transform = None;
        self.transformation_matrix = None;
        self
    }

    /// Set the map origin (upper-left corner X, Y).
    pub fn origin(mut self, x: f64, y: f64) -> Self {
        self.tiepoint = Some([0.0, 0.0, 0.0, x, y, 0.0]);
        self.tiepoint_is_origin = true;
        self.affine_transform = None;
        self.transformation_matrix = None;
        self
    }

    /// Set an explicit tiepoint (I, J, K, X, Y, Z).
    pub fn tiepoint(mut self, tiepoint: [f64; 6]) -> Self {
        self.tiepoint = Some(tiepoint);
        self.tiepoint_is_origin = false;
        self.affine_transform = None;
        self.transformation_matrix = None;
        self
    }

    /// Set a full affine transform. Takes precedence over pixel_scale + origin.
    ///
    /// The transform is corner-based, as returned by
    /// `GeoTiffFile::transform`. It's converted for the raster type when the
    /// tags are written, so [`Self::raster_type`] may be called before or after.
    pub fn transform(mut self, transform: GeoTransform) -> Self {
        self.affine_transform = Some(transform);
        self.tiepoint = None;
        self.pixel_scale = None;
        self.tiepoint_is_origin = false;
        self.transformation_matrix = None;
        self
    }

    /// Set a 4x4 model transformation matrix.
    pub fn transformation_matrix(mut self, matrix: [f64; 16]) -> Self {
        self.transformation_matrix = Some(matrix);
        self.affine_transform = None;
        self.tiepoint = None;
        self.pixel_scale = None;
        self.tiepoint_is_origin = false;
        self
    }

    /// Set the NoData value (written to GDAL_NODATA tag 42113).
    pub fn nodata(mut self, value: &str) -> Self {
        self.nodata = Some(value.to_string());
        self
    }

    /// Set compression algorithm.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        if !matches!(compression, Compression::Lerc) {
            self.lerc_options = None;
        }
        if !matches!(compression, Compression::Jpeg) {
            self.jpeg_options = None;
        }
        if matches!(compression, Compression::Lerc | Compression::Jpeg) {
            self.predictor = Predictor::None;
        }
        self
    }

    /// Set the Deflate compression level (0-9).
    ///
    /// Applies to `Compression::Deflate` output. The additional Deflate layer
    /// of `LERC+Deflate` always uses the codec default level.
    pub fn deflate_level(mut self, level: u32) -> Self {
        self.deflate_level = Some(level);
        self
    }

    /// Skip all-zero blocks on write (GDAL `SPARSE_OK` semantics).
    ///
    /// Sparse blocks are recorded with zero offsets and byte counts and read
    /// back as zero fill. Negative float zero is treated as zero.
    pub fn sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Set predictor (requires compression != None).
    pub fn predictor(mut self, predictor: Predictor) -> Self {
        self.predictor = predictor;
        self
    }

    /// Set LERC compression with the given options.
    ///
    /// This sets `compression = Lerc` and `predictor = None` (LERC performs
    /// its own quantization and does not use TIFF predictors).
    pub fn lerc_options(mut self, options: tiff_writer::LercOptions) -> Self {
        self.compression = Compression::Lerc;
        self.predictor = Predictor::None;
        self.lerc_options = Some(options);
        self.jpeg_options = None;
        self
    }

    /// Set JPEG compression with the given options.
    ///
    /// This sets `compression = Jpeg` and `predictor = None` (JPEG uses its
    /// own transform and entropy coding pipeline rather than TIFF predictors).
    ///
    /// Multi-band JPEG requires `planar_configuration(Planar)` so each encoded
    /// strip/tile is a single grayscale component.
    pub fn jpeg_options(mut self, options: JpegOptions) -> Self {
        self.compression = Compression::Jpeg;
        self.predictor = Predictor::None;
        self.jpeg_options = Some(options);
        self.lerc_options = None;
        self
    }

    /// Set planar configuration for multi-band output.
    pub fn planar_configuration(mut self, planar_configuration: PlanarConfiguration) -> Self {
        self.planar_configuration = planar_configuration;
        self
    }

    /// Enable tiling with given tile dimensions (must be multiples of 16).
    pub fn tile_size(mut self, tile_width: u32, tile_height: u32) -> Self {
        self.tile_width = Some(tile_width);
        self.tile_height = Some(tile_height);
        self
    }

    /// Set photometric interpretation.
    pub fn photometric(mut self, p: PhotometricInterpretation) -> Self {
        self.photometric = p;
        self
    }

    /// Set TIFF ExtraSamples semantics for channels beyond the base color model.
    pub fn extra_samples(mut self, extra_samples: Vec<ExtraSample>) -> Self {
        self.extra_samples = extra_samples;
        self
    }

    /// Set a palette ColorMap for `PhotometricInterpretation::Palette`.
    pub fn color_map(mut self, color_map: ColorMap) -> Self {
        self.color_map = Some(color_map);
        self
    }

    /// Set the InkSet tag for separated photometric data.
    pub fn ink_set(mut self, ink_set: InkSet) -> Self {
        self.ink_set = Some(ink_set);
        self
    }

    /// Set TIFF YCbCr chroma subsampling factors.
    pub fn ycbcr_subsampling(mut self, subsampling: [u16; 2]) -> Self {
        self.ycbcr_subsampling = Some(subsampling);
        self
    }

    /// Set TIFF YCbCr sample positioning.
    pub fn ycbcr_positioning(mut self, positioning: YCbCrPositioning) -> Self {
        self.ycbcr_positioning = Some(positioning);
        self
    }

    /// Select the TIFF container variant for emitted output.
    pub fn tiff_variant(mut self, variant: TiffVariant) -> Self {
        self.tiff_variant = variant;
        self
    }

    /// Build the GeoTIFF extra tags from the current metadata.
    pub(crate) fn build_extra_tags(&self) -> Result<Vec<Tag>> {
        self.validate_georeferencing()?;
        let mut extra = Vec::new();
        let writes_georeferencing = self.transformation_matrix.is_some()
            || self.affine_transform.is_some()
            || self.pixel_scale.is_some()
            || self.tiepoint.is_some();

        // Georeferencing tags
        if let Some(matrix) = &self.transformation_matrix {
            extra.push(Tag::new(
                tags::TAG_MODEL_TRANSFORMATION,
                TagValue::Double(matrix.to_vec()),
            ));
        } else if let Some(transform) = self.affine_transform {
            let raster_type = self.current_raster_type();
            if let Some((tp, ps)) = transform.to_tiepoint_and_scale_with_raster_type(raster_type) {
                extra.push(Tag::new(
                    tags::TAG_MODEL_PIXEL_SCALE,
                    TagValue::Double(ps.to_vec()),
                ));
                extra.push(Tag::new(
                    tags::TAG_MODEL_TIEPOINT,
                    TagValue::Double(tp.to_vec()),
                ));
            } else {
                let matrix = transform.to_transformation_matrix_with_raster_type(raster_type);
                extra.push(Tag::new(
                    tags::TAG_MODEL_TRANSFORMATION,
                    TagValue::Double(matrix.to_vec()),
                ));
            }
        } else {
            if let Some(ps) = &self.pixel_scale {
                extra.push(Tag::new(
                    tags::TAG_MODEL_PIXEL_SCALE,
                    TagValue::Double(ps.to_vec()),
                ));
            }
            if let Some(tp) = &self.tiepoint {
                let tiepoint = if self.tiepoint_is_origin {
                    self.origin_tiepoint_for_raster_type(tp)
                } else {
                    *tp
                };
                extra.push(Tag::new(
                    tags::TAG_MODEL_TIEPOINT,
                    TagValue::Double(tiepoint.to_vec()),
                ));
            }
        }

        // GeoKey directory
        if writes_georeferencing || !self.geokeys.keys.is_empty() {
            let (directory, double_params, ascii_params) = self.geokeys.serialize()?;
            extra.push(Tag::new(
                tags::TAG_GEO_KEY_DIRECTORY,
                TagValue::Short(directory),
            ));
            if !double_params.is_empty() {
                extra.push(Tag::new(
                    tags::TAG_GEO_DOUBLE_PARAMS,
                    TagValue::Double(double_params),
                ));
            }
            if !ascii_params.is_empty() {
                extra.push(Tag::new(
                    tags::TAG_GEO_ASCII_PARAMS,
                    TagValue::Ascii(ascii_params),
                ));
            }
        }

        // NoData
        if let Some(ref nd) = self.nodata {
            extra.push(Tag::new(tags::TAG_GDAL_NODATA, TagValue::Ascii(nd.clone())));
        }

        Ok(extra)
    }

    fn validate_georeferencing(&self) -> Result<()> {
        if let Some(matrix) = self.transformation_matrix {
            if !matrix.iter().all(|value| value.is_finite()) {
                return Err(Error::InvalidConfig(
                    "model transformation matrix values must be finite".into(),
                ));
            }
            let transform = GeoTransform::from_transformation_matrix(&matrix);
            if transform
                .geo_to_pixel(transform.origin_x, transform.origin_y)
                .is_none()
            {
                return Err(Error::InvalidConfig(
                    "model transformation matrix must contain an invertible 2D affine transform"
                        .into(),
                ));
            }
        }
        if let Some(transform) = self.affine_transform {
            let values = [
                transform.origin_x,
                transform.pixel_width,
                transform.skew_x,
                transform.origin_y,
                transform.skew_y,
                transform.pixel_height,
            ];
            if !values.iter().all(|value| value.is_finite())
                || transform
                    .geo_to_pixel(transform.origin_x, transform.origin_y)
                    .is_none()
            {
                return Err(Error::InvalidConfig(
                    "affine transform must be finite and invertible".into(),
                ));
            }
        }
        if let Some(pixel_scale) = self.pixel_scale {
            if !pixel_scale.iter().all(|value| value.is_finite())
                || pixel_scale[0] <= 0.0
                || pixel_scale[1] <= 0.0
                || pixel_scale[2] < 0.0
            {
                return Err(Error::InvalidConfig(
                    "model pixel scale must contain finite positive X/Y and non-negative Z values"
                        .into(),
                ));
            }
            if self.tiepoint.is_none() {
                return Err(Error::InvalidConfig(
                    "model pixel scale requires a model tiepoint or origin".into(),
                ));
            }
        }
        if self
            .tiepoint
            .is_some_and(|tiepoint| !tiepoint.iter().all(|value| value.is_finite()))
        {
            return Err(Error::InvalidConfig(
                "model tiepoint values must be finite".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn with_overview_georeferencing(&self, level: u32) -> Self {
        let factor = level as f64;
        if factor == 1.0 {
            return self.clone();
        }

        let mut builder = self.clone();
        if let Some(matrix) = self.transformation_matrix {
            builder.transformation_matrix = Some(scale_transformation_matrix(matrix, factor));
            builder.affine_transform = None;
            builder.pixel_scale = None;
            builder.tiepoint = None;
            builder.tiepoint_is_origin = false;
        } else if let Some(transform) = self.affine_transform {
            builder.affine_transform = Some(scale_transform_for_overview(transform, factor));
        } else if let (Some(tiepoint), Some(pixel_scale)) = (self.tiepoint, self.pixel_scale) {
            let transform = if self.tiepoint_is_origin {
                GeoTransform::from_origin_and_pixel_size(
                    tiepoint[3],
                    tiepoint[4],
                    pixel_scale[0],
                    -pixel_scale[1],
                )
            } else {
                GeoTransform::from_tiepoint_and_scale_with_raster_type(
                    &tiepoint,
                    &pixel_scale,
                    self.current_raster_type(),
                )
            };
            builder.affine_transform = Some(scale_transform_for_overview(transform, factor));
            builder.transformation_matrix = None;
            builder.pixel_scale = None;
            builder.tiepoint = None;
            builder.tiepoint_is_origin = false;
        } else if let Some(mut pixel_scale) = self.pixel_scale {
            pixel_scale[0] *= factor;
            pixel_scale[1] *= factor;
            builder.pixel_scale = Some(pixel_scale);
        }

        builder
    }

    fn current_raster_type(&self) -> RasterType {
        self.geokeys
            .get_short(geokeys::GT_RASTER_TYPE)
            .map(RasterType::from_code)
            .unwrap_or(RasterType::PixelIsArea)
    }

    fn origin_tiepoint_for_raster_type(&self, tiepoint: &[f64; 6]) -> [f64; 6] {
        let Some(pixel_scale) = self.pixel_scale else {
            return *tiepoint;
        };
        let transform = GeoTransform::from_origin_and_pixel_size(
            tiepoint[3],
            tiepoint[4],
            pixel_scale[0],
            -pixel_scale[1],
        );
        transform
            .to_tiepoint_and_scale_with_raster_type(self.current_raster_type())
            .map(|(tiepoint, _)| tiepoint)
            .unwrap_or(*tiepoint)
    }

    /// Build an ImageBuilder from this GeoTiffBuilder for a given sample type.
    pub(crate) fn to_image_builder<T: WriteSample>(&self) -> Result<ImageBuilder> {
        self.to_sized_image_builder::<T>(self.width, self.height)
    }

    /// Build an ImageBuilder with overridden raster dimensions while
    /// preserving codec, color-model, layout, and GeoTIFF metadata settings.
    pub(crate) fn to_sized_image_builder<T: WriteSample>(
        &self,
        width: u32,
        height: u32,
    ) -> Result<ImageBuilder> {
        let samples_per_pixel = u16::try_from(self.bands).map_err(|_| {
            Error::InvalidConfig(format!(
                "band count {} exceeds TIFF SamplesPerPixel limit {}",
                self.bands,
                u16::MAX
            ))
        })?;
        let mut ib = ImageBuilder::new(width, height)
            .sample_type::<T>()
            .samples_per_pixel(samples_per_pixel)
            .compression(self.compression)
            .predictor(self.predictor)
            .planar_configuration(self.planar_configuration)
            .photometric(self.photometric);

        if !self.extra_samples.is_empty() {
            ib = ib.extra_samples(self.extra_samples.clone());
        }
        if let Some(color_map) = &self.color_map {
            ib = ib.color_map(color_map.clone());
        }
        if let Some(ink_set) = self.ink_set {
            ib = ib.ink_set(ink_set);
        }
        if let Some(subsampling) = self.ycbcr_subsampling {
            ib = ib.ycbcr_subsampling(subsampling);
        }
        if let Some(positioning) = self.ycbcr_positioning {
            ib = ib.ycbcr_positioning(positioning);
        }

        if let Some(opts) = self.lerc_options {
            ib = ib.lerc_options(opts);
        }
        if let Some(opts) = self.jpeg_options {
            ib = ib.jpeg_options(opts);
        }
        if let Some(level) = self.deflate_level {
            ib = ib.deflate_level(level);
        }

        if let (Some(tw), Some(th)) = (self.tile_width, self.tile_height) {
            ib = ib.tiles(tw, th);
        }

        for tag in self.build_extra_tags()? {
            ib = ib.tag(tag);
        }

        Ok(ib)
    }

    fn expected_2d_sample_count(&self) -> Result<usize> {
        let height = checked_builder_dimension(self.height, "height")?;
        let width = checked_builder_dimension(self.width, "width")?;
        checked_sample_count(&[height, width], "expected raster")
    }

    fn expected_3d_sample_count(&self) -> Result<usize> {
        let height = checked_builder_dimension(self.height, "height")?;
        let width = checked_builder_dimension(self.width, "width")?;
        let bands = checked_builder_dimension(self.bands, "band")?;
        checked_sample_count(&[height, width, bands], "expected raster")
    }

    pub(crate) fn validate_2d_data_shape(&self, height: usize, width: usize) -> Result<()> {
        if !dimension_matches(width, self.width) || !dimension_matches(height, self.height) {
            return Err(Error::DataSizeMismatch {
                expected: self.expected_2d_sample_count()?,
                actual: checked_sample_count(&[height, width], "actual raster")?,
            });
        }

        Ok(())
    }

    pub(crate) fn validate_3d_data_shape(
        &self,
        height: usize,
        width: usize,
        bands: usize,
    ) -> Result<()> {
        if !dimension_matches(width, self.width)
            || !dimension_matches(height, self.height)
            || !dimension_matches(bands, self.bands)
        {
            return Err(Error::DataSizeMismatch {
                expected: self.expected_3d_sample_count()?,
                actual: checked_sample_count(&[height, width, bands], "actual raster")?,
            });
        }

        Ok(())
    }

    // ---- Write methods ----

    /// Write a single-band 2D array to a file path.
    pub fn write_2d<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_2d_to(writer, data)
    }

    /// Write a single-band 2D array to any Write+Seek target.
    pub fn write_2d_to<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let (height, width) = data.dim();
        self.validate_2d_data_shape(height, width)?;

        let ib = self.to_image_builder::<T>()?;
        let block_count = ib.checked_block_count()?;
        let layout = ib.data_layout();
        let fill_value = nodata_fill_or_zero::<T>(&self.nodata)?;
        let mut writer = TiffWriter::new(
            sink,
            WriteOptions {
                byte_order: tiff_core::ByteOrder::LittleEndian,
                variant: self.tiff_variant,
            },
        )?;
        let handle = writer.add_image(ib)?;

        for block_idx in 0..block_count {
            let samples = self.extract_block_2d(&data, block_idx, layout, fill_value);
            if self.sparse && samples.iter().all(|&value| value == T::zero()) {
                writer.write_block_sparse(&handle, block_idx)?;
            } else {
                writer.write_block(&handle, block_idx, &samples)?;
            }
        }

        writer.finish()?;
        Ok(())
    }

    /// Write a multi-band 3D array [rows, cols, bands] to a file path.
    pub fn write_3d<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView3<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_3d_to(writer, data)
    }

    /// Write a multi-band 3D array to any Write+Seek target.
    pub fn write_3d_to<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView3<T>,
    ) -> Result<()> {
        let (height, width, bands) = data.dim();
        self.validate_3d_data_shape(height, width, bands)?;

        let ib = self.to_image_builder::<T>()?;
        let block_count = ib.checked_block_count()?;
        let layout = ib.data_layout();
        let fill_value = nodata_fill_or_zero::<T>(&self.nodata)?;
        let mut writer = TiffWriter::new(
            sink,
            WriteOptions {
                byte_order: tiff_core::ByteOrder::LittleEndian,
                variant: self.tiff_variant,
            },
        )?;
        let handle = writer.add_image(ib)?;

        for block_idx in 0..block_count {
            let samples = self.extract_block_3d(&data, block_idx, layout, fill_value);
            if self.sparse && samples.iter().all(|&value| value == T::zero()) {
                writer.write_block_sparse(&handle, block_idx)?;
            } else {
                writer.write_block(&handle, block_idx, &samples)?;
            }
        }

        writer.finish()?;
        Ok(())
    }

    /// Create a streaming tile writer for incremental writes.
    pub fn tile_writer<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
    ) -> Result<StreamingTileWriter<T, W>> {
        StreamingTileWriter::new(self.clone(), sink)
    }

    /// Create a streaming tile writer that writes to a file path.
    pub fn tile_writer_file<T: NumericSample, P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<StreamingTileWriter<T, BufWriter<File>>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.tile_writer(writer)
    }

    fn extract_block_2d<T: NumericSample>(
        &self,
        data: &ArrayView2<T>,
        block_idx: usize,
        layout: DataLayout,
        fill_value: T,
    ) -> Vec<T> {
        if let DataLayout::Tiles { width, height } = layout {
            let tw = width as usize;
            let th = height as usize;
            let tiles_across = (self.width as usize).div_ceil(tw);
            let tile_row = block_idx / tiles_across;
            let tile_col = block_idx % tiles_across;
            let start_row = tile_row * th;
            let start_col = tile_col * tw;
            let rows = th.min((self.height as usize).saturating_sub(start_row));
            let cols = tw.min((self.width as usize).saturating_sub(start_col));

            let mut tile_data = vec![fill_value; tw * th];
            crate::raster_copy::copy_2d_region_into(
                data,
                crate::raster_copy::Region {
                    row_start: start_row,
                    col_start: start_col,
                    rows,
                    cols,
                },
                &mut tile_data,
                tw,
            );
            tile_data
        } else {
            let rps = strips_rows_per_strip(layout);
            let start_row = block_idx * rps;
            let end_row = ((block_idx + 1) * rps).min(self.height as usize);
            let w = self.width as usize;

            let mut samples = vec![fill_value; (end_row - start_row) * w];
            crate::raster_copy::copy_2d_region_into(
                data,
                crate::raster_copy::Region {
                    row_start: start_row,
                    col_start: 0,
                    rows: end_row - start_row,
                    cols: w,
                },
                &mut samples,
                w,
            );
            samples
        }
    }

    fn extract_block_3d<T: NumericSample>(
        &self,
        data: &ArrayView3<T>,
        block_idx: usize,
        layout: DataLayout,
        fill_value: T,
    ) -> Vec<T> {
        let bands = self.bands as usize;

        if let DataLayout::Tiles { width, height } = layout {
            let tw = width as usize;
            let th = height as usize;
            let tiles_across = (self.width as usize).div_ceil(tw);
            let tiles_down = (self.height as usize).div_ceil(th);
            let tiles_per_plane = tiles_across * tiles_down;
            let (plane, plane_block_index) =
                self.plane_and_block_index(block_idx, tiles_per_plane, bands);
            let tile_row = plane_block_index / tiles_across;
            let tile_col = plane_block_index % tiles_across;
            let start_row = tile_row * th;
            let start_col = tile_col * tw;

            let rows = th.min((self.height as usize).saturating_sub(start_row));
            let cols = tw.min((self.width as usize).saturating_sub(start_col));
            if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
                let mut tile_data = vec![fill_value; tw * th];
                crate::raster_copy::copy_3d_band_region_into(
                    data,
                    plane,
                    crate::raster_copy::Region {
                        row_start: start_row,
                        col_start: start_col,
                        rows,
                        cols,
                    },
                    &mut tile_data,
                    tw,
                );
                tile_data
            } else {
                let mut tile_data = vec![fill_value; tw * th * bands];
                crate::raster_copy::copy_3d_chunky_region_into(
                    data,
                    crate::raster_copy::Region {
                        row_start: start_row,
                        col_start: start_col,
                        rows,
                        cols,
                    },
                    &mut tile_data,
                    tw * bands,
                );
                tile_data
            }
        } else {
            let rps = strips_rows_per_strip(layout);
            let strips_per_plane = (self.height as usize).div_ceil(rps);
            let (plane, plane_block_index) =
                self.plane_and_block_index(block_idx, strips_per_plane, bands);
            let start_row = plane_block_index * rps;
            let end_row = ((plane_block_index + 1) * rps).min(self.height as usize);
            let w = self.width as usize;

            let rows = end_row - start_row;
            if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
                let mut samples = vec![fill_value; rows * w];
                crate::raster_copy::copy_3d_band_region_into(
                    data,
                    plane,
                    crate::raster_copy::Region {
                        row_start: start_row,
                        col_start: 0,
                        rows,
                        cols: w,
                    },
                    &mut samples,
                    w,
                );
                samples
            } else {
                let mut samples = vec![fill_value; rows * w * bands];
                crate::raster_copy::copy_3d_chunky_region_into(
                    data,
                    crate::raster_copy::Region {
                        row_start: start_row,
                        col_start: 0,
                        rows,
                        cols: w,
                    },
                    &mut samples,
                    w * bands,
                );
                samples
            }
        }
    }

    fn plane_and_block_index(
        &self,
        block_idx: usize,
        blocks_per_plane: usize,
        bands: usize,
    ) -> (usize, usize) {
        if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
            let plane = (block_idx / blocks_per_plane).min(bands.saturating_sub(1));
            (plane, block_idx % blocks_per_plane)
        } else {
            (0, block_idx)
        }
    }
}

fn strips_rows_per_strip(layout: DataLayout) -> usize {
    match layout {
        DataLayout::Strips { rows_per_strip } => (rows_per_strip as usize).max(1),
        DataLayout::Tiles { .. } => 1,
    }
}

fn scale_transform_for_overview(transform: GeoTransform, factor: f64) -> GeoTransform {
    GeoTransform {
        origin_x: transform.origin_x,
        pixel_width: transform.pixel_width * factor,
        skew_x: transform.skew_x * factor,
        origin_y: transform.origin_y,
        skew_y: transform.skew_y * factor,
        pixel_height: transform.pixel_height * factor,
    }
}

fn scale_transformation_matrix(mut matrix: [f64; 16], factor: f64) -> [f64; 16] {
    // Compose with diag(factor, factor, 1, 1) on the pixel-coordinate side.
    // Row-major storage means that scales columns 0 and 1 while preserving
    // the model-space translation.
    for index in [0usize, 1, 4, 5, 8, 9, 12, 13] {
        matrix[index] *= factor;
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sample_count_rejects_overflow() {
        let err = checked_sample_count(&[usize::MAX, 2], "actual raster").unwrap_err();
        assert!(
            matches!(err, Error::Other(message) if message.contains("sample count overflows usize"))
        );
    }

    #[test]
    fn later_scale_and_origin_settings_replace_a_transformation_matrix() {
        let builder = GeoTiffBuilder::new(1, 1)
            .transformation_matrix(
                GeoTransform::from_origin_and_pixel_size(10.0, 20.0, 2.0, -2.0)
                    .to_transformation_matrix(),
            )
            .pixel_scale(3.0, 4.0)
            .origin(30.0, 40.0);
        let tags = builder.build_extra_tags().unwrap();

        assert!(tags
            .iter()
            .all(|tag| tag.code != tags::TAG_MODEL_TRANSFORMATION));
        assert!(tags
            .iter()
            .any(|tag| tag.code == tags::TAG_MODEL_PIXEL_SCALE));
        assert!(tags.iter().any(|tag| tag.code == tags::TAG_MODEL_TIEPOINT));
    }

    #[test]
    fn invalid_georeferencing_is_rejected_before_writing() {
        assert!(GeoTiffBuilder::new(1, 1)
            .pixel_scale(0.0, 1.0)
            .origin(0.0, 0.0)
            .build_extra_tags()
            .is_err());
        assert!(GeoTiffBuilder::new(1, 1)
            .pixel_scale(1.0, 1.0)
            .build_extra_tags()
            .is_err());
        assert!(GeoTiffBuilder::new(1, 1)
            .transformation_matrix([0.0; 16])
            .build_extra_tags()
            .is_err());
    }
}
