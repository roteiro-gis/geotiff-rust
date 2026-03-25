//! GeoTiffBuilder: fluent API for constructing GeoTIFF files.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use geotiff_core::geokeys::{self, GeoKeyDirectory, GeoKeyValue};
use geotiff_core::tags;
use geotiff_core::transform::GeoTransform;
use geotiff_core::{ModelType, RasterType};
use ndarray::{ArrayView2, ArrayView3};
use tiff_core::{
    Compression, PhotometricInterpretation, PlanarConfiguration, Predictor, Tag, TagValue,
};
use tiff_writer::{ImageBuilder, TiffWriter, WriteOptions};

use crate::error::{Error, Result};
use crate::sample::WriteSample;
use crate::tile_writer::StreamingTileWriter;

/// Builder for constructing GeoTIFF files with metadata.
#[derive(Debug, Clone)]
pub struct GeoTiffBuilder {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bands: u32,
    pub(crate) geokeys: GeoKeyDirectory,
    pub(crate) pixel_scale: Option<[f64; 3]>,
    pub(crate) tiepoint: Option<[f64; 6]>,
    pub(crate) transformation_matrix: Option<[f64; 16]>,
    pub(crate) nodata: Option<String>,
    pub(crate) compression: Compression,
    pub(crate) predictor: Predictor,
    pub(crate) planar_configuration: PlanarConfiguration,
    pub(crate) tile_width: Option<u32>,
    pub(crate) tile_height: Option<u32>,
    pub(crate) photometric: PhotometricInterpretation,
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
            transformation_matrix: None,
            nodata: None,
            compression: Compression::None,
            predictor: Predictor::None,
            planar_configuration: PlanarConfiguration::Chunky,
            tile_width: None,
            tile_height: None,
            photometric: PhotometricInterpretation::MinIsBlack,
        }
    }

    /// Set the number of bands (samples per pixel). Default: 1.
    pub fn bands(mut self, bands: u32) -> Self {
        self.bands = bands;
        self
    }

    /// Set CRS by EPSG code. Auto-detects Geographic vs Projected.
    pub fn epsg(mut self, code: u16) -> Self {
        let is_geographic = (4000..5000).contains(&code);
        if is_geographic {
            self.geokeys.set(
                geokeys::GT_MODEL_TYPE,
                GeoKeyValue::Short(ModelType::Geographic.code()),
            );
            self.geokeys
                .set(geokeys::GEOGRAPHIC_TYPE, GeoKeyValue::Short(code));
        } else {
            self.geokeys.set(
                geokeys::GT_MODEL_TYPE,
                GeoKeyValue::Short(ModelType::Projected.code()),
            );
            self.geokeys
                .set(geokeys::PROJECTED_CS_TYPE, GeoKeyValue::Short(code));
        }
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
        self
    }

    /// Set the map origin (upper-left corner X, Y).
    pub fn origin(mut self, x: f64, y: f64) -> Self {
        self.tiepoint = Some([0.0, 0.0, 0.0, x, y, 0.0]);
        self
    }

    /// Set an explicit tiepoint (I, J, K, X, Y, Z).
    pub fn tiepoint(mut self, tiepoint: [f64; 6]) -> Self {
        self.tiepoint = Some(tiepoint);
        self
    }

    /// Set a full affine transform. Takes precedence over pixel_scale + origin.
    pub fn transform(mut self, transform: GeoTransform) -> Self {
        if let Some((tp, scale)) = transform.to_tiepoint_and_scale() {
            self.tiepoint = Some(tp);
            self.pixel_scale = Some(scale);
            self.transformation_matrix = None;
        } else {
            self.transformation_matrix = Some(transform.to_transformation_matrix());
            self.tiepoint = None;
            self.pixel_scale = None;
        }
        self
    }

    /// Set a 4x4 model transformation matrix.
    pub fn transformation_matrix(mut self, matrix: [f64; 16]) -> Self {
        self.transformation_matrix = Some(matrix);
        self.tiepoint = None;
        self.pixel_scale = None;
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
        self
    }

    /// Set predictor (requires compression != None).
    pub fn predictor(mut self, predictor: Predictor) -> Self {
        self.predictor = predictor;
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

    /// Build the GeoTIFF extra tags from the current metadata.
    pub(crate) fn build_extra_tags(&self) -> Vec<Tag> {
        let mut extra = Vec::new();

        // Georeferencing tags
        if let Some(matrix) = &self.transformation_matrix {
            extra.push(Tag::new(
                tags::TAG_MODEL_TRANSFORMATION,
                TagValue::Double(matrix.to_vec()),
            ));
        } else {
            if let Some(ps) = &self.pixel_scale {
                extra.push(Tag::new(
                    tags::TAG_MODEL_PIXEL_SCALE,
                    TagValue::Double(ps.to_vec()),
                ));
            }
            if let Some(tp) = &self.tiepoint {
                extra.push(Tag::new(
                    tags::TAG_MODEL_TIEPOINT,
                    TagValue::Double(tp.to_vec()),
                ));
            }
        }

        // GeoKey directory
        if !self.geokeys.keys.is_empty() {
            let (directory, double_params, ascii_params) = self.geokeys.serialize();
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

        extra
    }

    /// Build an ImageBuilder from this GeoTiffBuilder for a given sample type.
    pub(crate) fn to_image_builder<T: WriteSample>(&self) -> ImageBuilder {
        let mut ib = ImageBuilder::new(self.width, self.height)
            .sample_type::<T>()
            .samples_per_pixel(self.bands as u16)
            .compression(self.compression)
            .predictor(self.predictor)
            .planar_configuration(self.planar_configuration)
            .photometric(self.photometric);

        if let (Some(tw), Some(th)) = (self.tile_width, self.tile_height) {
            ib = ib.tiles(tw, th);
        }

        for tag in self.build_extra_tags() {
            ib = ib.tag(tag);
        }

        ib
    }

    // ---- Write methods ----

    /// Write a single-band 2D array to a file path.
    pub fn write_2d<T: WriteSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_2d_to(writer, data)
    }

    /// Write a single-band 2D array to any Write+Seek target.
    pub fn write_2d_to<T: WriteSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView2<T>,
    ) -> Result<()> {
        let (height, width) = data.dim();
        if width as u32 != self.width || height as u32 != self.height {
            return Err(Error::DataSizeMismatch {
                expected: (self.height as usize) * (self.width as usize),
                actual: height * width,
            });
        }

        let ib = self.to_image_builder::<T>();
        let mut writer = TiffWriter::new(sink, WriteOptions::default())?;
        let handle = writer.add_image(ib)?;

        let block_count = self.images_block_count::<T>();

        for block_idx in 0..block_count {
            let samples = self.extract_block_2d(&data, block_idx);
            writer.write_block(&handle, block_idx, &samples)?;
        }

        writer.finish()?;
        Ok(())
    }

    /// Write a multi-band 3D array [rows, cols, bands] to a file path.
    pub fn write_3d<T: WriteSample, P: AsRef<Path>>(
        &self,
        path: P,
        data: ArrayView3<T>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_3d_to(writer, data)
    }

    /// Write a multi-band 3D array to any Write+Seek target.
    pub fn write_3d_to<T: WriteSample, W: Write + Seek>(
        &self,
        sink: W,
        data: ArrayView3<T>,
    ) -> Result<()> {
        let (height, width, bands) = data.dim();
        if width as u32 != self.width || height as u32 != self.height || bands as u32 != self.bands
        {
            return Err(Error::DataSizeMismatch {
                expected: self.height as usize * self.width as usize * self.bands as usize,
                actual: height * width * bands,
            });
        }

        let ib = self.to_image_builder::<T>();
        let mut writer = TiffWriter::new(sink, WriteOptions::default())?;
        let handle = writer.add_image(ib)?;

        let block_count = self.images_block_count::<T>();
        for block_idx in 0..block_count {
            let samples = self.extract_block_3d(&data, block_idx);
            writer.write_block(&handle, block_idx, &samples)?;
        }

        writer.finish()?;
        Ok(())
    }

    /// Create a streaming tile writer for incremental writes.
    pub fn tile_writer<T: WriteSample, W: Write + Seek>(
        &self,
        sink: W,
    ) -> Result<StreamingTileWriter<T, W>> {
        StreamingTileWriter::new(self.clone(), sink)
    }

    /// Create a streaming tile writer that writes to a file path.
    pub fn tile_writer_file<T: WriteSample, P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<StreamingTileWriter<T, BufWriter<File>>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.tile_writer(writer)
    }

    fn images_block_count<T: WriteSample>(&self) -> usize {
        self.to_image_builder::<T>().block_count()
    }

    fn extract_block_2d<T: WriteSample>(&self, data: &ArrayView2<T>, block_idx: usize) -> Vec<T> {
        let zero = T::decode_many(&vec![0u8; T::BYTES_PER_SAMPLE])[0];
        if let (Some(tw), Some(th)) = (self.tile_width, self.tile_height) {
            let tw = tw as usize;
            let th = th as usize;
            let tiles_across = (self.width as usize).div_ceil(tw);
            let tile_row = block_idx / tiles_across;
            let tile_col = block_idx % tiles_across;
            let start_row = tile_row * th;
            let start_col = tile_col * tw;

            let mut tile_data = vec![zero; tw * th];
            for row in 0..th {
                let src_row = start_row + row;
                if src_row >= self.height as usize {
                    break;
                }
                for col in 0..tw {
                    let src_col = start_col + col;
                    if src_col >= self.width as usize {
                        break;
                    }
                    tile_data[row * tw + col] = data[[src_row, src_col]];
                }
            }
            tile_data
        } else {
            let rps = self.height.min(256) as usize;
            let start_row = block_idx * rps;
            let end_row = ((block_idx + 1) * rps).min(self.height as usize);
            let w = self.width as usize;

            let mut samples = Vec::with_capacity((end_row - start_row) * w);
            for row in start_row..end_row {
                for col in 0..w {
                    samples.push(data[[row, col]]);
                }
            }
            samples
        }
    }

    fn extract_block_3d<T: WriteSample>(&self, data: &ArrayView3<T>, block_idx: usize) -> Vec<T> {
        let zero = T::decode_many(&vec![0u8; T::BYTES_PER_SAMPLE])[0];
        let bands = self.bands as usize;

        if let (Some(tw), Some(th)) = (self.tile_width, self.tile_height) {
            let tw = tw as usize;
            let th = th as usize;
            let tiles_across = (self.width as usize).div_ceil(tw);
            let tiles_down = (self.height as usize).div_ceil(th);
            let tiles_per_plane = tiles_across * tiles_down;
            let (plane, plane_block_index) =
                self.plane_and_block_index(block_idx, tiles_per_plane, bands);
            let tile_row = plane_block_index / tiles_across;
            let tile_col = plane_block_index % tiles_across;
            let start_row = tile_row * th;
            let start_col = tile_col * tw;

            if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
                let mut tile_data = vec![zero; tw * th];
                for row in 0..th {
                    let src_row = start_row + row;
                    if src_row >= self.height as usize {
                        break;
                    }
                    for col in 0..tw {
                        let src_col = start_col + col;
                        if src_col >= self.width as usize {
                            break;
                        }
                        tile_data[row * tw + col] = data[[src_row, src_col, plane]];
                    }
                }
                tile_data
            } else {
                let mut tile_data = vec![zero; tw * th * bands];
                for row in 0..th {
                    let src_row = start_row + row;
                    if src_row >= self.height as usize {
                        break;
                    }
                    for col in 0..tw {
                        let src_col = start_col + col;
                        if src_col >= self.width as usize {
                            break;
                        }
                        for band in 0..bands {
                            tile_data[(row * tw + col) * bands + band] =
                                data[[src_row, src_col, band]];
                        }
                    }
                }
                tile_data
            }
        } else {
            let rps = self.rows_per_strip();
            let strips_per_plane = (self.height as usize).div_ceil(rps);
            let (plane, plane_block_index) =
                self.plane_and_block_index(block_idx, strips_per_plane, bands);
            let start_row = plane_block_index * rps;
            let end_row = ((plane_block_index + 1) * rps).min(self.height as usize);
            let w = self.width as usize;

            if matches!(self.planar_configuration, PlanarConfiguration::Planar) {
                let mut samples = Vec::with_capacity((end_row - start_row) * w);
                for row in start_row..end_row {
                    for col in 0..w {
                        samples.push(data[[row, col, plane]]);
                    }
                }
                samples
            } else {
                let mut samples = Vec::with_capacity((end_row - start_row) * w * bands);
                for row in start_row..end_row {
                    for col in 0..w {
                        for band in 0..bands {
                            samples.push(data[[row, col, band]]);
                        }
                    }
                }
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

    fn rows_per_strip(&self) -> usize {
        self.height.min(256) as usize
    }
}
