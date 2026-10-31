//! Pure-Rust GeoTIFF reader with optional HTTP range-backed remote access.
//!
//! Supports:
//! - **GeoTIFF**: TIFF files with GeoKey metadata (EPSG codes, CRS, tiepoints, pixel scale)
//! - **COG**: overview discovery plus optional remote open via HTTP range requests
//! - **Reads**: full rasters, windows, overviews, and single storage-domain bands
//! - **Compression passthrough**: any compression supported by `tiff-reader`, including TIFF
//!   `LERC`, `LERC+DEFLATE`, and, with the `zstd` feature enabled on `tiff-reader`, `LERC+ZSTD`
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "local")]
//! # fn main() -> Result<(), geotiff_reader::Error> {
//! use geotiff_reader::GeoTiffFile;
//!
//! let file = GeoTiffFile::open("dem.tif")?;
//! println!("EPSG: {:?}", file.epsg());
//! println!("bounds: {:?}", file.geo_bounds());
//! println!("size: {}x{}", file.width(), file.height());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "local"))]
//! # fn main() {}
//! ```

pub mod crs;
pub mod error;
pub mod geokeys;
pub mod transform;

#[cfg(feature = "cog")]
pub mod cog;

#[cfg(feature = "cog-async")]
pub mod cog_async;

#[cfg(any(feature = "cog", feature = "cog-async"))]
mod http_range;

#[cfg(all(test, any(feature = "cog", feature = "cog-async")))]
mod http_test_support;

pub use error::{Error, Result};

#[cfg(feature = "local")]
use crs::CrsInfo;
#[cfg(feature = "local")]
use geokeys::GeoKeyDirectory;
#[cfg(feature = "local")]
use ndarray::ArrayD;
#[cfg(feature = "local")]
use std::collections::HashSet;
#[cfg(feature = "local")]
use std::path::Path;
#[cfg(feature = "local")]
use tiff_reader::{OpenOptions as TiffOpenOptions, TagValue, TiffFile, TiffSample};
#[cfg(feature = "local")]
use transform::GeoTransform;

#[cfg(feature = "local")]
use geotiff_core::tags::{
    TAG_GDAL_NODATA, TAG_GEO_ASCII_PARAMS, TAG_GEO_DOUBLE_PARAMS, TAG_GEO_KEY_DIRECTORY,
    TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT, TAG_MODEL_TRANSFORMATION, TAG_NEW_SUBFILE_TYPE,
    TAG_SUBFILE_TYPE,
};

#[cfg(feature = "local")]
const MAX_SUBIFD_OVERVIEW_NODES: usize = 1024;
#[cfg(feature = "local")]
const MAX_SUBIFD_OVERVIEW_DEPTH: usize = 32;

/// A GeoTIFF file handle with geospatial metadata.
#[cfg(feature = "local")]
pub struct GeoTiffFile {
    tiff: TiffFile,
    geo_metadata: GeoMetadata,
    crs: CrsInfo,
    geokeys: GeoKeyDirectory,
    transform: Option<GeoTransform>,
    base_ifd_index: usize,
    overview_ifds: Vec<GeoImageIfd>,
}

#[cfg(feature = "local")]
#[derive(Debug, Clone)]
struct GeoImageIfd {
    top_level_ifd_index: Option<usize>,
    ifd: tiff_reader::Ifd,
}

#[cfg(feature = "local")]
pub use tiff_reader::OpenOptions as GeoTiffOpenOptions;

pub use geotiff_core::GeoMetadata;

#[cfg(feature = "local")]
impl GeoTiffFile {
    /// Open a GeoTIFF file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, TiffOpenOptions::default())
    }

    /// Open a GeoTIFF file from disk with explicit TIFF decoder options.
    pub fn open_with_options<P: AsRef<Path>>(path: P, options: GeoTiffOpenOptions) -> Result<Self> {
        let tiff = TiffFile::open_with_options(path, options)?;
        Self::from_tiff(tiff)
    }

    /// Open a GeoTIFF file from disk using memory-mapped I/O.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the mapped file is not mutated or
    /// truncated while the returned `GeoTiffFile` is alive. This includes
    /// writes through other file handles and writes from other processes.
    pub unsafe fn open_mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        unsafe { Self::open_mmap_with_options(path, TiffOpenOptions::default()) }
    }

    /// Open a GeoTIFF file from disk using memory-mapped I/O with explicit TIFF decoder options.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the mapped file is not mutated or
    /// truncated while the returned `GeoTiffFile` is alive. This includes
    /// writes through other file handles and writes from other processes.
    pub unsafe fn open_mmap_with_options<P: AsRef<Path>>(
        path: P,
        options: GeoTiffOpenOptions,
    ) -> Result<Self> {
        let tiff = unsafe { TiffFile::open_mmap_with_options(path, options)? };
        Self::from_tiff(tiff)
    }

    /// Open a GeoTIFF from an owned byte buffer.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        Self::from_bytes_with_options(data, TiffOpenOptions::default())
    }

    /// Open a GeoTIFF from bytes with explicit TIFF decoder options.
    pub fn from_bytes_with_options(data: Vec<u8>, options: GeoTiffOpenOptions) -> Result<Self> {
        let tiff = TiffFile::from_bytes_with_options(data, options)?;
        Self::from_tiff(tiff)
    }

    pub(crate) fn from_tiff(tiff: TiffFile) -> Result<Self> {
        let metadata_ifd_index = find_metadata_ifd_index(tiff.ifds())?;
        let metadata_ifd = tiff.ifd(metadata_ifd_index)?;
        let geokeys = parse_geokey_directory(metadata_ifd)?;
        let crs = CrsInfo::from_geokeys(&geokeys);
        let epsg = crs.epsg();
        let tiepoints = parse_tiepoints(metadata_ifd)?;
        let pixel_scale = parse_fixed_len_double_tag::<3>(metadata_ifd, TAG_MODEL_PIXEL_SCALE)?;
        let transformation =
            parse_fixed_len_double_tag::<16>(metadata_ifd, TAG_MODEL_TRANSFORMATION)?;
        validate_model_georeferencing(&tiepoints, pixel_scale.as_ref(), transformation.as_ref())?;
        let transform = transformation
            .as_ref()
            .map(GeoTransform::from_transformation_matrix)
            .or_else(|| {
                let tiepoint = tiepoints.first()?;
                let scale = pixel_scale.as_ref()?;
                Some(GeoTransform::from_tiepoint_and_scale_with_raster_type(
                    tiepoint,
                    scale,
                    crs.raster_type_enum(),
                ))
            });
        let base_ifd_index = find_base_ifd_index(tiff.ifds(), metadata_ifd_index);
        let base_ifd = tiff.ifd(base_ifd_index)?;
        let overview_ifds =
            collect_overview_ifds(&tiff, base_ifd, base_ifd_index, metadata_ifd_index)?;
        let geo_bounds = transform
            .as_ref()
            .map(|gt| gt.bounds(base_ifd.width(), base_ifd.height()));

        let geo_metadata = GeoMetadata {
            epsg,
            tiepoints,
            pixel_scale,
            transformation,
            nodata: parse_nodata(metadata_ifd)?,
            band_count: base_ifd.samples_per_pixel() as u32,
            width: base_ifd.width(),
            height: base_ifd.height(),
            geo_bounds,
        };

        Ok(Self {
            tiff,
            geo_metadata,
            crs,
            geokeys,
            transform,
            base_ifd_index,
            overview_ifds,
        })
    }

    /// Returns the underlying TIFF file.
    pub fn tiff(&self) -> &TiffFile {
        &self.tiff
    }

    /// Returns the parsed GeoTIFF metadata.
    pub fn metadata(&self) -> &GeoMetadata {
        &self.geo_metadata
    }

    /// Returns the EPSG code of the coordinate reference system, if present.
    pub fn epsg(&self) -> Option<u32> {
        self.geo_metadata.epsg
    }

    /// Returns the extracted CRS information.
    pub fn crs(&self) -> &CrsInfo {
        &self.crs
    }

    /// Returns the parsed GeoKey directory.
    pub fn geokeys(&self) -> &GeoKeyDirectory {
        &self.geokeys
    }

    /// Returns the affine transform, if present.
    pub fn transform(&self) -> Option<&GeoTransform> {
        self.transform.as_ref()
    }

    /// Returns the geographic bounds as `(min_x, min_y, max_x, max_y)`.
    pub fn geo_bounds(&self) -> Option<[f64; 4]> {
        self.geo_metadata.geo_bounds
    }

    /// Convert a pixel coordinate to map coordinates.
    pub fn pixel_to_geo(&self, col: f64, row: f64) -> Option<(f64, f64)> {
        self.transform
            .map(|transform| transform.pixel_to_geo(col, row))
    }

    /// Convert map coordinates to pixel coordinates.
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.transform
            .and_then(|transform| transform.geo_to_pixel(x, y))
    }

    /// Returns the image width in pixels.
    pub fn width(&self) -> u32 {
        self.geo_metadata.width
    }

    /// Returns the image height in pixels.
    pub fn height(&self) -> u32 {
        self.geo_metadata.height
    }

    /// Returns the number of bands.
    pub fn band_count(&self) -> u32 {
        self.geo_metadata.band_count
    }

    /// Returns the nodata value, if set.
    pub fn nodata(&self) -> Option<&str> {
        self.geo_metadata.nodata.as_deref()
    }

    /// Returns the number of internal overview IFDs.
    pub fn overview_count(&self) -> usize {
        self.overview_ifds.len()
    }

    /// Returns the top-level TIFF IFD index of the requested overview.
    ///
    /// Overviews stored in SubIFDs return
    /// `Error::OverviewHasNoTopLevelIfdIndex`.
    pub fn overview_ifd_index(&self, overview_index: usize) -> Result<usize> {
        self.overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?
            .top_level_ifd_index
            .ok_or(Error::OverviewHasNoTopLevelIfdIndex(overview_index))
    }

    /// Returns the parsed TIFF IFD metadata for the requested overview.
    pub fn overview_ifd(&self, overview_index: usize) -> Result<&tiff_reader::Ifd> {
        self.overview_ifds
            .get(overview_index)
            .map(|overview| &overview.ifd)
            .ok_or(Error::OverviewNotFound(overview_index))
    }

    /// Returns the TIFF IFD index of the base-resolution image.
    pub fn base_ifd_index(&self) -> usize {
        self.base_ifd_index
    }

    /// Decode the base-resolution raster into storage-domain typed samples.
    pub fn read_raster<T: TiffSample>(&self) -> Result<ArrayD<T>> {
        self.tiff
            .read_image::<T>(self.base_ifd_index)
            .map_err(Into::into)
    }

    /// Decode a base-resolution pixel window into storage-domain typed samples.
    pub fn read_window<T: TiffSample>(
        &self,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        self.tiff
            .read_window::<T>(self.base_ifd_index, row_off, col_off, rows, cols)
            .map_err(Into::into)
    }

    /// Decode one base-resolution storage-domain band into a typed
    /// `[height, width]` ndarray.
    pub fn read_band<T: TiffSample>(&self, band_index: usize) -> Result<ArrayD<T>> {
        self.tiff
            .read_band::<T>(self.base_ifd_index, band_index)
            .map_err(Into::into)
    }

    /// Decode a base-resolution window from one storage-domain band into a
    /// typed `[rows, cols]` ndarray.
    pub fn read_band_window<T: TiffSample>(
        &self,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        self.tiff
            .read_band_window::<T>(
                self.base_ifd_index,
                band_index,
                row_off,
                col_off,
                rows,
                cols,
            )
            .map_err(Into::into)
    }

    /// Decode the base-resolution raster into color-decoded typed pixels.
    pub fn read_decoded_raster<T: TiffSample>(&self) -> Result<ArrayD<T>> {
        self.tiff
            .read_decoded_image::<T>(self.base_ifd_index)
            .map_err(Into::into)
    }

    /// Decode a base-resolution pixel window into color-decoded typed pixels.
    pub fn read_decoded_window<T: TiffSample>(
        &self,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        self.tiff
            .read_decoded_window::<T>(self.base_ifd_index, row_off, col_off, rows, cols)
            .map_err(Into::into)
    }

    /// Decode an overview raster into storage-domain typed samples.
    pub fn read_overview<T: TiffSample>(&self, overview_index: usize) -> Result<ArrayD<T>> {
        let overview = self
            .overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?;
        self.tiff
            .read_image_from_ifd::<T>(&overview.ifd)
            .map_err(Into::into)
    }

    /// Decode an overview raster into color-decoded typed pixels.
    pub fn read_decoded_overview<T: TiffSample>(&self, overview_index: usize) -> Result<ArrayD<T>> {
        let overview = self
            .overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?;
        self.tiff
            .read_decoded_image_from_ifd::<T>(&overview.ifd)
            .map_err(Into::into)
    }

    /// Decode one overview storage-domain band into a typed `[height, width]`
    /// ndarray.
    pub fn read_overview_band<T: TiffSample>(
        &self,
        overview_index: usize,
        band_index: usize,
    ) -> Result<ArrayD<T>> {
        let overview = self
            .overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?;
        self.tiff
            .read_band_from_ifd::<T>(&overview.ifd, band_index)
            .map_err(Into::into)
    }

    /// Decode an overview pixel window into storage-domain typed samples.
    pub fn read_overview_window<T: TiffSample>(
        &self,
        overview_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let overview = self
            .overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?;
        self.tiff
            .read_window_from_ifd::<T>(&overview.ifd, row_off, col_off, rows, cols)
            .map_err(Into::into)
    }

    /// Decode an overview pixel window into color-decoded typed pixels.
    pub fn read_decoded_overview_window<T: TiffSample>(
        &self,
        overview_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let overview = self
            .overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?;
        self.tiff
            .read_decoded_window_from_ifd::<T>(&overview.ifd, row_off, col_off, rows, cols)
            .map_err(Into::into)
    }

    /// Decode an overview pixel window from one storage-domain band into a
    /// typed `[rows, cols]` ndarray.
    pub fn read_overview_band_window<T: TiffSample>(
        &self,
        overview_index: usize,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayD<T>> {
        let overview = self
            .overview_ifds
            .get(overview_index)
            .ok_or(Error::OverviewNotFound(overview_index))?;
        self.tiff
            .read_band_window_from_ifd::<T>(&overview.ifd, band_index, row_off, col_off, rows, cols)
            .map_err(Into::into)
    }
}

#[cfg(feature = "local")]
fn is_overview_ifd(base: &tiff_reader::Ifd, candidate: &tiff_reader::Ifd) -> bool {
    let smaller = candidate.width() <= base.width()
        && candidate.height() <= base.height()
        && (candidate.width() < base.width() || candidate.height() < base.height());
    if !smaller {
        return false;
    }

    let same_layout = candidate.samples_per_pixel() == base.samples_per_pixel()
        && candidate.bits_per_sample().ok() == base.bits_per_sample().ok()
        && candidate.sample_format().ok() == base.sample_format().ok()
        && candidate.photometric_interpretation() == base.photometric_interpretation();
    if !same_layout {
        return false;
    }

    has_reduced_resolution_flag(candidate)
        || (candidate.tag(TAG_NEW_SUBFILE_TYPE).is_none()
            && candidate.tag(TAG_SUBFILE_TYPE).is_none())
}

#[cfg(feature = "local")]
fn collect_overview_ifds(
    tiff: &TiffFile,
    base_ifd: &tiff_reader::Ifd,
    base_ifd_index: usize,
    metadata_ifd_index: usize,
) -> Result<Vec<GeoImageIfd>> {
    let mut seen_offsets = HashSet::from([base_ifd.offset()]);
    let mut overviews: Vec<GeoImageIfd> = Vec::new();
    for (index, candidate) in tiff.ifds().iter().enumerate() {
        if index != base_ifd_index
            && index != metadata_ifd_index
            && is_overview_ifd(base_ifd, candidate)
            && seen_offsets.insert(candidate.offset())
        {
            overviews.push(GeoImageIfd {
                top_level_ifd_index: Some(index),
                ifd: candidate.clone(),
            });
        }
    }

    if let Some(offsets) = base_ifd.sub_ifd_offsets() {
        collect_subifd_overviews(tiff, base_ifd, &offsets, &mut seen_offsets, &mut overviews)?;
    }

    overviews.sort_by(|lhs, rhs| {
        rhs.ifd
            .width()
            .cmp(&lhs.ifd.width())
            .then_with(|| rhs.ifd.height().cmp(&lhs.ifd.height()))
            .then_with(|| lhs.top_level_ifd_index.cmp(&rhs.top_level_ifd_index))
    });

    Ok(overviews)
}

#[cfg(feature = "local")]
fn collect_subifd_overviews(
    tiff: &TiffFile,
    base_ifd: &tiff_reader::Ifd,
    offsets: &[u64],
    seen_offsets: &mut HashSet<u64>,
    overviews: &mut Vec<GeoImageIfd>,
) -> Result<()> {
    let mut pending = Vec::new();
    let mut visited_nodes = 0usize;
    push_subifd_offsets(offsets, 1, seen_offsets, &mut pending, &mut visited_nodes)?;

    while let Some((offset, depth)) = pending.pop() {
        let candidate = tiff.read_ifd_at_offset(offset)?;
        if is_overview_ifd(base_ifd, &candidate) {
            overviews.push(GeoImageIfd {
                top_level_ifd_index: None,
                ifd: candidate.clone(),
            });
        }

        if let Some(child_offsets) = candidate.sub_ifd_offsets() {
            push_subifd_offsets(
                &child_offsets,
                depth + 1,
                seen_offsets,
                &mut pending,
                &mut visited_nodes,
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "local")]
fn push_subifd_offsets(
    offsets: &[u64],
    depth: usize,
    seen_offsets: &mut HashSet<u64>,
    pending: &mut Vec<(u64, usize)>,
    visited_nodes: &mut usize,
) -> Result<()> {
    if depth > MAX_SUBIFD_OVERVIEW_DEPTH {
        return Err(Error::Other(format!(
            "SubIFD overview traversal exceeds depth budget of {MAX_SUBIFD_OVERVIEW_DEPTH}"
        )));
    }

    for &offset in offsets.iter().rev() {
        if *visited_nodes >= MAX_SUBIFD_OVERVIEW_NODES {
            return Err(Error::Other(format!(
                "SubIFD overview traversal exceeds node budget of {MAX_SUBIFD_OVERVIEW_NODES}"
            )));
        }
        *visited_nodes += 1;

        if seen_offsets.insert(offset) {
            pending.push((offset, depth));
        }
    }

    Ok(())
}

#[cfg(feature = "local")]
fn find_metadata_ifd_index(ifds: &[tiff_reader::Ifd]) -> Result<usize> {
    ifds.iter()
        .position(|ifd| ifd.tag(TAG_GEO_KEY_DIRECTORY).is_some())
        .or_else(|| ifds.iter().position(has_model_georeferencing))
        .ok_or(Error::NotGeoTiff)
}

#[cfg(feature = "local")]
fn has_model_georeferencing(ifd: &tiff_reader::Ifd) -> bool {
    ifd.tag(TAG_MODEL_TRANSFORMATION).is_some()
        || (ifd.tag(TAG_MODEL_TIEPOINT).is_some() && ifd.tag(TAG_MODEL_PIXEL_SCALE).is_some())
}

#[cfg(feature = "local")]
fn find_base_ifd_index(ifds: &[tiff_reader::Ifd], metadata_ifd_index: usize) -> usize {
    let metadata_ifd = &ifds[metadata_ifd_index];
    if !has_reduced_resolution_flag(metadata_ifd) {
        return metadata_ifd_index;
    }

    ifds.iter()
        .enumerate()
        .skip(metadata_ifd_index + 1)
        .find_map(|(index, ifd)| (!has_reduced_resolution_flag(ifd)).then_some(index))
        .unwrap_or(metadata_ifd_index)
}

#[cfg(feature = "local")]
fn has_reduced_resolution_flag(ifd: &tiff_reader::Ifd) -> bool {
    ifd.tag(TAG_NEW_SUBFILE_TYPE)
        .and_then(|tag| tag.value.as_u64())
        .map(|flags| flags & 0x1 != 0)
        .or_else(|| {
            ifd.tag(TAG_SUBFILE_TYPE)
                .and_then(|tag| tag.value.as_u16())
                .map(|value| value == 2)
        })
        .unwrap_or(false)
}

#[cfg(feature = "local")]
fn parse_geokey_directory(ifd: &tiff_reader::Ifd) -> Result<GeoKeyDirectory> {
    let Some(tag) = ifd.tag(TAG_GEO_KEY_DIRECTORY) else {
        return Ok(GeoKeyDirectory::new());
    };
    let TagValue::Short(directory) = &tag.value else {
        return Err(invalid_geotiff_tag(
            TAG_GEO_KEY_DIRECTORY,
            "GeoKeyDirectoryTag must use SHORT values",
        ));
    };
    let double_params = match ifd.tag(TAG_GEO_DOUBLE_PARAMS) {
        Some(tag) => match &tag.value {
            TagValue::Double(values) => values.as_slice(),
            _ => {
                return Err(invalid_geotiff_tag(
                    TAG_GEO_DOUBLE_PARAMS,
                    "GeoDoubleParamsTag must use DOUBLE values",
                ))
            }
        },
        None => &[],
    };
    let ascii_params = match ifd.tag(TAG_GEO_ASCII_PARAMS) {
        Some(tag) => match &tag.value {
            TagValue::Ascii(value) => value.as_str(),
            _ => {
                return Err(invalid_geotiff_tag(
                    TAG_GEO_ASCII_PARAMS,
                    "GeoAsciiParamsTag must use ASCII values",
                ))
            }
        },
        None => "",
    };
    GeoKeyDirectory::parse(directory, double_params, ascii_params)
        .ok_or(Error::InvalidGeoKeyDirectory)
}

#[cfg(feature = "local")]
fn parse_fixed_len_double_tag<const N: usize>(
    ifd: &tiff_reader::Ifd,
    tag_code: u16,
) -> Result<Option<[f64; N]>> {
    let Some(tag) = ifd.tag(tag_code) else {
        return Ok(None);
    };
    let TagValue::Double(values) = &tag.value else {
        return Err(invalid_geotiff_tag(tag_code, "tag must use DOUBLE values"));
    };
    if values.len() != N {
        return Err(invalid_geotiff_tag(
            tag_code,
            format!("expected exactly {N} values, got {}", values.len()),
        ));
    }
    let mut out = [0.0; N];
    out.copy_from_slice(values);
    Ok(Some(out))
}

#[cfg(feature = "local")]
fn parse_tiepoints(ifd: &tiff_reader::Ifd) -> Result<Vec<[f64; 6]>> {
    let Some(tag) = ifd.tag(TAG_MODEL_TIEPOINT) else {
        return Ok(Vec::new());
    };
    let TagValue::Double(values) = &tag.value else {
        return Err(invalid_geotiff_tag(
            TAG_MODEL_TIEPOINT,
            "ModelTiepointTag must use DOUBLE values",
        ));
    };
    if values.is_empty() || values.len() % 6 != 0 {
        return Err(invalid_geotiff_tag(
            TAG_MODEL_TIEPOINT,
            format!(
                "expected a non-empty multiple of 6 values, got {}",
                values.len()
            ),
        ));
    }
    Ok(values
        .chunks_exact(6)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5]])
        .collect())
}

#[cfg(feature = "local")]
fn parse_nodata(ifd: &tiff_reader::Ifd) -> Result<Option<String>> {
    let Some(tag) = ifd.tag(TAG_GDAL_NODATA) else {
        return Ok(None);
    };
    match &tag.value {
        TagValue::Ascii(value) => Ok(Some(value.clone())),
        _ => Err(invalid_geotiff_tag(
            TAG_GDAL_NODATA,
            "GDAL_NODATA must use an ASCII value",
        )),
    }
}

#[cfg(feature = "local")]
fn invalid_geotiff_tag(tag: u16, reason: impl Into<String>) -> Error {
    Error::InvalidGeoTiffTag {
        tag,
        reason: reason.into(),
    }
}

#[cfg(feature = "local")]
fn validate_model_georeferencing(
    tiepoints: &[[f64; 6]],
    pixel_scale: Option<&[f64; 3]>,
    transformation: Option<&[f64; 16]>,
) -> Result<()> {
    if tiepoints.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid_geotiff_tag(
            TAG_MODEL_TIEPOINT,
            "tiepoint values must be finite",
        ));
    }
    if let Some(scale) = pixel_scale {
        if !scale.iter().all(|value| value.is_finite())
            || scale[0] <= 0.0
            || scale[1] <= 0.0
            || scale[2] < 0.0
        {
            return Err(invalid_geotiff_tag(
                TAG_MODEL_PIXEL_SCALE,
                "pixel scale must contain finite positive X/Y and non-negative Z values",
            ));
        }
        if tiepoints.is_empty() {
            return Err(invalid_geotiff_tag(
                TAG_MODEL_PIXEL_SCALE,
                "pixel scale requires at least one model tiepoint",
            ));
        }
    }
    if let Some(matrix) = transformation {
        if !matrix.iter().all(|value| value.is_finite()) {
            return Err(invalid_geotiff_tag(
                TAG_MODEL_TRANSFORMATION,
                "transformation matrix values must be finite",
            ));
        }
        let transform = GeoTransform::from_transformation_matrix(matrix);
        if transform
            .geo_to_pixel(transform.origin_x, transform.origin_y)
            .is_none()
        {
            return Err(invalid_geotiff_tag(
                TAG_MODEL_TRANSFORMATION,
                "transformation matrix must contain an invertible 2D affine transform",
            ));
        }
        if pixel_scale.is_some() || !tiepoints.is_empty() {
            return Err(invalid_geotiff_tag(
                TAG_MODEL_TRANSFORMATION,
                "ModelTransformationTag is mutually exclusive with ModelPixelScaleTag and ModelTiepointTag",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "local")]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        GeoTiffFile, MAX_SUBIFD_OVERVIEW_DEPTH, MAX_SUBIFD_OVERVIEW_NODES, TAG_GDAL_NODATA,
        TAG_MODEL_PIXEL_SCALE,
    };

    #[derive(Clone)]
    struct TestIfdSpec {
        entries: Vec<(u16, u16, u32, Vec<u8>)>,
        image_data: Vec<u8>,
    }

    fn le_u16(value: u16) -> [u8; 2] {
        value.to_le_bytes()
    }

    fn le_u32(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    fn le_f64(value: f64) -> [u8; 8] {
        value.to_le_bytes()
    }

    fn temp_geotiff_path(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "geotiff-rust-{test_name}-{}-{nanos}.tif",
            std::process::id()
        ))
    }

    fn inline_short(value: u16) -> Vec<u8> {
        let mut bytes = [0u8; 4];
        bytes[..2].copy_from_slice(&le_u16(value));
        bytes.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn build_lerc2_header_v2(
        width: u32,
        height: u32,
        valid_pixel_count: u32,
        image_type: i32,
        max_z_error: f64,
        z_min: f64,
        z_max: f64,
        payload_len: usize,
    ) -> Vec<u8> {
        let blob_size = 58 + 4 + payload_len;
        let mut bytes = Vec::with_capacity(blob_size);
        bytes.extend_from_slice(b"Lerc2 ");
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
        bytes.extend_from_slice(&image_type.to_le_bytes());
        bytes.extend_from_slice(&max_z_error.to_le_bytes());
        bytes.extend_from_slice(&z_min.to_le_bytes());
        bytes.extend_from_slice(&z_max.to_le_bytes());
        bytes
    }

    fn build_classic_tiff(ifds: &[TestIfdSpec]) -> Vec<u8> {
        let mut ifd_offsets = Vec::with_capacity(ifds.len());
        let mut cursor = 8usize;
        for ifd in ifds {
            ifd_offsets.push(cursor as u32);
            let deferred_len: usize = ifd
                .entries
                .iter()
                .filter(|(tag, _, _, value)| *tag != 273 && value.len() > 4)
                .map(|(_, _, _, value)| value.len())
                .sum();
            cursor += 2 + ifd.entries.len() * 12 + 4 + ifd.image_data.len() + deferred_len;
        }

        let mut bytes = Vec::with_capacity(cursor);
        bytes.extend_from_slice(b"II");
        bytes.extend_from_slice(&le_u16(42));
        bytes.extend_from_slice(&le_u32(ifd_offsets.first().copied().unwrap_or(0)));

        for (ifd_index, ifd) in ifds.iter().enumerate() {
            let ifd_offset = ifd_offsets[ifd_index] as usize;
            debug_assert_eq!(bytes.len(), ifd_offset);

            let ifd_size = 2 + ifd.entries.len() * 12 + 4;
            let mut next_data_offset = ifd_offset + ifd_size;
            let image_offset = next_data_offset as u32;
            next_data_offset += ifd.image_data.len();

            bytes.extend_from_slice(&le_u16(ifd.entries.len() as u16));
            let mut deferred = Vec::new();
            for (tag, ty, count, value) in &ifd.entries {
                bytes.extend_from_slice(&le_u16(*tag));
                bytes.extend_from_slice(&le_u16(*ty));
                bytes.extend_from_slice(&le_u32(*count));
                if *tag == 273 {
                    bytes.extend_from_slice(&le_u32(image_offset));
                } else if value.len() <= 4 {
                    let mut inline = [0u8; 4];
                    inline[..value.len()].copy_from_slice(value);
                    bytes.extend_from_slice(&inline);
                } else {
                    bytes.extend_from_slice(&le_u32(next_data_offset as u32));
                    next_data_offset += value.len();
                    deferred.push(value.clone());
                }
            }

            let next_ifd_offset = ifd_offsets.get(ifd_index + 1).copied().unwrap_or(0);
            bytes.extend_from_slice(&le_u32(next_ifd_offset));
            bytes.extend_from_slice(&ifd.image_data);
            for value in deferred {
                bytes.extend_from_slice(&value);
            }
            debug_assert_eq!(bytes.len(), next_data_offset);
        }

        bytes
    }

    fn build_simple_geotiff(pixel_is_point: bool) -> Vec<u8> {
        let image_data = vec![10u8, 20, 30, 40];
        let tiepoints = [0.0, 0.0, 0.0, 100.0, 200.0, 0.0];
        let scales = [2.0, 2.0, 0.0];
        let geo_keys = if pixel_is_point {
            vec![
                1, 1, 0, 3, // header
                1024, 0, 1, 2, // model type = Geographic
                1025, 0, 1, 2, // raster type = PixelIsPoint
                2048, 0, 1, 4326, // EPSG:4326
            ]
        } else {
            vec![
                1, 1, 0, 2, // header
                1024, 0, 1, 2, // model type = Geographic
                2048, 0, 1, 4326, // EPSG:4326
            ]
        };
        let nodata = b"-9999\0".to_vec();

        build_classic_tiff(&[TestIfdSpec {
            image_data,
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(4).to_vec()),
                (
                    33550u16,
                    12u16,
                    3u32,
                    scales.iter().flat_map(|value| le_f64(*value)).collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    tiepoints.iter().flat_map(|value| le_f64(*value)).collect(),
                ),
                (
                    34735u16,
                    3u16,
                    geo_keys.len() as u32,
                    geo_keys.iter().flat_map(|value| le_u16(*value)).collect(),
                ),
                (42113u16, 2u16, nodata.len() as u32, nodata),
            ],
        }])
    }

    fn build_geotiff_with_geokeys(geo_keys: &[u16], ascii_params: &[u8]) -> Vec<u8> {
        let tiepoints = [0.0, 0.0, 0.0, 100.0, 200.0, 0.0];
        let scales = [2.0, 2.0, 0.0];
        build_classic_tiff(&[TestIfdSpec {
            image_data: vec![10u8, 20, 30, 40],
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(4).to_vec()),
                (
                    33550u16,
                    12u16,
                    3u32,
                    scales.iter().flat_map(|value| le_f64(*value)).collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    tiepoints.iter().flat_map(|value| le_f64(*value)).collect(),
                ),
                (
                    34735u16,
                    3u16,
                    geo_keys.len() as u32,
                    geo_keys.iter().flat_map(|value| le_u16(*value)).collect(),
                ),
                (
                    34737u16,
                    2u16,
                    ascii_params.len() as u32,
                    ascii_params.to_vec(),
                ),
            ],
        }])
    }

    fn build_simple_lerc_geotiff() -> Vec<u8> {
        let tiepoints = [0.0, 0.0, 0.0, 100.0, 200.0, 0.0];
        let scales = [2.0, 2.0, 0.0];
        let geo_keys = vec![
            1, 1, 0, 2, // header
            1024, 0, 1, 2, // model type = Geographic
            2048, 0, 1, 4326, // EPSG:4326
        ];

        let mut image_data = build_lerc2_header_v2(2, 2, 4, 6, 0.0, 1.0, 4.0, 1 + 16);
        image_data.extend_from_slice(&0u32.to_le_bytes());
        image_data.push(1);
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            image_data.extend_from_slice(&value.to_le_bytes());
        }
        let image_len = image_data.len() as u32;

        build_classic_tiff(&[TestIfdSpec {
            image_data,
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, inline_short(32)),
                (259u16, 3u16, 1u32, inline_short(34887)),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, inline_short(1)),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(image_len).to_vec()),
                (339u16, 3u16, 1u32, inline_short(3)),
                (
                    33550u16,
                    12u16,
                    3u32,
                    scales.iter().flat_map(|value| le_f64(*value)).collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    tiepoints.iter().flat_map(|value| le_f64(*value)).collect(),
                ),
                (
                    34735u16,
                    3u16,
                    geo_keys.len() as u32,
                    geo_keys.iter().flat_map(|value| le_u16(*value)).collect(),
                ),
            ],
        }])
    }

    fn overwrite_classic_inline_long_tag(bytes: &mut [u8], tag_code: u16, value: u32) {
        let entry_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let mut offset = 10usize;
        for _ in 0..entry_count {
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code == tag_code {
                bytes[offset + 8..offset + 12].copy_from_slice(&le_u32(value));
                return;
            }
            offset += 12;
        }
        panic!("tag {tag_code} not found in classic TIFF");
    }

    fn classic_inline_long_tag(bytes: &[u8], tag_code: u16) -> u32 {
        let entry_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let mut offset = 10usize;
        for _ in 0..entry_count {
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code == tag_code {
                return u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
            }
            offset += 12;
        }
        panic!("tag {tag_code} not found in classic TIFF");
    }

    fn overwrite_classic_tag_type_or_count(
        bytes: &mut [u8],
        tag_code: u16,
        tag_type: Option<u16>,
        count: Option<u32>,
    ) {
        let entry_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let mut offset = 10usize;
        for _ in 0..entry_count {
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code == tag_code {
                if let Some(tag_type) = tag_type {
                    bytes[offset + 2..offset + 4].copy_from_slice(&le_u16(tag_type));
                }
                if let Some(count) = count {
                    bytes[offset + 4..offset + 8].copy_from_slice(&le_u32(count));
                }
                return;
            }
            offset += 12;
        }
        panic!("tag {tag_code} not found in classic TIFF");
    }

    fn overwrite_first_ifd_next_pointer(bytes: &mut [u8], value: u32) {
        let entry_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let pointer_offset = 10 + entry_count * 12;
        bytes[pointer_offset..pointer_offset + 4].copy_from_slice(&le_u32(value));
    }

    fn overwrite_classic_inline_long_tag_at(
        bytes: &mut [u8],
        ifd_offset: usize,
        tag_code: u16,
        value: u32,
    ) {
        let entry_count = u16::from_le_bytes([bytes[ifd_offset], bytes[ifd_offset + 1]]) as usize;
        let mut offset = ifd_offset + 2;
        for _ in 0..entry_count {
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code == tag_code {
                bytes[offset + 8..offset + 12].copy_from_slice(&le_u32(value));
                return;
            }
            offset += 12;
        }
        panic!("tag {tag_code} not found in classic TIFF at offset {ifd_offset}");
    }

    fn first_ifd_next_pointer(bytes: &[u8]) -> u32 {
        let entry_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let pointer_offset = 10 + entry_count * 12;
        u32::from_le_bytes([
            bytes[pointer_offset],
            bytes[pointer_offset + 1],
            bytes[pointer_offset + 2],
            bytes[pointer_offset + 3],
        ])
    }

    fn ifd_next_pointer(bytes: &[u8], ifd_offset: usize) -> u32 {
        let entry_count = u16::from_le_bytes([bytes[ifd_offset], bytes[ifd_offset + 1]]) as usize;
        let pointer_offset = ifd_offset + 2 + entry_count * 12;
        u32::from_le_bytes([
            bytes[pointer_offset],
            bytes[pointer_offset + 1],
            bytes[pointer_offset + 2],
            bytes[pointer_offset + 3],
        ])
    }

    fn overwrite_classic_long_tag_values(bytes: &mut [u8], tag_code: u16, values: &[u32]) {
        overwrite_classic_long_tag_values_at(bytes, 8, tag_code, values);
    }

    fn overwrite_classic_long_tag_values_at(
        bytes: &mut [u8],
        ifd_offset: usize,
        tag_code: u16,
        values: &[u32],
    ) {
        let entry_count = u16::from_le_bytes([bytes[ifd_offset], bytes[ifd_offset + 1]]) as usize;
        let mut offset = ifd_offset + 2;
        for _ in 0..entry_count {
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code == tag_code {
                let type_code = u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]);
                let count = u32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]) as usize;
                assert_eq!(type_code, 4, "tag {tag_code} is not a LONG tag");
                assert_eq!(count, values.len(), "tag {tag_code} value count mismatch");

                if values.len() == 1 {
                    bytes[offset + 8..offset + 12].copy_from_slice(&le_u32(values[0]));
                } else {
                    let value_offset = u32::from_le_bytes([
                        bytes[offset + 8],
                        bytes[offset + 9],
                        bytes[offset + 10],
                        bytes[offset + 11],
                    ]) as usize;
                    for (index, value) in values.iter().enumerate() {
                        let value_offset = value_offset + index * 4;
                        bytes[value_offset..value_offset + 4].copy_from_slice(&le_u32(*value));
                    }
                }
                return;
            }
            offset += 12;
        }
        panic!("tag {tag_code} not found in classic TIFF at offset {ifd_offset}");
    }

    fn top_level_ifd_offsets_after_first(bytes: &[u8], count: usize) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(count);
        let mut offset = first_ifd_next_pointer(bytes);
        while offset != 0 && offsets.len() < count {
            offsets.push(offset);
            offset = ifd_next_pointer(bytes, offset as usize);
        }
        assert_eq!(offsets.len(), count);
        offsets
    }

    fn subifd_budget_base(subifd_count: u32) -> TestIfdSpec {
        let geo_keys = [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326];
        TestIfdSpec {
            image_data: vec![0u8],
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(64).to_vec()),
                (257u16, 4u16, 1u32, le_u32(64).to_vec()),
                (258u16, 3u16, 1u32, inline_short(8)),
                (259u16, 3u16, 1u32, inline_short(1)),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, inline_short(1)),
                (278u16, 4u16, 1u32, le_u32(64).to_vec()),
                (279u16, 4u16, 1u32, le_u32(1).to_vec()),
                (
                    330u16,
                    4u16,
                    subifd_count,
                    vec![0; subifd_count as usize * 4],
                ),
                (
                    33550u16,
                    12u16,
                    3u32,
                    [2.0, 2.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    [0.0, 0.0, 0.0, 100.0, 200.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    34735u16,
                    3u16,
                    geo_keys.len() as u32,
                    geo_keys.iter().flat_map(|value| le_u16(*value)).collect(),
                ),
            ],
        }
    }

    fn subifd_budget_overview(has_child: bool) -> TestIfdSpec {
        let mut entries = vec![
            (254u16, 4u16, 1u32, le_u32(1).to_vec()),
            (256u16, 4u16, 1u32, le_u32(1).to_vec()),
            (257u16, 4u16, 1u32, le_u32(1).to_vec()),
            (258u16, 3u16, 1u32, inline_short(8)),
            (259u16, 3u16, 1u32, inline_short(1)),
            (273u16, 4u16, 1u32, vec![]),
            (277u16, 3u16, 1u32, inline_short(1)),
            (278u16, 4u16, 1u32, le_u32(1).to_vec()),
            (279u16, 4u16, 1u32, le_u32(1).to_vec()),
        ];
        if has_child {
            entries.push((330u16, 4u16, 1u32, le_u32(0).to_vec()));
        }
        TestIfdSpec {
            image_data: vec![0u8],
            entries,
        }
    }

    fn build_geotiff_exceeding_subifd_node_budget() -> Vec<u8> {
        let child_count = MAX_SUBIFD_OVERVIEW_NODES + 1;
        let mut ifds = Vec::with_capacity(child_count + 1);
        ifds.push(subifd_budget_base(child_count as u32));
        ifds.extend((0..child_count).map(|_| subifd_budget_overview(false)));

        let mut bytes = build_classic_tiff(&ifds);
        let child_offsets = top_level_ifd_offsets_after_first(&bytes, child_count);
        overwrite_classic_long_tag_values(&mut bytes, 330, &child_offsets);
        overwrite_first_ifd_next_pointer(&mut bytes, 0);
        bytes
    }

    fn build_geotiff_exceeding_subifd_depth_budget() -> Vec<u8> {
        let child_count = MAX_SUBIFD_OVERVIEW_DEPTH + 1;
        let mut ifds = Vec::with_capacity(child_count + 1);
        ifds.push(subifd_budget_base(1));
        ifds.extend((0..child_count).map(|index| subifd_budget_overview(index + 1 < child_count)));

        let mut bytes = build_classic_tiff(&ifds);
        let child_offsets = top_level_ifd_offsets_after_first(&bytes, child_count);
        overwrite_classic_long_tag_values(&mut bytes, 330, &child_offsets[..1]);
        for offsets in child_offsets.windows(2) {
            overwrite_classic_long_tag_values_at(
                &mut bytes,
                offsets[0] as usize,
                330,
                &offsets[1..],
            );
        }
        overwrite_first_ifd_next_pointer(&mut bytes, 0);
        bytes
    }

    fn build_geotiff_with_overview() -> Vec<u8> {
        let base = TestIfdSpec {
            image_data: vec![10u8, 20, 30, 40],
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(4).to_vec()),
                (
                    33550u16,
                    12u16,
                    3u32,
                    [2.0, 2.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    [0.0, 0.0, 0.0, 100.0, 200.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    34735u16,
                    3u16,
                    12u32,
                    [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326]
                        .iter()
                        .flat_map(|value| le_u16(*value))
                        .collect(),
                ),
            ],
        };
        let overview = TestIfdSpec {
            image_data: vec![99u8],
            entries: vec![
                (254u16, 4u16, 1u32, le_u32(1).to_vec()),
                (256u16, 4u16, 1u32, le_u32(1).to_vec()),
                (257u16, 4u16, 1u32, le_u32(1).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(1).to_vec()),
                (279u16, 4u16, 1u32, le_u32(1).to_vec()),
            ],
        };

        build_classic_tiff(&[base, overview])
    }

    fn build_geotiff_with_subifd_overview() -> Vec<u8> {
        let base = TestIfdSpec {
            image_data: vec![10u8, 20, 30, 40],
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(4).to_vec()),
                (330u16, 4u16, 1u32, le_u32(0).to_vec()),
                (
                    33550u16,
                    12u16,
                    3u32,
                    [2.0, 2.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    [0.0, 0.0, 0.0, 100.0, 200.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    34735u16,
                    3u16,
                    12u32,
                    [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326]
                        .iter()
                        .flat_map(|value| le_u16(*value))
                        .collect(),
                ),
            ],
        };
        let overview = TestIfdSpec {
            image_data: vec![99u8],
            entries: vec![
                (254u16, 4u16, 1u32, le_u32(1).to_vec()),
                (256u16, 4u16, 1u32, le_u32(1).to_vec()),
                (257u16, 4u16, 1u32, le_u32(1).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(1).to_vec()),
                (279u16, 4u16, 1u32, le_u32(1).to_vec()),
            ],
        };

        let mut bytes = build_classic_tiff(&[base, overview]);
        let child_ifd_offset = first_ifd_next_pointer(&bytes);
        overwrite_classic_inline_long_tag(&mut bytes, 330, child_ifd_offset);
        overwrite_first_ifd_next_pointer(&mut bytes, 0);
        bytes
    }

    fn build_geotiff_with_nested_subifd_overviews() -> Vec<u8> {
        let base = TestIfdSpec {
            image_data: (1u8..=16).collect(),
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(4).to_vec()),
                (257u16, 4u16, 1u32, le_u32(4).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(4).to_vec()),
                (279u16, 4u16, 1u32, le_u32(16).to_vec()),
                (330u16, 4u16, 1u32, le_u32(0).to_vec()),
                (
                    33550u16,
                    12u16,
                    3u32,
                    [2.0, 2.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    [0.0, 0.0, 0.0, 100.0, 200.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    34735u16,
                    3u16,
                    12u32,
                    [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326]
                        .iter()
                        .flat_map(|value| le_u16(*value))
                        .collect(),
                ),
            ],
        };
        let overview = TestIfdSpec {
            image_data: vec![50u8, 60, 70, 80],
            entries: vec![
                (254u16, 4u16, 1u32, le_u32(1).to_vec()),
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(4).to_vec()),
                (330u16, 4u16, 1u32, le_u32(0).to_vec()),
            ],
        };
        let nested = TestIfdSpec {
            image_data: vec![99u8],
            entries: vec![
                (254u16, 4u16, 1u32, le_u32(1).to_vec()),
                (256u16, 4u16, 1u32, le_u32(1).to_vec()),
                (257u16, 4u16, 1u32, le_u32(1).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(1).to_vec()),
                (279u16, 4u16, 1u32, le_u32(1).to_vec()),
            ],
        };

        let mut bytes = build_classic_tiff(&[base, overview, nested]);
        let child_ifd_offset = first_ifd_next_pointer(&bytes);
        let grandchild_ifd_offset = ifd_next_pointer(&bytes, child_ifd_offset as usize);
        overwrite_classic_inline_long_tag(&mut bytes, 330, child_ifd_offset);
        overwrite_classic_inline_long_tag_at(
            &mut bytes,
            child_ifd_offset as usize,
            330,
            grandchild_ifd_offset,
        );
        overwrite_first_ifd_next_pointer(&mut bytes, 0);
        bytes
    }

    fn build_cog_like_geotiff_with_ghost_ifd() -> Vec<u8> {
        let geo_keys = [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326];
        let ghost = TestIfdSpec {
            image_data: vec![0u8],
            entries: vec![
                (254u16, 4u16, 1u32, le_u32(1).to_vec()),
                (256u16, 4u16, 1u32, le_u32(1).to_vec()),
                (257u16, 4u16, 1u32, le_u32(1).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(1).to_vec()),
                (279u16, 4u16, 1u32, le_u32(1).to_vec()),
                (
                    33550u16,
                    12u16,
                    3u32,
                    [2.0, 2.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    33922u16,
                    12u16,
                    6u32,
                    [0.0, 0.0, 0.0, 100.0, 200.0, 0.0]
                        .iter()
                        .flat_map(|value| le_f64(*value))
                        .collect(),
                ),
                (
                    34735u16,
                    3u16,
                    geo_keys.len() as u32,
                    geo_keys.iter().flat_map(|value| le_u16(*value)).collect(),
                ),
            ],
        };
        let overview = TestIfdSpec {
            image_data: vec![50u8, 60, 70, 80],
            entries: vec![
                (254u16, 4u16, 1u32, le_u32(1).to_vec()),
                (256u16, 4u16, 1u32, le_u32(2).to_vec()),
                (257u16, 4u16, 1u32, le_u32(2).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(2).to_vec()),
                (279u16, 4u16, 1u32, le_u32(4).to_vec()),
            ],
        };
        let base = TestIfdSpec {
            image_data: (1u8..=16).collect(),
            entries: vec![
                (256u16, 4u16, 1u32, le_u32(4).to_vec()),
                (257u16, 4u16, 1u32, le_u32(4).to_vec()),
                (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
                (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (273u16, 4u16, 1u32, vec![]),
                (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
                (278u16, 4u16, 1u32, le_u32(4).to_vec()),
                (279u16, 4u16, 1u32, le_u32(16).to_vec()),
            ],
        };

        build_classic_tiff(&[ghost, overview, base])
    }

    #[test]
    fn opens_files_whose_geokeys_only_gdal_would_tolerate() {
        const CITATION: u16 = 2049;
        // KeyRevision 1.2, a Latin-1 citation, and a key with an unknown value
        // location, none of which should stop the raster from opening.
        let geo_keys = [
            1, 1, 2, 4, // header
            1024, 0, 1, 2, // model type = Geographic
            CITATION, 34737, 9, 0, // citation
            2048, 0, 1, 4326, // EPSG:4326
            2054, 12345, 1, 0, // unknown location
        ];
        let file =
            GeoTiffFile::from_bytes(build_geotiff_with_geokeys(&geo_keys, b"Geogr\xe1fic|\0"))
                .unwrap();

        assert_eq!(file.epsg(), Some(4326));
        assert_eq!(file.geokeys().keys.len(), 3);
        assert_eq!(file.geokeys().get_ascii(CITATION), Some("Geogr\u{fffd}fic"));
        let (values, _) = file.read_raster::<u8>().unwrap().into_raw_vec_and_offset();
        assert_eq!(values, vec![10, 20, 30, 40]);
    }

    #[test]
    fn parses_geotiff_metadata_and_reads_raster() {
        let file = GeoTiffFile::from_bytes(build_simple_geotiff(false)).unwrap();
        assert_eq!(file.epsg(), Some(4326));
        assert_eq!(file.width(), 2);
        assert_eq!(file.height(), 2);
        assert_eq!(file.band_count(), 1);
        assert_eq!(file.nodata(), Some("-9999"));
        assert_eq!(file.geo_bounds(), Some([100.0, 196.0, 104.0, 200.0]));

        let raster = file.read_raster::<u8>().unwrap();
        assert_eq!(raster.shape(), &[2, 2]);
        let (values, offset) = raster.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![10, 20, 30, 40]);
    }

    #[test]
    fn open_mmap_reads_geotiff_from_file() {
        let bytes = build_simple_geotiff(false);
        let path = temp_geotiff_path("open_mmap_reads_geotiff_from_file");
        fs::write(&path, &bytes).unwrap();

        let file = unsafe { GeoTiffFile::open_mmap(&path).unwrap() };
        assert_eq!(file.epsg(), Some(4326));
        assert_eq!(file.tiff().raw_bytes(), Some(bytes.as_slice()));

        let raster = file.read_raster::<u8>().unwrap();
        let (values, offset) = raster.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![10, 20, 30, 40]);

        drop(file);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_geotiff_metadata_and_reads_lerc_raster() {
        let file = GeoTiffFile::from_bytes(build_simple_lerc_geotiff()).unwrap();
        assert_eq!(file.epsg(), Some(4326));
        assert_eq!(file.width(), 2);
        assert_eq!(file.height(), 2);

        let raster = file.read_raster::<f32>().unwrap();
        assert_eq!(raster.shape(), &[2, 2]);
        let (values, offset) = raster.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pixel_is_point_metadata_shifts_bounds_to_outer_edges() {
        let file = GeoTiffFile::from_bytes(build_simple_geotiff(true)).unwrap();
        assert_eq!(file.geo_bounds(), Some([99.0, 197.0, 103.0, 201.0]));

        let transform = file.transform().unwrap();
        let (center_x, center_y) = transform.pixel_to_geo(0.5, 0.5);
        assert_eq!((center_x, center_y), (100.0, 200.0));
    }

    #[test]
    fn discovers_reduced_resolution_overviews() {
        let file = GeoTiffFile::from_bytes(build_geotiff_with_overview()).unwrap();
        assert_eq!(file.overview_count(), 1);
        assert_eq!(file.overview_ifd_index(0).unwrap(), 1);

        let overview = file.read_overview::<u8>(0).unwrap();
        assert_eq!(overview.shape(), &[1, 1]);
        let (values, offset) = overview.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![99]);
    }

    #[test]
    fn discovers_and_reads_subifd_overviews() {
        let file = GeoTiffFile::from_bytes(build_geotiff_with_subifd_overview()).unwrap();
        assert_eq!(file.overview_count(), 1);
        assert!(matches!(
            file.overview_ifd_index(0).unwrap_err(),
            crate::error::Error::OverviewHasNoTopLevelIfdIndex(0)
        ));
        assert_eq!(file.overview_ifd(0).unwrap().width(), 1);
        assert_eq!(file.overview_ifd(0).unwrap().height(), 1);

        let overview = file.read_overview::<u8>(0).unwrap();
        assert_eq!(overview.shape(), &[1, 1]);
        let (values, offset) = overview.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![99]);
    }

    #[test]
    fn deduplicates_overviews_referenced_by_both_ifd_chains() {
        let mut bytes = build_geotiff_with_subifd_overview();
        let child_ifd_offset = classic_inline_long_tag(&bytes, 330);
        overwrite_first_ifd_next_pointer(&mut bytes, child_ifd_offset);

        let file = GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(file.overview_count(), 1);
        assert_eq!(file.overview_ifd_index(0).unwrap(), 1);
    }

    #[test]
    fn rejects_malformed_geotiff_tag_types_and_counts() {
        let mut wrong_type = build_simple_geotiff(false);
        overwrite_classic_tag_type_or_count(&mut wrong_type, TAG_GDAL_NODATA, Some(1), None);
        assert!(matches!(
            GeoTiffFile::from_bytes(wrong_type),
            Err(crate::error::Error::InvalidGeoTiffTag {
                tag: TAG_GDAL_NODATA,
                ..
            })
        ));

        let mut wrong_count = build_simple_geotiff(false);
        overwrite_classic_tag_type_or_count(&mut wrong_count, TAG_MODEL_PIXEL_SCALE, None, Some(4));
        assert!(matches!(
            GeoTiffFile::from_bytes(wrong_count),
            Err(crate::error::Error::InvalidGeoTiffTag {
                tag: TAG_MODEL_PIXEL_SCALE,
                ..
            })
        ));
    }

    #[test]
    fn overview_candidates_must_not_grow_either_dimension() {
        let mut bytes = build_geotiff_with_overview();
        let overview_offset = first_ifd_next_pointer(&bytes) as usize;
        overwrite_classic_inline_long_tag_at(&mut bytes, overview_offset, 257, 3);

        let file = GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(file.overview_count(), 0);
    }

    #[test]
    fn discovers_nested_subifd_overviews() {
        let file = GeoTiffFile::from_bytes(build_geotiff_with_nested_subifd_overviews()).unwrap();
        assert_eq!(file.overview_count(), 2);
        assert_eq!(file.overview_ifd(0).unwrap().width(), 2);
        assert_eq!(file.overview_ifd(1).unwrap().width(), 1);
        assert!(matches!(
            file.overview_ifd_index(0).unwrap_err(),
            crate::error::Error::OverviewHasNoTopLevelIfdIndex(0)
        ));
        assert!(matches!(
            file.overview_ifd_index(1).unwrap_err(),
            crate::error::Error::OverviewHasNoTopLevelIfdIndex(1)
        ));

        let first = file.read_overview::<u8>(0).unwrap();
        assert_eq!(first.shape(), &[2, 2]);
        let second = file.read_overview::<u8>(1).unwrap();
        assert_eq!(second.shape(), &[1, 1]);
        assert_eq!(second[[0, 0]], 99);
    }

    #[test]
    fn rejects_subifd_overview_node_budget() {
        let error = match GeoTiffFile::from_bytes(build_geotiff_exceeding_subifd_node_budget()) {
            Ok(_) => panic!("expected SubIFD overview node budget error"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("SubIFD overview traversal exceeds node budget"));
    }

    #[test]
    fn rejects_subifd_overview_depth_budget() {
        let error = match GeoTiffFile::from_bytes(build_geotiff_exceeding_subifd_depth_budget()) {
            Ok(_) => panic!("expected SubIFD overview depth budget error"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("SubIFD overview traversal exceeds depth budget"));
    }

    #[test]
    fn reads_base_raster_window() {
        let file = GeoTiffFile::from_bytes(build_simple_geotiff(false)).unwrap();
        let window = file.read_window::<u8>(1, 0, 1, 2).unwrap();
        assert_eq!(window.shape(), &[1, 2]);
        let (values, offset) = window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![30, 40]);
    }

    #[test]
    fn reads_overview_window() {
        let file = GeoTiffFile::from_bytes(build_geotiff_with_overview()).unwrap();
        let window = file.read_overview_window::<u8>(0, 0, 0, 1, 1).unwrap();
        assert_eq!(window.shape(), &[1, 1]);
        let (values, offset) = window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![99]);
    }

    #[test]
    fn prefers_non_ghost_base_ifd_for_cog_like_layouts() {
        let file = GeoTiffFile::from_bytes(build_cog_like_geotiff_with_ghost_ifd()).unwrap();
        assert_eq!(file.base_ifd_index(), 2);
        assert_eq!(file.width(), 4);
        assert_eq!(file.height(), 4);
        assert_eq!(file.geo_bounds(), Some([100.0, 192.0, 108.0, 200.0]));
        assert_eq!(file.overview_count(), 1);
        assert_eq!(file.overview_ifd_index(0).unwrap(), 1);

        let base = file.read_raster::<u8>().unwrap();
        assert_eq!(base.shape(), &[4, 4]);
        let (values, offset) = base.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, (1u8..=16).collect::<Vec<_>>());

        let overview = file.read_overview::<u8>(0).unwrap();
        assert_eq!(overview.shape(), &[2, 2]);
        let (values, offset) = overview.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![50, 60, 70, 80]);
    }

    #[test]
    fn rejects_zero_rows_per_strip_without_panicking() {
        let mut bytes = build_simple_geotiff(false);
        overwrite_classic_inline_long_tag(&mut bytes, 278, 0);

        let file = GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(file.epsg(), Some(4326));

        let error = file.tiff().read_image_bytes(0).unwrap_err();
        assert!(error.to_string().contains("RowsPerStrip"));
    }
}
