use std::fs::File;
use std::io::BufWriter;

use ndarray::ArrayD;
use tempfile::NamedTempFile;
use tiff_core::{Compression, PhotometricInterpretation, PlanarConfiguration, Tag, TagValue};
use tiff_writer::{ImageBuilder, TiffWriter, WriteOptions};

use geotiff_core::geokeys::{self, GeoKeyDirectory, GeoKeyValue};
use geotiff_core::{tags, RasterType};
use geotiff_reader::GeoTiffFile;

#[path = "../../../test-support/reference.rs"]
mod reference;

#[derive(Clone, Copy)]
enum SampleKind {
    U8,
    I8,
}

fn fixture(path: &str) -> std::path::PathBuf {
    reference::fixture(env!("CARGO_MANIFEST_DIR"), path)
}

fn transform_tuple(file: &GeoTiffFile) -> Option<[f64; 6]> {
    file.transform().map(|transform| {
        [
            transform.origin_x,
            transform.pixel_width,
            transform.skew_x,
            transform.origin_y,
            transform.skew_y,
            transform.pixel_height,
        ]
    })
}

fn area_or_point(file: &GeoTiffFile) -> &'static str {
    match file.crs().raster_type_enum() {
        RasterType::PixelIsPoint => "Point",
        RasterType::PixelIsArea | RasterType::Unknown(_) => "Area",
    }
}

fn assert_gdal_hash_matches(
    path: &std::path::Path,
    overview_index: Option<usize>,
    sample_kind: SampleKind,
) {
    let path_str = path.to_str().unwrap();
    let overview_arg = overview_index.map(|index| index.to_string());
    let reference_json = if let Some(ref index) = overview_arg {
        reference::run_reference_json(
            env!("CARGO_MANIFEST_DIR"),
            &["hash", path_str, "--overview", index],
        )
    } else {
        reference::run_reference_json(env!("CARGO_MANIFEST_DIR"), &["hash", path_str])
    };
    let file = GeoTiffFile::open(path).unwrap();
    let width = reference_json["width"].as_u64().unwrap();
    let height = reference_json["height"].as_u64().unwrap();
    let band_count = reference_json["band_count"].as_u64().unwrap();
    let byte_len = reference_json["byte_len"].as_u64().unwrap() as usize;
    let expected_hash = reference_json["hash"].as_str().unwrap();

    match (overview_index, sample_kind) {
        (None, SampleKind::U8) => {
            let raster: ArrayD<u8> = file.read_raster().unwrap();
            assert_shape(&raster, width as u32, height as u32, band_count);
            let (actual_len, actual_hash) = reference::array_hash(&raster);
            assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
            assert_eq!(
                actual_hash, expected_hash,
                "pixel hash mismatch for {path_str}"
            );
        }
        (Some(index), SampleKind::U8) => {
            let raster: ArrayD<u8> = file.read_overview(index).unwrap();
            assert_shape(&raster, width as u32, height as u32, band_count);
            let (actual_len, actual_hash) = reference::array_hash(&raster);
            assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
            assert_eq!(
                actual_hash, expected_hash,
                "pixel hash mismatch for {path_str}"
            );
        }
        (None, SampleKind::I8) => {
            let raster: ArrayD<i8> = file.read_raster().unwrap();
            assert_shape(&raster, width as u32, height as u32, band_count);
            let (actual_len, actual_hash) = reference::array_hash(&raster);
            assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
            assert_eq!(
                actual_hash, expected_hash,
                "pixel hash mismatch for {path_str}"
            );
        }
        (Some(index), SampleKind::I8) => {
            let raster: ArrayD<i8> = file.read_overview(index).unwrap();
            assert_shape(&raster, width as u32, height as u32, band_count);
            let (actual_len, actual_hash) = reference::array_hash(&raster);
            assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
            assert_eq!(
                actual_hash, expected_hash,
                "pixel hash mismatch for {path_str}"
            );
        }
    }
}

fn assert_gdal_u8_pixels_close(path: &std::path::Path, overview_index: Option<usize>) {
    let path_str = path.to_str().unwrap();
    let overview_arg = overview_index.map(|index| index.to_string());
    let expected = if let Some(ref index) = overview_arg {
        reference::run_reference_bytes(
            env!("CARGO_MANIFEST_DIR"),
            &["bytes", path_str, "--overview", index],
        )
    } else {
        reference::run_reference_bytes(env!("CARGO_MANIFEST_DIR"), &["bytes", path_str])
    };
    let file = GeoTiffFile::open(path).unwrap();
    let raster: ArrayD<u8> = match overview_index {
        Some(index) => file.read_overview(index).unwrap(),
        None => file.read_raster().unwrap(),
    };
    let (actual, offset) = raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0), "unexpected array offset for {path_str}");

    let tolerance = reference::fixture_lossy_u8_tolerance(path)
        .unwrap_or_else(|| panic!("missing lossy tolerance for fixture: {path_str}"));
    reference::assert_u8_bytes_close(
        &actual,
        &expected,
        tolerance.max_abs_delta,
        tolerance.max_diff_pixels,
        path_str,
    );
}

fn assert_shape<T>(array: &ArrayD<T>, width: u32, height: u32, band_count: u64) {
    if band_count == 1 {
        assert_eq!(array.shape(), &[height as usize, width as usize]);
    } else {
        assert_eq!(
            array.shape(),
            &[height as usize, width as usize, band_count as usize]
        );
    }
}

fn build_geo_tags() -> Vec<Tag> {
    let mut geokeys = GeoKeyDirectory::new();
    geokeys.set(
        geokeys::GT_MODEL_TYPE,
        GeoKeyValue::Short(geotiff_core::ModelType::Projected.code()),
    );
    geokeys.set(
        geokeys::GT_RASTER_TYPE,
        GeoKeyValue::Short(geotiff_core::RasterType::PixelIsArea.code()),
    );
    geokeys.set(geokeys::PROJECTED_CS_TYPE, GeoKeyValue::Short(32615));
    geokeys.set(geokeys::PROJ_LINEAR_UNITS, GeoKeyValue::Short(9001));

    let (directory, double_params, ascii_params) = geokeys.serialize();
    let mut tags_out = vec![
        Tag::new(
            tags::TAG_MODEL_PIXEL_SCALE,
            TagValue::Double(vec![30.0, 30.0, 0.0]),
        ),
        Tag::new(
            tags::TAG_MODEL_TIEPOINT,
            TagValue::Double(vec![0.0, 0.0, 0.0, 500_000.0, 4_500_000.0, 0.0]),
        ),
        Tag::new(tags::TAG_GEO_KEY_DIRECTORY, TagValue::Short(directory)),
    ];
    if !double_params.is_empty() {
        tags_out.push(Tag::new(
            tags::TAG_GEO_DOUBLE_PARAMS,
            TagValue::Double(double_params),
        ));
    }
    if !ascii_params.is_empty() {
        tags_out.push(Tag::new(
            tags::TAG_GEO_ASCII_PARAMS,
            TagValue::Ascii(ascii_params),
        ));
    }
    tags_out
}

fn write_generated_planar_geotiff(path: &std::path::Path) {
    let width = 16u32;
    let height = 16u32;
    let tile = 16u32;

    let file = File::create(path).unwrap();
    let writer = BufWriter::new(file);
    let mut tiff_writer = TiffWriter::new(writer, WriteOptions::default()).unwrap();

    let mut image = ImageBuilder::new(width, height)
        .sample_type::<u8>()
        .samples_per_pixel(3)
        .photometric(PhotometricInterpretation::Rgb)
        .planar_configuration(PlanarConfiguration::Planar)
        .compression(Compression::Deflate)
        .tiles(tile, tile);
    for tag in build_geo_tags() {
        image = image.tag(tag);
    }

    let handle = tiff_writer.add_image(image).unwrap();
    for band in 0..3u8 {
        let mut plane = vec![0u8; (tile * tile) as usize];
        for row in 0..height as usize {
            for col in 0..width as usize {
                plane[row * tile as usize + col] =
                    ((row * 13 + col * 5 + band as usize * 29) % 251) as u8;
            }
        }
        tiff_writer
            .write_block(&handle, band as usize, &plane)
            .unwrap();
    }

    tiff_writer.finish().unwrap();
}

#[test]
fn matches_gdal_geotiff_metadata_for_interoperability_corpus() {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL GeoTIFF parity test because Python GDAL bindings are unavailable");
        return;
    }

    let cases = [
        ("gdal/gcore/data/byte.tif", true),
        ("gdal/gcore/data/byte_point.tif", true),
        ("gdal/gcore/data/WGS_1984_Web_Mercator.tif", false),
        ("gdal/gcore/data/byte_with_ovr.tif", true),
        ("gdal/gcore/data/cog/byte_little_endian_golden.tif", true),
    ];

    for (relative_path, assert_epsg) in cases {
        let path = fixture(relative_path);
        let path_str = path.to_str().unwrap();
        let reference_json =
            reference::run_reference_json(env!("CARGO_MANIFEST_DIR"), &["metadata", path_str]);
        let file = GeoTiffFile::open(&path).unwrap();

        assert_eq!(
            file.width() as u64,
            reference_json["width"].as_u64().unwrap(),
            "{relative_path}"
        );
        assert_eq!(
            file.height() as u64,
            reference_json["height"].as_u64().unwrap(),
            "{relative_path}"
        );
        assert_eq!(
            file.band_count() as u64,
            reference_json["band_count"].as_u64().unwrap(),
            "{relative_path}"
        );
        assert_eq!(
            file.overview_count() as u64,
            reference_json["overview_count"].as_u64().unwrap(),
            "{relative_path}"
        );
        assert_eq!(
            area_or_point(&file),
            reference_json["area_or_point"].as_str().unwrap_or("Area"),
            "{relative_path}"
        );

        if assert_epsg {
            assert_eq!(
                file.epsg(),
                reference_json["epsg"].as_u64().map(|value| value as u32),
                "{relative_path}"
            );
        }

        let expected_transform = reference_json["geo_transform"]
            .as_array()
            .unwrap_or_else(|| panic!("missing geo_transform for {relative_path}"));
        let actual_transform = transform_tuple(&file)
            .unwrap_or_else(|| panic!("missing transform for {relative_path}"));
        for (index, expected) in expected_transform.iter().enumerate() {
            reference::assert_close(
                actual_transform[index],
                expected.as_f64().unwrap(),
                1e-9,
                relative_path,
            );
        }
    }
}

#[test]
fn matches_gdal_decoded_pixels_and_overviews() {
    if !reference::python_gdal_available() {
        eprintln!(
            "skipping GDAL GeoTIFF pixel parity test because Python GDAL bindings are unavailable"
        );
        return;
    }

    let cases = vec![
        (fixture("gdal/gcore/data/byte.tif"), None, SampleKind::U8),
        (
            fixture("gdal/gcore/data/byte_point.tif"),
            None,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/WGS_1984_Web_Mercator.tif"),
            None,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/byte_with_ovr.tif"),
            None,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/byte_with_ovr.tif"),
            Some(0usize),
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/cog/byte_little_endian_golden.tif"),
            None,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gdrivers/data/gtiff/int8.tif"),
            None,
            SampleKind::I8,
        ),
        (
            fixture("gdal/gcore/data/gtiff/byte_JPEG.tif"),
            None,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/byte_zstd.tif"),
            None,
            SampleKind::U8,
        ),
    ];

    for (path, overview_index, sample_kind) in cases {
        if path.ends_with("byte_JPEG.tif") {
            assert_gdal_u8_pixels_close(&path, overview_index);
        } else {
            assert_gdal_hash_matches(&path, overview_index, sample_kind);
        }
    }
}

#[test]
fn matches_gdal_for_generated_planar_geotiff() {
    if !reference::python_gdal_available() {
        eprintln!(
            "skipping GDAL planar GeoTIFF parity test because Python GDAL bindings are unavailable"
        );
        return;
    }

    let fixture = NamedTempFile::new().unwrap();
    write_generated_planar_geotiff(fixture.path());

    let path_str = fixture.path().to_str().unwrap();
    let reference_json =
        reference::run_reference_json(env!("CARGO_MANIFEST_DIR"), &["metadata", path_str]);
    let file = GeoTiffFile::open(fixture.path()).unwrap();

    assert_eq!(file.epsg(), Some(32615));
    assert_eq!(file.width(), 16);
    assert_eq!(file.height(), 16);
    assert_eq!(file.band_count(), 3);
    assert_eq!(file.tiff().ifd(0).unwrap().planar_configuration(), 2);
    assert_eq!(reference_json["epsg"].as_u64(), Some(32615));
    assert_eq!(reference_json["band_count"].as_u64(), Some(3));

    assert_gdal_hash_matches(fixture.path(), None, SampleKind::U8);
}
