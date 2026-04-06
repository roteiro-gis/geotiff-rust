use std::path::Path;

use geotiff_reader::GeoTiffFile;
use geotiff_writer::{
    CogBuilder, Compression, GeoTiffBuilder, PhotometricInterpretation, PlanarConfiguration,
    Resampling,
};
use ndarray::{Array3, ArrayD};
use tempfile::NamedTempFile;

#[path = "../../../test-support/reference.rs"]
mod reference;

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

fn write_generated_planar_multiband_cog(path: &Path) {
    let mut data = Array3::<u8>::zeros((64, 64, 3));
    for row in 0..64 {
        for col in 0..64 {
            data[[row, col, 0]] = ((row * 5 + col * 3) % 251) as u8;
            data[[row, col, 1]] = ((row * 11 + col * 7) % 251) as u8;
            data[[row, col, 2]] = ((row * 13 + col * 17) % 251) as u8;
        }
    }

    let builder = GeoTiffBuilder::new(64, 64)
        .bands(3)
        .photometric(PhotometricInterpretation::Rgb)
        .planar_configuration(PlanarConfiguration::Planar)
        .tile_size(16, 16)
        .compression(Compression::Deflate)
        .epsg(4326)
        .pixel_scale(1.0, 1.0)
        .origin(-180.0, 90.0);

    CogBuilder::new(builder)
        .overview_levels(vec![2, 4])
        .resampling(Resampling::NearestNeighbor)
        .write_3d(path, data.view())
        .unwrap();
}

fn assert_gdal_hash_matches(path: &Path, overview_index: Option<usize>) {
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
    let width = reference_json["width"].as_u64().unwrap() as u32;
    let height = reference_json["height"].as_u64().unwrap() as u32;
    let band_count = reference_json["band_count"].as_u64().unwrap();
    let byte_len = reference_json["byte_len"].as_u64().unwrap() as usize;
    let expected_hash = reference_json["hash"].as_str().unwrap();

    let raster: ArrayD<u8> = match overview_index {
        Some(index) => file.read_overview(index).unwrap(),
        None => file.read_raster().unwrap(),
    };
    assert_shape(&raster, width, height, band_count);
    let (actual_len, actual_hash) = reference::array_hash(&raster);
    assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
    assert_eq!(
        actual_hash, expected_hash,
        "pixel hash mismatch for {path_str}"
    );
}

#[test]
fn matches_gdal_for_generated_planar_multiband_cog() {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL parity test because Python GDAL bindings are unavailable");
        return;
    }

    let fixture = NamedTempFile::new().unwrap();
    write_generated_planar_multiband_cog(fixture.path());

    let path_str = fixture.path().to_str().unwrap();
    let reference_json =
        reference::run_reference_json(env!("CARGO_MANIFEST_DIR"), &["metadata", path_str]);
    let file = GeoTiffFile::open(fixture.path()).unwrap();

    assert_eq!(file.epsg(), Some(4326));
    assert_eq!(
        file.width() as u64,
        reference_json["width"].as_u64().unwrap()
    );
    assert_eq!(
        file.height() as u64,
        reference_json["height"].as_u64().unwrap()
    );
    assert_eq!(
        file.band_count() as u64,
        reference_json["band_count"].as_u64().unwrap()
    );
    assert_eq!(
        file.overview_count() as u64,
        reference_json["overview_count"].as_u64().unwrap()
    );
    assert_eq!(reference_json["interleave"].as_str(), Some("BAND"));

    assert_gdal_hash_matches(fixture.path(), None);
    for overview_index in 0..file.overview_count() {
        assert_gdal_hash_matches(fixture.path(), Some(overview_index));
    }
}
