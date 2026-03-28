use std::path::Path;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use geotiff_reader::GeoTiffFile;
use geotiff_writer::{
    CogBuilder, Compression, GeoTiffBuilder, PhotometricInterpretation, PlanarConfiguration,
};
use ndarray::{Array3, ArrayD};
use tempfile::NamedTempFile;

#[path = "../../test-support/reference.rs"]
mod reference;

const RUST_IMPL_NAME: &str = "geotiff-rust";
const REFERENCE_IMPL_NAME: &str = "gdal";

fn write_multiband_planar_cog_fixture(path: &Path) {
    let mut data = Array3::<u16>::zeros((1024, 1024, 3));
    for row in 0..1024 {
        for col in 0..1024 {
            data[[row, col, 0]] = ((row * 17 + col * 3) % u16::MAX as usize) as u16;
            data[[row, col, 1]] = ((row * 7 + col * 19) % u16::MAX as usize) as u16;
            data[[row, col, 2]] = ((row * 23 + col * 11) % u16::MAX as usize) as u16;
        }
    }

    let builder = GeoTiffBuilder::new(1024, 1024)
        .bands(3)
        .photometric(PhotometricInterpretation::Rgb)
        .planar_configuration(PlanarConfiguration::Planar)
        .tile_size(256, 256)
        .compression(Compression::Deflate)
        .epsg(4326)
        .pixel_scale(1.0, 1.0)
        .origin(-180.0, 90.0);

    CogBuilder::new(builder)
        .overview_levels(vec![2, 4])
        .write_3d(path, data.view())
        .unwrap();
}

fn rust_hash(path: &Path) -> (usize, String) {
    let file = GeoTiffFile::open(path).unwrap();
    let raster: ArrayD<u16> = file.read_raster().unwrap();
    reference::array_hash(&raster)
}

fn gdal_hash(manifest_dir: &str, fixture_path: &str) -> (usize, String) {
    let reference_json = reference::run_reference_json(manifest_dir, &["hash", fixture_path]);
    (
        reference_json["byte_len"].as_u64().unwrap() as usize,
        reference_json["hash"].as_str().unwrap().to_string(),
    )
}

fn gdal_benchmark_duration(
    manifest_dir: &str,
    fixture_path: &str,
    iterations: usize,
    expected: &(usize, String),
) -> Duration {
    let iteration_arg = iterations.to_string();
    let reference_json = reference::run_reference_json(
        manifest_dir,
        &["benchmark", fixture_path, "--iterations", &iteration_arg],
    );

    assert_eq!(
        reference_json["byte_len"].as_u64().unwrap() as usize,
        expected.0
    );
    assert_eq!(reference_json["hash"].as_str().unwrap(), expected.1);

    Duration::from_secs_f64(reference_json["total_seconds"].as_f64().unwrap())
}

fn bench_open_and_full_decode_multiband_planar_cog(c: &mut Criterion) {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL benchmark because Python GDAL bindings are unavailable");
        return;
    }

    let fixture = NamedTempFile::new().unwrap();
    write_multiband_planar_cog_fixture(fixture.path());

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_path = fixture.path().to_str().unwrap().to_string();
    let expected = rust_hash(fixture.path());
    assert_eq!(gdal_hash(manifest_dir, &fixture_path), expected);

    let mut group = c.benchmark_group("geotiff-writer/multiband-planar-cog-decode-vs-gdal");
    group.throughput(Throughput::Bytes(expected.0 as u64));

    group.bench_function(BenchmarkId::new(RUST_IMPL_NAME, "geotiff-reader"), |b| {
        b.iter_custom(|iters| {
            let iterations = usize::try_from(iters).expect("criterion iteration count overflowed");
            let start = Instant::now();
            for _ in 0..iterations {
                let file = GeoTiffFile::open(fixture.path()).unwrap();
                let raster: ArrayD<u16> = file.read_raster().unwrap();
                black_box(raster);
            }
            start.elapsed()
        });
    });

    group.bench_function(
        BenchmarkId::new(REFERENCE_IMPL_NAME, "geotiff-reader"),
        |b| {
            b.iter_custom(|iters| {
                let iterations =
                    usize::try_from(iters).expect("criterion iteration count overflowed");
                gdal_benchmark_duration(manifest_dir, &fixture_path, iterations, &expected)
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_open_and_full_decode_multiband_planar_cog);
criterion_main!(benches);
