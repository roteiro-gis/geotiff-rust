#![cfg(feature = "local")]

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use geotiff_core::geokeys::{self, GeoKeyDirectory, GeoKeyValue};
use geotiff_core::tags;
use geotiff_reader::GeoTiffFile;
use ndarray::ArrayD;
use tempfile::NamedTempFile;
use tiff_core::{Compression, PhotometricInterpretation, PlanarConfiguration, Tag, TagValue};
use tiff_writer::{ImageBuilder, TiffWriter, WriteOptions};

#[path = "../../test-support/reference.rs"]
mod reference;

const RUST_IMPL_NAME: &str = "geotiff-rust";
const REFERENCE_IMPL_NAME: &str = "gdal";

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

fn write_benchmark_fixture(path: &Path) {
    let width = 2048u32;
    let height = 2048u32;
    let tile = 256u32;

    let file = File::create(path).unwrap();
    let writer = BufWriter::new(file);
    let mut tiff_writer = TiffWriter::new(writer, WriteOptions::default()).unwrap();

    let mut image = ImageBuilder::new(width, height)
        .sample_type::<u16>()
        .compression(Compression::Deflate)
        .tiles(tile, tile);
    for tag in build_geo_tags() {
        image = image.tag(tag);
    }

    let handle = tiff_writer.add_image(image).unwrap();
    let tiles_across = width / tile;
    let tiles_down = height / tile;
    let mut block_index = 0usize;
    for tile_row in 0..tiles_down {
        for tile_col in 0..tiles_across {
            let mut block = vec![0u16; (tile * tile) as usize];
            for local_row in 0..tile {
                for local_col in 0..tile {
                    let row = tile_row * tile + local_row;
                    let col = tile_col * tile + local_col;
                    let offset = (local_row * tile + local_col) as usize;
                    block[offset] = ((row * 97 + col * 13) % u16::MAX as u32) as u16;
                }
            }
            tiff_writer
                .write_block(&handle, block_index, &block)
                .unwrap();
            block_index += 1;
        }
    }

    tiff_writer.finish().unwrap();
}

fn write_planar_benchmark_fixture(path: &Path) {
    let width = 1024u32;
    let height = 1024u32;
    let tile = 256u32;

    let file = File::create(path).unwrap();
    let writer = BufWriter::new(file);
    let mut tiff_writer = TiffWriter::new(writer, WriteOptions::default()).unwrap();

    let mut image = ImageBuilder::new(width, height)
        .sample_type::<u16>()
        .samples_per_pixel(3)
        .photometric(PhotometricInterpretation::Rgb)
        .planar_configuration(PlanarConfiguration::Planar)
        .compression(Compression::Deflate)
        .tiles(tile, tile);
    for tag in build_geo_tags() {
        image = image.tag(tag);
    }

    let handle = tiff_writer.add_image(image).unwrap();
    let tiles_across = width / tile;
    let tiles_down = height / tile;
    let tiles_per_plane = (tiles_across * tiles_down) as usize;
    for band in 0..3u32 {
        for tile_row in 0..tiles_down {
            for tile_col in 0..tiles_across {
                let mut block = vec![0u16; (tile * tile) as usize];
                for local_row in 0..tile {
                    for local_col in 0..tile {
                        let row = tile_row * tile + local_row;
                        let col = tile_col * tile + local_col;
                        let offset = (local_row * tile + local_col) as usize;
                        block[offset] =
                            ((row * 41 + col * 17 + band * 101) % u16::MAX as u32) as u16;
                    }
                }
                let tile_index = (tile_row * tiles_across + tile_col) as usize;
                let block_index = band as usize * tiles_per_plane + tile_index;
                tiff_writer
                    .write_block(&handle, block_index, &block)
                    .unwrap();
            }
        }
    }

    tiff_writer.finish().unwrap();
}

fn rust_hash(path: &Path) -> (usize, String) {
    let file = GeoTiffFile::open(path).unwrap();
    assert_eq!(file.epsg(), Some(32615));
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

fn bench_open_and_full_decode(c: &mut Criterion) {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL benchmark because Python GDAL bindings are unavailable");
        return;
    }

    let fixture = NamedTempFile::new().unwrap();
    write_benchmark_fixture(fixture.path());

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_path = fixture.path().to_str().unwrap().to_string();
    let expected = rust_hash(fixture.path());
    assert_eq!(gdal_hash(manifest_dir, &fixture_path), expected);

    let mut group = c.benchmark_group("geotiff-reader/open-and-full-decode-vs-gdal");
    group.throughput(Throughput::Bytes(expected.0 as u64));

    group.bench_function(BenchmarkId::new(RUST_IMPL_NAME, "geotiff-reader"), |b| {
        b.iter_custom(|iters| {
            let iterations = usize::try_from(iters).expect("criterion iteration count overflowed");
            let start = Instant::now();
            for _ in 0..iterations {
                let file = GeoTiffFile::open(fixture.path()).unwrap();
                assert_eq!(file.epsg(), Some(32615));
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

fn bench_open_and_full_decode_planar(c: &mut Criterion) {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL planar benchmark because Python GDAL bindings are unavailable");
        return;
    }

    let fixture = NamedTempFile::new().unwrap();
    write_planar_benchmark_fixture(fixture.path());

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_path = fixture.path().to_str().unwrap().to_string();
    let expected = rust_hash(fixture.path());
    assert_eq!(gdal_hash(manifest_dir, &fixture_path), expected);

    let mut group = c.benchmark_group("geotiff-reader/open-and-full-decode-planar-vs-gdal");
    group.throughput(Throughput::Bytes(expected.0 as u64));

    group.bench_function(
        BenchmarkId::new(RUST_IMPL_NAME, "geotiff-reader-planar"),
        |b| {
            b.iter_custom(|iters| {
                let iterations =
                    usize::try_from(iters).expect("criterion iteration count overflowed");
                let start = Instant::now();
                for _ in 0..iterations {
                    let file = GeoTiffFile::open(fixture.path()).unwrap();
                    assert_eq!(file.epsg(), Some(32615));
                    let raster: ArrayD<u16> = file.read_raster().unwrap();
                    black_box(raster);
                }
                start.elapsed()
            });
        },
    );

    group.bench_function(
        BenchmarkId::new(REFERENCE_IMPL_NAME, "geotiff-reader-planar"),
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

criterion_group!(
    benches,
    bench_open_and_full_decode,
    bench_open_and_full_decode_planar
);
criterion_main!(benches);
