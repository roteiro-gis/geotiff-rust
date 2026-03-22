use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use ndarray::ArrayD;
use tempfile::NamedTempFile;
use tiff_core::Compression;
use tiff_reader::TiffFile;
use tiff_writer::{ImageBuilder, TiffWriter, WriteOptions};

#[path = "../../test-support/reference.rs"]
mod reference;

fn write_benchmark_fixture(path: &std::path::Path) {
    let width = 2048u32;
    let height = 2048u32;
    let tile = 256u32;

    let file = File::create(path).unwrap();
    let writer = BufWriter::new(file);
    let mut tiff_writer = TiffWriter::new(writer, WriteOptions::default()).unwrap();
    let image = ImageBuilder::new(width, height)
        .sample_type::<u16>()
        .compression(Compression::Deflate)
        .tiles(tile, tile);
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
                    block[offset] = ((row * 31 + col * 17) % u16::MAX as u32) as u16;
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

fn median_seconds(samples: &[f64]) -> f64 {
    let mut values = samples.to_vec();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    values[values.len() / 2]
}

#[test]
#[ignore = "runs an explicit timing comparison against GDAL"]
fn compares_full_decode_throughput_against_gdal() {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL benchmark because Python GDAL bindings are unavailable");
        return;
    }

    let fixture = NamedTempFile::new().unwrap();
    write_benchmark_fixture(fixture.path());

    let iterations = std::env::var("GEOTIFF_RUST_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let fixture_path = fixture.path().to_str().unwrap().to_string();
    let iteration_arg = iterations.to_string();
    let reference_json = reference::run_reference_json(
        env!("CARGO_MANIFEST_DIR"),
        &["benchmark", &fixture_path, "--iterations", &iteration_arg],
    );

    let mut timings = Vec::with_capacity(iterations);
    let mut hashes = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let file = TiffFile::open(fixture.path()).unwrap();
        let raster: ArrayD<u16> = file.read_image(0).unwrap();
        timings.push(start.elapsed().as_secs_f64());
        hashes.push(reference::array_hash(&raster));
    }

    let (byte_len, hash) = hashes[0].clone();
    assert!(hashes.iter().all(|value| value == &hashes[0]));
    assert_eq!(
        reference_json["byte_len"].as_u64().unwrap() as usize,
        byte_len
    );
    assert_eq!(reference_json["hash"].as_str().unwrap(), hash);

    let rust_median = median_seconds(&timings);
    let gdal_median = reference_json["median_seconds"].as_f64().unwrap();
    let slowdown = rust_median / gdal_median;

    println!(
        "tiff-reader median={rust_median:.6}s gdal median={gdal_median:.6}s slowdown={slowdown:.3}x"
    );

    if let Some(limit) = std::env::var("GEOTIFF_RUST_BENCH_MAX_SLOWDOWN")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
    {
        assert!(
            slowdown <= limit,
            "tiff-reader slowdown {slowdown:.3}x exceeded configured limit {limit:.3}x"
        );
    }
}
