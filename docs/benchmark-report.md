# Benchmark Report

Date: 2026-03-21

This report summarizes the current Dockerized parity and comparison benchmark
suite for `geotiff-rust` against GDAL and libtiff. It captures the current
reference-parity status and the performance shape of the reader comparison
benches.

## System Under Test

- Machine: Apple M1
- CPU topology: 8 logical CPUs
- Memory: 16 GiB
- OS: macOS 13.0
- Architecture: `arm64`
- Rust toolchain: `rustc 1.92.0`
- Reference environment: Docker image with Rust 1.85, `gdal-bin`,
  `python3-gdal`, and `libtiff-tools`

These measurements reflect this machine. GDAL and libtiff ran in Docker, but
the timings still reflect the same host CPU and storage stack.

## Scope

- Dockerized parity run covering workspace tests plus GDAL/libtiff-backed
  reference parity tests
- `tiff-reader` full-decode comparison against the repo's GDAL helper
- `geotiff-reader` open-plus-full-decode comparison against the repo's GDAL helper

## Methodology

Commands used for this report:

```sh
./scripts/run-reference-parity.sh
./scripts/run-reference-benchmarks.sh
```

Notes:

- The parity run completed cleanly inside Docker.
- The `tiff-reader` benchmark uses a synthetic 2048x2048 tiled,
  Deflate-compressed `u16` TIFF fixture generated at benchmark time.
- The `geotiff-reader` benchmark uses a matching synthetic GeoTIFF fixture with
  `EPSG:32615` metadata.
- Both benches validate byte length and raster hash equality against the GDAL
  helper before timing.
- The comparison target is the repo's Python GDAL helper, not a direct GDAL C API benchmark.

## Current Results

### Parity

- `cargo test --workspace` passed inside the Docker reference environment
- `tiff-reader` GDAL/libtiff reference parity tests passed
- `geotiff-reader` GDAL reference parity tests passed

### Summary

| workload | geotiff-rust | GDAL | result |
| --- | ---: | ---: | --- |
| `tiff-reader` full decode | 4.80 ms | 11.21 ms | `geotiff-rust` 2.33x faster |
| `tiff-reader` throughput | 1.63 GiB/s | 0.70 GiB/s | `geotiff-rust` 2.33x higher throughput |
| `geotiff-reader` open + full decode | 6.54 ms | 16.69 ms | `geotiff-rust` 2.55x faster |
| `geotiff-reader` throughput | 1.19 GiB/s | 0.47 GiB/s | `geotiff-rust` 2.55x higher throughput |

## Interpretation

- Both current reader comparison benches favor `geotiff-rust` on this host.
- The GeoTIFF path is slower than the raw TIFF path for both implementations,
  which is consistent with the additional open and metadata work.
- The parity run confirms that the faster Rust timings still correspond to
  matching decoded raster content and metadata against the reference stack.
- Because the benchmark target is the repo's Python GDAL helper, these numbers
  should be read as the current end-to-end reference-harness cost rather than a
  direct GDAL C API ceiling.

## Limits

- This report reflects one machine.
- The benchmark fixtures are synthetic and intentionally narrow; they do not
  cover every real-world TIFF or GeoTIFF workload shape.
- The parity suite uses real interoperability fixtures, but the benchmark suite
  does not currently time that broader corpus.
- Docker improves reproducibility here, but containerized results remain host-specific.
