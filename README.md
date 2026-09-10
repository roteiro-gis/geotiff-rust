# geotiff-rust

[![tiff-core crates.io](https://img.shields.io/crates/v/tiff-core.svg)](https://crates.io/crates/tiff-core)
[![tiff-core docs.rs](https://docs.rs/tiff-core/badge.svg)](https://docs.rs/tiff-core)
[![tiff-reader crates.io](https://img.shields.io/crates/v/tiff-reader.svg)](https://crates.io/crates/tiff-reader)
[![tiff-reader docs.rs](https://docs.rs/tiff-reader/badge.svg)](https://docs.rs/tiff-reader)
[![tiff-writer crates.io](https://img.shields.io/crates/v/tiff-writer.svg)](https://crates.io/crates/tiff-writer)
[![tiff-writer docs.rs](https://docs.rs/tiff-writer/badge.svg)](https://docs.rs/tiff-writer)
[![geotiff-core crates.io](https://img.shields.io/crates/v/geotiff-core.svg)](https://crates.io/crates/geotiff-core)
[![geotiff-core docs.rs](https://docs.rs/geotiff-core/badge.svg)](https://docs.rs/geotiff-core)
[![geotiff-reader crates.io](https://img.shields.io/crates/v/geotiff-reader.svg)](https://crates.io/crates/geotiff-reader)
[![geotiff-reader docs.rs](https://docs.rs/geotiff-reader/badge.svg)](https://docs.rs/geotiff-reader)
[![geotiff-writer crates.io](https://img.shields.io/crates/v/geotiff-writer.svg)](https://crates.io/crates/geotiff-writer)
[![geotiff-writer docs.rs](https://docs.rs/geotiff-writer/badge.svg)](https://docs.rs/geotiff-writer)

Pure-Rust TIFF/BigTIFF and GeoTIFF/COG readers and writers. No C libraries, no build scripts; unsafe is limited to explicitly opt-in `memmap2` reading.

## Crates

| Crate | Description |
|---|---|
| `tiff-core` | Shared TIFF types: ByteOrder, tags, sample traits, compression/predictor enums, and color-model metadata |
| `tiff-reader` | TIFF/BigTIFF decoder with safe file-backed random access, opt-in mmap, strip/tile reads, storage-domain reads, and explicit decoded pixel access |
| `tiff-writer` | TIFF/BigTIFF encoder with streaming writes, compression, predictors, and BigTIFF |
| `geotiff-core` | Shared GeoTIFF types: GeoKeyDirectory, CRS, GeoTransform, tag constants |
| `geotiff-reader` | GeoTIFF reader with CRS/transform extraction, overview discovery, and optional HTTP COG access |
| `geotiff-writer` | GeoTIFF/COG writer with fluent builder, tile-wise GeoTIFF writes, and overview generation |

## Reading

```rust
use geotiff_reader::GeoTiffFile;

let file = GeoTiffFile::open("dem.tif")?;
println!("EPSG: {:?}, bounds: {:?}", file.epsg(), file.geo_bounds());
let raster: ndarray::ArrayD<f32> = file.read_raster()?;
```

Use `read_decoded_raster` / `read_decoded_window` on `GeoTiffFile` and
`read_decoded_image` / `read_decoded_window` on `TiffFile` when you want
palette expansion or color-space conversion (for example palette TIFF,
YCbCr, or CMYK) instead of storage-domain samples.

Use `read_band` / `read_band_window` on `GeoTiffFile` and `TiffFile` when
you only need one storage-domain band as a `[rows, cols]` array.

Enable the non-default `f16` feature on the reader or writer crate to use
`half::f16` rasters. These files use TIFF `SampleFormat=Float` with
`BitsPerSample=16`; byte order, Deflate/LZW/ZSTD compression, and the
floating-point predictor are supported, while JPEG and LERC are not.

## Writing

```rust
use geotiff_writer::{GeoTiffBuilder, Compression};
use ndarray::Array2;

let data = Array2::<f32>::zeros((256, 256));
GeoTiffBuilder::new(256, 256)
    .epsg(4326)
    .pixel_scale(0.01, 0.01)
    .origin(-180.0, 90.0)
    .nodata("-9999")
    .compression(Compression::Deflate)
    .write_2d("output.tif", data.view())?;
```

For separate-planar multiband output, set
`planar_configuration(PlanarConfiguration::Planar)` on `ImageBuilder` or
`GeoTiffBuilder`.

### Streaming tile writes

```rust
use geotiff_writer::GeoTiffBuilder;
use ndarray::Array2;

let builder = GeoTiffBuilder::new(512, 512)
    .tile_size(256, 256)
    .epsg(4326);
let mut writer = builder.tile_writer_file::<f32, _>("large.tif")?;
for (x, y, tile) in tiles {
    writer.write_tile(x, y, &tile.view())?;
}
writer.finish()?;
```

### COG with overviews

```rust
use geotiff_writer::{GeoTiffBuilder, CogBuilder, Resampling, Compression};
use ndarray::Array2;

let data = Array2::<u8>::zeros((1024, 1024));
CogBuilder::new(
    GeoTiffBuilder::new(1024, 1024)
        .tile_size(256, 256)
        .compression(Compression::Deflate)
        .epsg(4326)
)
.overview_levels(vec![2, 4, 8])
.resampling(Resampling::Average)
.write_2d("output.tif", data.view())?;
```

For multi-band COG output, use `write_3d`/`write_3d_to` or `write_tile_3d`
with `bands(...)` and optional
`planar_configuration(PlanarConfiguration::Planar)`.

## Features

**Read**
- Classic TIFF and BigTIFF
- Little-endian and big-endian byte orders
- Strip and tile data access with windowed reads
- Chunky and separate planar sample layouts
- Full-raster and windowed single-band reads, optimized for separate-planar rasters
- Compression: Deflate, LZW, PackBits, LERC, LERC+DEFLATE, JPEG (optional), ZSTD (optional), LERC+ZSTD (optional), WebP decode (optional)
- Bounded IFD parsing and block decompression budgets for untrusted input
- Parallel decompression via Rayon
- Storage-domain typed sample reads via `read_image` / `read_window` / `read_band*`
- Explicit decoded pixel reads via `read_decoded_*` for standard TIFF color models, including palette expansion, YCbCr/CMYK conversion, and sub-byte grayscale/palette decode
- Structured photometric/color-model metadata: palette `ColorMap`, `ExtraSamples`, CMYK, and YCbCr
- GeoKey directory, structured CRS metadata (projected, geographic, geocentric, vertical, compound), transforms, NoData
- Overview discovery from both reduced-resolution top-level IFDs and recursive base-image SubIFD-backed overview trees
- Optional HTTP range-backed remote COG access, blocking (`cog`) or async Tokio-based (`cog-async`)

**Write**
- Classic TIFF and BigTIFF with auto-detection
- Strip and tile layouts
- Compression: Deflate, LZW, JPEG (optional), LERC, LERC+DEFLATE, ZSTD (optional), LERC+ZSTD (optional)
- Predictors: horizontal differencing, floating-point byte interleaving
- Chunky and separate planar multi-band layouts and all sample types (u8 through f64)
- Photometric/color-model tags: palette `ColorMap`, `ExtraSamples` alpha, CMYK (`Separated` + `InkSet`), and YCbCr 4:4:4 or JPEG 4:2:0
- Streaming tile-by-tile GeoTIFF writes for large rasters
- GeoTIFF metadata: GeoTIFF 1.1 key-directory emission, projected/geographic/geocentric/vertical compound CRS keys, pixel scale, origin, affine transforms, NoData
- COG output with GDAL-compatible ghost-area metadata, overview generation (nearest-neighbor, average), top-level or SubIFD-backed overview IFDs, and multi-band chunky/planar rasters
- Tile-wise COG assembly via `CogTileWriter` (base tiles are staged in a temporary raw tile store before final emission)
- `wasm32-unknown-unknown` support: COG assembly stages blocks in memory instead of a temporary file

## Codec Notes

`JPEG`-in-TIFF write uses standard compression code `7` with full JPEG
interchange streams per strip/tile. Supported interoperable layouts include
single-band chunky output, multi-band separate-planar output, and chunky
three-sample YCbCr with 4:4:4 or 4:2:0 chroma sampling. The emitted TIFF,
GeoTIFF, and COG files do not require TIFF-side shared `JPEGTables`.

TIFF `LERC` writing records the registered LERC2 2.4 parameter version used by
GDAL/libtiff. If the encoder produces a different LERC2 container version, the
writer rejects the block before writing incompatible TIFF metadata.

## Robustness Notes

`TiffFile::open_with_options`, `from_bytes_with_options`, and
`from_source_with_options` accept `OpenOptions` with `ParseBudgets` for bounding
IFD chain length, tag counts, and metadata payload bytes. `GeoTiffFile`
exposes the same settings as `GeoTiffOpenOptions`, and HTTP COG opens pass them
through `HttpOpenOptions::tiff_options`. GeoTIFF SubIFD overview discovery is
also bounded by explicit node and depth limits before following arbitrary
SubIFD offsets. The reader derives encoded strip/tile read limits and
decompressed output limits from the raster layout before reading block
payloads. `OpenOptions::decode_output_bytes` bounds each decoded output buffer
allocation; raise it only when intentionally decoding larger windows or full
rasters.

Remote COG opens use bounded connect, read, and per-request timeouts by default.
`HttpOpenOptions` also accepts per-request headers and an optional custom
`reqwest::blocking::Client` for authenticated object stores, proxies, or custom
TLS policy.

Range requests are deduplicated and coalesced. Concurrent readers that miss the
same cached chunk share a single in-flight fetch rather than each issuing their
own request, and a read spanning several uncached chunks is served by one
coalesced request per contiguous run instead of one request per chunk.
`max_coalesced_chunks` bounds how much a single request may cover; set it to 1
to disable coalescing.

Local `open` constructors use safe file-backed random access by default.
Memory-mapped local opens are available through `unsafe` `open_mmap`
constructors when the caller can guarantee that the mapped file will not be
mutated or truncated while it is open.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `local` | yes | Local file reading via `tiff-reader` (geotiff-reader) |
| `rayon` | yes | Parallel strip/tile decompression (tiff-reader, geotiff-reader) |
| `jpeg` | yes | JPEG-in-TIFF read/write support (tiff-reader, tiff-writer) |
| `zstd` | yes | Pure-Rust ZSTD compression via `ruzstd`, including TIFF `LERC+ZSTD` read/write support (tiff-reader, tiff-writer) |
| `webp` | yes | Pure-Rust WebP-in-TIFF decoding (tiff-reader) |
| `f16` | no | IEEE 16-bit floating-point raster samples (all TIFF/GeoTIFF crates) |
| `cog` | no | HTTP range-backed remote COG open with rustls TLS by default (geotiff-reader) |
| `cog-async` | no | Async Tokio-based HTTP range-backed remote COG open (geotiff-reader) |
| `rustls-tls` | no | Rustls-backed HTTPS transport for `geotiff-reader` remote COG opens |

## Testing

```sh
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Reference-library parity tests compare this workspace against GDAL/libtiff
when those tools are available locally; otherwise they self-skip. Reader tests
cover TIFF and GeoTIFF fixtures, and writer tests validate generated TIFF,
GeoTIFF, and COG outputs through reference-library metadata and decoded-pixel
checks. Lossless codecs use exact byte and hash parity. JPEG cases use strict
bounded-delta checks because compliant decoders can differ slightly.

For a reproducible reference environment, run the Docker harness:

```sh
./scripts/run-reference-parity.sh
```

For reference comparisons and current benchmark results against GDAL/libtiff,
see [docs/benchmark-report.md](docs/benchmark-report.md).

For the workspace release order and package verification notes, see
[docs/publishing.md](docs/publishing.md).

## License

MIT OR Apache-2.0
