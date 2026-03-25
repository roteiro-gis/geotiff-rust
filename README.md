# geotiff-rust

Pure-Rust TIFF/BigTIFF and GeoTIFF/COG readers and writers. No C libraries, no build scripts, no unsafe beyond `memmap2`.

## Crates

| Crate | Description |
|---|---|
| `tiff-core` | Shared types: ByteOrder, TagType, TagValue, TiffSample, compression/predictor enums |
| `tiff-reader` | TIFF/BigTIFF decoder with mmap, strip/tile reads, and typed raster decode |
| `tiff-writer` | TIFF/BigTIFF encoder with streaming writes, compression, predictors, and BigTIFF |
| `geotiff-core` | Shared GeoTIFF types: GeoKeyDirectory, CRS, GeoTransform, tag constants |
| `geotiff-reader` | GeoTIFF reader with CRS/transform extraction, overview discovery, and optional HTTP COG access |
| `geotiff-writer` | GeoTIFF/COG writer with fluent builder, streaming tile writes, and overview generation |

## Reading

```rust
use geotiff_reader::GeoTiffFile;

let file = GeoTiffFile::open("dem.tif")?;
println!("EPSG: {:?}, bounds: {:?}", file.epsg(), file.geo_bounds());
let raster: ndarray::ArrayD<f32> = file.read_raster()?;
```

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

## Features

**Read**
- Classic TIFF and BigTIFF
- Little-endian and big-endian byte orders
- Strip and tile data access with windowed reads
- Chunky and separate planar sample layouts
- Compression: Deflate, LZW, PackBits, JPEG (optional), ZSTD (optional)
- Parallel decompression via Rayon
- Typed raster reads into `ndarray::ArrayD` (u8 through f64)
- GeoKey directory, CRS/EPSG, transforms, NoData, overview discovery
- Optional HTTP range-backed remote COG access

**Write**
- Classic TIFF and BigTIFF with auto-detection
- Strip and tile layouts
- Compression: Deflate, LZW, ZSTD (optional)
- Predictors: horizontal differencing, floating-point byte interleaving
- Chunky and separate planar multi-band layouts (RGB/RGBA) and all sample types (u8 through f64)
- Streaming tile-by-tile writes for large rasters
- GeoTIFF metadata: EPSG, pixel scale, origin, affine transforms, NoData
- COG output with ghost IFD, overview generation (nearest-neighbor, average)

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `local` | yes | Local file reading via `tiff-reader` (geotiff-reader) |
| `rayon` | yes | Parallel strip/tile decompression (tiff-reader, geotiff-reader) |
| `jpeg` | yes | JPEG-in-TIFF support (tiff-reader) |
| `zstd` | yes | ZSTD compression (tiff-reader, tiff-writer) |
| `cog` | no | HTTP range-backed remote COG open (geotiff-reader) |

## Testing

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Reference-library parity tests are included for `tiff-reader` and
`geotiff-reader`. They compare this workspace against GDAL/libtiff when those
tools are available locally; otherwise they self-skip. Lossless codecs use
exact byte and hash parity. The JPEG fixture uses a strict bounded-delta check
because compliant decoders can differ by +/-1 in a small number of samples.

For a reproducible reference environment, run the Docker harness:

```sh
./scripts/run-reference-parity.sh
```

For reference comparisons and current benchmark results against GDAL/libtiff,
see [docs/benchmark-report.md](docs/benchmark-report.md).

## License

MIT OR Apache-2.0
