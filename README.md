# geotiff-rust

Pure-Rust, read-only decoders for TIFF/BigTIFF and GeoTIFF/COG. No C libraries, no build scripts, and no unsafe beyond `memmap2`.

## Crates

| Crate | Description |
|---|---|
| `tiff-reader` | TIFF/BigTIFF decoder with random-access sources, strip/tile reads, and typed raster decode |
| `geotiff-reader` | GeoTIFF reader with CRS/transform extraction, overview discovery, and optional HTTP range-backed remote open |

## Usage

```rust
use geotiff_reader::GeoTiffFile;

let file = GeoTiffFile::open("dem.tif")?;
println!("CRS EPSG: {:?}", file.epsg());
println!("bounds: {:?}", file.geo_bounds());
println!("size: {}x{} ({} bands)", file.width(), file.height(), file.band_count());
println!("nodata: {:?}", file.nodata());
```

Using `tiff-reader` directly:

```rust
use tiff_reader::TiffFile;

let file = TiffFile::open("image.tif")?;
println!("byte order: {:?}", file.byte_order());
println!("BigTIFF: {}", file.is_bigtiff());

for i in 0..file.ifd_count() {
    let ifd = file.ifd(i)?;
    println!("  IFD {}: {}x{}, compression={}", i, ifd.width(), ifd.height(), ifd.compression());
    println!("    tiled: {}, bands: {}", ifd.is_tiled(), ifd.samples_per_pixel());
}

let pixels: ndarray::ArrayD<u16> = file.read_image(0)?;
```

## Features

**TIFF**
- Classic TIFF and BigTIFF support
- Little-endian and big-endian byte orders
- IFD chain traversal (multi-page/multi-image)
- Random-access decode from mmap, in-memory bytes, or custom sources
- Strip and tile data access
- Compression: Deflate, LZW, PackBits, JPEG-in-TIFF compression 7 (optional), ZSTD (optional)
- Parallel strip/tile decompression via Rayon
- All standard tag types (BYTE through IFD8)
- Typed raster reads into `ndarray::ArrayD`

**GeoTIFF**
- GeoKey directory parsing (tag 34735)
- CRS/EPSG extraction (ProjectedCSType, GeographicType)
- Model tiepoint and pixel scale (tags 33922, 33550)
- Model transformation matrix (tag 34264)
- Correct `PixelIsArea`/`PixelIsPoint` normalization for affine bounds and transforms
- Nodata from GDAL_NODATA tag (42113)
- Band interleaving detection
- Pixel-to-geographic coordinate transforms
- Overview discovery for reduced-resolution GeoTIFF IFDs
- Optional HTTP range-backed remote COG/GeoTIFF access

## Feature flags

```toml
[dependencies]
geotiff-reader = "0.1"              # local file reading (default)
geotiff-reader = { version = "0.1", features = ["cog"] }  # + HTTP range-backed remote open
```

| Flag | Default | Description |
|---|---|---|
| `local` | yes | Local file reading via `tiff-reader` |
| `rayon` | yes | Parallel strip/tile decompression |
| `jpeg` | yes | JPEG-in-TIFF compression 7 support, including `JPEGTables`-backed abbreviated streams |
| `zstd` | yes | ZSTD compression support (tiff-reader) |
| `cog` | no | Enable HTTP range-backed remote GeoTIFF/COG open; implies `local` |

## Testing

```sh
cargo test --all-features
```

## License

MIT OR Apache-2.0
