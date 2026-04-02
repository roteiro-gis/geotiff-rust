# Changelog

## 0.2.5

- Add pure-Rust TIFF/GeoTIFF `LERC` read support through the published `lerc-rust` crates.
- Add TIFF `LercParameters` parsing and support for TIFF-side `LERC+DEFLATE` and `LERC+ZSTD`.
- Add real GDAL interoperability fixtures for plain `LERC`, `LERC+DEFLATE`, `LERC+ZSTD`, and tiled separate-planar RGB `LERC`.
- Preserve the existing write surface; TIFF `LERC` write is not part of this release.
