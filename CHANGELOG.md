# Changelog

## 0.4.0

- Add JPEG-in-TIFF write support across `tiff-writer`, `geotiff-writer`, and COG output using standard compression code `7`.
- Add explicit decoded-pixel read APIs while preserving storage-domain sample reads, including palette expansion, sub-byte grayscale/palette decoding, YCbCr conversion, and CMYK conversion.
- Add structured TIFF color-model metadata for `ColorMap`, `ExtraSamples`, CMYK `InkSet`, YCbCr tags, and extended photometric interpretations.
- Add richer GeoTIFF CRS modeling for projected, geographic, geocentric, vertical, and compound CRS metadata.
- Discover overviews from both reduced-resolution top-level IFDs and recursive SubIFD overview trees.
- Improve COG generation with exact BigTIFF auto-selection, disk-backed `CogTileWriter` assembly, GDAL-compatible block ordering/ghost metadata, and nodata-aware average overviews.
- Harden writer validation for streaming tile offsets, band-count mismatches, unsupported YCbCr subsampling, and JPEG layouts that are not interoperable with GDAL/libtiff.
- Prepare crates.io publishing metadata for the workspace crates and use the published `lerc-rust` 0.3 crates from the registry.

## 0.3.1

- Move cross-crate release tests into non-publishable integration crates so publishable package tarballs stay focused.
- Fix release-time dev-dependency constraints for the workspace test crates.

## 0.3.0

- Add pure-Rust TIFF `LERC` write support through the published `lerc-rust` 0.3 crates.
- Add GeoTIFF and COG `LERC`, `LERC+DEFLATE`, and `LERC+ZSTD` write support.
- Move `LercOptions` into `tiff-writer` and expose consistent builder configuration for TIFF and GeoTIFF writers.
- Add roundtrip and reference coverage for LERC writer behavior.

## 0.2.5

- Add pure-Rust TIFF/GeoTIFF `LERC` read support through the published `lerc-rust` crates.
- Add TIFF `LercParameters` parsing and support for TIFF-side `LERC+DEFLATE` and `LERC+ZSTD`.
- Add real GDAL interoperability fixtures for plain `LERC`, `LERC+DEFLATE`, `LERC+ZSTD`, and tiled separate-planar RGB `LERC`.
- Preserve the existing write surface; TIFF `LERC` write is not part of this release.
