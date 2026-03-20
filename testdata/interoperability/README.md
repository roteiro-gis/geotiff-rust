# Interoperability Corpus

This directory is the authoritative real-world TIFF/GeoTIFF interoperability corpus for the workspace.

All fixtures here are sourced from the public GDAL test-data repository:

- Repository: `https://github.com/OSGeo/gdal`
- License: inherited from the upstream GDAL project and its test data

The checked-in corpus is intentionally small and targeted. It covers the decoder features that this workspace claims to support:

- baseline classic TIFF / GeoTIFF decode
- `PixelIsPoint` vs `PixelIsArea` behavior
- GeoTIFF CRS extraction
- internal overview discovery
- BigTIFF parsing
- signed 8-bit samples
- JPEG-in-TIFF compression 7
- ZSTD compression
- COG-style local layout suitable for HTTP range tests

`manifest.tsv` records the source URL, checksum, and intended consumers for each fixture. Use `scripts/fetch-interoperability-corpus.sh` to refresh or re-verify the corpus.
