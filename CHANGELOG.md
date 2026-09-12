# Changelog

## Unreleased

- upgrade `weezl` to 0.2.1 and raise the minimum supported Rust version to 1.88
- deduplicate concurrent remote COG range fetches: the blocking and async range
  caches now track in-flight chunks, so parallel tile decode shares one request
  per chunk instead of racing. Reading a 5.6 MB tiled COG previously transferred
  7.7 MB across 31 requests for 18 distinct ranges; it now transfers 4.5 MB
  across 18 requests with no range fetched twice
- coalesce contiguous uncached chunks into a single range request instead of
  fetching them one at a time, so a read spanning several chunks costs one round
  trip rather than one per chunk; bounded by the new `max_coalesced_chunks`
  field on `HttpOpenOptions` and `AsyncHttpOpenOptions` (default 16, set to 1 to
  disable). Adding the field is a breaking change for exhaustive struct literals
  that do not use `..Default::default()`
- make the HTTP test server handle each connection on its own thread and record
  served byte ranges, so duplicated and sequential range fetches are visible to
  the test suite; the previous serial accept loop serialized clients and hid
  both. Accepted sockets are now explicitly set to blocking mode, which also
  addresses intermittent request-read failures under parallel test load

## 0.8.1 - 2026-08-11

- support COG writing on `wasm32-unknown-unknown` by staging blocks and raw tiles in memory while retaining temporary-file spooling on native targets
- pad the GDAL COG ghost area to a 2-byte boundary so the first IFD is word-aligned and accepted by GDAL's COG validator across classic TIFF, BigTIFF, chunky, planar, one-shot, and tile-wise output
- make the GDAL COG validator parity test detect the separately packaged `osgeo_utils` module before running

## 0.8.0 - 2026-07-25

Breaking changes:

- raise the minimum supported Rust version from 1.77 to 1.85
- remove the `ImageBuilder` helpers deprecated since 0.6.0 (`block_count`, `block_sample_count`, `estimated_uncompressed_bytes`, `layout_tags`, `build_tags`); use the `checked_*` equivalents
- collapse duplicate reader methods: `read_*_samples` and `read_*_sample_bytes` aliases fold into `read_image` / `read_window` / `read_band*` and the `*_bytes` variants on `TiffFile` and `GeoTiffFile`
- `Ifd::rows_per_strip` returns `u32` (it always resolved to a value); `Ifd::bits_per_sample` / `Ifd::sample_format` are now the validating `Result` accessors introduced in this release
- `Ifd::index` is now `Option<usize>`: top-level chain IFDs have `Some(index)`, while IFDs read directly by offset (including SubIFDs) have no fabricated chain index
- `GeoTiffBuilder::write_2d`/`write_3d` require `NumericSample` (implemented for every supported sample type) so plain GeoTIFF writes validate nodata and pad edge blocks with the nodata fill
- `BlockKey.ifd_index` is now `BlockKey.ifd_offset: u64` and `BlockEncodingOptions` gains a `deflate_level` field
- `WriteOptions::auto()` no longer accepts an unused estimated-size argument; final TIFF/BigTIFF selection always uses the exact completed layout
- `tiff_writer::encoder::estimate_ifd_size` is fallible and reports arithmetic overflow instead of wrapping an unrepresentable layout
- predictor setters no longer silently discard requests made after selecting JPEG or LERC; incompatible combinations are retained and rejected during validation
- remove unused `geotiff_reader::Error` variants (`UnsupportedModelType`, `UnknownEpsg`, `BandOutOfBounds`, `NoGeoTransform`); `lru`/`parking_lot` are only pulled in by remote COG features and direct `memmap2`/`smallvec` dependencies are dropped

Other changes:

- add optional `f16` features to the TIFF and GeoTIFF readers/writers, enabling `half::f16` rasters encoded as `SampleFormat=Float` with 16 bits per sample
- add an async Tokio-based remote COG reader behind a new `cog-async` feature: `AsyncHttpGeoTiffFile` fetches ranges with the async `reqwest` client and runs TIFF parsing/decoding on the blocking pool, mirroring the blocking reader's chunk cache, Content-Range validation, and timeouts
- decode WebP-compressed TIFF blocks (compression 50001) behind a new default `webp` feature on `tiff-reader`, using the pure-Rust `image-webp` crate with the standard decoded-size budget checks
- write interleaved YCbCr JPEG (the standard web-visualization COG configuration): chunky 3-sample JPEG blocks with `PhotometricInterpretation::YCbCr` encode with 2x2 chroma subsampling by default, emit the matching `YCbCrSubsampling` tag, and validate against GDAL; `BlockEncodingOptions` gains a `jpeg_sampling` field
- add sparse tile/strip writing via `GeoTiffBuilder::sparse(true)` (GDAL `SPARSE_OK` semantics): all-zero blocks are recorded with zero offsets and byte counts across plain, streaming, and COG write paths, and `TiffWriter::write_block_sparse` exposes the primitive
- add a property-based writer→reader roundtrip suite covering sample types, compressions, predictors, planar layouts, strips/tiles, and both byte orders; a big-endian GDAL parity test for writer output; and synthetic sparse-block and video-range-YCbCr fuzz seeds
- restructure the GDAL ghost-area block reader into explicit wrapped/direct phases with a candidate-outcome test matrix, and move SubIFD tag-offset math into `tiff_writer::encoder::find_tag_value_offset`
- add `deflate_level(0..=9)` to `ImageBuilder` and `GeoTiffBuilder` for controlling Deflate output size/speed, plus `compress_with_level` in `tiff-writer::compress`; `BlockEncodingOptions` gains a `deflate_level` field
- speed up decode and write paths: switch the Deflate backend to the pure-Rust `zlib-rs` (~20-40% faster on predictor-compressed rasters), stream overview resampling by source row spans (~6x faster streaming COG average overviews), copy raster regions by row slice instead of per-element indexing (~4.5x faster plain multiband writes), resolve per-block decode metadata once per read, encode writer tag values once, and reuse the floating-point predictor scratch buffer across rows
- pad one-shot COG edge tiles with the configured nodata fill value instead of zero, matching the streaming tile writer and overview generation paths
- reject writer configurations that pair the horizontal predictor with float samples, the floating-point predictor with integer samples, or unsupported float widths; 16-bit float samples require the optional `f16` feature
- decode `BitsPerSample`/`SampleFormat` tags stored with nonstandard BYTE or LONG encodings instead of silently falling back to 1-bit defaults, and reject other unexpected tag types via new `Ifd::checked_bits_per_sample` / `Ifd::checked_sample_format` used by all decode paths
- round integer overview resampling to the nearest value instead of truncating toward zero, and reject GeoTIFF/COG nodata strings that are out of range or fractional for the raster sample type instead of silently saturating the fill value
- fix decoded YCbCr chroma scaling to honor the `ReferenceBlackWhite` chroma ranges per TIFF 6.0 (`127/(ReferenceMax - ReferenceZero)` for 8-bit samples) instead of always dividing by the full-scale range; output is unchanged for files using the default headroom-free references
- read sparse strips and tiles (zero offset or zero byte count, as written by GDAL `SPARSE_OK=TRUE`) as implicit zero-filled blocks instead of failing with a decode error
- fix decoded-block cache collisions between top-level chain IFDs and IFDs parsed at explicit file offsets (such as SubIFD overviews) by keying the cache on the IFD file offset; `BlockKey.ifd_index` is now `BlockKey.ifd_offset: u64` and `Ifd::offset()` exposes the owning file offset
- enforce the decoded-output budget for every intersecting storage block, and reject JPEG payloads whose encoded dimensions do not match the TIFF strip/tile geometry
- validate BigTIFF offset-size/reserved header fields, duplicate IFD tags, scalar LONG-to-SHORT range conversions, writer tag type/count coherence, block row geometry, and extra-tag conflicts before mutation or allocation
- reject unsupported or feature-disabled writer codecs and predictors without compression during image validation instead of failing after streaming begins
- bind image handles to their originating writer and reject repeated sparse, compressed, or raw block writes before appending payload bytes
- validate GeoKey headers, duplicate keys, inline counts, ASCII parameters, model tag types/counts, finite invertible transforms, and positive model pixel scales; preserve flipped/tiny-skew transforms with ModelTransformation instead of lossy tiepoint conversion
- fix overview discovery so candidates cannot grow either raster dimension and the same IFD referenced by both the top-level and SubIFD chains is reported only once
- preserve exact `u64`/`i64` nodata text at numeric limits, reject rounded out-of-range boundaries, and make later pixel-scale/origin/tiepoint builder calls replace an earlier transformation matrix
- keep the async-only COG feature free of reqwest's blocking client implementation and document the complete `f16`, WebP, and async COG feature surface

## 0.7.0 - 2026-06-20

- make memory-mapped local reads explicit through `open_mmap` / `open_mmap_with_options` and keep default file opens on safe file-backed I/O
- add decoded output allocation budgets to TIFF `OpenOptions` and enforce them before full-image, window, and color-decoded output buffers are allocated
- check tiled-read tile counts and tile indexes for `usize` overflow before reading tile payloads, and reject GeoTIFF writer input shapes whose dimensions or sample counts cannot be represented without overflow or truncation
- restore HTTPS-ready remote COG reads with a Rustls TLS feature, default connect/read/request timeouts, custom request headers, and preconfigured blocking client support
- reject GeoTIFF and COG band counts above the TIFF `SamplesPerPixel` limit instead of narrowing them into invalid metadata
- stream one-shot COG overview generation tile-by-tile so overviews no longer require full intermediate overview arrays
- add a CodeQL workflow, run integration benchmark targets from `scripts/run-reference-benchmarks.sh`, and keep benchmark-only mmap paths behind explicit mmap open helpers

## 0.6.1 - 2026-06-10

- replace C-backed TIFF ZSTD read/write dependencies with pure-Rust `ruzstd` for `ZSTD` and `LERC+ZSTD` blocks
- disable reqwest TLS features for COG reads so the default dependency graph does not include native TLS, `ring`, or `aws-lc`
- regenerate `fuzz/Cargo.lock` so locked fuzz checks use the same `ruzstd` dependency graph as the workspace

## 0.6.0 - 2026-06-08

- add TIFF reader parse budgets through `OpenOptions`, bounding IFD chain length, per-IFD tag entries, per-tag value bytes, and aggregate metadata value bytes
- bound strip/tile payload reads before source access and enforce decompressed block budgets across Deflate, LZW, PackBits, JPEG, ZSTD, LERC, `LERC+DEFLATE`, and `LERC+ZSTD`
- reject oversized BigTIFF `LONG8` scalar dimensions, zero strip/tile dimensions, oversized TIFF block byte counts, and LERC payloads that conflict with TIFF metadata
- fix TIFF LERC writer metadata to use the registered LERC2 2.4 parameter version and reject incompatible encoder output versions before writing
- emit GeoTIFF 1.1 GeoKey directory minor revision by default, preserving legacy 1.0 only when compatible and promoting vertical GeoKeys to 1.1
- validate writer strip/tile dimensions, block counts, sample counts, and estimated byte counts for zero values and overflow, and deprecate legacy infallible `ImageBuilder` helpers after making them best-effort instead of panicking on invalid builders
- pin release CI toolchains and runners, build cargo-fuzz with Rust 1.85, and add regression coverage for metadata budgets, block budgets, writer layout checks, and GDAL LERC version compatibility

## 0.5.0 - 2026-05-17

- add storage-domain single-band read APIs to `tiff-reader` and `geotiff-reader`, including full-image and windowed reads that return `[rows, cols]` arrays
- optimize separate-planar band reads so the reader only decodes the requested band plane instead of every plane
- optimize windowed strip/tile reads so small windows enumerate only intersecting storage blocks
- fix `block_cache_slots = 0` and HTTP `cache_slots = 0` so zero slots consistently disables cache storage
- fix PixelIsPoint GeoTIFF writing so transform serialization preserves normalized coordinates without a half-pixel shift
- support transform-only GeoTIFF metadata by accepting model georeferencing without a GeoKey directory and emitting a minimal GeoKey directory when writing transforms without CRS keys
- add SubIFD-backed COG overview writing alongside the existing top-level overview IFD layout, and scale overview georeferencing to each overview level
- remove duplicate GeoTIFF tags from COG overview IFDs
- make GeoKey serialization fallible instead of truncating oversized key counts and parameter offsets
- update `lerc-rust` dependencies to `0.4.2`
- add coverage for chunky and separate-planar band reads, GeoTIFF band windows, disabled zero-slot range caches, and LERC interoperability fuzz seeds

## 0.4.0 - 2026-04-19

- add JPEG-in-TIFF write support across `tiff-writer`, `geotiff-writer`, and COG output using standard compression code `7`
- add explicit decoded-pixel read APIs while preserving storage-domain sample reads, including palette expansion, sub-byte grayscale/palette decoding, YCbCr conversion, and CMYK conversion
- add structured TIFF color-model metadata for `ColorMap`, `ExtraSamples`, CMYK `InkSet`, YCbCr tags, and extended photometric interpretations
- add richer GeoTIFF CRS modeling for projected, geographic, geocentric, vertical, and compound CRS metadata
- discover overviews from both reduced-resolution top-level IFDs and recursive SubIFD overview trees
- improve COG generation with exact BigTIFF auto-selection, disk-backed `CogTileWriter` assembly, GDAL-compatible block ordering/ghost metadata, and nodata-aware average overviews
- reject streaming tile offsets, band-count mismatches, unsupported YCbCr subsampling, and JPEG layouts that are not interoperable with GDAL/libtiff
- prepare crates.io publishing metadata for the workspace crates and use the published `lerc-rust` 0.3 crates from the registry

## 0.3.1 - 2026-04-06

- move cross-crate release tests into non-publishable integration crates so publishable package tarballs stay focused
- fix release-time dev-dependency constraints for the workspace test crates

## 0.3.0 - 2026-04-06

- add pure-Rust TIFF `LERC` write support through the published `lerc-rust` 0.3 crates
- add GeoTIFF and COG `LERC`, `LERC+DEFLATE`, and `LERC+ZSTD` write support
- move `LercOptions` into `tiff-writer` and expose consistent builder configuration for TIFF and GeoTIFF writers
- add roundtrip and reference coverage for LERC writer behavior

## 0.2.5 - 2026-04-02

- add pure-Rust TIFF/GeoTIFF `LERC` read support through the published `lerc-rust` crates
- add TIFF `LercParameters` parsing and support for TIFF-side `LERC+DEFLATE` and `LERC+ZSTD`
- add real GDAL interoperability fixtures for plain `LERC`, `LERC+DEFLATE`, `LERC+ZSTD`, and tiled separate-planar RGB `LERC`
- preserve the existing write surface; TIFF `LERC` write is not part of this release
