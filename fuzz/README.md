# Fuzzing

This workspace uses `cargo-fuzz`.

Setup:

```sh
cargo install cargo-fuzz
./scripts/seed-fuzz-corpus.sh
```

Run the parser/decode fuzzers:

```sh
cargo fuzz run tiff_open fuzz/corpus/tiff_open
cargo fuzz run geotiff_open fuzz/corpus/geotiff_open
```

The seed corpus is derived from `testdata/interoperability`, which is the single source of truth for real-world fixtures.
