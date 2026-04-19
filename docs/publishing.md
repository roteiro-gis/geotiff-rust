# Publishing

This workspace publishes six crates. Publish them in dependency order so Cargo
can verify each downstream package against the newly published registry
versions:

1. `tiff-core`
2. `geotiff-core`
3. `tiff-reader`
4. `tiff-writer`
5. `geotiff-reader`
6. `geotiff-writer`

Run the release checks before publishing:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
cargo package -p tiff-core
cargo package -p geotiff-core
```

After `tiff-core` and `geotiff-core` are published, run `cargo package` for
the dependent crates in the order above, then publish each crate with:

```sh
cargo publish -p <crate>
```

Cargo verifies package tarballs using registry dependencies rather than local
path dependencies, so dependent crates cannot complete a full `cargo package`
verification until their internal `0.4.0` dependencies are available on
crates.io.

Before those internal versions are live, you can still locally verify the
downstream tarballs with temporary patches:

```sh
cargo package -p tiff-reader \
  --config 'patch.crates-io.tiff-core.path="tiff-core"'
cargo package -p tiff-writer \
  --config 'patch.crates-io.tiff-core.path="tiff-core"'
cargo package -p geotiff-reader \
  --config 'patch.crates-io.geotiff-core.path="geotiff-core"' \
  --config 'patch.crates-io.tiff-core.path="tiff-core"' \
  --config 'patch.crates-io.tiff-reader.path="tiff-reader"'
cargo package -p geotiff-writer \
  --config 'patch.crates-io.geotiff-core.path="geotiff-core"' \
  --config 'patch.crates-io.tiff-core.path="tiff-core"' \
  --config 'patch.crates-io.tiff-writer.path="tiff-writer"'
```
