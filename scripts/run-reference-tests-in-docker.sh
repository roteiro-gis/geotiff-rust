#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${GEOTIFF_RUST_REFERENCE_IMAGE:-geotiff-rust-reference}"

docker build -f "$repo_root/docker/reference.Dockerfile" -t "$image_name" "$repo_root"

docker run --rm \
    -e GEOTIFF_RUST_BENCH_ITERATIONS="${GEOTIFF_RUST_BENCH_ITERATIONS:-5}" \
    -e GEOTIFF_RUST_BENCH_MAX_SLOWDOWN="${GEOTIFF_RUST_BENCH_MAX_SLOWDOWN:-}" \
    -v "$repo_root:/workspace" \
    -w /workspace \
    "$image_name" \
    bash -c '
        cargo test --workspace
        cargo test -p tiff-reader --test reference_benchmark -- --ignored --nocapture
        cargo test -p geotiff-reader --test reference_benchmark -- --ignored --nocapture
    '
