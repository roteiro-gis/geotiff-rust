#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
corpus_root="$repo_root/testdata/interoperability"
manifest="$corpus_root/manifest.tsv"
fuzz_root="$repo_root/fuzz/corpus"

rm -rf "$fuzz_root/tiff_open" "$fuzz_root/geotiff_open"
mkdir -p "$fuzz_root/tiff_open" "$fuzz_root/geotiff_open"

while IFS=$'\t' read -r relative_path _expected_sha targets _source_url; do
    [[ -z "${relative_path}" || "${relative_path}" == \#* ]] && continue

    source_path="$corpus_root/$relative_path"
    file_name="$(basename "$relative_path")"

    if [[ "$targets" == *"tiff_open"* ]]; then
        cp "$source_path" "$fuzz_root/tiff_open/$file_name"
    fi
    if [[ "$targets" == *"geotiff_open"* ]]; then
        cp "$source_path" "$fuzz_root/geotiff_open/$file_name"
    fi
done < "$manifest"
