#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
corpus_root="$repo_root/testdata/interoperability"
manifest="$corpus_root/manifest.tsv"
verify_only="${1:-}"

if [[ -n "$verify_only" && "$verify_only" != "--verify-only" ]]; then
    echo "usage: $0 [--verify-only]" >&2
    exit 1
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{ print $1 }'
        return
    fi

    echo "missing checksum tool: install sha256sum or shasum" >&2
    exit 1
}

verify_file() {
    local relative_path="$1"
    local expected_sha="$2"
    local destination="$corpus_root/$relative_path"

    if [[ ! -f "$destination" ]]; then
        echo "missing fixture: $relative_path" >&2
        return 1
    fi

    local actual_sha
    actual_sha="$(sha256_file "$destination")"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "checksum mismatch for $relative_path" >&2
        echo "expected: $expected_sha" >&2
        echo "actual:   $actual_sha" >&2
        return 1
    fi

    return 0
}

while IFS=$'\t' read -r relative_path expected_sha _targets source_url; do
    [[ -z "${relative_path}" || "${relative_path}" == \#* ]] && continue

    destination="$corpus_root/$relative_path"
    if verify_file "$relative_path" "$expected_sha"; then
        continue
    fi
    if [[ "$verify_only" == "--verify-only" ]]; then
        exit 1
    fi

    mkdir -p "$(dirname "$destination")"
    curl -L --max-redirs 5 -o "$destination" "$source_url"
    verify_file "$relative_path" "$expected_sha"
done < "$manifest"
