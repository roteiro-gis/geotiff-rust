#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use ndarray::ArrayD;
use serde_json::Value;

pub fn workspace_root(manifest_dir: &str) -> PathBuf {
    Path::new(manifest_dir)
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(manifest_dir).join(".."))
}

pub fn fixture(manifest_dir: &str, relative_path: &str) -> PathBuf {
    workspace_root(manifest_dir)
        .join("testdata/interoperability")
        .join(relative_path)
}

pub fn reference_script(manifest_dir: &str) -> PathBuf {
    workspace_root(manifest_dir).join("scripts/reference_gdal.py")
}

pub fn python_gdal_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("python3")
            .args(["-c", "from osgeo import gdal"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    })
}

pub fn tiffdump_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| Command::new("tiffdump").output().is_ok())
}

pub fn run_reference_json(manifest_dir: &str, args: &[&str]) -> Value {
    let output = Command::new("python3")
        .arg(reference_script(manifest_dir))
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run GDAL reference helper: {err}"));
    assert!(
        output.status.success(),
        "GDAL reference helper failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|err| panic!("failed to parse GDAL reference JSON: {err}"))
}

pub fn run_reference_bytes(manifest_dir: &str, args: &[&str]) -> Vec<u8> {
    let output = Command::new("python3")
        .arg(reference_script(manifest_dir))
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run GDAL reference helper: {err}"));
    assert!(
        output.status.success(),
        "GDAL reference helper failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

pub fn run_tiffdump(path: &Path) -> String {
    let output = Command::new("tiffdump")
        .arg(path)
        .output()
        .unwrap_or_else(|err| panic!("failed to run tiffdump: {err}"));
    assert!(
        output.status.success(),
        "tiffdump failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap_or_else(|err| panic!("tiffdump emitted non-utf8 output: {err}"))
}

pub fn fnv1a64(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

pub trait SampleBytes {
    fn append_ne_bytes(&self, out: &mut Vec<u8>);
}

impl SampleBytes for u8 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.push(*self);
    }
}

impl SampleBytes for i8 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}

impl SampleBytes for u16 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i16 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for u32 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i32 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for f32 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for u64 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i64 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for f64 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

pub fn array_hash<T: SampleBytes>(array: &ArrayD<T>) -> (usize, String) {
    let element_size = std::mem::size_of::<T>();
    let mut bytes = Vec::with_capacity(array.len() * element_size);
    for value in array {
        value.append_ne_bytes(&mut bytes);
    }
    let len = bytes.len();
    (len, fnv1a64(&bytes))
}

pub fn assert_close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "{context}: expected {expected}, got {actual}, delta {delta}"
    );
}

pub fn assert_u8_bytes_close(
    actual: &[u8],
    expected: &[u8],
    max_abs_delta: u8,
    max_diff_pixels: usize,
    context: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: byte length mismatch"
    );

    let mut diff_count = 0usize;
    let mut max_delta = 0u8;
    let mut first_diffs = Vec::new();
    for (index, (&actual_byte, &expected_byte)) in actual.iter().zip(expected).enumerate() {
        let delta = actual_byte.abs_diff(expected_byte);
        if delta != 0 {
            diff_count += 1;
            max_delta = max_delta.max(delta);
            if first_diffs.len() < 8 {
                first_diffs.push((index, actual_byte, expected_byte, delta));
            }
        }
    }

    assert!(
        max_delta <= max_abs_delta,
        "{context}: max abs delta {max_delta} exceeded {max_abs_delta}; first diffs: {first_diffs:?}"
    );
    assert!(
        diff_count <= max_diff_pixels,
        "{context}: differing pixels {diff_count} exceeded {max_diff_pixels}; first diffs: {first_diffs:?}"
    );
}
