use std::path::Path;

use tiff_reader::TiffFile;

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/regressions")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn source_cmyk(row: usize, col: usize) -> [usize; 4] {
    [
        (col * 3) % 200 + 20,
        (row * 3) % 200 + 20,
        ((col + row) * 2) % 200 + 20,
        40,
    ]
}

fn assert_near(actual: usize, expected: usize, context: &str) {
    assert!(
        actual.abs_diff(expected) <= 3,
        "{context}: {actual} vs {expected}"
    );
}

/// libtiff hands back CMYK JPEG components unchanged, while a standalone JPEG
/// decoder inverts them to undo Adobe's convention.
#[test]
fn cmyk_jpeg_samples_are_not_inverted() {
    let file = TiffFile::from_bytes(fixture("gdal_jpeg_cmyk.tif")).unwrap();

    let samples = file.read_image_bytes(0).unwrap();

    for (index, pixel) in samples.chunks_exact(4).enumerate() {
        let (row, col) = (index / 64, index % 64);
        for (actual, expected) in pixel.iter().zip(source_cmyk(row, col)) {
            assert_near(usize::from(*actual), expected, "sample");
        }
    }
}

#[test]
fn cmyk_jpeg_decodes_to_the_same_rgb_as_gdal() {
    let file = TiffFile::from_bytes(fixture("gdal_jpeg_cmyk.tif")).unwrap();

    let pixels = file.read_decoded_image_bytes(0).unwrap();

    let at = |row: usize, col: usize| &pixels[(row * 64 + col) * 3..(row * 64 + col) * 3 + 3];
    // GDAL 3.12 reads pixel (10, 10) as (172, 172, 164).
    assert_near(usize::from(at(10, 10)[0]), 172, "red");
    assert_near(usize::from(at(10, 10)[1]), 172, "green");
    assert_near(usize::from(at(10, 10)[2]), 164, "blue");
}
