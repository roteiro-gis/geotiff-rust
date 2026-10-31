use std::path::Path;

use tiff_reader::TiffFile;

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/regressions")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// GDAL writes lossless 8-bit LERC blocks with Huffman coding when that beats
/// bit stuffing, so this needs a real libLerc blob rather than one of ours.
#[test]
fn reads_gdal_written_u8_lerc_huffman_blocks() {
    let file = TiffFile::from_bytes(fixture("gdal_lerc_u8_huffman.tif")).unwrap();

    let image = file.read_image::<u8>(0).unwrap();

    let (height, width) = (image.shape()[0], image.shape()[1]);
    assert_eq!((height, width), (48, 64));
    for ((row, col), value) in (0..height)
        .flat_map(|row| (0..width).map(move |col| (row, col)))
        .zip(image.iter().copied())
    {
        let expected = (col * 3 + row * 2) % 251 + (col * 7 + row * 13) % 4;
        assert_eq!(usize::from(value), expected, "pixel ({row}, {col})");
    }
}
