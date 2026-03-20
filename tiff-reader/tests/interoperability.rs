use std::path::{Path, PathBuf};

use ndarray::ArrayD;
use tiff_reader::TiffFile;

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../testdata/interoperability")
        .join(path)
}

#[test]
fn decodes_real_world_tiff_corpus() {
    let byte = TiffFile::open(fixture("gdal/gcore/data/byte.tif")).unwrap();
    assert!(!byte.is_bigtiff());
    assert_eq!(byte.ifd(0).unwrap().compression(), 1);
    let raster: ArrayD<u8> = byte.read_image(0).unwrap();
    assert_eq!(raster.shape(), &[20, 20]);

    let signed = TiffFile::open(fixture("gdal/gdrivers/data/gtiff/int8.tif")).unwrap();
    assert_eq!(signed.ifd(0).unwrap().sample_format(), vec![2]);
    let raster: ArrayD<i8> = signed.read_image(0).unwrap();
    assert_eq!(raster.shape(), &[20, 20]);

    #[cfg(feature = "jpeg")]
    let jpeg = TiffFile::open(fixture("gdal/gcore/data/gtiff/byte_JPEG.tif")).unwrap();
    #[cfg(feature = "jpeg")]
    assert_eq!(jpeg.ifd(0).unwrap().compression(), 7);
    #[cfg(feature = "jpeg")]
    let raster: ArrayD<u8> = jpeg.read_image(0).unwrap();
    #[cfg(feature = "jpeg")]
    assert_eq!(raster.shape(), &[20, 20]);

    #[cfg(feature = "zstd")]
    let zstd = TiffFile::open(fixture("gdal/gcore/data/byte_zstd.tif")).unwrap();
    #[cfg(feature = "zstd")]
    assert_eq!(zstd.ifd(0).unwrap().compression(), 50000);
    #[cfg(feature = "zstd")]
    let raster: ArrayD<u8> = zstd.read_image(0).unwrap();
    #[cfg(feature = "zstd")]
    assert_eq!(raster.shape(), &[20, 20]);
}

#[test]
fn opens_real_world_bigtiff_and_cog_layouts() {
    let bigtiff = TiffFile::open(fixture("gdal/gcore/data/bigtiff_one_strip_long.tif")).unwrap();
    assert!(bigtiff.is_bigtiff());
    let raster: ArrayD<u8> = bigtiff.read_image(0).unwrap();
    assert_eq!(raster.shape(), &[1, 1]);

    let tiled = TiffFile::open(fixture("gdal/gcore/data/gtiff/byte_NONE_tiled.tif")).unwrap();
    assert!(tiled.ifd(0).unwrap().is_tiled());
    let raster: ArrayD<u8> = tiled.read_image(0).unwrap();
    assert_eq!(raster.shape(), &[20, 20]);
}

#[test]
fn opens_real_world_tiff_with_internal_overviews() {
    let file = TiffFile::open(fixture("gdal/gcore/data/byte_with_ovr.tif")).unwrap();
    assert!(file.ifd_count() > 1);

    let base: ArrayD<u8> = file.read_image(0).unwrap();
    assert_eq!(base.shape(), &[20, 20]);

    let overview: ArrayD<u8> = file.read_image(1).unwrap();
    assert!(!overview.is_empty());
}
