//! Pure-Rust GeoTIFF and COG writer with compression, tiling, and overview support.
//!
//! Compression: None, LZW, Deflate, LERC, LERC+Deflate, LERC+Zstd, ZSTD (feature).
//!
//! # Example
//!
//! ```no_run
//! use geotiff_writer::GeoTiffBuilder;
//! use ndarray::Array2;
//!
//! let data = Array2::<f32>::zeros((100, 100));
//! GeoTiffBuilder::new(100, 100)
//!     .epsg(4326)
//!     .pixel_scale(0.01, 0.01)
//!     .origin(-180.0, 90.0)
//!     .nodata("-9999")
//!     .write_2d("output.tif", data.view())
//!     .unwrap();
//! ```

pub mod builder;
pub mod cog;
pub mod error;
pub mod sample;
pub mod tile_writer;

pub use builder::GeoTiffBuilder;
pub use cog::{CogBuilder, CogTileWriter, Resampling};
pub use error::{Error, Result};
pub use sample::{NumericSample, WriteSample};
pub use tile_writer::StreamingTileWriter;

// Re-export core types for convenience
pub use geotiff_core::{
    CrsInfo, GeoKeyDirectory, GeoKeyValue, GeoTransform, ModelType, RasterType,
};
pub use tiff_core::{
    Compression, LercAdditionalCompression, PhotometricInterpretation, PlanarConfiguration,
    Predictor,
};
pub use tiff_writer::LercOptions;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::io::Cursor;

    #[test]
    fn write_and_read_simple_f64() {
        let mut data = Array2::<f64>::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                data[[r, c]] = (r * 4 + c + 1) as f64;
            }
        }

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(4, 4)
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        let img = file.read_image::<f64>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        let expected: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn write_and_read_with_metadata() {
        let data = Array2::<f32>::from_elem((2, 2), 42.0);

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(2, 2)
            .epsg(4326)
            .pixel_scale(1.0, 1.0)
            .origin(100.0, 200.0)
            .nodata("-9999")
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();

        // Verify pixel data
        let file = tiff_reader::TiffFile::from_bytes(bytes.clone()).unwrap();
        let img = file.read_image::<f32>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![42.0f32; 4]);

        // Verify GeoTIFF metadata via geotiff-reader
        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(geo.epsg(), Some(4326));
        assert_eq!(geo.nodata(), Some("-9999"));

        let transform = geo.transform().unwrap();
        let (x, y) = transform.pixel_to_geo(0.0, 0.0);
        assert!((x - 100.0).abs() < 1e-10);
        assert!((y - 200.0).abs() < 1e-10);
    }

    #[test]
    fn write_and_read_compressed_with_geotiff() {
        let data = Array2::<u16>::from_elem((8, 8), 1000);

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(8, 8)
            .compression(Compression::Deflate)
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        let img = file.read_image::<u16>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1000u16; 64]);
    }

    #[test]
    fn write_with_transform() {
        let data = Array2::<u8>::from_elem((2, 2), 1);
        let transform = GeoTransform::from_origin_and_pixel_size(-180.0, 90.0, 0.5, -0.5);

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(2, 2)
            .transform(transform)
            .epsg(4326)
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        let gt = geo.transform().unwrap();
        let (x, y) = gt.pixel_to_geo(0.0, 0.0);
        assert!((x - (-180.0)).abs() < 1e-10);
        assert!((y - 90.0).abs() < 1e-10);
        let (x2, y2) = gt.pixel_to_geo(1.0, 1.0);
        assert!((x2 - (-179.5)).abs() < 1e-10);
        assert!((y2 - 89.5).abs() < 1e-10);
    }

    #[test]
    fn streaming_tile_writer() {
        let mut buf = Cursor::new(Vec::new());

        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16).epsg(4326);

        let mut tw = builder.tile_writer::<u8, _>(&mut buf).unwrap();

        // Write 4 tiles (2x2 grid of 16x16 tiles)
        for tile_row in 0..2 {
            for tile_col in 0..2 {
                let val = (tile_row * 2 + tile_col + 1) as u8;
                let tile = Array2::from_elem((16, 16), val);
                tw.write_tile(tile_col * 16, tile_row * 16, &tile.view())
                    .unwrap();
            }
        }

        tw.finish().unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[32, 32]);

        // Check corners of each tile
        assert_eq!(img[[0, 0]], 1); // top-left tile
        assert_eq!(img[[0, 16]], 2); // top-right tile
        assert_eq!(img[[16, 0]], 3); // bottom-left tile
        assert_eq!(img[[16, 16]], 4); // bottom-right tile
    }

    #[test]
    fn streaming_vs_oneshot_produce_same_pixels() {
        let mut data = ndarray::Array2::<u8>::zeros((32, 32));
        for r in 0..32 {
            for c in 0..32 {
                data[[r, c]] = ((r * 32 + c) % 256) as u8;
            }
        }

        // One-shot write
        let mut oneshot_buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(32, 32)
            .tile_size(16, 16)
            .write_2d_to(&mut oneshot_buf, data.view())
            .unwrap();

        // Streaming write
        let mut streaming_buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);
        let mut tw = builder.tile_writer::<u8, _>(&mut streaming_buf).unwrap();
        for tile_row in 0..2u32 {
            for tile_col in 0..2u32 {
                let y_off = (tile_row * 16) as usize;
                let x_off = (tile_col * 16) as usize;
                let tile = data
                    .slice(ndarray::s![y_off..y_off + 16, x_off..x_off + 16])
                    .to_owned();
                tw.write_tile(x_off, y_off, &tile.view()).unwrap();
            }
        }
        tw.finish().unwrap();

        // Read both and compare pixels
        let oneshot_file = tiff_reader::TiffFile::from_bytes(oneshot_buf.into_inner()).unwrap();
        let streaming_file = tiff_reader::TiffFile::from_bytes(streaming_buf.into_inner()).unwrap();

        let oneshot_img = oneshot_file.read_image::<u8>(0).unwrap();
        let streaming_img = streaming_file.read_image::<u8>(0).unwrap();

        assert_eq!(oneshot_img.shape(), streaming_img.shape());
        let (ov, _) = oneshot_img.into_raw_vec_and_offset();
        let (sv, _) = streaming_img.into_raw_vec_and_offset();
        assert_eq!(ov, sv);
    }

    #[test]
    fn write_and_read_multiband_rgb() {
        let mut data = ndarray::Array3::<u8>::zeros((4, 4, 3));
        for r in 0..4 {
            for c in 0..4 {
                data[[r, c, 0]] = 255; // red
                data[[r, c, 1]] = 0; // green
                data[[r, c, 2]] = (r * 64) as u8; // blue gradient
            }
        }

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(4, 4)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .write_3d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[4, 4, 3]);
        let (values, _) = img.into_raw_vec_and_offset();
        // First pixel: R=255, G=0, B=0
        assert_eq!(&values[0..3], &[255, 0, 0]);
        // Third row first pixel: R=255, G=0, B=128
        assert_eq!(&values[24..27], &[255, 0, 128]);
    }

    #[test]
    fn write_and_read_multiband_rgb_planar() {
        let mut data = ndarray::Array3::<u8>::zeros((2, 2, 3));
        data[[0, 0, 0]] = 1;
        data[[0, 1, 0]] = 2;
        data[[1, 0, 0]] = 3;
        data[[1, 1, 0]] = 4;
        data[[0, 0, 1]] = 10;
        data[[0, 1, 1]] = 20;
        data[[1, 0, 1]] = 30;
        data[[1, 1, 1]] = 40;
        data[[0, 0, 2]] = 100;
        data[[0, 1, 2]] = 110;
        data[[1, 0, 2]] = 120;
        data[[1, 1, 2]] = 130;

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(2, 2)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .planar_configuration(PlanarConfiguration::Planar)
            .epsg(4326)
            .write_3d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let tiff = tiff_reader::TiffFile::from_bytes(bytes.clone()).unwrap();
        assert_eq!(tiff.ifd(0).unwrap().planar_configuration(), 2);
        let img = tiff.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[2, 2, 3]);
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1, 10, 100, 2, 20, 110, 3, 30, 120, 4, 40, 130]);

        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(geo.epsg(), Some(4326));
    }

    #[test]
    fn streaming_tile_writer_planar_matches_oneshot() {
        let mut data = ndarray::Array3::<u8>::zeros((32, 32, 3));
        for r in 0..32 {
            for c in 0..32 {
                data[[r, c, 0]] = ((r * 32 + c) % 256) as u8;
                data[[r, c, 1]] = ((r * 7 + c * 5) % 256) as u8;
                data[[r, c, 2]] = ((r * 11 + c * 3) % 256) as u8;
            }
        }

        let mut oneshot_buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(32, 32)
            .bands(3)
            .tile_size(16, 16)
            .photometric(PhotometricInterpretation::Rgb)
            .planar_configuration(PlanarConfiguration::Planar)
            .write_3d_to(&mut oneshot_buf, data.view())
            .unwrap();

        let mut streaming_buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .bands(3)
            .tile_size(16, 16)
            .photometric(PhotometricInterpretation::Rgb)
            .planar_configuration(PlanarConfiguration::Planar);
        let mut tw = builder.tile_writer::<u8, _>(&mut streaming_buf).unwrap();
        for tile_row in 0..2usize {
            for tile_col in 0..2usize {
                let y_off = tile_row * 16;
                let x_off = tile_col * 16;
                let tile = data
                    .slice(ndarray::s![y_off..y_off + 16, x_off..x_off + 16, ..])
                    .to_owned();
                tw.write_tile_3d(x_off, y_off, &tile.view()).unwrap();
            }
        }
        tw.finish().unwrap();

        let oneshot_file = tiff_reader::TiffFile::from_bytes(oneshot_buf.into_inner()).unwrap();
        let streaming_file = tiff_reader::TiffFile::from_bytes(streaming_buf.into_inner()).unwrap();

        assert_eq!(oneshot_file.ifd(0).unwrap().planar_configuration(), 2);
        assert_eq!(streaming_file.ifd(0).unwrap().planar_configuration(), 2);

        let oneshot_img = oneshot_file.read_image::<u8>(0).unwrap();
        let streaming_img = streaming_file.read_image::<u8>(0).unwrap();

        assert_eq!(oneshot_img.shape(), streaming_img.shape());
        let (ov, _) = oneshot_img.into_raw_vec_and_offset();
        let (sv, _) = streaming_img.into_raw_vec_and_offset();
        assert_eq!(ov, sv);
    }

    // -- LERC compression --

    #[test]
    fn write_and_read_lerc_geotiff() {
        let data = Array2::<f32>::from_elem((8, 8), 42.5);
        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(8, 8)
            .lerc_options(LercOptions::default())
            .epsg(4326)
            .pixel_scale(1.0, 1.0)
            .origin(0.0, 8.0)
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes.clone()).unwrap();
        let img = file.read_image::<f32>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![42.5f32; 64]);

        let geo = geotiff_reader::GeoTiffFile::from_bytes(bytes).unwrap();
        assert_eq!(geo.epsg(), Some(4326));
    }

    #[test]
    fn write_and_read_lerc_deflate_geotiff() {
        let data = Array2::<u16>::from_elem((8, 8), 1000);
        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(8, 8)
            .lerc_options(LercOptions {
                max_z_error: 0.0,
                additional_compression: LercAdditionalCompression::Deflate,
            })
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u16>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1000u16; 64]);
    }

    #[test]
    fn write_and_read_lerc_multiband_geotiff() {
        let mut data = ndarray::Array3::<u8>::zeros((4, 4, 3));
        for r in 0..4 {
            for c in 0..4 {
                data[[r, c, 0]] = 255;
                data[[r, c, 1]] = 0;
                data[[r, c, 2]] = (r * 64) as u8;
            }
        }

        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(4, 4)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .lerc_options(LercOptions::default())
            .write_3d_to(&mut buf, data.view())
            .unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[4, 4, 3]);
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(&values[0..3], &[255, 0, 0]);
        assert_eq!(&values[24..27], &[255, 0, 128]);
    }

    #[test]
    fn cog_lerc_with_overviews() {
        let mut data = Array2::<f32>::zeros((32, 32));
        for r in 0..32 {
            for c in 0..32 {
                data[[r, c]] = (r * 32 + c) as f32;
            }
        }

        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .tile_size(16, 16)
            .lerc_options(LercOptions::default())
            .epsg(4326);

        CogBuilder::new(builder)
            .overview_levels(vec![2, 4])
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();

        // Base + 2 overviews
        assert_eq!(file.ifd_count(), 3);
        assert_eq!(file.ifd(0).unwrap().width(), 32);

        let base = file.read_image::<f32>(0).unwrap();
        assert_eq!(base.shape(), &[32, 32]);
        assert_eq!(base[[0, 0]], 0.0);
        assert_eq!(base[[1, 1]], 33.0);
    }

    #[test]
    fn streaming_tile_writer_lerc() {
        let mut buf = Cursor::new(Vec::new());
        let builder = GeoTiffBuilder::new(32, 32)
            .tile_size(16, 16)
            .lerc_options(LercOptions::default())
            .epsg(4326);

        let mut tw = builder.tile_writer::<u8, _>(&mut buf).unwrap();

        for tile_row in 0..2 {
            for tile_col in 0..2 {
                let val = (tile_row * 2 + tile_col + 1) as u8;
                let tile = ndarray::Array2::from_elem((16, 16), val);
                tw.write_tile(tile_col * 16, tile_row * 16, &tile.view())
                    .unwrap();
            }
        }
        tw.finish().unwrap();

        let bytes = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(bytes).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[32, 32]);
        assert_eq!(img[[0, 0]], 1);
        assert_eq!(img[[0, 16]], 2);
        assert_eq!(img[[16, 0]], 3);
        assert_eq!(img[[16, 16]], 4);
    }

    #[test]
    fn write_and_read_lerc_lossy_geotiff() {
        let data = Array2::<f32>::from_shape_fn((8, 8), |(r, c)| (r * 8 + c) as f32 * 1.1);
        let mut buf = Cursor::new(Vec::new());
        GeoTiffBuilder::new(8, 8)
            .lerc_options(LercOptions {
                max_z_error: 0.5,
                additional_compression: LercAdditionalCompression::None,
            })
            .write_2d_to(&mut buf, data.view())
            .unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<f32>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        for (actual, &expected) in values.iter().zip(data.iter()) {
            assert!(
                (actual - expected).abs() <= 0.5,
                "lossy error {:.3} exceeds 0.5",
                (actual - expected).abs()
            );
        }
    }
}
