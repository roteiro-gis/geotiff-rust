//! Pure-Rust TIFF/BigTIFF encoder with compression, tiling, and streaming writes.
//!
//! # Example
//!
//! ```no_run
//! use tiff_writer::{TiffWriter, WriteOptions, ImageBuilder};
//! use tiff_core::Compression;
//! use std::io::Cursor;
//!
//! let mut buf = Cursor::new(Vec::new());
//! let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
//!
//! let image = ImageBuilder::new(4, 4).sample_type::<u8>();
//! let handle = writer.add_image(image).unwrap();
//! writer.write_block(&handle, 0, &[0u8; 16]).unwrap();
//! writer.finish().unwrap();
//! ```

pub mod builder;
pub mod compress;
pub mod encoder;
pub mod error;
pub mod sample;
pub mod writer;

pub use builder::{DataLayout, ImageBuilder};
pub use error::{Error, Result};
pub use sample::TiffWriteSample;
pub use writer::{ImageHandle, TiffVariant, TiffWriter, WriteOptions};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn write_and_read_stripped_u8() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(2, 2).sample_type::<u8>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[1u8, 2, 3, 4]).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        assert_eq!(file.ifd_count(), 1);
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[2, 2]);
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn write_and_read_stripped_u16() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(3, 2).sample_type::<u16>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[100u16, 200, 300, 400, 500, 600])
            .unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<u16>(0).unwrap();
        assert_eq!(img.shape(), &[2, 3]);
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![100, 200, 300, 400, 500, 600]);
    }

    #[test]
    fn write_and_read_stripped_f32() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(2, 2).sample_type::<f32>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[1.5f32, 2.5, 3.5, 4.5])
            .unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<f32>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn write_and_read_stripped_f64() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(2, 2).sample_type::<f64>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<f64>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn write_and_read_multi_strip() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(4, 4).sample_type::<u8>().strips(1);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[1u8, 2, 3, 4]).unwrap();
        writer.write_block(&handle, 1, &[5u8, 6, 7, 8]).unwrap();
        writer.write_block(&handle, 2, &[9u8, 10, 11, 12]).unwrap();
        writer.write_block(&handle, 3, &[13u8, 14, 15, 16]).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let window = file.read_window::<u8>(0, 1, 1, 2, 2).unwrap();
        let (values, _) = window.into_raw_vec_and_offset();
        assert_eq!(values, vec![6, 7, 10, 11]);
    }

    #[test]
    fn write_and_read_tiled() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        // 4x4 image with 16x16 tiles = 1 tile (padded)
        let image = ImageBuilder::new(4, 4).sample_type::<u8>().tiles(16, 16);
        let handle = writer.add_image(image).unwrap();

        // Tile must be full tile_width * tile_height = 256 samples
        let mut tile_data = vec![0u8; 256];
        // Fill the 4x4 actual pixels (top-left of the 16x16 tile)
        for row in 0..4 {
            for col in 0..4 {
                tile_data[row * 16 + col] = (row * 4 + col + 1) as u8;
            }
        }
        writer.write_block(&handle, 0, &tile_data).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[4, 4]);
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(
            values,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn write_and_read_lzw_compressed() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(4, 4)
            .sample_type::<u8>()
            .compression(tiff_core::Compression::Lzw)
            .strips(4);
        let handle = writer.add_image(image).unwrap();
        let pixels: Vec<u8> = (1..=16).collect();
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, pixels);
    }

    #[test]
    fn write_and_read_deflate_compressed() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(4, 4)
            .sample_type::<u8>()
            .compression(tiff_core::Compression::Deflate)
            .strips(4);
        let handle = writer.add_image(image).unwrap();
        let pixels: Vec<u8> = (1..=16).collect();
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, pixels);
    }

    #[test]
    fn write_multi_ifd() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        // Base image
        let base = ImageBuilder::new(2, 2).sample_type::<u8>().strips(2);
        let h1 = writer.add_image(base).unwrap();
        writer.write_block(&h1, 0, &[10u8, 20, 30, 40]).unwrap();

        // Overview
        let ovr = ImageBuilder::new(1, 1)
            .sample_type::<u8>()
            .strips(1)
            .overview();
        let h2 = writer.add_image(ovr).unwrap();
        writer.write_block(&h2, 0, &[99u8]).unwrap();

        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        assert_eq!(file.ifd_count(), 2);

        let img0 = file.read_image::<u8>(0).unwrap();
        let (v0, _) = img0.into_raw_vec_and_offset();
        assert_eq!(v0, vec![10, 20, 30, 40]);

        let img1 = file.read_image::<u8>(1).unwrap();
        let (v1, _) = img1.into_raw_vec_and_offset();
        assert_eq!(v1, vec![99]);
    }

    #[test]
    fn write_and_read_horizontal_predictor_u16() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(3, 1)
            .sample_type::<u16>()
            .compression(tiff_core::Compression::Deflate)
            .predictor(tiff_core::Predictor::Horizontal)
            .strips(1);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[1u16, 2, 4]).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<u16>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, vec![1, 2, 4]);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn write_and_read_zstd_compressed() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

        let image = ImageBuilder::new(4, 4)
            .sample_type::<u8>()
            .compression(tiff_core::Compression::Zstd)
            .strips(4);
        let handle = writer.add_image(image).unwrap();
        let pixels: Vec<u8> = (1..=16).collect();
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        let (values, _) = img.into_raw_vec_and_offset();
        assert_eq!(values, pixels);
    }

    // -- Additional data type coverage --

    #[test]
    fn write_and_read_i8() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2).sample_type::<i8>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[-1i8, 0, 1, 127]).unwrap();
        writer.finish().unwrap();
        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<i8>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![-1i8, 0, 1, 127]);
    }

    #[test]
    fn write_and_read_i16() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2).sample_type::<i16>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[-100i16, 0, 100, 32000])
            .unwrap();
        writer.finish().unwrap();
        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<i16>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![-100i16, 0, 100, 32000]);
    }

    #[test]
    fn write_and_read_u32() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2).sample_type::<u32>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[0u32, 1, 1000000, u32::MAX])
            .unwrap();
        writer.finish().unwrap();
        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u32>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![0u32, 1, 1000000, u32::MAX]);
    }

    #[test]
    fn write_and_read_i32() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2).sample_type::<i32>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[i32::MIN, -1, 0, i32::MAX])
            .unwrap();
        writer.finish().unwrap();
        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<i32>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![i32::MIN, -1, 0, i32::MAX]);
    }

    #[test]
    fn write_and_read_u64() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 1).sample_type::<u64>().strips(1);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[0u64, 999999]).unwrap();
        writer.finish().unwrap();
        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u64>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![0u64, 999999]);
    }

    #[test]
    fn write_and_read_i64() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 1).sample_type::<i64>().strips(1);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[-1i64, 42]).unwrap();
        writer.finish().unwrap();
        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<i64>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![-1i64, 42]);
    }

    // -- Multi-band --

    #[test]
    fn write_and_read_rgb_u8() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2)
            .sample_type::<u8>()
            .samples_per_pixel(3)
            .photometric(tiff_core::PhotometricInterpretation::Rgb)
            .strips(2);
        let handle = writer.add_image(image).unwrap();
        // 2x2 RGB: 12 samples
        let pixels = vec![
            255u8, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            128, 128, 128, // gray
        ];
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[2, 2, 3]);
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, pixels);
    }

    // -- Edge cases --

    #[test]
    fn write_1x1_image() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(1, 1).sample_type::<f64>().strips(1);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[42.0f64]).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<f64>(0).unwrap();
        assert_eq!(img.shape(), &[1, 1]);
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![42.0]);
    }

    #[test]
    fn write_non_tile_aligned_dimensions() {
        // 5x3 image with 16x16 tiles = 1 tile
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(5, 3).sample_type::<u8>().tiles(16, 16);
        let handle = writer.add_image(image).unwrap();
        // Full 16x16 tile with actual data in top-left 5x3
        let mut tile = vec![0u8; 256];
        for r in 0..3 {
            for c in 0..5 {
                tile[r * 16 + c] = (r * 5 + c + 1) as u8;
            }
        }
        writer.write_block(&handle, 0, &tile).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[3, 5]);
        let (v, _) = img.into_raw_vec_and_offset();
        let expected: Vec<u8> = (1..=15).collect();
        assert_eq!(v, expected);
    }

    #[test]
    fn write_and_read_planar_stripped_rgb() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2)
            .sample_type::<u8>()
            .samples_per_pixel(3)
            .photometric(tiff_core::PhotometricInterpretation::Rgb)
            .planar_configuration(tiff_core::PlanarConfiguration::Planar)
            .strips(1);
        let handle = writer.add_image(image).unwrap();

        writer.write_block(&handle, 0, &[255u8, 1]).unwrap();
        writer.write_block(&handle, 1, &[2u8, 3]).unwrap();
        writer.write_block(&handle, 2, &[10u8, 20]).unwrap();
        writer.write_block(&handle, 3, &[30u8, 40]).unwrap();
        writer.write_block(&handle, 4, &[100u8, 101]).unwrap();
        writer.write_block(&handle, 5, &[102u8, 103]).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        assert_eq!(file.ifd(0).unwrap().planar_configuration(), 2);
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[2, 2, 3]);
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![255, 10, 100, 1, 20, 101, 2, 30, 102, 3, 40, 103]);
    }

    #[test]
    fn write_and_read_planar_tiled_rgb() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(2, 2)
            .sample_type::<u8>()
            .samples_per_pixel(3)
            .photometric(tiff_core::PhotometricInterpretation::Rgb)
            .planar_configuration(tiff_core::PlanarConfiguration::Planar)
            .tiles(16, 16);
        let handle = writer.add_image(image).unwrap();

        let mut red = vec![0u8; 16 * 16];
        let mut green = vec![0u8; 16 * 16];
        let mut blue = vec![0u8; 16 * 16];
        red[0] = 1;
        red[1] = 2;
        red[16] = 3;
        red[17] = 4;
        green[0] = 10;
        green[1] = 20;
        green[16] = 30;
        green[17] = 40;
        blue[0] = 100;
        blue[1] = 110;
        blue[16] = 120;
        blue[17] = 130;

        writer.write_block(&handle, 0, &red).unwrap();
        writer.write_block(&handle, 1, &green).unwrap();
        writer.write_block(&handle, 2, &blue).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        assert_eq!(file.ifd(0).unwrap().planar_configuration(), 2);
        let img = file.read_image::<u8>(0).unwrap();
        assert_eq!(img.shape(), &[2, 2, 3]);
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![1, 10, 100, 2, 20, 110, 3, 30, 120, 4, 40, 130]);
    }

    #[test]
    fn write_and_read_planar_horizontal_predictor_u16() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(3, 1)
            .sample_type::<u16>()
            .samples_per_pixel(2)
            .planar_configuration(tiff_core::PlanarConfiguration::Planar)
            .compression(tiff_core::Compression::Deflate)
            .predictor(tiff_core::Predictor::Horizontal)
            .strips(1);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[1u16, 2, 4]).unwrap();
        writer.write_block(&handle, 1, &[100u16, 102, 105]).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u16>(0).unwrap();
        assert_eq!(img.shape(), &[1, 3, 2]);
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![1, 100, 2, 102, 4, 105]);
    }

    // -- BigTIFF --

    #[test]
    fn write_and_read_bigtiff() {
        let mut buf = Cursor::new(Vec::new());
        let opts = WriteOptions {
            byte_order: tiff_core::ByteOrder::LittleEndian,
            variant: TiffVariant::BigTiff,
        };
        let mut writer = TiffWriter::new(&mut buf, opts).unwrap();
        let image = ImageBuilder::new(2, 2).sample_type::<u8>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[10u8, 20, 30, 40]).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        assert!(file.is_bigtiff());
        let img = file.read_image::<u8>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![10, 20, 30, 40]);
    }

    #[test]
    fn write_and_read_bigtiff_f64_multistrip() {
        let mut buf = Cursor::new(Vec::new());
        let opts = WriteOptions {
            byte_order: tiff_core::ByteOrder::LittleEndian,
            variant: TiffVariant::BigTiff,
        };
        let mut writer = TiffWriter::new(&mut buf, opts).unwrap();
        let image = ImageBuilder::new(4, 4).sample_type::<f64>().strips(2);
        let handle = writer.add_image(image).unwrap();
        let row01: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let row23: Vec<f64> = (9..=16).map(|x| x as f64).collect();
        writer.write_block(&handle, 0, &row01).unwrap();
        writer.write_block(&handle, 1, &row23).unwrap();
        writer.finish().unwrap();

        let data = buf.into_inner();
        let file = tiff_reader::TiffFile::from_bytes(data).unwrap();
        assert!(file.is_bigtiff());
        let img = file.read_image::<f64>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        let expected: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        assert_eq!(v, expected);
    }

    // -- Horizontal predictor with compression for f32 --

    #[test]
    fn write_and_read_horizontal_predictor_f32() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(4, 1)
            .sample_type::<f32>()
            .compression(tiff_core::Compression::Deflate)
            .predictor(tiff_core::Predictor::Horizontal)
            .strips(1);
        let handle = writer.add_image(image).unwrap();
        writer
            .write_block(&handle, 0, &[1.0f32, 2.0, 4.0, 8.0])
            .unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<f32>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, vec![1.0f32, 2.0, 4.0, 8.0]);
    }

    // -- LZW with horizontal predictor u8 --

    #[test]
    fn write_and_read_lzw_horizontal_predictor_u8() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(8, 2)
            .sample_type::<u8>()
            .compression(tiff_core::Compression::Lzw)
            .predictor(tiff_core::Predictor::Horizontal)
            .strips(2);
        let handle = writer.add_image(image).unwrap();
        let pixels: Vec<u8> = (0..16).collect();
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<u8>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, pixels);
    }

    // -- Floating-point predictor --

    #[test]
    fn write_and_read_float_predictor_f32() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(4, 2)
            .sample_type::<f32>()
            .compression(tiff_core::Compression::Deflate)
            .predictor(tiff_core::Predictor::FloatingPoint)
            .strips(2);
        let handle = writer.add_image(image).unwrap();
        let pixels: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<f32>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, pixels);
    }

    #[test]
    fn write_and_read_float_predictor_f64() {
        let mut buf = Cursor::new(Vec::new());
        let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
        let image = ImageBuilder::new(4, 1)
            .sample_type::<f64>()
            .compression(tiff_core::Compression::Lzw)
            .predictor(tiff_core::Predictor::FloatingPoint)
            .strips(1);
        let handle = writer.add_image(image).unwrap();
        let pixels: Vec<f64> = vec![100.0, 200.5, 300.25, 400.125];
        writer.write_block(&handle, 0, &pixels).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        let img = file.read_image::<f64>(0).unwrap();
        let (v, _) = img.into_raw_vec_and_offset();
        assert_eq!(v, pixels);
    }

    // -- Auto variant --

    #[test]
    fn auto_variant_uses_classic_for_small_images() {
        let mut buf = Cursor::new(Vec::new());
        let opts = WriteOptions::auto(1024); // tiny
        let mut writer = TiffWriter::new(&mut buf, opts).unwrap();
        let image = ImageBuilder::new(2, 2).sample_type::<u8>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[1u8, 2, 3, 4]).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        assert!(!file.is_bigtiff());
    }

    #[test]
    fn auto_variant_uses_bigtiff_for_large_images() {
        let mut buf = Cursor::new(Vec::new());
        let opts = WriteOptions::auto(5_000_000_000); // 5 GB
        let mut writer = TiffWriter::new(&mut buf, opts).unwrap();
        // Just a small image to verify header format
        let image = ImageBuilder::new(2, 2).sample_type::<u8>().strips(2);
        let handle = writer.add_image(image).unwrap();
        writer.write_block(&handle, 0, &[1u8, 2, 3, 4]).unwrap();
        writer.finish().unwrap();

        let file = tiff_reader::TiffFile::from_bytes(buf.into_inner()).unwrap();
        assert!(file.is_bigtiff());
    }
}
