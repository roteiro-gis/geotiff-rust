use std::fmt::Debug;
use std::io::Cursor;

use tiff_core::{Compression, Predictor};
use tiff_reader::{TiffFile, TiffSample};
use tiff_writer::{
    ImageBuilder, JpegOptions, LercOptions, TiffVariant, TiffWriteSample, TiffWriter, WriteOptions,
};

fn roundtrip_image<T>(image: ImageBuilder, block_index: usize, block: &[T]) -> Vec<T>
where
    T: TiffWriteSample + TiffSample + Debug + PartialEq,
{
    let mut buf = Cursor::new(Vec::new());
    let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();
    let handle = writer.add_image(image).unwrap();
    writer.write_block(&handle, block_index, block).unwrap();
    writer.finish().unwrap();

    let file = TiffFile::from_bytes(buf.into_inner()).unwrap();
    let image = file.read_image::<T>(0).unwrap();
    let (values, offset) = image.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    values
}

fn padded_tile<T: Copy + Default>(
    width: usize,
    height: usize,
    tile_width: usize,
    pixels: &[T],
) -> Vec<T> {
    let mut tile = vec![T::default(); tile_width * tile_width];
    for row in 0..height {
        let src_start = row * width;
        let src_end = src_start + width;
        let dst_start = row * tile_width;
        let dst_end = dst_start + width;
        tile[dst_start..dst_end].copy_from_slice(&pixels[src_start..src_end]);
    }
    tile
}

fn assert_u8_bytes_close(
    actual: &[u8],
    expected: &[u8],
    max_abs_delta: u8,
    max_diff_pixels: usize,
) {
    assert_eq!(actual.len(), expected.len(), "byte length mismatch");

    let mut diff_pixels = 0usize;
    let mut max_seen_delta = 0u8;
    for (&actual_byte, &expected_byte) in actual.iter().zip(expected.iter()) {
        let delta = actual_byte.abs_diff(expected_byte);
        if delta != 0 {
            diff_pixels += 1;
            max_seen_delta = max_seen_delta.max(delta);
        }
    }

    assert!(
        max_seen_delta <= max_abs_delta,
        "max abs delta {max_seen_delta} exceeded {max_abs_delta}"
    );
    assert!(
        diff_pixels <= max_diff_pixels,
        "differing pixels {diff_pixels} exceeded {max_diff_pixels}"
    );
}

#[test]
fn stripped_roundtrips_cover_core_sample_types() {
    let u8_values = roundtrip_image(
        ImageBuilder::new(2, 2).sample_type::<u8>().strips(2),
        0,
        &[1u8, 2, 3, 4],
    );
    assert_eq!(u8_values, vec![1, 2, 3, 4]);

    let u16_values = roundtrip_image(
        ImageBuilder::new(3, 2).sample_type::<u16>().strips(2),
        0,
        &[100u16, 200, 300, 400, 500, 600],
    );
    assert_eq!(u16_values, vec![100, 200, 300, 400, 500, 600]);

    let f32_values = roundtrip_image(
        ImageBuilder::new(2, 2).sample_type::<f32>().strips(2),
        0,
        &[1.5f32, 2.5, 3.5, 4.5],
    );
    assert_eq!(f32_values, vec![1.5, 2.5, 3.5, 4.5]);

    let f64_values = roundtrip_image(
        ImageBuilder::new(2, 2).sample_type::<f64>().strips(2),
        0,
        &[1.0f64, 2.0, 3.0, 4.0],
    );
    assert_eq!(f64_values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn multi_strip_window_roundtrips() {
    let mut buf = Cursor::new(Vec::new());
    let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

    let image = ImageBuilder::new(4, 4).sample_type::<u8>().strips(1);
    let handle = writer.add_image(image).unwrap();
    writer.write_block(&handle, 0, &[1u8, 2, 3, 4]).unwrap();
    writer.write_block(&handle, 1, &[5u8, 6, 7, 8]).unwrap();
    writer.write_block(&handle, 2, &[9u8, 10, 11, 12]).unwrap();
    writer.write_block(&handle, 3, &[13u8, 14, 15, 16]).unwrap();
    writer.finish().unwrap();

    let file = TiffFile::from_bytes(buf.into_inner()).unwrap();
    let window = file.read_window::<u8>(0, 1, 1, 2, 2).unwrap();
    let (values, offset) = window.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    assert_eq!(values, vec![6, 7, 10, 11]);
}

#[test]
fn tiled_and_compressed_images_roundtrip() {
    let mut tile_data = vec![0u8; 16 * 16];
    for row in 0..4 {
        for col in 0..4 {
            tile_data[row * 16 + col] = (row * 4 + col + 1) as u8;
        }
    }

    let tiled = roundtrip_image(
        ImageBuilder::new(4, 4).sample_type::<u8>().tiles(16, 16),
        0,
        &tile_data,
    );
    assert_eq!(
        tiled,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );

    let pixels: Vec<u8> = (1..=16).collect();
    let lzw = roundtrip_image(
        ImageBuilder::new(4, 4)
            .sample_type::<u8>()
            .compression(Compression::Lzw)
            .strips(4),
        0,
        &pixels,
    );
    assert_eq!(lzw, pixels);

    let deflate = roundtrip_image(
        ImageBuilder::new(4, 4)
            .sample_type::<u8>()
            .compression(Compression::Deflate)
            .strips(4),
        0,
        &pixels,
    );
    assert_eq!(deflate, pixels);
}

#[test]
fn jpeg_strips_and_rgb_tiles_roundtrip() {
    let grayscale_rows = [
        [32u8, 32, 32, 32, 192, 192, 192, 192],
        [32, 32, 32, 32, 192, 192, 192, 192],
        [32, 32, 32, 32, 192, 192, 192, 192],
        [32, 32, 32, 32, 192, 192, 192, 192],
        [96, 96, 96, 96, 224, 224, 224, 224],
        [96, 96, 96, 96, 224, 224, 224, 224],
        [96, 96, 96, 96, 224, 224, 224, 224],
        [96, 96, 96, 96, 224, 224, 224, 224],
    ];
    let grayscale: Vec<u8> = grayscale_rows.into_iter().flatten().collect();

    let mut grayscale_buf = Cursor::new(Vec::new());
    let mut grayscale_writer =
        TiffWriter::new(&mut grayscale_buf, WriteOptions::default()).unwrap();
    let grayscale_handle = grayscale_writer
        .add_image(
            ImageBuilder::new(8, 8)
                .sample_type::<u8>()
                .compression(Compression::Jpeg)
                .jpeg_options(JpegOptions { quality: 90 })
                .strips(4),
        )
        .unwrap();
    grayscale_writer
        .write_block(&grayscale_handle, 0, &grayscale[..32])
        .unwrap();
    grayscale_writer
        .write_block(&grayscale_handle, 1, &grayscale[32..])
        .unwrap();
    grayscale_writer.finish().unwrap();

    let grayscale_file = TiffFile::from_bytes(grayscale_buf.into_inner()).unwrap();
    let grayscale_ifd = grayscale_file.ifd(0).unwrap();
    assert_eq!(grayscale_ifd.compression(), Compression::Jpeg.to_code());
    assert!(grayscale_ifd.tag(tiff_core::TAG_JPEG_TABLES).is_none());
    let grayscale_image = grayscale_file.read_image::<u8>(0).unwrap();
    let (grayscale_values, grayscale_offset) = grayscale_image.into_raw_vec_and_offset();
    assert_eq!(grayscale_offset, Some(0));
    assert_u8_bytes_close(&grayscale_values, &grayscale, 2, 32);

    let mut rgb = vec![0u8; 16 * 16 * 3];
    for row in 0..16usize {
        for col in 0..16usize {
            let pixel = (row * 16 + col) * 3;
            let color = match (row / 8, col / 8) {
                (0, 0) => [255, 0, 0],
                (0, 1) => [0, 255, 0],
                (1, 0) => [0, 0, 255],
                _ => [240, 240, 32],
            };
            rgb[pixel..pixel + 3].copy_from_slice(&color);
        }
    }

    let mut rgb_buf = Cursor::new(Vec::new());
    let mut rgb_writer = TiffWriter::new(&mut rgb_buf, WriteOptions::default()).unwrap();
    let rgb_handle = rgb_writer
        .add_image(
            ImageBuilder::new(16, 16)
                .sample_type::<u8>()
                .samples_per_pixel(3)
                .photometric(tiff_core::PhotometricInterpretation::Rgb)
                .compression(Compression::Jpeg)
                .jpeg_options(JpegOptions { quality: 90 })
                .tiles(16, 16),
        )
        .unwrap();
    rgb_writer.write_block(&rgb_handle, 0, &rgb).unwrap();
    rgb_writer.finish().unwrap();

    let rgb_file = TiffFile::from_bytes(rgb_buf.into_inner()).unwrap();
    let rgb_ifd = rgb_file.ifd(0).unwrap();
    assert_eq!(rgb_ifd.compression(), Compression::Jpeg.to_code());
    assert!(rgb_ifd.tag(tiff_core::TAG_JPEG_TABLES).is_none());
    let rgb_image = rgb_file.read_image::<u8>(0).unwrap();
    let (rgb_values, rgb_offset) = rgb_image.into_raw_vec_and_offset();
    assert_eq!(rgb_offset, Some(0));
    assert_u8_bytes_close(&rgb_values, &rgb, 6, 256);
}

#[test]
fn multi_ifd_and_planar_rgb_roundtrip() {
    let mut buf = Cursor::new(Vec::new());
    let mut writer = TiffWriter::new(&mut buf, WriteOptions::default()).unwrap();

    let base = ImageBuilder::new(2, 2).sample_type::<u8>().strips(2);
    let base_handle = writer.add_image(base).unwrap();
    writer
        .write_block(&base_handle, 0, &[10u8, 20, 30, 40])
        .unwrap();

    let overview = ImageBuilder::new(1, 1)
        .sample_type::<u8>()
        .overview()
        .strips(1);
    let overview_handle = writer.add_image(overview).unwrap();
    writer.write_block(&overview_handle, 0, &[99u8]).unwrap();

    let planar = ImageBuilder::new(2, 2)
        .sample_type::<u8>()
        .samples_per_pixel(3)
        .photometric(tiff_core::PhotometricInterpretation::Rgb)
        .planar_configuration(tiff_core::PlanarConfiguration::Planar)
        .tiles(16, 16);
    let planar_handle = writer.add_image(planar).unwrap();
    for band in 0..3usize {
        let mut planar_tile = vec![0u8; 16 * 16];
        for row in 0..2usize {
            for col in 0..2usize {
                let index = row * 16 + col;
                planar_tile[index] = (band * 10 + row * 2 + col + 1) as u8;
            }
        }
        writer
            .write_block(&planar_handle, band, &planar_tile)
            .unwrap();
    }
    writer.finish().unwrap();

    let file = TiffFile::from_bytes(buf.into_inner()).unwrap();
    assert_eq!(file.ifd_count(), 3);

    let base_image = file.read_image::<u8>(0).unwrap();
    assert_eq!(base_image[[1, 1]], 40);

    let reduced = file.read_image::<u8>(1).unwrap();
    assert_eq!(reduced[[0, 0]], 99);

    let rgb = file.read_image::<u8>(2).unwrap();
    assert_eq!(rgb.shape(), &[2, 2, 3]);
    assert_eq!(rgb[[0, 0, 0]], 1);
    assert_eq!(rgb[[0, 0, 1]], 11);
    assert_eq!(rgb[[0, 0, 2]], 21);
}

#[test]
fn lerc_roundtrip_and_builder_state_behave_consistently() {
    let data: Vec<f32> = (0..16).map(|value| value as f32 * 1.1).collect();
    let values = roundtrip_image(
        ImageBuilder::new(4, 4)
            .sample_type::<f32>()
            .lerc_options(LercOptions::default())
            .predictor(Predictor::Horizontal)
            .tiles(16, 16),
        0,
        &padded_tile(4, 4, 16, &data),
    );
    assert_eq!(values.len(), 16);
    for (actual, expected) in values.iter().zip(data.iter()) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    let ib = ImageBuilder::new(4, 4)
        .sample_type::<u8>()
        .lerc_options(LercOptions::default())
        .compression(Compression::Deflate);
    assert!(ib.lerc_parameters_tag().is_none());

    let predictor_roundtrip = roundtrip_image(
        ImageBuilder::new(4, 4)
            .sample_type::<f32>()
            .lerc_options(LercOptions::default())
            .predictor(Predictor::Horizontal)
            .tiles(16, 16),
        0,
        &padded_tile(4, 4, 16, &data),
    );
    for (actual, expected) in predictor_roundtrip.iter().zip(data.iter()) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    let ib = ImageBuilder::new(4, 4)
        .sample_type::<u8>()
        .compression(Compression::Lerc);
    assert!(ib.lerc_parameters_tag().is_some());
}

#[test]
fn writer_validation_rejects_zero_samples_and_rgb_band_mismatches() {
    let mut zero_spp_buf = Cursor::new(Vec::new());
    let mut zero_spp_writer = TiffWriter::new(&mut zero_spp_buf, WriteOptions::default()).unwrap();
    let err = zero_spp_writer
        .add_image(
            ImageBuilder::new(1, 1)
                .sample_type::<u8>()
                .samples_per_pixel(0),
        )
        .unwrap_err();
    assert!(
        matches!(err, tiff_writer::Error::InvalidConfig(message) if message.contains("samples_per_pixel"))
    );

    let mut rgb_buf = Cursor::new(Vec::new());
    let mut rgb_writer = TiffWriter::new(&mut rgb_buf, WriteOptions::default()).unwrap();
    let err = rgb_writer
        .add_image(
            ImageBuilder::new(1, 1)
                .sample_type::<u8>()
                .samples_per_pixel(1)
                .photometric(tiff_core::PhotometricInterpretation::Rgb),
        )
        .unwrap_err();
    assert!(
        matches!(err, tiff_writer::Error::InvalidConfig(message) if message.contains("RGB photometric interpretation"))
    );

    let mut jpeg_u16_buf = Cursor::new(Vec::new());
    let mut jpeg_u16_writer = TiffWriter::new(&mut jpeg_u16_buf, WriteOptions::default()).unwrap();
    let err = jpeg_u16_writer
        .add_image(
            ImageBuilder::new(1, 1)
                .sample_type::<u16>()
                .compression(Compression::Jpeg),
        )
        .unwrap_err();
    assert!(
        matches!(err, tiff_writer::Error::InvalidConfig(message) if message.contains("8-bit samples"))
    );

    let mut jpeg_chunky_four_band_buf = Cursor::new(Vec::new());
    let mut jpeg_chunky_four_band_writer =
        TiffWriter::new(&mut jpeg_chunky_four_band_buf, WriteOptions::default()).unwrap();
    let err = jpeg_chunky_four_band_writer
        .add_image(
            ImageBuilder::new(1, 1)
                .sample_type::<u8>()
                .samples_per_pixel(4)
                .compression(Compression::Jpeg),
        )
        .unwrap_err();
    assert!(
        matches!(err, tiff_writer::Error::InvalidConfig(message) if message.contains("1 or 3 samples per block"))
    );

    let mut jpeg_rgb_buf = Cursor::new(Vec::new());
    let mut jpeg_rgb_writer = TiffWriter::new(&mut jpeg_rgb_buf, WriteOptions::default()).unwrap();
    let err = jpeg_rgb_writer
        .add_image(
            ImageBuilder::new(1, 1)
                .sample_type::<u8>()
                .samples_per_pixel(3)
                .compression(Compression::Jpeg),
        )
        .unwrap_err();
    assert!(
        matches!(err, tiff_writer::Error::InvalidConfig(message) if message.contains("RGB photometric interpretation"))
    );

    let mut jpeg_wide_buf = Cursor::new(Vec::new());
    let mut jpeg_wide_writer =
        TiffWriter::new(&mut jpeg_wide_buf, WriteOptions::default()).unwrap();
    let err = jpeg_wide_writer
        .add_image(
            ImageBuilder::new(70_000, 1)
                .sample_type::<u8>()
                .compression(Compression::Jpeg),
        )
        .unwrap_err();
    assert!(
        matches!(err, tiff_writer::Error::InvalidConfig(message) if message.contains("block width"))
    );
}

#[test]
fn explicit_bigtiff_roundtrips_small_images() {
    let mut buf = Cursor::new(Vec::new());
    let mut writer = TiffWriter::new(
        &mut buf,
        WriteOptions {
            byte_order: tiff_core::ByteOrder::LittleEndian,
            variant: TiffVariant::BigTiff,
        },
    )
    .unwrap();

    let handle = writer
        .add_image(ImageBuilder::new(2, 2).sample_type::<u8>().strips(2))
        .unwrap();
    writer.write_block(&handle, 0, &[1u8, 2, 3, 4]).unwrap();
    writer.finish().unwrap();

    let file = TiffFile::from_bytes(buf.into_inner()).unwrap();
    assert!(file.is_bigtiff());
    let image = file.read_image::<u8>(0).unwrap();
    let (values, offset) = image.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    assert_eq!(values, vec![1, 2, 3, 4]);
}
