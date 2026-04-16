use std::io::Cursor;

use geotiff_reader::GeoTiffFile;
use geotiff_writer::{
    ColorMap, ColorModel, Compression, Error as GeoTiffWriteError, ExtraSample, GeoTiffBuilder,
    InkSet, JpegOptions, LercAdditionalCompression, LercOptions, ModelType, PlanarConfiguration,
    TiffVariant,
};
use ndarray::{Array2, Array3};
use tiff_reader::TiffFile;

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

fn sample_color_map() -> ColorMap {
    let red = (0u16..=255).map(|value| value * 257).collect();
    let green = (0u16..=255).map(|value| 65_535 - value * 257).collect();
    let blue = (0u16..=255).map(|value| (value / 2) * 257).collect();
    ColorMap::new(red, green, blue).unwrap()
}

#[test]
fn geotiff_roundtrips_pixels_metadata_and_transform() {
    let mut data = Array2::<f64>::zeros((4, 4));
    for row in 0..4 {
        for col in 0..4 {
            data[[row, col]] = (row * 4 + col + 1) as f64;
        }
    }

    let mut buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(4, 4)
        .epsg(4326)
        .pixel_scale(1.0, 1.0)
        .origin(100.0, 200.0)
        .nodata("-9999")
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let bytes = buf.into_inner();
    let tiff = TiffFile::from_bytes(bytes.clone()).unwrap();
    let raster = tiff.read_image::<f64>(0).unwrap();
    let (values, offset) = raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    assert_eq!(
        values,
        (1..=16).map(|value| value as f64).collect::<Vec<_>>()
    );

    let geo = GeoTiffFile::from_bytes(bytes).unwrap();
    assert_eq!(geo.epsg(), Some(4326));
    assert_eq!(geo.nodata(), Some("-9999"));

    let transform = geo.transform().unwrap();
    let (x, y) = transform.pixel_to_geo(0.0, 0.0);
    assert!((x - 100.0).abs() < 1e-10);
    assert!((y - 200.0).abs() < 1e-10);
}

#[test]
fn geotiff_compressed_and_streaming_outputs_roundtrip() {
    let data = Array2::<u16>::from_elem((8, 8), 1000);
    let mut compressed = Cursor::new(Vec::new());
    GeoTiffBuilder::new(8, 8)
        .compression(Compression::Deflate)
        .write_2d_to(&mut compressed, data.view())
        .unwrap();

    let compressed_file = TiffFile::from_bytes(compressed.into_inner()).unwrap();
    let compressed_raster = compressed_file.read_image::<u16>(0).unwrap();
    let (compressed_values, offset) = compressed_raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    assert_eq!(compressed_values, vec![1000u16; 64]);

    let mut oneshot_data = ndarray::Array2::<u8>::zeros((32, 32));
    for row in 0..32 {
        for col in 0..32 {
            oneshot_data[[row, col]] = ((row * 32 + col) % 256) as u8;
        }
    }

    let mut oneshot_buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(32, 32)
        .tile_size(16, 16)
        .write_2d_to(&mut oneshot_buf, oneshot_data.view())
        .unwrap();

    let mut streaming_buf = Cursor::new(Vec::new());
    let builder = GeoTiffBuilder::new(32, 32).tile_size(16, 16);
    let mut writer = builder.tile_writer::<u8, _>(&mut streaming_buf).unwrap();
    for tile_row in 0..2usize {
        for tile_col in 0..2usize {
            let y_off = tile_row * 16;
            let x_off = tile_col * 16;
            let tile = oneshot_data
                .slice(ndarray::s![y_off..y_off + 16, x_off..x_off + 16])
                .to_owned();
            writer.write_tile(x_off, y_off, &tile.view()).unwrap();
        }
    }
    writer.finish().unwrap();

    let oneshot = TiffFile::from_bytes(oneshot_buf.into_inner()).unwrap();
    let streaming = TiffFile::from_bytes(streaming_buf.into_inner()).unwrap();
    let oneshot_image = oneshot.read_image::<u8>(0).unwrap();
    let streaming_image = streaming.read_image::<u8>(0).unwrap();
    let (oneshot_values, oneshot_offset) = oneshot_image.into_raw_vec_and_offset();
    let (streaming_values, streaming_offset) = streaming_image.into_raw_vec_and_offset();
    assert_eq!(oneshot_offset, Some(0));
    assert_eq!(streaming_offset, Some(0));
    assert_eq!(oneshot_values, streaming_values);
}

#[test]
fn planar_and_lerc_geotiffs_roundtrip() {
    let mut planar = ndarray::Array3::<u8>::zeros((16, 16, 3));
    for row in 0..16 {
        for col in 0..16 {
            planar[[row, col, 0]] = ((row * 3 + col * 5) % 251) as u8;
            planar[[row, col, 1]] = ((row * 11 + col * 7) % 251) as u8;
            planar[[row, col, 2]] = ((row * 17 + col * 13) % 251) as u8;
        }
    }

    let mut planar_buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(16, 16)
        .bands(3)
        .tile_size(16, 16)
        .planar_configuration(PlanarConfiguration::Planar)
        .compression(Compression::Deflate)
        .write_3d_to(&mut planar_buf, planar.view())
        .unwrap();

    let planar_file = TiffFile::from_bytes(planar_buf.into_inner()).unwrap();
    let planar_raster = planar_file.read_image::<u8>(0).unwrap();
    assert_eq!(planar_raster.shape(), &[16, 16, 3]);
    assert_eq!(planar_raster[[5, 7, 0]], planar[[5, 7, 0]]);
    assert_eq!(planar_raster[[5, 7, 1]], planar[[5, 7, 1]]);
    assert_eq!(planar_raster[[5, 7, 2]], planar[[5, 7, 2]]);

    let lerc_data = Array2::<f32>::from_shape_fn((8, 8), |(row, col)| (row * 8 + col) as f32 * 1.1);
    let mut lerc_buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(8, 8)
        .lerc_options(LercOptions {
            max_z_error: 0.5,
            additional_compression: LercAdditionalCompression::None,
        })
        .write_2d_to(&mut lerc_buf, lerc_data.view())
        .unwrap();

    let lerc_file = TiffFile::from_bytes(lerc_buf.into_inner()).unwrap();
    let lerc_raster = lerc_file.read_image::<f32>(0).unwrap();
    let (values, offset) = lerc_raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    for (actual, expected) in values.iter().zip(lerc_data.iter()) {
        assert!((actual - expected).abs() <= 0.5);
    }
}

#[test]
fn geotiff_writer_emits_small_bigtiff_when_requested() {
    let data = Array2::<u8>::from_elem((4, 4), 9);
    let mut buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(4, 4)
        .tiff_variant(TiffVariant::BigTiff)
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let bytes = buf.into_inner();
    assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 43);
    let file = TiffFile::from_bytes(bytes).unwrap();
    assert!(file.is_bigtiff());
}

#[test]
fn jpeg_geotiff_roundtrips_pixels_metadata_and_streaming_tiles() {
    let mut rgb = Array3::<u8>::zeros((16, 16, 3));
    for row in 0..16usize {
        for col in 0..16usize {
            let color = match (row / 8, col / 8) {
                (0, 0) => [255, 0, 0],
                (0, 1) => [0, 255, 0],
                (1, 0) => [0, 0, 255],
                _ => [240, 240, 32],
            };
            rgb[[row, col, 0]] = color[0];
            rgb[[row, col, 1]] = color[1];
            rgb[[row, col, 2]] = color[2];
        }
    }

    let mut buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(16, 16)
        .bands(3)
        .photometric(tiff_core::PhotometricInterpretation::Rgb)
        .planar_configuration(PlanarConfiguration::Planar)
        .tile_size(16, 16)
        .jpeg_options(JpegOptions { quality: 90 })
        .epsg(4326)
        .pixel_scale(1.0, 1.0)
        .origin(-180.0, 90.0)
        .write_3d_to(&mut buf, rgb.view())
        .unwrap();

    let bytes = buf.into_inner();
    let tiff = TiffFile::from_bytes(bytes.clone()).unwrap();
    let ifd = tiff.ifd(0).unwrap();
    assert_eq!(ifd.compression(), Compression::Jpeg.to_code());
    assert!(ifd.tag(tiff_core::TAG_JPEG_TABLES).is_none());

    let raster = tiff.read_image::<u8>(0).unwrap();
    let (values, offset) = raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    let expected = rgb.iter().copied().collect::<Vec<_>>();
    assert_u8_bytes_close(&values, &expected, 2, 0);

    let geo = GeoTiffFile::from_bytes(bytes).unwrap();
    assert_eq!(geo.epsg(), Some(4326));
    assert_eq!(geo.band_count(), 3);

    let mut single_band = Array2::<u8>::zeros((32, 32));
    for row in 0..32usize {
        for col in 0..32usize {
            single_band[[row, col]] = match (row / 16, col / 16) {
                (0, 0) => 24,
                (0, 1) => 96,
                (1, 0) => 160,
                _ => 224,
            };
        }
    }

    let mut streaming_buf = Cursor::new(Vec::new());
    let mut writer = GeoTiffBuilder::new(32, 32)
        .tile_size(16, 16)
        .jpeg_options(JpegOptions { quality: 90 })
        .tile_writer::<u8, _>(&mut streaming_buf)
        .unwrap();
    for tile_row in 0..2usize {
        for tile_col in 0..2usize {
            let y_off = tile_row * 16;
            let x_off = tile_col * 16;
            let tile = single_band
                .slice(ndarray::s![y_off..y_off + 16, x_off..x_off + 16])
                .to_owned();
            writer.write_tile(x_off, y_off, &tile.view()).unwrap();
        }
    }
    writer.finish().unwrap();

    let streaming = TiffFile::from_bytes(streaming_buf.into_inner()).unwrap();
    let ifd = streaming.ifd(0).unwrap();
    assert_eq!(ifd.compression(), Compression::Jpeg.to_code());
    let image = streaming.read_image::<u8>(0).unwrap();
    let (streaming_values, streaming_offset) = image.into_raw_vec_and_offset();
    assert_eq!(streaming_offset, Some(0));
    let expected = single_band.iter().copied().collect::<Vec<_>>();
    assert_u8_bytes_close(&streaming_values, &expected, 2, 32);
}

#[test]
fn streaming_tile_writer_rejects_misaligned_offsets_and_band_mismatches() {
    let mut single_band_buf = Cursor::new(Vec::new());
    let mut single_band_writer = GeoTiffBuilder::new(32, 32)
        .tile_size(16, 16)
        .tile_writer::<u8, _>(&mut single_band_buf)
        .unwrap();
    let single_band_tile = Array2::<u8>::zeros((16, 16));
    let err = single_band_writer
        .write_tile(1, 0, &single_band_tile.view())
        .unwrap_err();
    assert!(
        matches!(err, GeoTiffWriteError::Other(message) if message.contains("align to tile boundaries"))
    );

    let mut planar_buf = Cursor::new(Vec::new());
    let mut planar_writer = GeoTiffBuilder::new(32, 32)
        .bands(3)
        .tile_size(16, 16)
        .planar_configuration(PlanarConfiguration::Planar)
        .tile_writer::<u8, _>(&mut planar_buf)
        .unwrap();
    let wrong_bands = Array3::<u8>::zeros((16, 16, 2));
    let err = planar_writer
        .write_tile_3d(0, 0, &wrong_bands.view())
        .unwrap_err();
    assert!(matches!(err, GeoTiffWriteError::DataSizeMismatch { .. }));
}

#[test]
fn geocentric_epsg_roundtrips_through_reader_metadata() {
    let data = Array2::<u8>::from_elem((1, 1), 7);
    let mut buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(1, 1)
        .epsg(4978)
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let geo = GeoTiffFile::from_bytes(buf.into_inner()).unwrap();
    assert_eq!(geo.epsg(), Some(4978));
    assert_eq!(geo.crs().model_type_enum(), ModelType::Geocentric);
    assert_eq!(geo.crs().geocentric_epsg, Some(4978));
}

#[test]
fn geotiff_color_model_builder_passthrough_roundtrips() {
    let palette = Array3::from_shape_vec((2, 2, 2), vec![0u8, 255, 1, 192, 2, 128, 3, 64]).unwrap();
    let mut palette_buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(2, 2)
        .bands(2)
        .photometric(tiff_core::PhotometricInterpretation::Palette)
        .extra_samples(vec![ExtraSample::UnassociatedAlpha])
        .color_map(sample_color_map())
        .write_3d_to(&mut palette_buf, palette.view())
        .unwrap();

    let palette_file = TiffFile::from_bytes(palette_buf.into_inner()).unwrap();
    let palette_ifd = palette_file.ifd(0).unwrap();
    assert!(matches!(
        palette_ifd.color_model().unwrap(),
        ColorModel::Palette {
            color_map,
            extra_samples
        } if color_map.len() == 256 && extra_samples == vec![ExtraSample::UnassociatedAlpha]
    ));

    let cmyk = Array3::from_shape_vec((1, 2, 4), vec![0u8, 64, 128, 255, 255, 128, 64, 0]).unwrap();
    let mut cmyk_buf = Cursor::new(Vec::new());
    GeoTiffBuilder::new(2, 1)
        .bands(4)
        .photometric(tiff_core::PhotometricInterpretation::Separated)
        .ink_set(InkSet::Cmyk)
        .write_3d_to(&mut cmyk_buf, cmyk.view())
        .unwrap();

    let cmyk_file = TiffFile::from_bytes(cmyk_buf.into_inner()).unwrap();
    let cmyk_ifd = cmyk_file.ifd(0).unwrap();
    assert!(matches!(
        cmyk_ifd.color_model().unwrap(),
        ColorModel::Cmyk { extra_samples } if extra_samples.is_empty()
    ));
}

#[test]
fn geotiff_writer_rejects_unsupported_ycbcr_subsampling() {
    let data = Array3::<u8>::zeros((1, 1, 3));
    let mut buf = Cursor::new(Vec::new());
    let err = GeoTiffBuilder::new(1, 1)
        .bands(3)
        .photometric(tiff_core::PhotometricInterpretation::YCbCr)
        .ycbcr_subsampling([2, 2])
        .write_3d_to(&mut buf, data.view())
        .unwrap_err();
    assert!(
        matches!(err, GeoTiffWriteError::Tiff(tiff_writer::Error::InvalidConfig(message)) if message.contains("YCbCr subsampling"))
    );
}
