use std::io::Cursor;

use geotiff_reader::GeoTiffFile;
use geotiff_writer::{
    CogBuilder, Compression, Error as GeoTiffWriteError, GeoTiffBuilder, JpegOptions,
    PhotometricInterpretation, PlanarConfiguration, Resampling, TiffVariant,
};
use ndarray::{Array2, Array3};
use tiff_reader::TiffFile;

fn gdal_structural_metadata_bytes(planar_configuration: PlanarConfiguration) -> Vec<u8> {
    let mut payload = String::from(
        "LAYOUT=IFDS_BEFORE_DATA\n\
BLOCK_ORDER=ROW_MAJOR\n\
BLOCK_LEADER=SIZE_AS_UINT4\n\
BLOCK_TRAILER=LAST_4_BYTES_REPEATED\n\
KNOWN_INCOMPATIBLE_EDITION=NO\n",
    );
    if matches!(planar_configuration, PlanarConfiguration::Planar) {
        payload.push_str("INTERLEAVE=BAND\n");
    }
    payload.push(' ');
    format!(
        "GDAL_STRUCTURAL_METADATA_SIZE={:06} bytes\n{}",
        payload.len(),
        payload
    )
    .into_bytes()
}

fn assert_strictly_increasing_offsets(offsets: &[u64], context: &str) {
    for window in offsets.windows(2) {
        assert!(
            window[0] < window[1],
            "{context}: offsets are not strictly increasing"
        );
    }
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
fn cog_layout_and_overview_discovery_roundtrip() {
    let data = Array2::<u8>::from_elem((64, 64), 42);
    let mut buf = Cursor::new(Vec::new());
    let builder = GeoTiffBuilder::new(64, 64)
        .tile_size(32, 32)
        .epsg(4326)
        .pixel_scale(1.0, 1.0)
        .origin(0.0, 64.0);

    CogBuilder::new(builder)
        .overview_levels(vec![2, 4])
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let bytes = buf.into_inner();
    let tiff = TiffFile::from_bytes(bytes.clone()).unwrap();
    assert_eq!(tiff.ifd(0).unwrap().width(), 64);
    assert!(tiff.ifd_count() >= 3);
    assert_eq!(
        &bytes[8..8 + gdal_structural_metadata_bytes(PlanarConfiguration::Chunky).len()],
        gdal_structural_metadata_bytes(PlanarConfiguration::Chunky).as_slice()
    );

    let geo = GeoTiffFile::from_bytes(bytes).unwrap();
    assert_eq!(geo.epsg(), Some(4326));
    assert_eq!(geo.base_ifd_index(), 0);
    assert_eq!(geo.overview_count(), 2);
    assert_eq!(geo.read_raster::<u8>().unwrap().shape(), &[64, 64]);
    assert_eq!(geo.read_overview::<u8>(0).unwrap().shape(), &[32, 32]);
    assert_eq!(geo.read_overview::<u8>(1).unwrap().shape(), &[16, 16]);
}

#[test]
fn cog_resampling_compression_and_ifd_order_hold() {
    let data = Array2::<u16>::from_elem((32, 32), 5000);
    let mut buf = Cursor::new(Vec::new());
    let builder = GeoTiffBuilder::new(32, 32)
        .tile_size(16, 16)
        .compression(Compression::Deflate);

    CogBuilder::new(builder)
        .overview_levels(vec![2])
        .resampling(Resampling::Average)
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let file = TiffFile::from_bytes(buf.into_inner()).unwrap();
    assert_eq!(file.ifd(0).unwrap().width(), 32);

    let base = file.read_image::<u16>(0).unwrap();
    let (values, offset) = base.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    assert!(values.iter().all(|&value| value == 5000));

    let overview_idx = (0..file.ifd_count())
        .find(|&index| file.ifd(index).unwrap().width() == 16)
        .unwrap();
    let overview = file.read_image::<u16>(overview_idx).unwrap();
    assert_eq!(overview.shape(), &[16, 16]);

    let mut min_data_offset = u64::MAX;
    for index in 0..file.ifd_count() {
        let ifd = file.ifd(index).unwrap();
        if let Some(offsets) = ifd.tile_offsets() {
            for offset in offsets {
                if offset > 0 {
                    min_data_offset = min_data_offset.min(offset);
                }
            }
        }
    }
    assert!(min_data_offset > 50);
}

#[test]
fn cog_streaming_and_multiband_roundtrip_match_expectations() {
    let mut rgb = Array3::<u8>::zeros((32, 32, 3));
    for row in 0..32 {
        for col in 0..32 {
            rgb[[row, col, 0]] = ((row * 7 + col * 3) % 251) as u8;
            rgb[[row, col, 1]] = ((row * 5 + col * 11) % 251) as u8;
            rgb[[row, col, 2]] = ((row * 13 + col * 17) % 251) as u8;
        }
    }

    let chunky_builder = GeoTiffBuilder::new(32, 32)
        .bands(3)
        .photometric(PhotometricInterpretation::Rgb)
        .tile_size(16, 16)
        .epsg(4326)
        .pixel_scale(1.0, 1.0)
        .origin(0.0, 32.0);
    let mut chunky_buf = Cursor::new(Vec::new());
    CogBuilder::new(chunky_builder)
        .overview_levels(vec![2])
        .resampling(Resampling::NearestNeighbor)
        .write_3d_to(&mut chunky_buf, rgb.view())
        .unwrap();
    let chunky_geo = GeoTiffFile::from_bytes(chunky_buf.into_inner()).unwrap();
    let base = chunky_geo.read_raster::<u8>().unwrap();
    assert_eq!(base.shape(), &[32, 32, 3]);
    assert_eq!(base[[5, 9, 0]], rgb[[5, 9, 0]]);
    assert_eq!(
        chunky_geo.read_overview::<u8>(0).unwrap().shape(),
        &[16, 16, 3]
    );

    let mut planar = Array3::<u8>::zeros((32, 32, 3));
    for row in 0..32 {
        for col in 0..32 {
            planar[[row, col, 0]] = ((row * 19 + col * 3) % 251) as u8;
            planar[[row, col, 1]] = ((row * 7 + col * 23) % 251) as u8;
            planar[[row, col, 2]] = ((row * 13 + col * 29) % 251) as u8;
        }
    }

    let planar_builder = GeoTiffBuilder::new(32, 32)
        .bands(3)
        .photometric(PhotometricInterpretation::Rgb)
        .planar_configuration(PlanarConfiguration::Planar)
        .tile_size(16, 16)
        .compression(Compression::Deflate)
        .epsg(4326);

    let mut oneshot_buf = Cursor::new(Vec::new());
    CogBuilder::new(planar_builder.clone())
        .overview_levels(vec![2])
        .write_3d_to(&mut oneshot_buf, planar.view())
        .unwrap();

    let mut streaming_buf = Cursor::new(Vec::new());
    let mut writer = CogBuilder::new(planar_builder)
        .overview_levels(vec![2])
        .tile_writer::<u8, _>(&mut streaming_buf)
        .unwrap();
    for tile_row in 0..2usize {
        for tile_col in 0..2usize {
            let y_off = tile_row * 16;
            let x_off = tile_col * 16;
            let tile = planar
                .slice(ndarray::s![y_off..y_off + 16, x_off..x_off + 16, ..])
                .to_owned();
            writer.write_tile_3d(x_off, y_off, &tile.view()).unwrap();
        }
    }
    writer.finish().unwrap();

    let oneshot = GeoTiffFile::from_bytes(oneshot_buf.into_inner()).unwrap();
    let streaming = GeoTiffFile::from_bytes(streaming_buf.into_inner()).unwrap();
    assert_strictly_increasing_offsets(
        &oneshot.tiff().ifd(0).unwrap().tile_offsets().unwrap(),
        "oneshot planar COG base image",
    );
    assert_strictly_increasing_offsets(
        &streaming.tiff().ifd(0).unwrap().tile_offsets().unwrap(),
        "streaming planar COG base image",
    );

    let oneshot_base = oneshot.read_raster::<u8>().unwrap();
    let streaming_base = streaming.read_raster::<u8>().unwrap();
    let (oneshot_values, oneshot_offset) = oneshot_base.into_raw_vec_and_offset();
    let (streaming_values, streaming_offset) = streaming_base.into_raw_vec_and_offset();
    assert_eq!(oneshot_offset, Some(0));
    assert_eq!(streaming_offset, Some(0));
    assert_eq!(oneshot_values, streaming_values);

    let overview = streaming.read_overview::<u8>(0).unwrap();
    assert_eq!(overview.shape(), &[16, 16, 3]);
}

#[test]
fn cog_validates_and_dedupes_overview_levels() {
    let data = Array2::<u8>::from_elem((8, 8), 1);

    let mut invalid_buf = Cursor::new(Vec::new());
    let err = CogBuilder::new(GeoTiffBuilder::new(8, 8).tile_size(16, 16))
        .overview_levels(vec![2, 0, 2])
        .write_2d_to(&mut invalid_buf, data.view())
        .unwrap_err();
    assert!(
        matches!(err, GeoTiffWriteError::InvalidConfig(message) if message.contains("greater than 1"))
    );

    let mut deduped_buf = Cursor::new(Vec::new());
    CogBuilder::new(GeoTiffBuilder::new(8, 8).tile_size(16, 16))
        .overview_levels(vec![4, 2, 2, 4])
        .write_2d_to(&mut deduped_buf, data.view())
        .unwrap();
    let tiff = TiffFile::from_bytes(deduped_buf.into_inner()).unwrap();
    assert_eq!(tiff.ifd_count(), 3);
}

#[test]
fn cog_reuses_writer_validation_for_invalid_layouts() {
    let rgb = Array3::<u8>::zeros((16, 16, 3));
    let mut chunky_jpeg_buf = Cursor::new(Vec::new());
    let err = CogBuilder::new(
        GeoTiffBuilder::new(16, 16)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .tile_size(16, 16)
            .jpeg_options(JpegOptions { quality: 90 }),
    )
    .write_3d_to(&mut chunky_jpeg_buf, rgb.view())
    .unwrap_err();
    assert!(
        matches!(err, GeoTiffWriteError::Tiff(tiff_writer::Error::InvalidConfig(message)) if message.contains("one sample per encoded block"))
    );

    let mut chunky_jpeg_streaming_buf = Cursor::new(Vec::new());
    let err = match CogBuilder::new(
        GeoTiffBuilder::new(16, 16)
            .bands(3)
            .photometric(PhotometricInterpretation::Rgb)
            .tile_size(16, 16)
            .jpeg_options(JpegOptions { quality: 90 }),
    )
    .tile_writer::<u8, _>(&mut chunky_jpeg_streaming_buf)
    {
        Ok(_) => panic!("expected invalid chunky JPEG COG tile writer to fail validation"),
        Err(err) => err,
    };
    assert!(
        matches!(err, GeoTiffWriteError::Tiff(tiff_writer::Error::InvalidConfig(message)) if message.contains("one sample per encoded block"))
    );

    let palette = Array2::<u8>::zeros((16, 16));
    let mut palette_buf = Cursor::new(Vec::new());
    let err = CogBuilder::new(
        GeoTiffBuilder::new(16, 16)
            .photometric(PhotometricInterpretation::Palette)
            .tile_size(16, 16),
    )
    .write_2d_to(&mut palette_buf, palette.view())
    .unwrap_err();
    assert!(
        matches!(err, GeoTiffWriteError::Tiff(tiff_writer::Error::InvalidConfig(message)) if message.contains("ColorMap"))
    );
}

#[test]
fn cog_average_overviews_ignore_nodata_for_oneshot_and_streaming_writes() {
    let nodata = -1.0f32;
    let oneshot = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, nodata, 2.0, nodata, nodata, nodata, nodata, nodata, 3.0, 3.0, 4.0, 4.0, 3.0,
            nodata, nodata, nodata,
        ],
    )
    .unwrap();

    let builder = GeoTiffBuilder::new(4, 4).tile_size(16, 16).nodata("-1");

    let mut oneshot_buf = Cursor::new(Vec::new());
    CogBuilder::new(builder.clone())
        .overview_levels(vec![2])
        .resampling(Resampling::Average)
        .write_2d_to(&mut oneshot_buf, oneshot.view())
        .unwrap();
    let oneshot_tiff = TiffFile::from_bytes(oneshot_buf.into_inner()).unwrap();
    let oneshot_overview = oneshot_tiff.read_image::<f32>(1).unwrap();
    assert_eq!(oneshot_overview[[0, 0]], 1.0);
    assert_eq!(oneshot_overview[[0, 1]], 2.0);
    assert_eq!(oneshot_overview[[1, 0]], 3.0);
    assert_eq!(oneshot_overview[[1, 1]], 4.0);

    let mut streaming_buf = Cursor::new(Vec::new());
    let mut writer = CogBuilder::new(builder)
        .overview_levels(vec![2])
        .resampling(Resampling::Average)
        .tile_writer::<f32, _>(&mut streaming_buf)
        .unwrap();
    let sparse_tile = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    writer.write_tile(0, 0, &sparse_tile.view()).unwrap();
    writer.finish().unwrap();

    let streaming_tiff = TiffFile::from_bytes(streaming_buf.into_inner()).unwrap();
    let streaming_overview = streaming_tiff.read_image::<f32>(1).unwrap();
    assert_eq!(streaming_overview[[0, 0]], 2.5);
    assert_eq!(streaming_overview[[0, 1]], nodata);
    assert_eq!(streaming_overview[[1, 0]], nodata);
    assert_eq!(streaming_overview[[1, 1]], nodata);
}

#[test]
fn cog_emits_bigtiff_when_requested() {
    let data = Array2::<u8>::from_elem((32, 32), 7);
    let mut buf = Cursor::new(Vec::new());
    let builder = GeoTiffBuilder::new(32, 32)
        .tile_size(16, 16)
        .tiff_variant(TiffVariant::BigTiff);

    CogBuilder::new(builder)
        .overview_levels(vec![2])
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let bytes = buf.into_inner();
    assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 43);
    assert_eq!(
        &bytes[16..16 + gdal_structural_metadata_bytes(PlanarConfiguration::Chunky).len()],
        gdal_structural_metadata_bytes(PlanarConfiguration::Chunky).as_slice()
    );

    let tiff = TiffFile::from_bytes(bytes).unwrap();
    assert!(tiff.is_bigtiff());
}

#[test]
fn cog_jpeg_compression_roundtrips_without_jpeg_tables() {
    let mut data = Array2::<u8>::zeros((32, 32));
    for row in 0..32usize {
        for col in 0..32usize {
            data[[row, col]] = match (row / 16, col / 16) {
                (0, 0) => 24,
                (0, 1) => 96,
                (1, 0) => 160,
                _ => 224,
            };
        }
    }

    let mut buf = Cursor::new(Vec::new());
    let builder = GeoTiffBuilder::new(32, 32)
        .tile_size(16, 16)
        .epsg(4326)
        .jpeg_options(JpegOptions { quality: 90 });

    CogBuilder::new(builder)
        .overview_levels(vec![2])
        .write_2d_to(&mut buf, data.view())
        .unwrap();

    let bytes = buf.into_inner();
    let tiff = TiffFile::from_bytes(bytes.clone()).unwrap();
    let base_ifd = tiff.ifd(0).unwrap();
    let overview_ifd = tiff.ifd(1).unwrap();
    assert_eq!(base_ifd.compression(), Compression::Jpeg.to_code());
    assert_eq!(overview_ifd.compression(), Compression::Jpeg.to_code());
    assert!(base_ifd.tag(tiff_core::TAG_JPEG_TABLES).is_none());
    assert!(overview_ifd.tag(tiff_core::TAG_JPEG_TABLES).is_none());

    let raster = tiff.read_image::<u8>(0).unwrap();
    let (values, offset) = raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    let expected = data.iter().copied().collect::<Vec<_>>();
    assert_u8_bytes_close(&values, &expected, 2, 32);

    let geo = GeoTiffFile::from_bytes(bytes).unwrap();
    assert_eq!(geo.overview_count(), 1);
    assert_eq!(geo.read_overview::<u8>(0).unwrap().shape(), &[16, 16]);
}
