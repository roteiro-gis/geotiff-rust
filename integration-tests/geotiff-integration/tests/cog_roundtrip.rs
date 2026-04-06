use std::io::Cursor;

use geotiff_reader::GeoTiffFile;
use geotiff_writer::{
    CogBuilder, Compression, GeoTiffBuilder, PhotometricInterpretation, PlanarConfiguration,
    Resampling,
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
        assert!(window[0] < window[1], "{context}: offsets are not strictly increasing");
    }
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
    assert_eq!(chunky_geo.read_overview::<u8>(0).unwrap().shape(), &[16, 16, 3]);

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
