use ndarray::ArrayD;
use tiff_reader::TiffFile;

#[path = "../../test-support/reference.rs"]
mod reference;

#[derive(Debug, Default)]
struct DirectoryExpectation {
    width: u32,
    height: u32,
    bits_per_sample: Vec<u16>,
    compression: u16,
    samples_per_pixel: u16,
    sample_format: Vec<u16>,
    rows_per_strip: Option<u32>,
    tile_width: Option<u32>,
    tile_height: Option<u32>,
}

#[derive(Debug, Default)]
struct TiffDumpExpectation {
    is_bigtiff: bool,
    directories: Vec<DirectoryExpectation>,
}

#[derive(Clone, Copy)]
enum SampleKind {
    U8,
    I8,
}

fn fixture(path: &str) -> std::path::PathBuf {
    reference::fixture(env!("CARGO_MANIFEST_DIR"), path)
}

fn parse_tiffdump(output: &str) -> TiffDumpExpectation {
    let mut parsed = TiffDumpExpectation::default();
    let mut current = None;

    for line in output.lines() {
        if line.contains("<BigTIFF>") {
            parsed.is_bigtiff = true;
        }
        if line.starts_with("Directory ") {
            if let Some(directory) = current.take() {
                parsed.directories.push(directory);
            }
            current = Some(DirectoryExpectation::default());
            continue;
        }

        let Some(directory) = current.as_mut() else {
            continue;
        };

        if line.starts_with("ImageWidth ") {
            directory.width = parse_scalar_u32(line);
        } else if line.starts_with("ImageLength ") {
            directory.height = parse_scalar_u32(line);
        } else if line.starts_with("BitsPerSample ") {
            directory.bits_per_sample = parse_u16_list(line);
        } else if line.starts_with("Compression ") {
            directory.compression = parse_scalar_u16(line);
        } else if line.starts_with("SamplesPerPixel ") {
            directory.samples_per_pixel = parse_scalar_u16(line);
        } else if line.starts_with("SampleFormat ") {
            directory.sample_format = parse_u16_list(line);
        } else if line.starts_with("RowsPerStrip ") {
            directory.rows_per_strip = Some(parse_scalar_u32(line));
        } else if line.starts_with("TileWidth ") {
            directory.tile_width = Some(parse_scalar_u32(line));
        } else if line.starts_with("TileLength ") {
            directory.tile_height = Some(parse_scalar_u32(line));
        }
    }

    if let Some(directory) = current.take() {
        parsed.directories.push(directory);
    }

    parsed
}

fn parse_scalar_u16(line: &str) -> u16 {
    parse_u16_list(line)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("missing u16 payload in line: {line}"))
}

fn parse_scalar_u32(line: &str) -> u32 {
    parse_u32_list(line)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("missing u32 payload in line: {line}"))
}

fn parse_u16_list(line: &str) -> Vec<u16> {
    parse_u32_list(line)
        .into_iter()
        .map(|value| u16::try_from(value).unwrap())
        .collect()
}

fn parse_u32_list(line: &str) -> Vec<u32> {
    let payload = line
        .split_once('<')
        .and_then(|(_, rest)| rest.strip_suffix('>'))
        .unwrap_or_else(|| panic!("missing angle-bracket payload in line: {line}"));
    payload
        .split_whitespace()
        .map(|value| value.parse::<u32>().unwrap())
        .collect()
}

fn assert_shape<T>(array: &ArrayD<T>, width: u32, height: u32, band_count: u64) {
    if band_count == 1 {
        assert_eq!(array.shape(), &[height as usize, width as usize]);
    } else {
        assert_eq!(
            array.shape(),
            &[height as usize, width as usize, band_count as usize]
        );
    }
}

fn assert_gdal_hash_matches(path: &std::path::Path, ifd_index: usize, sample_kind: SampleKind) {
    let path_str = path.to_str().unwrap();
    let overview_arg = (ifd_index != 0).then(|| (ifd_index - 1).to_string());
    let reference_json = if let Some(ref overview) = overview_arg {
        reference::run_reference_json(
            env!("CARGO_MANIFEST_DIR"),
            &["hash", path_str, "--overview", overview],
        )
    } else {
        reference::run_reference_json(env!("CARGO_MANIFEST_DIR"), &["hash", path_str])
    };
    let file = TiffFile::open(path).unwrap();
    let width = reference_json["width"].as_u64().unwrap();
    let height = reference_json["height"].as_u64().unwrap();
    let band_count = reference_json["band_count"].as_u64().unwrap();
    let byte_len = reference_json["byte_len"].as_u64().unwrap() as usize;
    let expected_hash = reference_json["hash"].as_str().unwrap();

    match sample_kind {
        SampleKind::U8 => {
            let raster: ArrayD<u8> = file.read_image(ifd_index).unwrap();
            assert_shape(&raster, width as u32, height as u32, band_count);
            let (actual_len, actual_hash) = reference::array_hash(&raster);
            assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
            assert_eq!(
                actual_hash, expected_hash,
                "pixel hash mismatch for {path_str}"
            );
        }
        SampleKind::I8 => {
            let raster: ArrayD<i8> = file.read_image(ifd_index).unwrap();
            assert_shape(&raster, width as u32, height as u32, band_count);
            let (actual_len, actual_hash) = reference::array_hash(&raster);
            assert_eq!(actual_len, byte_len, "byte length mismatch for {path_str}");
            assert_eq!(
                actual_hash, expected_hash,
                "pixel hash mismatch for {path_str}"
            );
        }
    }
}

fn assert_gdal_u8_pixels_close(path: &std::path::Path, ifd_index: usize) {
    let path_str = path.to_str().unwrap();
    let overview_arg = (ifd_index != 0).then(|| (ifd_index - 1).to_string());
    let expected = if let Some(ref overview) = overview_arg {
        reference::run_reference_bytes(
            env!("CARGO_MANIFEST_DIR"),
            &["bytes", path_str, "--overview", overview],
        )
    } else {
        reference::run_reference_bytes(env!("CARGO_MANIFEST_DIR"), &["bytes", path_str])
    };
    let file = TiffFile::open(path).unwrap();
    let raster: ArrayD<u8> = file.read_image(ifd_index).unwrap();
    let (actual, offset) = raster.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0), "unexpected array offset for {path_str}");

    // JPEG stays on a bounded-delta comparison because `jpeg-decoder` differs
    // from GDAL/libjpeg on `byte_JPEG.tif` by 3 grayscale samples, each +1.
    let max_diff_pixels = (expected.len() / 100).max(4);
    reference::assert_u8_bytes_close(&actual, &expected, 1, max_diff_pixels, path_str);
}

#[test]
fn matches_libtiff_directory_layout_for_interoperability_corpus() {
    if !reference::tiffdump_available() {
        eprintln!("skipping libtiff parity test because `tiffdump` is unavailable");
        return;
    }

    let fixtures = [
        "gdal/gcore/data/byte.tif",
        "gdal/gcore/data/WGS_1984_Web_Mercator.tif",
        "gdal/gcore/data/byte_with_ovr.tif",
        "gdal/gcore/data/bigtiff_one_strip_long.tif",
        "gdal/gcore/data/gtiff/byte_NONE_tiled.tif",
        "gdal/gdrivers/data/gtiff/int8.tif",
    ];

    for relative_path in fixtures {
        let path = fixture(relative_path);
        let expected = parse_tiffdump(&reference::run_tiffdump(&path));
        let file = TiffFile::open(&path).unwrap();

        assert_eq!(file.is_bigtiff(), expected.is_bigtiff, "{relative_path}");
        assert_eq!(
            file.ifd_count(),
            expected.directories.len(),
            "{relative_path}"
        );

        for (index, directory) in expected.directories.iter().enumerate() {
            let ifd = file.ifd(index).unwrap();
            assert_eq!(ifd.width(), directory.width, "{relative_path} ifd {index}");
            assert_eq!(
                ifd.height(),
                directory.height,
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.bits_per_sample(),
                directory.bits_per_sample,
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.compression(),
                directory.compression,
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.samples_per_pixel(),
                directory.samples_per_pixel,
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.sample_format(),
                directory.sample_format,
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.is_tiled(),
                directory.tile_width.is_some() && directory.tile_height.is_some(),
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.rows_per_strip(),
                directory.rows_per_strip.or(Some(directory.height)),
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.tile_width(),
                directory.tile_width,
                "{relative_path} ifd {index}"
            );
            assert_eq!(
                ifd.tile_height(),
                directory.tile_height,
                "{relative_path} ifd {index}"
            );
        }
    }
}

#[test]
fn matches_gdal_decoded_pixels_for_interoperability_corpus() {
    if !reference::python_gdal_available() {
        eprintln!("skipping GDAL pixel parity test because Python GDAL bindings are unavailable");
        return;
    }

    let cases = vec![
        (fixture("gdal/gcore/data/byte.tif"), 0usize, SampleKind::U8),
        (
            fixture("gdal/gcore/data/WGS_1984_Web_Mercator.tif"),
            0usize,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/byte_with_ovr.tif"),
            0usize,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/byte_with_ovr.tif"),
            1usize,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/bigtiff_one_strip_long.tif"),
            0usize,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gcore/data/gtiff/byte_NONE_tiled.tif"),
            0usize,
            SampleKind::U8,
        ),
        (
            fixture("gdal/gdrivers/data/gtiff/int8.tif"),
            0usize,
            SampleKind::I8,
        ),
        (
            fixture("gdal/gcore/data/gtiff/byte_JPEG.tif"),
            0usize,
            SampleKind::U8,
        ),
    ];

    #[cfg(feature = "zstd")]
    let mut cases = cases;

    #[cfg(feature = "zstd")]
    cases.push((
        fixture("gdal/gcore/data/byte_zstd.tif"),
        0usize,
        SampleKind::U8,
    ));

    for (path, ifd_index, sample_kind) in cases {
        if path.ends_with("byte_JPEG.tif") {
            assert_gdal_u8_pixels_close(&path, ifd_index);
        } else {
            assert_gdal_hash_matches(&path, ifd_index, sample_kind);
        }
    }
}
