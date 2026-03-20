#![no_main]

use geotiff_reader::GeoTiffFile;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let file = match GeoTiffFile::from_bytes(data.to_vec()) {
        Ok(file) => file,
        Err(_) => return,
    };

    let _ = file.metadata();
    let _ = file.epsg();
    let _ = file.geo_bounds();
    let _ = file.transform().map(|transform| {
        let _ = transform.pixel_to_geo(0.0, 0.0);
        let _ = transform.geo_to_pixel(0.0, 0.0);
    });

    let Ok(ifd) = file.tiff().ifd(0) else {
        return;
    };
    let Ok(layout) = ifd.raster_layout() else {
        return;
    };
    let Some(decoded_len) = layout.row_bytes().checked_mul(layout.height) else {
        return;
    };
    if decoded_len > 8 * 1024 * 1024 {
        return;
    }

    let _ = file.tiff().read_image_bytes(0);
});
