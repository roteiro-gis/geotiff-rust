#![no_main]

use libfuzzer_sys::fuzz_target;
use tiff_reader::{OpenOptions, TiffFile};

const MAX_DECODED_BYTES: usize = 8 * 1024 * 1024;
const MAX_IFDS: usize = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let file = match TiffFile::from_bytes_with_options(
        data.to_vec(),
        OpenOptions {
            block_cache_bytes: 0,
            block_cache_slots: 0,
        },
    ) {
        Ok(file) => file,
        Err(_) => return,
    };

    for ifd_index in 0..file.ifd_count().min(MAX_IFDS) {
        let Ok(ifd) = file.ifd(ifd_index) else {
            continue;
        };
        let Ok(layout) = ifd.raster_layout() else {
            continue;
        };
        let Some(decoded_len) = layout.row_bytes().checked_mul(layout.height) else {
            continue;
        };
        if decoded_len > MAX_DECODED_BYTES {
            continue;
        }

        let _ = file.read_image_bytes(ifd_index);
    }
});
