//! Pure-Rust TIFF/BigTIFF encoder with compression, tiling, and streaming writes.
//!
//! Supports:
//! - **Compression**: None, LZW, Deflate, LERC, LERC+Deflate, LERC+Zstd, ZSTD (feature)
//! - **Predictors**: Horizontal differencing, floating-point
//! - **Layouts**: strips, tiles, multi-IFD, BigTIFF
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

pub use builder::{DataLayout, ImageBuilder, LercOptions};
pub use error::{Error, Result};
pub use sample::TiffWriteSample;
pub use tiff_core::LercAdditionalCompression;
pub use writer::{ImageHandle, TiffVariant, TiffWriter, WriteOptions};
