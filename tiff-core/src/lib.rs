//! Shared TIFF/BigTIFF types used by both `tiff-reader` and `tiff-writer`.
//!
//! This crate provides the foundational types for TIFF file manipulation:
//! byte order, tag types, tag values, sample traits, raster layout,
//! and well-known constants and enums.

pub mod byte_order;
pub mod constants;
pub mod layout;
pub mod sample;
pub mod tag;

pub use byte_order::ByteOrder;
pub use constants::*;
pub use layout::RasterLayout;
pub use sample::TiffSample;
pub use tag::{Tag, TagType, TagValue};
