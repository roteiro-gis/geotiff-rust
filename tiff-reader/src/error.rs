use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error reading {1}: {0}")]
    Io(#[source] std::io::Error, String),

    #[error("not a TIFF file: invalid magic bytes")]
    InvalidMagic,

    #[error("unsupported TIFF version: {0}")]
    UnsupportedVersion(u16),

    #[error("IFD index {0} not found")]
    IfdNotFound(usize),

    #[error("tag {0} not found in IFD")]
    TagNotFound(u16),

    #[error("unexpected tag type {actual} for tag {tag}, expected {expected}")]
    UnexpectedTagType {
        tag: u16,
        expected: &'static str,
        actual: u16,
    },

    #[error("unsupported compression: {0}")]
    UnsupportedCompression(u16),

    #[error("unsupported predictor: {0}")]
    UnsupportedPredictor(u16),

    #[error("unsupported planar configuration: {0}")]
    UnsupportedPlanarConfiguration(u16),

    #[error("unsupported bits per sample: {0}")]
    UnsupportedBitsPerSample(u16),

    #[error("unsupported sample format: {0}")]
    UnsupportedSampleFormat(u16),

    #[error("decompression failed for strip/tile {index}: {reason}")]
    DecompressionFailed { index: usize, reason: String },

    #[error("data truncated at offset {offset}: need {needed} bytes, have {available}")]
    Truncated {
        offset: u64,
        needed: u64,
        available: u64,
    },

    #[error("offset {offset} with length {length} is out of bounds for a {data_len}-byte file")]
    OffsetOutOfBounds {
        offset: u64,
        length: u64,
        data_len: u64,
    },

    #[error("invalid TIFF tag {tag}: {reason}")]
    InvalidTagValue {
        tag: u16,
        reason: String,
    },

    #[error("invalid image layout: {0}")]
    InvalidImageLayout(String),

    #[error("type mismatch: expected {expected}, found {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },

    #[error("{0}")]
    Other(String),
}
