use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("compression failed for block {index}: {reason}")]
    CompressionFailed { index: usize, reason: String },

    #[error("block {index} has wrong sample count: expected {expected}, got {actual}")]
    BlockSizeMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("block index {index} is out of range (expected < {total})")]
    BlockIndexOutOfRange { index: usize, total: usize },

    #[error("not all blocks were written: wrote {written} of {total}")]
    IncompleteImage { written: usize, total: usize },

    #[error("writer has already been finalized")]
    AlreadyFinalized,

    #[error("classic TIFF offset {offset} exceeds 4 GiB limit; use TiffVariant::BigTiff")]
    ClassicOffsetOverflow { offset: u64 },

    #[error("{0}")]
    Other(String),
}
