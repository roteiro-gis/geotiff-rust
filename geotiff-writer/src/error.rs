use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TIFF writer error: {0}")]
    Tiff(#[from] tiff_writer::Error),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("tile ({x_off},{y_off}) out of bounds for {width}x{height} raster")]
    TileOutOfBounds {
        x_off: usize,
        y_off: usize,
        width: u32,
        height: u32,
    },

    #[error("data size mismatch: expected {expected}, got {actual}")]
    DataSizeMismatch { expected: usize, actual: usize },

    #[error("{0}")]
    Other(String),
}
