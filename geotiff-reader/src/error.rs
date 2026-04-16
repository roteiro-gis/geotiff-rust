use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error reading {1}: {0}")]
    Io(#[source] std::io::Error, String),

    #[error("TIFF error: {0}")]
    #[cfg(feature = "local")]
    Tiff(#[from] tiff_reader::TiffError),

    #[error("HTTP error: {0}")]
    #[cfg(feature = "cog")]
    Http(#[from] reqwest::Error),

    #[error("not a GeoTIFF: missing GeoKey directory (tag 34735)")]
    NotGeoTiff,

    #[error("invalid GeoKey directory")]
    InvalidGeoKeyDirectory,

    #[error("unsupported GeoKey model type: {0}")]
    UnsupportedModelType(u16),

    #[error("EPSG code {0} not recognized")]
    UnknownEpsg(u32),

    #[error("overview index {0} not found")]
    OverviewNotFound(usize),

    #[error("overview index {0} is stored in a SubIFD and has no top-level TIFF IFD index")]
    OverviewHasNoTopLevelIfdIndex(usize),

    #[error("band index {0} is out of bounds")]
    BandOutOfBounds(usize),

    #[error("no pixel scale or transformation matrix found")]
    NoGeoTransform,

    #[error("{0}")]
    Other(String),
}
