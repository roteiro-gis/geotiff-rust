// Well-known TIFF tag codes.
pub const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub const TAG_SUBFILE_TYPE: u16 = 255;
pub const TAG_IMAGE_WIDTH: u16 = 256;
pub const TAG_IMAGE_LENGTH: u16 = 257;
pub const TAG_BITS_PER_SAMPLE: u16 = 258;
pub const TAG_COMPRESSION: u16 = 259;
pub const TAG_PHOTOMETRIC_INTERPRETATION: u16 = 262;
pub const TAG_STRIP_OFFSETS: u16 = 273;
pub const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub const TAG_ROWS_PER_STRIP: u16 = 278;
pub const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub const TAG_PLANAR_CONFIGURATION: u16 = 284;
pub const TAG_PREDICTOR: u16 = 317;
pub const TAG_TILE_WIDTH: u16 = 322;
pub const TAG_TILE_LENGTH: u16 = 323;
pub const TAG_TILE_OFFSETS: u16 = 324;
pub const TAG_TILE_BYTE_COUNTS: u16 = 325;
pub const TAG_SAMPLE_FORMAT: u16 = 339;
pub const TAG_JPEG_TABLES: u16 = 347;

/// TIFF compression scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    None,
    Lzw,
    OldJpeg,
    Jpeg,
    Deflate,
    PackBits,
    DeflateOld,
    Zstd,
}

impl Compression {
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::None),
            5 => Some(Self::Lzw),
            6 => Some(Self::OldJpeg),
            7 => Some(Self::Jpeg),
            8 => Some(Self::Deflate),
            32773 => Some(Self::PackBits),
            32946 => Some(Self::DeflateOld),
            50000 => Some(Self::Zstd),
            _ => None,
        }
    }

    pub fn to_code(self) -> u16 {
        match self {
            Self::None => 1,
            Self::Lzw => 5,
            Self::OldJpeg => 6,
            Self::Jpeg => 7,
            Self::Deflate => 8,
            Self::PackBits => 32773,
            Self::DeflateOld => 32946,
            Self::Zstd => 50000,
        }
    }
}

/// TIFF predictor scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Predictor {
    None,
    Horizontal,
    FloatingPoint,
}

impl Predictor {
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::None),
            2 => Some(Self::Horizontal),
            3 => Some(Self::FloatingPoint),
            _ => None,
        }
    }

    pub fn to_code(self) -> u16 {
        match self {
            Self::None => 1,
            Self::Horizontal => 2,
            Self::FloatingPoint => 3,
        }
    }
}

/// TIFF sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    Uint,
    Int,
    Float,
}

impl SampleFormat {
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Uint),
            2 => Some(Self::Int),
            3 => Some(Self::Float),
            _ => None,
        }
    }

    pub fn to_code(self) -> u16 {
        match self {
            Self::Uint => 1,
            Self::Int => 2,
            Self::Float => 3,
        }
    }
}

/// TIFF photometric interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhotometricInterpretation {
    MinIsWhite,
    MinIsBlack,
    Rgb,
    Palette,
    Mask,
}

impl PhotometricInterpretation {
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            0 => Some(Self::MinIsWhite),
            1 => Some(Self::MinIsBlack),
            2 => Some(Self::Rgb),
            3 => Some(Self::Palette),
            4 => Some(Self::Mask),
            _ => None,
        }
    }

    pub fn to_code(self) -> u16 {
        match self {
            Self::MinIsWhite => 0,
            Self::MinIsBlack => 1,
            Self::Rgb => 2,
            Self::Palette => 3,
            Self::Mask => 4,
        }
    }
}

/// TIFF planar configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanarConfiguration {
    Chunky,
    Planar,
}

impl PlanarConfiguration {
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Chunky),
            2 => Some(Self::Planar),
            _ => None,
        }
    }

    pub fn to_code(self) -> u16 {
        match self {
            Self::Chunky => 1,
            Self::Planar => 2,
        }
    }
}
