use crate::byte_order::ByteOrder;

/// TIFF data type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagType {
    Byte,      // 1
    Ascii,     // 2
    Short,     // 3
    Long,      // 4
    Rational,  // 5
    SByte,     // 6
    Undefined, // 7
    SShort,    // 8
    SLong,     // 9
    SRational, // 10
    Float,     // 11
    Double,    // 12
    Long8,     // 16 (BigTIFF)
    SLong8,    // 17 (BigTIFF)
    Ifd8,      // 18 (BigTIFF)
    Unknown(u16),
}

impl TagType {
    pub fn from_code(code: u16) -> Self {
        match code {
            1 => Self::Byte,
            2 => Self::Ascii,
            3 => Self::Short,
            4 => Self::Long,
            5 => Self::Rational,
            6 => Self::SByte,
            7 => Self::Undefined,
            8 => Self::SShort,
            9 => Self::SLong,
            10 => Self::SRational,
            11 => Self::Float,
            12 => Self::Double,
            16 => Self::Long8,
            17 => Self::SLong8,
            18 => Self::Ifd8,
            _ => Self::Unknown(code),
        }
    }

    /// Convert back to the TIFF type code.
    pub fn to_code(self) -> u16 {
        match self {
            Self::Byte => 1,
            Self::Ascii => 2,
            Self::Short => 3,
            Self::Long => 4,
            Self::Rational => 5,
            Self::SByte => 6,
            Self::Undefined => 7,
            Self::SShort => 8,
            Self::SLong => 9,
            Self::SRational => 10,
            Self::Float => 11,
            Self::Double => 12,
            Self::Long8 => 16,
            Self::SLong8 => 17,
            Self::Ifd8 => 18,
            Self::Unknown(c) => c,
        }
    }

    /// Size in bytes of a single element of this type.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Byte | Self::Ascii | Self::SByte | Self::Undefined => 1,
            Self::Short | Self::SShort => 2,
            Self::Long | Self::SLong | Self::Float => 4,
            Self::Rational
            | Self::SRational
            | Self::Double
            | Self::Long8
            | Self::SLong8
            | Self::Ifd8 => 8,
            Self::Unknown(_) => 1,
        }
    }
}

/// A parsed TIFF tag.
#[derive(Debug, Clone)]
pub struct Tag {
    pub code: u16,
    pub tag_type: TagType,
    pub count: u64,
    pub value: TagValue,
}

impl Tag {
    /// Construct a tag from a code and value. Type and count are inferred.
    pub fn new(code: u16, value: TagValue) -> Self {
        Self {
            code,
            tag_type: value.tag_type(),
            count: value.count(),
            value,
        }
    }
}

/// Decoded tag value.
#[derive(Debug, Clone, PartialEq)]
pub enum TagValue {
    Byte(Vec<u8>),
    Ascii(String),
    Short(Vec<u16>),
    Long(Vec<u32>),
    Rational(Vec<[u32; 2]>),
    SByte(Vec<i8>),
    Undefined(Vec<u8>),
    SShort(Vec<i16>),
    SLong(Vec<i32>),
    SRational(Vec<[i32; 2]>),
    Float(Vec<f32>),
    Double(Vec<f64>),
    Long8(Vec<u64>),
    SLong8(Vec<i64>),
}

impl TagValue {
    /// Returns the TagType that matches this value variant.
    pub fn tag_type(&self) -> TagType {
        match self {
            Self::Byte(_) => TagType::Byte,
            Self::Ascii(_) => TagType::Ascii,
            Self::Short(_) => TagType::Short,
            Self::Long(_) => TagType::Long,
            Self::Rational(_) => TagType::Rational,
            Self::SByte(_) => TagType::SByte,
            Self::Undefined(_) => TagType::Undefined,
            Self::SShort(_) => TagType::SShort,
            Self::SLong(_) => TagType::SLong,
            Self::SRational(_) => TagType::SRational,
            Self::Float(_) => TagType::Float,
            Self::Double(_) => TagType::Double,
            Self::Long8(_) => TagType::Long8,
            Self::SLong8(_) => TagType::SLong8,
        }
    }

    /// Number of elements.
    pub fn count(&self) -> u64 {
        match self {
            Self::Byte(v) | Self::Undefined(v) => v.len() as u64,
            Self::Ascii(s) => (s.len() + 1) as u64, // +1 for NUL terminator
            Self::Short(v) => v.len() as u64,
            Self::Long(v) => v.len() as u64,
            Self::Rational(v) => v.len() as u64,
            Self::SByte(v) => v.len() as u64,
            Self::SShort(v) => v.len() as u64,
            Self::SLong(v) => v.len() as u64,
            Self::SRational(v) => v.len() as u64,
            Self::Float(v) => v.len() as u64,
            Self::Double(v) => v.len() as u64,
            Self::Long8(v) => v.len() as u64,
            Self::SLong8(v) => v.len() as u64,
        }
    }

    /// Total byte length when encoded.
    pub fn encoded_len(&self) -> usize {
        self.count() as usize * self.tag_type().element_size()
    }

    /// Encode the value into bytes using the given byte order.
    pub fn encode(&self, byte_order: ByteOrder) -> Vec<u8> {
        match self {
            Self::Byte(v) | Self::Undefined(v) => v.clone(),
            Self::Ascii(s) => {
                let mut bytes = s.as_bytes().to_vec();
                bytes.push(0); // NUL terminator
                bytes
            }
            Self::Short(v) => v.iter().flat_map(|&x| byte_order.write_u16(x)).collect(),
            Self::Long(v) => v.iter().flat_map(|&x| byte_order.write_u32(x)).collect(),
            Self::Rational(v) => v
                .iter()
                .flat_map(|&[n, d]| {
                    let mut bytes = Vec::with_capacity(8);
                    bytes.extend_from_slice(&byte_order.write_u32(n));
                    bytes.extend_from_slice(&byte_order.write_u32(d));
                    bytes
                })
                .collect(),
            Self::SByte(v) => v.iter().map(|&x| x as u8).collect(),
            Self::SShort(v) => v.iter().flat_map(|&x| byte_order.write_i16(x)).collect(),
            Self::SLong(v) => v.iter().flat_map(|&x| byte_order.write_i32(x)).collect(),
            Self::SRational(v) => v
                .iter()
                .flat_map(|&[n, d]| {
                    let mut bytes = Vec::with_capacity(8);
                    bytes.extend_from_slice(&byte_order.write_i32(n));
                    bytes.extend_from_slice(&byte_order.write_i32(d));
                    bytes
                })
                .collect(),
            Self::Float(v) => v.iter().flat_map(|&x| byte_order.write_f32(x)).collect(),
            Self::Double(v) => v.iter().flat_map(|&x| byte_order.write_f64(x)).collect(),
            Self::Long8(v) => v.iter().flat_map(|&x| byte_order.write_u64(x)).collect(),
            Self::SLong8(v) => v.iter().flat_map(|&x| byte_order.write_i64(x)).collect(),
        }
    }

    /// Extract a single u16 value.
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            Self::Short(v) => v.first().copied(),
            Self::Byte(v) => v.first().map(|&b| b as u16),
            Self::Long(v) => v.first().map(|&l| l as u16),
            _ => None,
        }
    }

    /// Extract a single u32 value.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Long(v) => v.first().copied(),
            Self::Short(v) => v.first().map(|&s| s as u32),
            Self::Long8(v) => v.first().map(|&l| l as u32),
            _ => None,
        }
    }

    /// Extract a single u64 value.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Long8(v) => v.first().copied(),
            Self::Long(v) => v.first().map(|&l| l as u64),
            Self::Short(v) => v.first().map(|&s| s as u64),
            _ => None,
        }
    }

    /// Extract a single f64 value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Double(v) => v.first().copied(),
            Self::Float(v) => v.first().map(|&f| f as f64),
            Self::Long(v) => v.first().map(|&l| l as f64),
            Self::Short(v) => v.first().map(|&s| s as f64),
            _ => None,
        }
    }

    /// Extract as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Ascii(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract raw bytes for byte-oriented tag payloads.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Byte(v) | Self::Undefined(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Extract as a slice of f64 values.
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            Self::Double(v) => Some(v.clone()),
            Self::Float(v) => Some(v.iter().map(|&f| f as f64).collect()),
            _ => None,
        }
    }

    /// Extract a value list as unsigned offsets/counts.
    pub fn as_u64_vec(&self) -> Option<Vec<u64>> {
        match self {
            Self::Byte(v) => Some(v.iter().map(|&x| x as u64).collect()),
            Self::Short(v) => Some(v.iter().map(|&x| x as u64).collect()),
            Self::Long(v) => Some(v.iter().map(|&x| x as u64).collect()),
            Self::Long8(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Extract a SHORT array without cloning when possible.
    pub fn as_u16_slice(&self) -> Option<&[u16]> {
        match self {
            Self::Short(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}
