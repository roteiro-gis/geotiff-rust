use crate::error::{Error, Result};
use crate::header::ByteOrder;
use crate::io::Cursor;
use crate::source::TiffSource;

/// TIFF data type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagType {
    Byte,       // 1
    Ascii,      // 2
    Short,      // 3
    Long,       // 4
    Rational,   // 5
    SByte,      // 6
    Undefined,  // 7
    SShort,     // 8
    SLong,      // 9
    SRational,  // 10
    Float,      // 11
    Double,     // 12
    Long8,      // 16 (BigTIFF)
    SLong8,     // 17 (BigTIFF)
    Ifd8,       // 18 (BigTIFF)
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

    /// Size in bytes of a single element of this type.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Byte | Self::Ascii | Self::SByte | Self::Undefined => 1,
            Self::Short | Self::SShort => 2,
            Self::Long | Self::SLong | Self::Float => 4,
            Self::Rational | Self::SRational | Self::Double | Self::Long8 | Self::SLong8 | Self::Ifd8 => 8,
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

/// Decoded tag value.
#[derive(Debug, Clone)]
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

impl Tag {
    /// Parse a classic TIFF tag entry (12-byte IFD entry).
    pub fn parse_classic(
        code: u16,
        type_code: u16,
        count: u64,
        value_offset_bytes: &[u8],
        source: &dyn TiffSource,
        byte_order: ByteOrder,
    ) -> Result<Self> {
        let tag_type = TagType::from_code(type_code);
        let total_size = value_len(code, count, tag_type.element_size())?;

        let owned;
        let value_bytes = if total_size <= 4 {
            &value_offset_bytes[..total_size]
        } else {
            let offset = match byte_order {
                ByteOrder::LittleEndian => u32::from_le_bytes(value_offset_bytes.try_into().unwrap()),
                ByteOrder::BigEndian => u32::from_be_bytes(value_offset_bytes.try_into().unwrap()),
            } as u64;
            owned = read_value_bytes(source, offset, total_size)?;
            owned.as_slice()
        };

        let value = decode_value(&tag_type, count, value_bytes, byte_order)?;
        Ok(Self {
            code,
            tag_type,
            count,
            value,
        })
    }

    /// Parse a BigTIFF tag entry (20-byte IFD entry).
    pub fn parse_bigtiff(
        code: u16,
        type_code: u16,
        count: u64,
        value_offset_bytes: &[u8],
        source: &dyn TiffSource,
        byte_order: ByteOrder,
    ) -> Result<Self> {
        let tag_type = TagType::from_code(type_code);
        let total_size = value_len(code, count, tag_type.element_size())?;

        let owned;
        let value_bytes = if total_size <= 8 {
            &value_offset_bytes[..total_size]
        } else {
            let offset = match byte_order {
                ByteOrder::LittleEndian => u64::from_le_bytes(value_offset_bytes.try_into().unwrap()),
                ByteOrder::BigEndian => u64::from_be_bytes(value_offset_bytes.try_into().unwrap()),
            };
            owned = read_value_bytes(source, offset, total_size)?;
            owned.as_slice()
        };

        let value = decode_value(&tag_type, count, value_bytes, byte_order)?;
        Ok(Self {
            code,
            tag_type,
            count,
            value,
        })
    }
}

fn read_value_bytes(source: &dyn TiffSource, offset: u64, len: usize) -> Result<Vec<u8>> {
    if let Some(data) = source.as_slice() {
        return Ok(slice_at(data, offset, len)?.to_vec());
    }
    source.read_exact_at(offset, len)
}

fn value_len(tag: u16, count: u64, element_size: usize) -> Result<usize> {
    let count = usize::try_from(count).map_err(|_| Error::InvalidTagValue {
        tag,
        reason: "value count does not fit in memory".into(),
    })?;
    count.checked_mul(element_size).ok_or_else(|| Error::InvalidTagValue {
        tag,
        reason: "value byte length overflows usize".into(),
    })
}

fn slice_at(data: &[u8], offset: u64, len: usize) -> Result<&[u8]> {
    let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds {
        offset,
        length: len as u64,
        data_len: data.len() as u64,
    })?;
    let end = start.checked_add(len).ok_or(Error::OffsetOutOfBounds {
        offset,
        length: len as u64,
        data_len: data.len() as u64,
    })?;
    if end > data.len() {
        return Err(Error::OffsetOutOfBounds {
            offset,
            length: len as u64,
            data_len: data.len() as u64,
        });
    }
    Ok(&data[start..end])
}

fn decode_value(
    tag_type: &TagType,
    count: u64,
    bytes: &[u8],
    byte_order: ByteOrder,
) -> Result<TagValue> {
    let mut cursor = Cursor::new(bytes, byte_order);
    let n = count as usize;

    Ok(match tag_type {
        TagType::Byte | TagType::Unknown(_) => {
            TagValue::Byte(cursor.read_bytes(n)?.to_vec())
        }
        TagType::Ascii => {
            let raw = cursor.read_bytes(n)?;
            let s = String::from_utf8_lossy(raw)
                .trim_end_matches('\0')
                .to_string();
            TagValue::Ascii(s)
        }
        TagType::Short => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_u16()?);
            }
            TagValue::Short(v)
        }
        TagType::Long => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_u32()?);
            }
            TagValue::Long(v)
        }
        TagType::Rational => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let num = cursor.read_u32()?;
                let den = cursor.read_u32()?;
                v.push([num, den]);
            }
            TagValue::Rational(v)
        }
        TagType::SByte => {
            let raw = cursor.read_bytes(n)?;
            TagValue::SByte(raw.iter().map(|&b| b as i8).collect())
        }
        TagType::Undefined => {
            TagValue::Undefined(cursor.read_bytes(n)?.to_vec())
        }
        TagType::SShort => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_u16()? as i16);
            }
            TagValue::SShort(v)
        }
        TagType::SLong => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_u32()? as i32);
            }
            TagValue::SLong(v)
        }
        TagType::SRational => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let num = cursor.read_u32()? as i32;
                let den = cursor.read_u32()? as i32;
                v.push([num, den]);
            }
            TagValue::SRational(v)
        }
        TagType::Float => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let bits = cursor.read_u32()?;
                v.push(f32::from_bits(bits));
            }
            TagValue::Float(v)
        }
        TagType::Double => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_f64()?);
            }
            TagValue::Double(v)
        }
        TagType::Long8 | TagType::Ifd8 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_u64()?);
            }
            TagValue::Long8(v)
        }
        TagType::SLong8 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(cursor.read_u64()? as i64);
            }
            TagValue::SLong8(v)
        }
    })
}
