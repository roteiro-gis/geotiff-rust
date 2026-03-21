use crate::error::{Error, Result};
use crate::header::ByteOrder;
use crate::io::Cursor;
use crate::source::TiffSource;

pub use tiff_core::{Tag, TagType, TagValue};

/// Parse a classic TIFF tag entry (12-byte IFD entry).
pub fn parse_tag_classic(
    code: u16,
    type_code: u16,
    count: u64,
    value_offset_bytes: &[u8],
    source: &dyn TiffSource,
    byte_order: ByteOrder,
) -> Result<Tag> {
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
    Ok(Tag {
        code,
        tag_type,
        count,
        value,
    })
}

/// Parse a BigTIFF tag entry (20-byte IFD entry).
pub fn parse_tag_bigtiff(
    code: u16,
    type_code: u16,
    count: u64,
    value_offset_bytes: &[u8],
    source: &dyn TiffSource,
    byte_order: ByteOrder,
) -> Result<Tag> {
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
    Ok(Tag {
        code,
        tag_type,
        count,
        value,
    })
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
    count
        .checked_mul(element_size)
        .ok_or_else(|| Error::InvalidTagValue {
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
        TagType::Byte | TagType::Unknown(_) => TagValue::Byte(cursor.read_bytes(n)?.to_vec()),
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
        TagType::Undefined => TagValue::Undefined(cursor.read_bytes(n)?.to_vec()),
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
