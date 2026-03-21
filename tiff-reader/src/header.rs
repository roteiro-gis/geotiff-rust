use crate::error::{Error, Result};

pub use tiff_core::ByteOrder;

/// Parsed TIFF/BigTIFF file header.
#[derive(Debug, Clone)]
pub struct TiffHeader {
    pub byte_order: ByteOrder,
    /// 42 for classic TIFF, 43 for BigTIFF.
    pub version: u16,
    /// Offset to the first IFD.
    pub first_ifd_offset: u64,
}

impl TiffHeader {
    /// Returns `true` if this is a BigTIFF file (version 43).
    pub fn is_bigtiff(&self) -> bool {
        self.version == 43
    }

    /// Parse the TIFF header from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::InvalidMagic);
        }

        let byte_order = match &data[0..2] {
            b"II" => ByteOrder::LittleEndian,
            b"MM" => ByteOrder::BigEndian,
            _ => return Err(Error::InvalidMagic),
        };

        let read_u16 = |offset: usize| -> u16 {
            let bytes = [data[offset], data[offset + 1]];
            match byte_order {
                ByteOrder::LittleEndian => u16::from_le_bytes(bytes),
                ByteOrder::BigEndian => u16::from_be_bytes(bytes),
            }
        };

        let version = read_u16(2);

        match version {
            42 => {
                // Classic TIFF: 4-byte IFD offset at position 4
                let read_u32 = |offset: usize| -> u32 {
                    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
                    match byte_order {
                        ByteOrder::LittleEndian => u32::from_le_bytes(bytes),
                        ByteOrder::BigEndian => u32::from_be_bytes(bytes),
                    }
                };
                let first_ifd_offset = read_u32(4) as u64;
                Ok(Self {
                    byte_order,
                    version,
                    first_ifd_offset,
                })
            }
            43 => {
                // BigTIFF: 2-byte offset size (must be 8), 2-byte reserved, 8-byte IFD offset
                if data.len() < 16 {
                    return Err(Error::InvalidMagic);
                }
                let offset_size = read_u16(4);
                if offset_size != 8 {
                    return Err(Error::UnsupportedVersion(version));
                }
                // bytes 6-7: reserved (must be 0)
                let read_u64 = |offset: usize| -> u64 {
                    let bytes: [u8; 8] = data[offset..offset + 8].try_into().unwrap();
                    match byte_order {
                        ByteOrder::LittleEndian => u64::from_le_bytes(bytes),
                        ByteOrder::BigEndian => u64::from_be_bytes(bytes),
                    }
                };
                let first_ifd_offset = read_u64(8);
                Ok(Self {
                    byte_order,
                    version,
                    first_ifd_offset,
                })
            }
            _ => Err(Error::UnsupportedVersion(version)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_little_endian_classic() {
        // II, version 42, IFD offset at 8
        let data = b"II\x2a\x00\x08\x00\x00\x00";
        let header = TiffHeader::parse(data).unwrap();
        assert_eq!(header.byte_order, ByteOrder::LittleEndian);
        assert_eq!(header.version, 42);
        assert_eq!(header.first_ifd_offset, 8);
        assert!(!header.is_bigtiff());
    }

    #[test]
    fn parse_big_endian_classic() {
        // MM, version 42, IFD offset at 8
        let data = b"MM\x00\x2a\x00\x00\x00\x08";
        let header = TiffHeader::parse(data).unwrap();
        assert_eq!(header.byte_order, ByteOrder::BigEndian);
        assert_eq!(header.version, 42);
        assert_eq!(header.first_ifd_offset, 8);
    }

    #[test]
    fn parse_bigtiff() {
        // II, version 43, offset size 8, reserved 0, IFD offset at 16
        let mut data = Vec::new();
        data.extend_from_slice(b"II"); // byte order
        data.extend_from_slice(&43u16.to_le_bytes()); // version
        data.extend_from_slice(&8u16.to_le_bytes()); // offset size
        data.extend_from_slice(&0u16.to_le_bytes()); // reserved
        data.extend_from_slice(&16u64.to_le_bytes()); // first IFD offset
        let header = TiffHeader::parse(&data).unwrap();
        assert!(header.is_bigtiff());
        assert_eq!(header.first_ifd_offset, 16);
    }

    #[test]
    fn reject_invalid_magic() {
        let data = b"XX\x2a\x00\x08\x00\x00\x00";
        assert!(TiffHeader::parse(data).is_err());
    }
}
