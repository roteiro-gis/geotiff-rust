/// Byte order indicator from the TIFF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

impl ByteOrder {
    /// Returns the magic bytes written at offset 0 of the TIFF file.
    pub fn magic(self) -> [u8; 2] {
        match self {
            Self::LittleEndian => *b"II",
            Self::BigEndian => *b"MM",
        }
    }

    /// Read a u16 from a 2-byte array in this byte order.
    pub fn read_u16(self, bytes: [u8; 2]) -> u16 {
        match self {
            Self::LittleEndian => u16::from_le_bytes(bytes),
            Self::BigEndian => u16::from_be_bytes(bytes),
        }
    }

    /// Read a u32 from a 4-byte array in this byte order.
    pub fn read_u32(self, bytes: [u8; 4]) -> u32 {
        match self {
            Self::LittleEndian => u32::from_le_bytes(bytes),
            Self::BigEndian => u32::from_be_bytes(bytes),
        }
    }

    /// Read a u64 from an 8-byte array in this byte order.
    pub fn read_u64(self, bytes: [u8; 8]) -> u64 {
        match self {
            Self::LittleEndian => u64::from_le_bytes(bytes),
            Self::BigEndian => u64::from_be_bytes(bytes),
        }
    }

    /// Read an f64 from an 8-byte array in this byte order.
    pub fn read_f64(self, bytes: [u8; 8]) -> f64 {
        f64::from_bits(self.read_u64(bytes))
    }

    /// Write a u16 in this byte order.
    pub fn write_u16(self, value: u16) -> [u8; 2] {
        match self {
            Self::LittleEndian => value.to_le_bytes(),
            Self::BigEndian => value.to_be_bytes(),
        }
    }

    /// Write a u32 in this byte order.
    pub fn write_u32(self, value: u32) -> [u8; 4] {
        match self {
            Self::LittleEndian => value.to_le_bytes(),
            Self::BigEndian => value.to_be_bytes(),
        }
    }

    /// Write a u64 in this byte order.
    pub fn write_u64(self, value: u64) -> [u8; 8] {
        match self {
            Self::LittleEndian => value.to_le_bytes(),
            Self::BigEndian => value.to_be_bytes(),
        }
    }

    /// Write an f64 in this byte order.
    pub fn write_f64(self, value: f64) -> [u8; 8] {
        self.write_u64(value.to_bits())
    }

    /// Write an f32 in this byte order.
    pub fn write_f32(self, value: f32) -> [u8; 4] {
        self.write_u32(value.to_bits())
    }

    /// Write an i16 in this byte order.
    pub fn write_i16(self, value: i16) -> [u8; 2] {
        self.write_u16(value as u16)
    }

    /// Write an i32 in this byte order.
    pub fn write_i32(self, value: i32) -> [u8; 4] {
        self.write_u32(value as u32)
    }

    /// Write an i64 in this byte order.
    pub fn write_i64(self, value: i64) -> [u8; 8] {
        self.write_u64(value as u64)
    }
}
