use crate::error::{Error, Result};
use crate::header::ByteOrder;

/// A cursor over a byte slice with byte-order-aware reads.
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    byte_order: ByteOrder,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8], byte_order: ByteOrder) -> Self {
        Self {
            data,
            pos: 0,
            byte_order,
        }
    }

    pub fn with_offset(data: &'a [u8], offset: usize, byte_order: ByteOrder) -> Result<Self> {
        if offset > data.len() {
            return Err(Error::Truncated {
                offset: offset as u64,
                needed: 0,
                available: data.len() as u64,
            });
        }
        Ok(Self {
            data,
            pos: offset,
            byte_order,
        })
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let bytes = [self.data[self.pos], self.data[self.pos + 1]];
        self.pos += 2;
        Ok(match self.byte_order {
            ByteOrder::LittleEndian => u16::from_le_bytes(bytes),
            ByteOrder::BigEndian => u16::from_be_bytes(bytes),
        })
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let bytes: [u8; 4] = self.data[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        Ok(match self.byte_order {
            ByteOrder::LittleEndian => u32::from_le_bytes(bytes),
            ByteOrder::BigEndian => u32::from_be_bytes(bytes),
        })
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        Ok(match self.byte_order {
            ByteOrder::LittleEndian => u64::from_le_bytes(bytes),
            ByteOrder::BigEndian => u64::from_be_bytes(bytes),
        })
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        self.ensure(8)?;
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        Ok(match self.byte_order {
            ByteOrder::LittleEndian => f64::from_le_bytes(bytes),
            ByteOrder::BigEndian => f64::from_be_bytes(bytes),
        })
    }

    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.ensure(n)?;
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.ensure(n)?;
        self.pos += n;
        Ok(())
    }

    fn ensure(&self, n: usize) -> Result<()> {
        if !matches!(self.pos.checked_add(n), Some(end) if end <= self.data.len()) {
            return Err(Error::Truncated {
                offset: self.pos as u64,
                needed: n as u64,
                available: self.data.len().saturating_sub(self.pos) as u64,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Cursor;
    use crate::header::ByteOrder;

    #[test]
    fn oversized_skip_returns_truncated_error_without_overflowing() {
        let mut cursor = Cursor::with_offset(&[0], 1, ByteOrder::LittleEndian).unwrap();

        assert!(cursor.skip(usize::MAX).is_err());
    }

    #[test]
    fn oversized_read_returns_truncated_error_without_overflowing() {
        let mut cursor = Cursor::with_offset(&[0], 1, ByteOrder::LittleEndian).unwrap();

        assert!(cursor.read_bytes(usize::MAX).is_err());
    }
}
