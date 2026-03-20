use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{Error, Result};

/// Random-access byte source for TIFF decoding.
pub trait TiffSource: Send + Sync {
    /// Total object length in bytes.
    fn len(&self) -> u64;

    /// Returns `true` when the source has no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `len` bytes starting at `offset`.
    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>>;

    /// Expose a whole-object slice when the source is fully resident in memory.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }
}

/// Shared source handle used by `TiffFile`.
pub type SharedSource = Arc<dyn TiffSource>;

/// Memory-mapped file source.
pub struct MmapSource {
    mmap: Mmap,
}

impl MmapSource {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io(e, path.display().to_string()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::Io(e, path.display().to_string()))?;
        Ok(Self { mmap })
    }
}

impl TiffSource for MmapSource {
    fn len(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds {
            offset,
            length: len as u64,
            data_len: self.len(),
        })?;
        let end = start.checked_add(len).ok_or(Error::OffsetOutOfBounds {
            offset,
            length: len as u64,
            data_len: self.len(),
        })?;
        if end > self.mmap.len() {
            return Err(Error::OffsetOutOfBounds {
                offset,
                length: len as u64,
                data_len: self.len(),
            });
        }
        Ok(self.mmap[start..end].to_vec())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(&self.mmap)
    }
}

/// In-memory byte-vector source.
pub struct BytesSource {
    bytes: Vec<u8>,
}

impl BytesSource {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

impl TiffSource for BytesSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds {
            offset,
            length: len as u64,
            data_len: self.len(),
        })?;
        let end = start.checked_add(len).ok_or(Error::OffsetOutOfBounds {
            offset,
            length: len as u64,
            data_len: self.len(),
        })?;
        if end > self.bytes.len() {
            return Err(Error::OffsetOutOfBounds {
                offset,
                length: len as u64,
                data_len: self.len(),
            });
        }
        Ok(self.bytes[start..end].to_vec())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(&self.bytes)
    }
}
