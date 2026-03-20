//! LRU cache for decompressed strips and tiles.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

/// Cache key for a decoded strip or tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub ifd_index: usize,
    pub kind: BlockKind,
    pub block_index: usize,
}

/// Whether the cached block came from a strip- or tile-backed image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
    Strip,
    Tile,
}

/// Thread-safe LRU cache for decoded block payloads.
pub struct BlockCache {
    inner: Mutex<BlockCacheState>,
    max_bytes: usize,
}

struct BlockCacheState {
    cache: LruCache<BlockKey, Arc<Vec<u8>>>,
    current_bytes: usize,
}

impl BlockCache {
    /// Create a new cache with byte and slot limits.
    pub fn new(max_bytes: usize, max_slots: usize) -> Self {
        let slots = NonZeroUsize::new(max_slots).unwrap_or_else(|| NonZeroUsize::new(257).unwrap());
        Self {
            inner: Mutex::new(BlockCacheState {
                cache: LruCache::new(slots),
                current_bytes: 0,
            }),
            max_bytes,
        }
    }

    /// Return a cached block and promote it in LRU order.
    pub fn get(&self, key: &BlockKey) -> Option<Arc<Vec<u8>>> {
        let mut state = self.inner.lock();
        state.cache.get(key).cloned()
    }

    /// Insert a decoded block into the cache.
    pub fn insert(&self, key: BlockKey, data: Vec<u8>) -> Arc<Vec<u8>> {
        let data_len = data.len();
        let value = Arc::new(data);

        if self.max_bytes == 0 || data_len > self.max_bytes {
            return value;
        }

        let mut state = self.inner.lock();
        while state.current_bytes + data_len > self.max_bytes && !state.cache.is_empty() {
            if let Some((_, evicted)) = state.cache.pop_lru() {
                state.current_bytes = state.current_bytes.saturating_sub(evicted.len());
            }
        }

        state.current_bytes += data_len;
        if let Some(previous) = state.cache.put(key, value.clone()) {
            state.current_bytes = state.current_bytes.saturating_sub(previous.len());
        }

        value
    }
}

impl Default for BlockCache {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024, 257)
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockCache, BlockKey, BlockKind};

    #[test]
    fn caches_and_promotes_entries() {
        let cache = BlockCache::new(12, 8);
        let a = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Strip,
            block_index: 0,
        };
        let b = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Strip,
            block_index: 1,
        };
        let c = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Strip,
            block_index: 2,
        };

        cache.insert(a, vec![0; 4]);
        cache.insert(b, vec![0; 4]);
        cache.insert(c, vec![0; 4]);

        let promoted = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Strip,
            block_index: 0,
        };
        assert!(cache.get(&promoted).is_some());

        let d = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Strip,
            block_index: 3,
        };
        cache.insert(d, vec![0; 4]);

        let evicted = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Strip,
            block_index: 1,
        };
        assert!(cache.get(&promoted).is_some());
        assert!(cache.get(&evicted).is_none());
    }

    #[test]
    fn disabled_cache_bypasses_storage() {
        let cache = BlockCache::new(0, 4);
        let key = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Tile,
            block_index: 0,
        };
        cache.insert(key, vec![1, 2, 3]);
        assert!(cache.get(&key).is_none());
    }
}
