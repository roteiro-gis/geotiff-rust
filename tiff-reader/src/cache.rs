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
    enabled: bool,
}

struct BlockCacheState {
    cache: LruCache<BlockKey, Arc<Vec<u8>>>,
    current_bytes: usize,
}

impl BlockCache {
    /// Create a new cache with byte and slot limits.
    pub fn new(max_bytes: usize, max_slots: usize) -> Self {
        let slots = NonZeroUsize::new(max_slots.max(1)).unwrap();
        Self {
            inner: Mutex::new(BlockCacheState {
                cache: LruCache::new(slots),
                current_bytes: 0,
            }),
            max_bytes,
            enabled: max_bytes > 0 && max_slots > 0,
        }
    }

    /// Return a cached block and promote it in LRU order.
    pub fn get(&self, key: &BlockKey) -> Option<Arc<Vec<u8>>> {
        if !self.enabled {
            return None;
        }
        let mut state = self.inner.lock();
        state.cache.get(key).cloned()
    }

    /// Insert a decoded block into the cache.
    pub fn insert(&self, key: BlockKey, data: Vec<u8>) -> Arc<Vec<u8>> {
        let data_len = data.len();
        let value = Arc::new(data);

        let mut state = self.inner.lock();
        if let Some(previous) = state.cache.pop(&key) {
            state.current_bytes = state.current_bytes.saturating_sub(previous.len());
        }

        if !self.enabled || data_len > self.max_bytes {
            return value;
        }

        while state.current_bytes > self.max_bytes - data_len && !state.cache.is_empty() {
            if let Some((_, evicted)) = state.cache.pop_lru() {
                state.current_bytes = state.current_bytes.saturating_sub(evicted.len());
            }
        }

        state.current_bytes += data_len;
        if let Some((_, evicted)) = state.cache.push(key, value.clone()) {
            state.current_bytes = state.current_bytes.saturating_sub(evicted.len());
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

    #[test]
    fn zero_slots_disable_cache_storage() {
        let cache = BlockCache::new(1024, 0);
        let key = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Tile,
            block_index: 0,
        };
        cache.insert(key, vec![1, 2, 3]);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.inner.lock().current_bytes, 0);
    }

    #[test]
    fn slot_eviction_updates_byte_accounting() {
        let cache = BlockCache::new(100, 2);
        for block_index in 0..3 {
            cache.insert(
                BlockKey {
                    ifd_index: 0,
                    kind: BlockKind::Strip,
                    block_index,
                },
                vec![0; 4],
            );
        }

        assert_eq!(cache.inner.lock().current_bytes, 8);
    }

    #[test]
    fn replacing_mru_entry_preserves_other_cached_blocks() {
        let cache = BlockCache::new(10, 8);
        let a = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Tile,
            block_index: 0,
        };
        let b = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Tile,
            block_index: 1,
        };

        cache.insert(a, vec![0; 8]);
        cache.insert(b, vec![0; 2]);
        assert!(cache.get(&a).is_some());

        cache.insert(a, vec![0; 7]);

        assert!(cache.get(&a).is_some());
        assert!(cache.get(&b).is_some());
        assert_eq!(cache.inner.lock().current_bytes, 9);
    }

    #[test]
    fn oversized_replacement_removes_stale_entry() {
        let cache = BlockCache::new(8, 8);
        let key = BlockKey {
            ifd_index: 0,
            kind: BlockKind::Tile,
            block_index: 0,
        };

        cache.insert(key, vec![0; 4]);
        cache.insert(key, vec![0; 9]);

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.inner.lock().current_bytes, 0);
    }
}
