//! HTTP range-backed remote GeoTIFF/COG access.
//!
//! This module opens remote objects through the same TIFF decoder core used for
//! local files by providing a random-access byte source backed by cached range
//! requests.

use std::collections::HashMap;
use std::io::Read;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lru::LruCache;
use parking_lot::{Condvar, Mutex};
use reqwest::blocking::{Client, RequestBuilder};
use reqwest::header::{HeaderMap, CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use reqwest::StatusCode;
use tiff_reader::source::{SharedSource, TiffSource};
use tiff_reader::{OpenOptions as TiffOpenOptions, TiffFile};

use crate::http_range::{probe_total_from_content_range, validate_content_range_header};
use crate::{Error, GeoTiffFile, Result};

/// Options for HTTP range-backed GeoTIFF access.
#[derive(Debug, Clone)]
pub struct HttpOpenOptions {
    /// Fixed byte-range chunk size.
    pub chunk_size: usize,
    /// Maximum bytes retained in the range cache.
    pub cache_bytes: usize,
    /// Maximum cached chunks.
    pub cache_slots: usize,
    /// Maximum number of adjacent missing chunks merged into one range request.
    ///
    /// A read spanning several uncached chunks is served by one coalesced GET
    /// per contiguous run rather than one request per chunk, so this bounds
    /// how much a single request may fetch. Set to 1 to disable coalescing.
    pub max_coalesced_chunks: usize,
    /// TCP connect timeout for clients built from these options.
    ///
    /// Ignored when `client` is provided; configure custom clients directly.
    pub connect_timeout: Option<Duration>,
    /// Timeout for each range response body read.
    pub read_timeout: Option<Duration>,
    /// Overall timeout applied to each HEAD or GET request.
    pub request_timeout: Option<Duration>,
    /// Headers sent on every HEAD and byte-range GET request.
    pub headers: HeaderMap,
    /// Optional preconfigured blocking client for custom TLS, proxy, redirect, or auth behavior.
    pub client: Option<Client>,
    /// TIFF decoder options applied after range reads are assembled.
    pub tiff_options: TiffOpenOptions,
}

impl Default for HttpOpenOptions {
    fn default() -> Self {
        Self {
            chunk_size: 256 * 1024,
            cache_bytes: 64 * 1024 * 1024,
            cache_slots: 257,
            max_coalesced_chunks: 16,
            connect_timeout: Some(Duration::from_secs(10)),
            read_timeout: Some(Duration::from_secs(30)),
            request_timeout: Some(Duration::from_secs(120)),
            headers: HeaderMap::new(),
            client: None,
            tiff_options: TiffOpenOptions::default(),
        }
    }
}

/// Remote GeoTIFF/COG handle backed by HTTP range requests.
pub struct HttpGeoTiffFile {
    url: String,
    inner: GeoTiffFile,
}

impl HttpGeoTiffFile {
    /// Open a remote GeoTIFF/COG using HTTP range requests.
    pub fn open(url: impl Into<String>) -> Result<Self> {
        Self::open_with_options(url, HttpOpenOptions::default())
    }

    /// Open a remote GeoTIFF/COG using explicit range-cache options.
    pub fn open_with_options(url: impl Into<String>, options: HttpOpenOptions) -> Result<Self> {
        let url = url.into();
        let tiff_options = options.tiff_options;
        let source: SharedSource = Arc::new(HttpRangeSource::open(url.clone(), options)?);
        let tiff = TiffFile::from_source_with_options(source, tiff_options)?;
        let inner = GeoTiffFile::from_tiff(tiff)?;
        Ok(Self { url, inner })
    }

    /// The source URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Access the decoded GeoTIFF.
    pub fn inner(&self) -> &GeoTiffFile {
        &self.inner
    }
}

struct HttpRangeSource {
    client: Client,
    url: String,
    len: u64,
    chunk_size: usize,
    max_coalesced_chunks: usize,
    headers: HeaderMap,
    read_timeout: Option<Duration>,
    request_timeout: Option<Duration>,
    cache: Mutex<RangeCacheState>,
    in_flight: Mutex<HashMap<u64, Arc<InFlightChunk>>>,
    max_bytes: usize,
    cache_enabled: bool,
}

struct RangeCacheState {
    cache: LruCache<u64, Arc<Vec<u8>>>,
    current_bytes: usize,
}

/// Rendezvous point for threads that all want the same uncached chunk.
///
/// Exactly one caller fetches; the rest block here until the result is
/// published. Errors are carried as text because the transport error types
/// are not `Clone`.
struct InFlightChunk {
    state: Mutex<Option<std::result::Result<Arc<Vec<u8>>, String>>>,
    ready: Condvar,
}

impl InFlightChunk {
    fn new() -> Self {
        Self {
            state: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn publish(&self, value: std::result::Result<Arc<Vec<u8>>, String>) {
        let mut state = self.state.lock();
        if state.is_none() {
            *state = Some(value);
        }
        self.ready.notify_all();
    }

    fn wait(&self) -> std::result::Result<Arc<Vec<u8>>, String> {
        let mut state = self.state.lock();
        loop {
            if let Some(value) = state.as_ref() {
                return value.clone();
            }
            self.ready.wait(&mut state);
        }
    }
}

/// Whether this caller owns fetching a chunk or is waiting on another caller.
enum ChunkClaim {
    Cached(Arc<Vec<u8>>),
    Lead(Arc<InFlightChunk>),
    Follow(Arc<InFlightChunk>),
}

/// Releases every led chunk on unwind so waiters can never be stranded.
struct LeadGuard<'a> {
    source: &'a HttpRangeSource,
    pending: Vec<(u64, Arc<InFlightChunk>)>,
}

impl LeadGuard<'_> {
    /// Hand a finished chunk to any waiters and stop guarding it.
    fn settle(&mut self, index: u64, value: std::result::Result<Arc<Vec<u8>>, String>) {
        if let Some(position) = self.pending.iter().position(|(key, _)| *key == index) {
            let (_, entry) = self.pending.remove(position);
            self.source.in_flight.lock().remove(&index);
            entry.publish(value);
        }
    }
}

impl Drop for LeadGuard<'_> {
    fn drop(&mut self) {
        for (index, entry) in self.pending.drain(..) {
            self.source.in_flight.lock().remove(&index);
            entry.publish(Err("chunk fetch abandoned before completion".to_string()));
        }
    }
}

impl HttpRangeSource {
    fn open(url: String, options: HttpOpenOptions) -> Result<Self> {
        let client = build_client(&options)?;
        let len = probe_content_length(&client, &url, &options.headers, options.request_timeout)?;
        let slots = NonZeroUsize::new(options.cache_slots.max(1)).unwrap();
        Ok(Self {
            client,
            url,
            len,
            chunk_size: options.chunk_size.max(1),
            max_coalesced_chunks: options.max_coalesced_chunks.max(1),
            headers: options.headers,
            read_timeout: options.read_timeout,
            request_timeout: options.request_timeout,
            cache: Mutex::new(RangeCacheState {
                cache: LruCache::new(slots),
                current_bytes: 0,
            }),
            in_flight: Mutex::new(HashMap::new()),
            max_bytes: options.cache_bytes,
            cache_enabled: options.cache_bytes > 0 && options.cache_slots > 0,
        })
    }

    fn cached(&self, index: u64) -> Option<Arc<Vec<u8>>> {
        if !self.cache_enabled {
            return None;
        }
        self.cache.lock().cache.get(&index).cloned()
    }

    /// Claim a chunk, re-checking the cache under the in-flight lock so a
    /// fetch that finished since the caller last looked is not repeated.
    ///
    /// Lock order is `in_flight` then `cache`; publishers take them in the
    /// opposite order but never hold both, so the two cannot deadlock.
    fn claim(&self, index: u64) -> ChunkClaim {
        let mut in_flight = self.in_flight.lock();
        if let Some(chunk) = self.cached(index) {
            return ChunkClaim::Cached(chunk);
        }
        if let Some(entry) = in_flight.get(&index) {
            return ChunkClaim::Follow(Arc::clone(entry));
        }
        let entry = Arc::new(InFlightChunk::new());
        in_flight.insert(index, Arc::clone(&entry));
        ChunkClaim::Lead(entry)
    }

    fn store(&self, index: u64, body: Vec<u8>) -> Arc<Vec<u8>> {
        let body_len = body.len();
        let value = Arc::new(body);

        let mut state = self.cache.lock();
        if let Some(previous) = state.cache.pop(&index) {
            state.current_bytes = state.current_bytes.saturating_sub(previous.len());
        }

        if !self.cache_enabled || body_len > self.max_bytes {
            return value;
        }

        while state.current_bytes > self.max_bytes - body_len && !state.cache.is_empty() {
            if let Some((_, evicted)) = state.cache.pop_lru() {
                state.current_bytes = state.current_bytes.saturating_sub(evicted.len());
            }
        }
        state.current_bytes += body_len;
        if let Some((_, evicted)) = state.cache.push(index, value.clone()) {
            state.current_bytes = state.current_bytes.saturating_sub(evicted.len());
        }
        value
    }

    fn chunk_bounds(&self, index: u64) -> Result<(u64, u64)> {
        let chunk_size = self.chunk_size as u64;
        let start = index
            .checked_mul(chunk_size)
            .ok_or_else(|| Error::Other("range chunk offset overflowed u64".into()))?;
        if start >= self.len {
            return Err(Error::Other(format!(
                "range chunk {index} starts beyond end of object"
            )));
        }
        let end = start.saturating_add(chunk_size).min(self.len) - 1;
        Ok((start, end))
    }

    /// Resolve a single chunk. Reads go through `chunk_span`; this is the
    /// one-chunk shorthand used by the cache-accounting tests.
    #[cfg(test)]
    fn chunk(&self, index: u64) -> Result<Arc<Vec<u8>>> {
        let mut chunks = self.chunk_span(index, index)?;
        Ok(chunks.remove(0))
    }

    /// Resolve every chunk in `first..=last`, fetching contiguous runs of
    /// missing chunks in one request each and sharing in-flight fetches with
    /// other threads.
    fn chunk_span(&self, first: u64, last: u64) -> Result<Vec<Arc<Vec<u8>>>> {
        let count = usize::try_from(last - first + 1)
            .map_err(|_| Error::Other("range chunk span overflowed usize".into()))?;
        let mut resolved: Vec<Option<Arc<Vec<u8>>>> = vec![None; count];
        let mut followers: Vec<(usize, Arc<InFlightChunk>)> = Vec::new();
        let mut guard = LeadGuard {
            source: self,
            pending: Vec::new(),
        };

        for (slot, resolved_slot) in resolved.iter_mut().enumerate() {
            let index = first + slot as u64;
            match self.claim(index) {
                ChunkClaim::Cached(chunk) => *resolved_slot = Some(chunk),
                ChunkClaim::Follow(entry) => followers.push((slot, entry)),
                ChunkClaim::Lead(entry) => guard.pending.push((index, entry)),
            }
        }

        // Fetch the chunks this caller leads as contiguous runs, capped so one
        // request cannot balloon past the configured limit.
        let led: Vec<u64> = guard.pending.iter().map(|(index, _)| *index).collect();
        let mut position = 0;
        while position < led.len() {
            let mut run_end = position;
            loop {
                let next = run_end + 1;
                let contiguous = next < led.len() && led[next] == led[run_end] + 1;
                if !contiguous || next - position + 1 > self.max_coalesced_chunks {
                    break;
                }
                run_end = next;
            }
            let run = &led[position..=run_end];
            match self.fetch_run(run[0], run.len()) {
                Ok(bodies) => {
                    for (offset, body) in bodies.into_iter().enumerate() {
                        let index = run[0] + offset as u64;
                        let stored = self.store(index, body);
                        resolved[(index - first) as usize] = Some(Arc::clone(&stored));
                        guard.settle(index, Ok(stored));
                    }
                }
                Err(error) => {
                    let message = error.to_string();
                    for index in run {
                        guard.settle(*index, Err(message.clone()));
                    }
                    return Err(error);
                }
            }
            position = run_end + 1;
        }

        for (slot, entry) in followers {
            let chunk = entry.wait().map_err(Error::Other)?;
            resolved[slot] = Some(chunk);
        }

        resolved
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::Other("range chunk span left an unresolved chunk".into()))
    }

    /// Issue one range request covering `count` consecutive chunks and split
    /// the response back into per-chunk bodies.
    fn fetch_run(&self, first: u64, count: usize) -> Result<Vec<Vec<u8>>> {
        let (start, _) = self.chunk_bounds(first)?;
        let (_, end) = self.chunk_bounds(first + count as u64 - 1)?;

        let response = request_with_options(
            self.client.get(&self.url),
            &self.headers,
            shorter_timeout(self.read_timeout, self.request_timeout),
        )
        .header(RANGE, format!("bytes={start}-{end}"))
        .send()?
        .error_for_status()?;
        if response.status() != StatusCode::PARTIAL_CONTENT {
            return Err(Error::Other(format!(
                "server did not honor byte-range request for {}: expected 206, got {}",
                self.url,
                response.status()
            )));
        }
        validate_content_range_header(
            response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|value| value.to_str().ok()),
            &self.url,
            start,
            end,
            Some(self.len),
        )?;
        let expected_len = usize::try_from(end - start + 1).unwrap_or(usize::MAX);
        let body = read_response_body(
            response,
            expected_len,
            self.request_timeout,
            &format!("{} bytes={start}-{end}", self.url),
        )?;
        if body.len() != expected_len {
            return Err(Error::Other(format!(
                "range response length mismatch for {}: expected {expected_len} bytes, got {}",
                self.url,
                body.len()
            )));
        }

        let mut bodies = Vec::with_capacity(count);
        let mut rest = body.as_slice();
        for offset in 0..count {
            let (chunk_start, chunk_end) = self.chunk_bounds(first + offset as u64)?;
            let take = usize::try_from(chunk_end - chunk_start + 1)
                .map_err(|_| Error::Other("range chunk length overflowed usize".into()))?;
            let (head, tail) = rest.split_at(take.min(rest.len()));
            bodies.push(head.to_vec());
            rest = tail;
        }
        Ok(bodies)
    }
}

fn build_client(options: &HttpOpenOptions) -> Result<Client> {
    if let Some(client) = &options.client {
        return Ok(client.clone());
    }

    let mut builder = Client::builder();
    if let Some(timeout) = options.connect_timeout {
        builder = builder.connect_timeout(timeout);
    }
    if let Some(timeout) = options.request_timeout {
        builder = builder.timeout(timeout);
    }
    Ok(builder.build()?)
}

fn request_with_options(
    request: RequestBuilder,
    headers: &HeaderMap,
    request_timeout: Option<Duration>,
) -> RequestBuilder {
    let mut request = if headers.is_empty() {
        request
    } else {
        request.headers(headers.clone())
    };
    if let Some(timeout) = request_timeout {
        request = request.timeout(timeout);
    }
    request
}

fn shorter_timeout(lhs: Option<Duration>, rhs: Option<Duration>) -> Option<Duration> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
        (Some(timeout), None) | (None, Some(timeout)) => Some(timeout),
        (None, None) => None,
    }
}

fn read_response_body(
    mut response: reqwest::blocking::Response,
    expected_len: usize,
    request_timeout: Option<Duration>,
    context: &str,
) -> Result<Vec<u8>> {
    let started = Instant::now();
    let mut body = Vec::with_capacity(expected_len);
    let mut buffer = [0u8; 8192];

    while body.len() < expected_len {
        if let Some(timeout) = request_timeout {
            if started.elapsed() >= timeout {
                return Err(Error::Other(format!(
                    "HTTP response body read exceeded overall timeout for {context}"
                )));
            }
        }

        let remaining = expected_len - body.len();
        let read_len = buffer.len().min(remaining);
        let read = response
            .read(&mut buffer[..read_len])
            .map_err(|err| Error::Io(err, context.to_string()))?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&buffer[..read]);
    }

    Ok(body)
}

impl TiffSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> tiff_reader::error::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        let end = offset.checked_add(len as u64).ok_or({
            tiff_reader::TiffError::OffsetOutOfBounds {
                offset,
                length: len as u64,
                data_len: self.len,
            }
        })?;
        if end > self.len {
            return Err(tiff_reader::TiffError::OffsetOutOfBounds {
                offset,
                length: len as u64,
                data_len: self.len,
            });
        }

        let first_chunk = offset / self.chunk_size as u64;
        let last_chunk = (end.saturating_sub(1)) / self.chunk_size as u64;
        let mut out = Vec::with_capacity(len);

        let chunks = self
            .chunk_span(first_chunk, last_chunk)
            .map_err(|e| tiff_reader::TiffError::Other(format!("HTTP range read failed: {e}")))?;

        for (position, chunk) in chunks.into_iter().enumerate() {
            let chunk_index = first_chunk + position as u64;
            let chunk_start = chunk_index * self.chunk_size as u64;
            let start_in_chunk = if chunk_index == first_chunk {
                usize::try_from(offset - chunk_start).unwrap_or(0)
            } else {
                0
            };
            let end_in_chunk = if chunk_index == last_chunk {
                usize::try_from(end - chunk_start).unwrap_or(chunk.len())
            } else {
                chunk.len()
            };
            out.extend_from_slice(&chunk[start_in_chunk..end_in_chunk]);
        }

        Ok(out)
    }
}

fn probe_content_length(
    client: &Client,
    url: &str,
    headers: &HeaderMap,
    request_timeout: Option<Duration>,
) -> Result<u64> {
    let head = request_with_options(client.head(url), headers, request_timeout).send()?;
    if head.status().is_success() {
        if let Some(value) = head.headers().get(CONTENT_LENGTH) {
            if let Ok(text) = value.to_str() {
                if let Ok(len) = text.parse::<u64>() {
                    return Ok(len);
                }
            }
        }
    }

    let response = request_with_options(client.get(url), headers, request_timeout)
        .header(RANGE, "bytes=0-0")
        .send()?
        .error_for_status()?;
    if response.status() != StatusCode::PARTIAL_CONTENT {
        return Err(Error::Other(format!(
            "server does not support HTTP range requests for {url}"
        )));
    }
    let content_range = response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| Error::Other(format!("missing Content-Range header for {url}")))?;
    probe_total_from_content_range(content_range, url)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use reqwest::blocking::Client;
    use reqwest::header::{HeaderMap, HeaderValue};
    use tiff_reader::source::TiffSource;

    use super::{HttpGeoTiffFile, HttpOpenOptions, HttpRangeSource};
    use crate::http_test_support::{build_simple_geotiff, TestServer};

    #[test]
    fn default_http_options_set_request_timeouts() {
        let options = HttpOpenOptions::default();

        assert_eq!(options.connect_timeout, Some(Duration::from_secs(10)));
        assert_eq!(options.read_timeout, Some(Duration::from_secs(30)));
        assert_eq!(options.request_timeout, Some(Duration::from_secs(120)));
        assert!(options.headers.is_empty());
        assert!(options.client.is_none());
    }

    #[test]
    fn opens_remote_geotiff_over_http_ranges() {
        let bytes = build_simple_geotiff();
        let Some(server) = TestServer::start(bytes) else {
            return;
        };
        let file = HttpGeoTiffFile::open_with_options(
            server.url(),
            HttpOpenOptions {
                chunk_size: 128,
                cache_bytes: 1024 * 1024,
                cache_slots: 16,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        assert_eq!(file.inner().epsg(), Some(4326));
        let raster = file.inner().read_raster::<u8>().unwrap();
        let (values, offset) = raster.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![10, 20, 30, 40]);
    }

    #[test]
    fn reads_real_cog_tile_bytes_exactly_over_small_ranges() {
        let Some(bytes) = real_cog_fixture() else {
            return;
        };
        let Some(server) = TestServer::start(bytes.clone()) else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 128,
                cache_bytes: 1024 * 1024,
                cache_slots: 16,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        let expected = &bytes[570..570 + 1223];
        let actual = source.read_exact_at(570, 1223).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn range_read_rejects_wrong_content_range_start_end() {
        let Some(server) = TestServer::start_with_content_range_offset(vec![0; 12], 1) else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 4,
                cache_bytes: 0,
                cache_slots: 0,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        let error = source.read_exact_at(0, 1).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("Content-Range"), "{message}");
        assert!(message.contains("expected bytes 0-3"), "{message}");
    }

    #[test]
    fn sends_custom_headers_with_custom_client_for_probe_and_range_requests() {
        let Some(server) = TestServer::start(vec![0; 12]) else {
            return;
        };
        let mut headers = HeaderMap::new();
        headers.insert("x-test-auth", HeaderValue::from_static("secret"));
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 4,
                cache_bytes: 0,
                cache_slots: 0,
                headers,
                client: Some(client),
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        source.read_exact_at(0, 1).unwrap();

        let requests = server.requests();
        assert!(
            requests.iter().any(|request| {
                request.starts_with("HEAD ")
                    && request.to_ascii_lowercase().contains("x-test-auth: secret")
            }),
            "HEAD request did not include custom header: {requests:?}"
        );
        assert!(
            requests.iter().any(|request| {
                let lower = request.to_ascii_lowercase();
                request.starts_with("GET ")
                    && lower.contains("range: bytes=0-3")
                    && lower.contains("x-test-auth: secret")
            }),
            "range GET request did not include custom header: {requests:?}"
        );
    }

    #[test]
    fn range_cache_slot_eviction_updates_byte_accounting() {
        let Some(server) = TestServer::start(vec![0; 12]) else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 4,
                cache_bytes: 100,
                cache_slots: 2,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        source.chunk(0).unwrap();
        source.chunk(1).unwrap();
        source.chunk(2).unwrap();

        assert_eq!(source.cache.lock().current_bytes, 8);
    }

    #[test]
    fn zero_range_cache_slots_disable_storage() {
        let Some(server) = TestServer::start(vec![0; 12]) else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 4,
                cache_bytes: 100,
                cache_slots: 0,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        source.chunk(0).unwrap();

        assert_eq!(source.cache.lock().current_bytes, 0);
    }

    #[test]
    fn zero_length_range_read_does_not_fetch_chunk() {
        let Some(server) = TestServer::start(vec![0; 12]) else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 4,
                cache_bytes: 100,
                cache_slots: 2,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        assert_eq!(source.read_exact_at(12, 0).unwrap(), Vec::<u8>::new());
        assert_eq!(source.cache.lock().current_bytes, 0);
    }

    #[test]
    fn concurrent_reads_of_one_chunk_issue_a_single_range_request() {
        let Some(server) = TestServer::start_with_response_delay(
            (0..2048u32).map(|value| value as u8).collect(),
            Duration::from_millis(50),
        ) else {
            return;
        };
        let source = Arc::new(
            HttpRangeSource::open(
                server.url(),
                HttpOpenOptions {
                    chunk_size: 256,
                    cache_bytes: 1024 * 1024,
                    cache_slots: 16,
                    ..HttpOpenOptions::default()
                },
            )
            .unwrap(),
        );

        // Eight threads race for the same chunk. Without single-flight each
        // one misses the cache and issues its own duplicate GET.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let source = Arc::clone(&source);
            handles.push(thread::spawn(move || source.chunk(3).unwrap()));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let expected = &source.chunk(3).unwrap();
        for result in &results {
            assert_eq!(result.as_slice(), expected.as_slice());
        }

        let (total, distinct) = server.range_fetch_counts();
        assert_eq!(distinct, 1, "expected one distinct chunk range");
        assert_eq!(
            total, 1,
            "chunk 3 was fetched {total} times instead of once; \
             concurrent cache misses are not being coalesced"
        );
    }

    #[test]
    fn concurrent_overlapping_reads_never_refetch_a_chunk() {
        // Models what parallel tile decode does to the source: many threads
        // reading overlapping spans that share underlying chunks.
        let object: Vec<u8> = (0..8192u32).map(|value| value as u8).collect();
        let Some(server) =
            TestServer::start_with_response_delay(object.clone(), Duration::from_millis(30))
        else {
            return;
        };
        let source = Arc::new(
            HttpRangeSource::open(
                server.url(),
                HttpOpenOptions {
                    chunk_size: 512,
                    cache_bytes: 1024 * 1024,
                    cache_slots: 256,
                    ..HttpOpenOptions::default()
                },
            )
            .unwrap(),
        );

        let mut handles = Vec::new();
        for worker in 0..12u64 {
            let source = Arc::clone(&source);
            // Overlapping 1 KiB windows walking 256 bytes at a time, so
            // adjacent workers share chunks.
            let offset = worker * 256;
            handles.push(thread::spawn(move || {
                (offset, source.read_exact_at(offset, 1024).unwrap())
            }));
        }
        for handle in handles {
            let (offset, actual) = handle.join().unwrap();
            let start = offset as usize;
            assert_eq!(actual, object[start..start + 1024], "bad read at {offset}");
        }

        let (served, covered) = server.byte_fetch_totals();
        assert_eq!(
            served, covered,
            "overlapping concurrent reads transferred {served} bytes to cover \
             only {covered} distinct bytes, so data was downloaded more than once"
        );
    }

    #[test]
    fn multi_chunk_read_is_coalesced_into_one_request() {
        let Some(server) = TestServer::start((0..4096u32).map(|value| value as u8).collect())
        else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 256,
                cache_bytes: 1024 * 1024,
                cache_slots: 64,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        // Spans chunks 1..=4, all uncached and contiguous.
        let actual = source.read_exact_at(300, 900).unwrap();
        let expected: Vec<u8> =
            (0..4096u32).map(|value| value as u8).collect::<Vec<_>>()[300..1200].to_vec();
        assert_eq!(actual, expected);

        let (total, _) = server.range_fetch_counts();
        assert_eq!(
            total, 1,
            "a contiguous 4-chunk read issued {total} sequential requests \
             instead of one coalesced request"
        );
    }

    #[test]
    fn coalesced_read_reuses_already_cached_chunks() {
        let Some(server) = TestServer::start((0..4096u32).map(|value| value as u8).collect())
        else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 256,
                cache_bytes: 1024 * 1024,
                cache_slots: 64,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        // Warm chunk 2, then read across chunks 1..=4. Chunk 2 must not be
        // refetched, which splits the span into two coalesced runs.
        source.chunk(2).unwrap();
        let actual = source.read_exact_at(300, 900).unwrap();
        let expected: Vec<u8> =
            (0..4096u32).map(|value| value as u8).collect::<Vec<_>>()[300..1200].to_vec();
        assert_eq!(actual, expected);

        let ranges = server.data_ranges();
        assert_eq!(ranges.len(), 3, "unexpected fetch pattern: {ranges:?}");
        assert_eq!(ranges[0], (512, 767), "warm-up should fetch chunk 2 only");
        assert!(
            ranges[1..].contains(&(256, 511)) && ranges[1..].contains(&(768, 1279)),
            "expected runs either side of the cached chunk: {ranges:?}"
        );
    }

    #[test]
    fn coalescing_is_bounded_by_max_coalesced_chunks() {
        let Some(server) = TestServer::start((0..4096u32).map(|value| value as u8).collect())
        else {
            return;
        };
        let source = HttpRangeSource::open(
            server.url(),
            HttpOpenOptions {
                chunk_size: 256,
                cache_bytes: 1024 * 1024,
                cache_slots: 64,
                max_coalesced_chunks: 2,
                ..HttpOpenOptions::default()
            },
        )
        .unwrap();

        let actual = source.read_exact_at(0, 1024).unwrap();
        let expected: Vec<u8> =
            (0..4096u32).map(|value| value as u8).collect::<Vec<_>>()[0..1024].to_vec();
        assert_eq!(actual, expected);

        let ranges = server.data_ranges();
        assert_eq!(
            ranges.len(),
            2,
            "four chunks capped at two per request should issue two requests: {ranges:?}"
        );
        for (start, end) in ranges {
            assert!(end - start < 512, "run exceeded the cap: {start}-{end}");
        }
    }

    fn real_cog_fixture() -> Option<Vec<u8>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../testdata/interoperability/gdal/gcore/data/cog/byte_little_endian_golden.tif");
        std::fs::read(path).ok()
    }
}
