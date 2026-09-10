//! Asynchronous HTTP range-backed remote GeoTIFF/COG access.
//!
//! Range requests use the async `reqwest` client while TIFF parsing and
//! block decoding run on the Tokio blocking pool, bridged through a
//! [`TiffSource`] whose reads block on in-flight range fetches. All decode
//! entry points on [`AsyncHttpGeoTiffFile`] are `async`; metadata accessors
//! are synchronous because everything they need is resolved at open time.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use lru::LruCache;
use parking_lot::Mutex;
use reqwest::header::{HeaderMap, CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use reqwest::{Client, RequestBuilder, StatusCode};
use tiff_reader::source::{SharedSource, TiffSource};
use tiff_reader::{OpenOptions as TiffOpenOptions, TiffFile, TiffSample};
use tokio::runtime::Handle;
use tokio::sync::Notify;

use crate::http_range::{probe_total_from_content_range, validate_content_range_header};
use crate::{Error, GeoTiffFile, Result};

/// Options for asynchronous HTTP range-backed GeoTIFF access.
#[derive(Debug, Clone)]
pub struct AsyncHttpOpenOptions {
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
    /// Overall timeout applied to each HEAD or GET request, including the
    /// response body.
    pub request_timeout: Option<Duration>,
    /// Headers sent on every HEAD and byte-range GET request.
    pub headers: HeaderMap,
    /// Optional preconfigured async client for custom TLS, proxy, redirect,
    /// or auth behavior.
    pub client: Option<Client>,
    /// TIFF decoder options applied after range reads are assembled.
    pub tiff_options: TiffOpenOptions,
}

impl Default for AsyncHttpOpenOptions {
    fn default() -> Self {
        Self {
            chunk_size: 256 * 1024,
            cache_bytes: 64 * 1024 * 1024,
            cache_slots: 257,
            max_coalesced_chunks: 16,
            connect_timeout: Some(Duration::from_secs(10)),
            request_timeout: Some(Duration::from_secs(120)),
            headers: HeaderMap::new(),
            client: None,
            tiff_options: TiffOpenOptions::default(),
        }
    }
}

/// Remote GeoTIFF/COG handle backed by asynchronous HTTP range requests.
pub struct AsyncHttpGeoTiffFile {
    url: String,
    inner: Arc<GeoTiffFile>,
}

impl AsyncHttpGeoTiffFile {
    /// Open a remote GeoTIFF/COG using asynchronous HTTP range requests.
    ///
    /// Must be called within a multi-thread Tokio runtime.
    pub async fn open(url: impl Into<String>) -> Result<Self> {
        Self::open_with_options(url, AsyncHttpOpenOptions::default()).await
    }

    /// Open a remote GeoTIFF/COG using explicit range-cache options.
    pub async fn open_with_options(
        url: impl Into<String>,
        options: AsyncHttpOpenOptions,
    ) -> Result<Self> {
        let url = url.into();
        let tiff_options = options.tiff_options;
        let source = Arc::new(AsyncHttpRangeSource::open(url.clone(), options).await?);
        let bridged: SharedSource = Arc::new(BridgedTiffSource {
            source,
            handle: Handle::current(),
        });
        let inner = spawn_decode(move || {
            let tiff = TiffFile::from_source_with_options(bridged, tiff_options)?;
            GeoTiffFile::from_tiff(tiff)
        })
        .await?;
        Ok(Self {
            url,
            inner: Arc::new(inner),
        })
    }

    /// The source URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Access the decoded GeoTIFF for synchronous metadata queries.
    ///
    /// Only metadata accessors are safe here: the inner file's blocking read
    /// methods bridge into the async runtime and panic when called from an
    /// async context. Use the `read_*` methods on this type instead.
    pub fn inner(&self) -> &GeoTiffFile {
        &self.inner
    }

    /// Decode the base-resolution raster into storage-domain typed samples.
    pub async fn read_raster<T: TiffSample + Send>(&self) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_raster::<T>()).await
    }

    /// Decode the base-resolution raster into color-decoded typed pixels.
    pub async fn read_decoded_raster<T: TiffSample + Send>(&self) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_decoded_raster::<T>()).await
    }

    /// Decode a base-resolution pixel window into storage-domain typed samples.
    pub async fn read_window<T: TiffSample + Send>(
        &self,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_window::<T>(row_off, col_off, rows, cols)).await
    }

    /// Decode a base-resolution pixel window into color-decoded typed pixels.
    pub async fn read_decoded_window<T: TiffSample + Send>(
        &self,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_decoded_window::<T>(row_off, col_off, rows, cols)).await
    }

    /// Decode one base-resolution storage-domain band.
    pub async fn read_band<T: TiffSample + Send>(
        &self,
        band_index: usize,
    ) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_band::<T>(band_index)).await
    }

    /// Decode a base-resolution window from one storage-domain band.
    pub async fn read_band_window<T: TiffSample + Send>(
        &self,
        band_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_band_window::<T>(band_index, row_off, col_off, rows, cols))
            .await
    }

    /// Decode an overview raster into storage-domain typed samples.
    pub async fn read_overview<T: TiffSample + Send>(
        &self,
        overview_index: usize,
    ) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || inner.read_overview::<T>(overview_index)).await
    }

    /// Decode an overview pixel window into storage-domain typed samples.
    pub async fn read_overview_window<T: TiffSample + Send>(
        &self,
        overview_index: usize,
        row_off: usize,
        col_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ndarray::ArrayD<T>> {
        let inner = Arc::clone(&self.inner);
        spawn_decode(move || {
            inner.read_overview_window::<T>(overview_index, row_off, col_off, rows, cols)
        })
        .await
    }
}

async fn spawn_decode<T, F>(decode: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(decode)
        .await
        .map_err(|error| Error::Other(format!("decode task failed: {error}")))?
}

/// Sync [`TiffSource`] facade over the async range source.
///
/// Reads block on the captured runtime handle, which is only legal from the
/// Tokio blocking pool; every decode entry point above routes through
/// `spawn_blocking` to guarantee that.
struct BridgedTiffSource {
    source: Arc<AsyncHttpRangeSource>,
    handle: Handle,
}

impl TiffSource for BridgedTiffSource {
    fn len(&self) -> u64 {
        self.source.len
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> tiff_reader::error::Result<Vec<u8>> {
        let source = Arc::clone(&self.source);
        self.handle
            .block_on(async move { source.read_exact_at(offset, len).await })
    }
}

struct AsyncHttpRangeSource {
    client: Client,
    url: String,
    len: u64,
    chunk_size: usize,
    max_coalesced_chunks: usize,
    headers: HeaderMap,
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

/// Rendezvous point for tasks that all want the same uncached chunk.
///
/// Exactly one task fetches; the rest await here until the result is
/// published. Errors are carried as text because the transport error types
/// are not `Clone`.
struct InFlightChunk {
    state: Mutex<Option<std::result::Result<Arc<Vec<u8>>, String>>>,
    ready: Notify,
}

impl InFlightChunk {
    fn new() -> Self {
        Self {
            state: Mutex::new(None),
            ready: Notify::new(),
        }
    }

    fn publish(&self, value: std::result::Result<Arc<Vec<u8>>, String>) {
        {
            let mut state = self.state.lock();
            if state.is_none() {
                *state = Some(value);
            }
        }
        self.ready.notify_waiters();
    }

    async fn wait(&self) -> std::result::Result<Arc<Vec<u8>>, String> {
        loop {
            // Register before checking so a publish that lands in between is
            // not missed.
            let notified = self.ready.notified();
            if let Some(value) = self.state.lock().as_ref() {
                return value.clone();
            }
            notified.await;
        }
    }
}

/// Whether this caller owns fetching a chunk or is waiting on another caller.
enum ChunkClaim {
    Cached(Arc<Vec<u8>>),
    Lead(Arc<InFlightChunk>),
    Follow(Arc<InFlightChunk>),
}

/// Releases every led chunk on drop so waiters can never be stranded, including
/// when the owning future is cancelled mid-fetch.
struct LeadGuard<'a> {
    source: &'a AsyncHttpRangeSource,
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

impl AsyncHttpRangeSource {
    async fn open(url: String, options: AsyncHttpOpenOptions) -> Result<Self> {
        let client = match &options.client {
            Some(client) => client.clone(),
            None => {
                let mut builder = Client::builder();
                if let Some(timeout) = options.connect_timeout {
                    builder = builder.connect_timeout(timeout);
                }
                if let Some(timeout) = options.request_timeout {
                    builder = builder.timeout(timeout);
                }
                builder.build()?
            }
        };
        let len =
            probe_content_length(&client, &url, &options.headers, options.request_timeout).await?;
        let slots = NonZeroUsize::new(options.cache_slots.max(1)).unwrap();
        Ok(Self {
            client,
            url,
            len,
            chunk_size: options.chunk_size.max(1),
            max_coalesced_chunks: options.max_coalesced_chunks.max(1),
            headers: options.headers,
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
    /// opposite order but never hold both, so the two cannot deadlock. Neither
    /// lock is ever held across an await.
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
    async fn chunk(&self, index: u64) -> Result<Arc<Vec<u8>>> {
        let mut chunks = self.chunk_span(index, index).await?;
        Ok(chunks.remove(0))
    }

    /// Resolve every chunk in `first..=last`, fetching contiguous runs of
    /// missing chunks in one request each and sharing in-flight fetches with
    /// other tasks.
    async fn chunk_span(&self, first: u64, last: u64) -> Result<Vec<Arc<Vec<u8>>>> {
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
            match self.fetch_run(run[0], run.len()).await {
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
            let chunk = entry.wait().await.map_err(Error::Other)?;
            resolved[slot] = Some(chunk);
        }

        resolved
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::Other("range chunk span left an unresolved chunk".into()))
    }

    /// Issue one range request covering `count` consecutive chunks and split
    /// the response back into per-chunk bodies.
    async fn fetch_run(&self, first: u64, count: usize) -> Result<Vec<Vec<u8>>> {
        let (start, _) = self.chunk_bounds(first)?;
        let (_, end) = self.chunk_bounds(first + count as u64 - 1)?;

        let response = request_with_options(
            self.client.get(&self.url),
            &self.headers,
            self.request_timeout,
        )
        .header(RANGE, format!("bytes={start}-{end}"))
        .send()
        .await?
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
        let body = read_response_body_bounded(
            response,
            expected_len,
            &format!("{} bytes={start}-{end}", self.url),
        )
        .await?;

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

    async fn read_exact_at(&self, offset: u64, len: usize) -> tiff_reader::error::Result<Vec<u8>> {
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
            .await
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

async fn read_response_body_bounded(
    mut response: reqwest::Response,
    expected_len: usize,
    context: &str,
) -> Result<Vec<u8>> {
    if let Some(content_len) = response.content_length() {
        let expected_len_u64 = u64::try_from(expected_len).unwrap_or(u64::MAX);
        if content_len > expected_len_u64 {
            return Err(Error::Other(format!(
                "HTTP response body for {context} exceeds the expected {expected_len}-byte range"
            )));
        }
    }

    let mut body = Vec::with_capacity(expected_len);
    while let Some(chunk) = response.chunk().await? {
        let remaining = expected_len - body.len();
        if chunk.len() > remaining {
            return Err(Error::Other(format!(
                "HTTP response body for {context} exceeds the expected {expected_len}-byte range"
            )));
        }
        body.extend_from_slice(&chunk);
    }

    if body.len() != expected_len {
        return Err(Error::Other(format!(
            "range response length mismatch for {context}: expected {expected_len} bytes, got {}",
            body.len()
        )));
    }
    Ok(body)
}

async fn probe_content_length(
    client: &Client,
    url: &str,
    headers: &HeaderMap,
    request_timeout: Option<Duration>,
) -> Result<u64> {
    let head = request_with_options(client.head(url), headers, request_timeout)
        .send()
        .await?;
    if head.status().is_success() {
        if let Some(len) = head
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|text| text.parse::<u64>().ok())
        {
            return Ok(len);
        }
    }

    let response = request_with_options(client.get(url), headers, request_timeout)
        .header(RANGE, "bytes=0-0")
        .send()
        .await?
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
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use reqwest::Client;

    use super::{AsyncHttpGeoTiffFile, AsyncHttpOpenOptions, AsyncHttpRangeSource};
    use crate::http_test_support::{build_simple_geotiff, TestServer};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn opens_remote_geotiff_over_async_http_ranges() {
        let bytes = build_simple_geotiff();
        let Some(server) = TestServer::start(bytes) else {
            return;
        };

        let file = AsyncHttpGeoTiffFile::open_with_options(
            server.url(),
            AsyncHttpOpenOptions {
                chunk_size: 128,
                cache_bytes: 1024 * 1024,
                cache_slots: 16,
                ..AsyncHttpOpenOptions::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(file.inner().epsg(), Some(4326));
        assert_eq!(file.inner().nodata(), Some("-9999"));

        let raster = file.read_raster::<u8>().await.unwrap();
        let (values, offset) = raster.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![10, 20, 30, 40]);

        let window = file.read_window::<u8>(1, 0, 1, 2).await.unwrap();
        let (values, offset) = window.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![30, 40]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_range_reads_send_custom_headers() {
        let Some(server) = TestServer::start(vec![0; 12]) else {
            return;
        };
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-test-auth",
            reqwest::header::HeaderValue::from_static("secret"),
        );

        // 12 zero bytes are not a TIFF; opening fails after the probe and
        // first range request, which is all this test needs.
        let result = AsyncHttpGeoTiffFile::open_with_options(
            server.url(),
            AsyncHttpOpenOptions {
                chunk_size: 4,
                headers,
                ..AsyncHttpOpenOptions::default()
            },
        )
        .await;
        assert!(result.is_err());

        let requests = server.requests();
        assert!(requests
            .iter()
            .any(|request| request.to_ascii_lowercase().contains("x-test-auth: secret")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn custom_async_client_honors_request_timeout() {
        let delay = Duration::from_millis(500);
        let Some(server) = TestServer::start_with_response_delay(vec![0; 12], delay) else {
            return;
        };
        let started = Instant::now();
        let result = AsyncHttpGeoTiffFile::open_with_options(
            server.url(),
            AsyncHttpOpenOptions {
                request_timeout: Some(Duration::from_millis(30)),
                client: Some(Client::builder().build().unwrap()),
                ..AsyncHttpOpenOptions::default()
            },
        )
        .await;

        assert!(result.is_err());
        assert!(started.elapsed() < delay, "custom client ignored timeout");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_range_read_rejects_oversized_body_before_buffering_it() {
        let Some(server) =
            TestServer::start_with_range_body_suffix(vec![0; 12], vec![1; 1024 * 1024])
        else {
            return;
        };
        let source = AsyncHttpRangeSource::open(
            server.url(),
            AsyncHttpOpenOptions {
                chunk_size: 4,
                cache_bytes: 0,
                cache_slots: 0,
                ..AsyncHttpOpenOptions::default()
            },
        )
        .await
        .unwrap();

        let error = source.chunk(0).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("exceeds the expected 4-byte range"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_async_reads_of_one_chunk_issue_a_single_range_request() {
        let Some(server) = TestServer::start_with_response_delay(
            (0..2048u32).map(|value| value as u8).collect(),
            Duration::from_millis(50),
        ) else {
            return;
        };
        let source = Arc::new(
            AsyncHttpRangeSource::open(
                server.url(),
                AsyncHttpOpenOptions {
                    chunk_size: 256,
                    cache_bytes: 1024 * 1024,
                    cache_slots: 16,
                    ..AsyncHttpOpenOptions::default()
                },
            )
            .await
            .unwrap(),
        );

        // Eight tasks race for the same chunk. Without single-flight each one
        // misses the cache and issues its own duplicate GET.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let source = Arc::clone(&source);
            handles.push(tokio::spawn(async move { source.chunk(3).await.unwrap() }));
        }
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }

        let expected = source.chunk(3).await.unwrap();
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_multi_chunk_read_is_coalesced_into_one_request() {
        let Some(server) = TestServer::start((0..4096u32).map(|value| value as u8).collect())
        else {
            return;
        };
        let source = AsyncHttpRangeSource::open(
            server.url(),
            AsyncHttpOpenOptions {
                chunk_size: 256,
                cache_bytes: 1024 * 1024,
                cache_slots: 64,
                ..AsyncHttpOpenOptions::default()
            },
        )
        .await
        .unwrap();

        // Spans chunks 1..=4, all uncached and contiguous.
        let actual = source.read_exact_at(300, 900).await.unwrap();
        let expected =
            (0..4096u32).map(|value| value as u8).collect::<Vec<_>>()[300..1200].to_vec();
        assert_eq!(actual, expected);

        let (total, _) = server.range_fetch_counts();
        assert_eq!(
            total, 1,
            "a contiguous 4-chunk read issued {total} sequential requests \
             instead of one coalesced request"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_async_overlapping_reads_never_refetch_a_chunk() {
        // Overlapping spans produce distinct ranges once coalescing is in play,
        // so this compares bytes served against bytes covered rather than
        // counting repeated ranges.
        let object: Vec<u8> = (0..8192u32).map(|value| value as u8).collect();
        let Some(server) =
            TestServer::start_with_response_delay(object.clone(), Duration::from_millis(30))
        else {
            return;
        };
        let source = Arc::new(
            AsyncHttpRangeSource::open(
                server.url(),
                AsyncHttpOpenOptions {
                    chunk_size: 512,
                    cache_bytes: 1024 * 1024,
                    cache_slots: 256,
                    ..AsyncHttpOpenOptions::default()
                },
            )
            .await
            .unwrap(),
        );

        let mut handles = Vec::new();
        for worker in 0..12u64 {
            let source = Arc::clone(&source);
            let offset = worker * 256;
            handles.push(tokio::spawn(async move {
                (offset, source.read_exact_at(offset, 1024).await.unwrap())
            }));
        }
        for handle in handles {
            let (offset, actual) = handle.await.unwrap();
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
}
