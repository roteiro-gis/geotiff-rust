//! HTTP range-backed remote GeoTIFF/COG access.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use reqwest::StatusCode;
use tiff_reader::source::{SharedSource, TiffSource};
use tiff_reader::TiffFile;

use crate::{Error, GeoTiffFile, Result};

/// Options for HTTP range-backed GeoTIFF access.
#[derive(Debug, Clone, Copy)]
pub struct HttpOpenOptions {
    /// Fixed byte-range chunk size.
    pub chunk_size: usize,
    /// Maximum bytes retained in the range cache.
    pub cache_bytes: usize,
    /// Maximum cached chunks.
    pub cache_slots: usize,
}

impl Default for HttpOpenOptions {
    fn default() -> Self {
        Self {
            chunk_size: 256 * 1024,
            cache_bytes: 64 * 1024 * 1024,
            cache_slots: 257,
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
        let source: SharedSource = Arc::new(HttpRangeSource::open(url.clone(), options)?);
        let tiff = TiffFile::from_source(source)?;
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
    cache: Mutex<RangeCacheState>,
    max_bytes: usize,
}

struct RangeCacheState {
    cache: LruCache<u64, Arc<Vec<u8>>>,
    current_bytes: usize,
}

impl HttpRangeSource {
    fn open(url: String, options: HttpOpenOptions) -> Result<Self> {
        let client = Client::builder().build()?;
        let len = probe_content_length(&client, &url)?;
        let slots = NonZeroUsize::new(options.cache_slots).unwrap_or_else(|| NonZeroUsize::new(257).unwrap());
        Ok(Self {
            client,
            url,
            len,
            chunk_size: options.chunk_size.max(4096),
            cache: Mutex::new(RangeCacheState {
                cache: LruCache::new(slots),
                current_bytes: 0,
            }),
            max_bytes: options.cache_bytes,
        })
    }

    fn chunk(&self, index: u64) -> Result<Arc<Vec<u8>>> {
        {
            let mut state = self.cache.lock();
            if let Some(chunk) = state.cache.get(&index) {
                return Ok(chunk.clone());
            }
        }

        let chunk_size = self.chunk_size as u64;
        let start = index
            .checked_mul(chunk_size)
            .ok_or_else(|| Error::Other("range chunk offset overflowed u64".into()))?;
        if start >= self.len {
            return Err(Error::Other(format!("range chunk {index} starts beyond end of object")));
        }
        let end = (start + chunk_size).min(self.len) - 1;
        let response = self
            .client
            .get(&self.url)
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
        let body = response.bytes()?.to_vec();
        let expected_len = usize::try_from(end - start + 1).unwrap_or(usize::MAX);
        if body.len() != expected_len {
            return Err(Error::Other(format!(
                "range response length mismatch for {}: expected {expected_len} bytes, got {}",
                self.url,
                body.len()
            )));
        }
        let body_len = body.len();
        let value = Arc::new(body);

        if self.max_bytes == 0 || body_len > self.max_bytes {
            return Ok(value);
        }

        let mut state = self.cache.lock();
        while state.current_bytes + body_len > self.max_bytes && !state.cache.is_empty() {
            if let Some((_, evicted)) = state.cache.pop_lru() {
                state.current_bytes = state.current_bytes.saturating_sub(evicted.len());
            }
        }
        if let Some(previous) = state.cache.put(index, value.clone()) {
            state.current_bytes = state.current_bytes.saturating_sub(previous.len());
        }
        state.current_bytes += body_len;
        Ok(value)
    }
}

impl TiffSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> tiff_reader::error::Result<Vec<u8>> {
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

        for chunk_index in first_chunk..=last_chunk {
            let chunk = self
                .chunk(chunk_index)
                .map_err(|e| tiff_reader::TiffError::Other(format!("HTTP range read failed: {e}")))?;
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

fn probe_content_length(client: &Client, url: &str) -> Result<u64> {
    let head = client.head(url).send()?;
    if head.status().is_success() {
        if let Some(value) = head.headers().get(CONTENT_LENGTH) {
            if let Ok(text) = value.to_str() {
                if let Ok(len) = text.parse::<u64>() {
                    return Ok(len);
                }
            }
        }
    }

    let response = client
        .get(url)
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
    parse_total_length(content_range).ok_or_else(|| {
        Error::Other(format!("unable to parse object size from Content-Range: {content_range}"))
    })
}

fn parse_total_length(content_range: &str) -> Option<u64> {
    let (_, total) = content_range.split_once('/')?;
    total.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::{parse_total_length, HttpGeoTiffFile, HttpOpenOptions};

    #[test]
    fn parses_total_length_from_content_range() {
        assert_eq!(parse_total_length("bytes 0-0/12345"), Some(12345));
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
            },
        )
        .unwrap();

        assert_eq!(file.inner().epsg(), Some(4326));
        let raster = file.inner().read_raster::<u8>().unwrap();
        let (values, offset) = raster.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        assert_eq!(values, vec![10, 20, 30, 40]);
    }

    fn build_simple_geotiff() -> Vec<u8> {
        fn le_u16(value: u16) -> [u8; 2] {
            value.to_le_bytes()
        }
        fn le_u32(value: u32) -> [u8; 4] {
            value.to_le_bytes()
        }
        fn le_f64(value: f64) -> [u8; 8] {
            value.to_le_bytes()
        }

        let image_data = vec![10u8, 20, 30, 40];
        let tiepoints = [0.0, 0.0, 0.0, 100.0, 200.0, 0.0];
        let scales = [2.0, 2.0, 0.0];
        let geo_keys: [u16; 12] = [1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326];
        let nodata = b"-9999\0".to_vec();

        let entries = vec![
            (256u16, 4u16, 1u32, le_u32(2).to_vec()),
            (257u16, 4u16, 1u32, le_u32(2).to_vec()),
            (258u16, 3u16, 1u32, [8, 0, 0, 0].to_vec()),
            (259u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
            (273u16, 4u16, 1u32, vec![]),
            (277u16, 3u16, 1u32, [1, 0, 0, 0].to_vec()),
            (278u16, 4u16, 1u32, le_u32(2).to_vec()),
            (279u16, 4u16, 1u32, le_u32(image_data.len() as u32).to_vec()),
            (33550u16, 12u16, 3u32, scales.iter().flat_map(|value| le_f64(*value)).collect()),
            (33922u16, 12u16, 6u32, tiepoints.iter().flat_map(|value| le_f64(*value)).collect()),
            (34735u16, 3u16, geo_keys.len() as u32, geo_keys.iter().flat_map(|value| le_u16(*value)).collect()),
            (42113u16, 2u16, nodata.len() as u32, nodata),
        ];

        let ifd_offset = 8u32;
        let ifd_size = 2 + entries.len() * 12 + 4;
        let mut next_data_offset = ifd_offset as usize + ifd_size;
        let image_offset = next_data_offset as u32;
        next_data_offset += image_data.len();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"II");
        bytes.extend_from_slice(&le_u16(42));
        bytes.extend_from_slice(&le_u32(ifd_offset));
        bytes.extend_from_slice(&le_u16(entries.len() as u16));

        let mut deferred = Vec::new();
        for (tag, ty, count, value) in entries {
            bytes.extend_from_slice(&le_u16(tag));
            bytes.extend_from_slice(&le_u16(ty));
            bytes.extend_from_slice(&le_u32(count));
            if tag == 273 {
                bytes.extend_from_slice(&le_u32(image_offset));
            } else if value.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..value.len()].copy_from_slice(&value);
                bytes.extend_from_slice(&inline);
            } else {
                bytes.extend_from_slice(&le_u32(next_data_offset as u32));
                next_data_offset += value.len();
                deferred.push(value);
            }
        }
        bytes.extend_from_slice(&le_u32(0));
        bytes.extend_from_slice(&image_data);
        for value in deferred {
            bytes.extend_from_slice(&value);
        }
        bytes
    }

    struct TestServer {
        addr: SocketAddr,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn start(bytes: Vec<u8>) -> Option<Self> {
            let listener = TcpListener::bind("127.0.0.1:0").ok()?;
            listener.set_nonblocking(true).ok()?;
            let addr = listener.local_addr().ok()?;
            let stop = Arc::new(AtomicBool::new(false));
            let stop_flag = stop.clone();

            let handle = thread::spawn(move || {
                while !stop_flag.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let mut request = [0u8; 4096];
                            let read = match stream.read(&mut request) {
                                Ok(read) => read,
                                Err(_) => continue,
                            };
                            let request = String::from_utf8_lossy(&request[..read]);
                            let mut lines = request.lines();
                            let request_line = lines.next().unwrap_or_default();
                            let mut range = None;
                            for line in lines {
                                let lower = line.to_ascii_lowercase();
                                if let Some(value) = lower.strip_prefix("range: bytes=") {
                                    range = Some(value.trim().to_string());
                                }
                            }

                            if request_line.starts_with("HEAD ") {
                                let response = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                    bytes.len()
                                );
                                let _ = stream.write_all(response.as_bytes());
                                continue;
                            }

                            if let Some(range_spec) = range {
                                let (start_s, end_s) = range_spec.split_once('-').unwrap();
                                let start: usize = start_s.parse().unwrap();
                                let end: usize = end_s.parse().unwrap();
                                let body = &bytes[start..=end];
                                let response = format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                    body.len(),
                                    start,
                                    end,
                                    bytes.len()
                                );
                                let _ = stream.write_all(response.as_bytes());
                                let _ = stream.write_all(body);
                            } else {
                                let response = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                    bytes.len()
                                );
                                let _ = stream.write_all(response.as_bytes());
                                let _ = stream.write_all(&bytes);
                            }
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });

            Some(Self {
                addr,
                stop,
                handle: Some(handle),
            })
        }

        fn url(&self) -> String {
            format!("http://{}", self.addr)
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            let _ = std::net::TcpStream::connect(self.addr);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
