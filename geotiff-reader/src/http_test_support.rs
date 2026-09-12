//! Shared HTTP test support: a minimal range-request server and a tiny
//! GeoTIFF fixture used by the blocking and async remote-read tests.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub(crate) fn build_simple_geotiff() -> Vec<u8> {
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
        (
            33550u16,
            12u16,
            3u32,
            scales.iter().flat_map(|value| le_f64(*value)).collect(),
        ),
        (
            33922u16,
            12u16,
            6u32,
            tiepoints.iter().flat_map(|value| le_f64(*value)).collect(),
        ),
        (
            34735u16,
            3u16,
            geo_keys.len() as u32,
            geo_keys.iter().flat_map(|value| le_u16(*value)).collect(),
        ),
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

type RequestedRange = (usize, usize);
type ParsedRequest = (String, Option<RequestedRange>, String);

pub(crate) struct TestServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<String>>>,
    ranges: Arc<Mutex<Vec<RequestedRange>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl TestServer {
    pub(crate) fn start(bytes: Vec<u8>) -> Option<Self> {
        Self::start_with_content_range_offset(bytes, 0)
    }

    pub(crate) fn start_with_response_delay(bytes: Vec<u8>, delay: Duration) -> Option<Self> {
        Self::start_configured(bytes, 0, Some(delay), Vec::new())
    }

    #[cfg(feature = "cog-async")]
    pub(crate) fn start_with_range_body_suffix(bytes: Vec<u8>, suffix: Vec<u8>) -> Option<Self> {
        Self::start_configured(bytes, 0, None, suffix)
    }

    pub(crate) fn start_with_content_range_offset(
        bytes: Vec<u8>,
        content_range_offset: usize,
    ) -> Option<Self> {
        Self::start_configured(bytes, content_range_offset, None, Vec::new())
    }

    fn start_configured(
        bytes: Vec<u8>,
        content_range_offset: usize,
        response_delay: Option<Duration>,
        range_body_suffix: Vec<u8>,
    ) -> Option<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        listener.set_nonblocking(true).ok()?;
        let addr = listener.local_addr().ok()?;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let ranges = Arc::new(Mutex::new(Vec::new()));

        let config = Arc::new(ResponseConfig {
            bytes,
            content_range_offset,
            response_delay,
            range_body_suffix,
        });
        let requests_worker = requests.clone();
        let ranges_worker = ranges.clone();

        // One thread per connection. A serial accept-and-respond loop would
        // serialize concurrent clients, which hides exactly the duplicated and
        // sequential range fetches these tests exist to catch.
        let handle = thread::spawn(move || {
            let mut connections = Vec::new();
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let config = config.clone();
                        let requests = requests_worker.clone();
                        let ranges = ranges_worker.clone();
                        connections.push(thread::spawn(move || {
                            serve_connection(stream, &config, &requests, &ranges);
                        }));
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
            for connection in connections {
                let _ = connection.join();
            }
        });

        Some(Self {
            addr,
            stop,
            requests,
            ranges,
            handle: Some(handle),
        })
    }

    pub(crate) fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub(crate) fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }

    /// Every byte range served, in request order, excluding the `bytes=0-0`
    /// object-size probe issued during open.
    pub(crate) fn data_ranges(&self) -> Vec<RequestedRange> {
        self.ranges
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|range| *range != (0, 0))
            .collect()
    }

    /// `(total_requests, distinct_ranges)` for served data ranges. These are
    /// equal only when no range was ever fetched more than once.
    pub(crate) fn range_fetch_counts(&self) -> (usize, usize) {
        let ranges = self.data_ranges();
        let mut distinct = ranges.clone();
        distinct.sort_unstable();
        distinct.dedup();
        (ranges.len(), distinct.len())
    }

    /// `(bytes_served, distinct_bytes_covered)`.
    ///
    /// Comparing range identity is not enough once requests are coalesced,
    /// because two overlapping reads produce two *different* ranges that still
    /// transfer some of the same bytes. These are equal only when no byte of
    /// the object was ever sent twice.
    pub(crate) fn byte_fetch_totals(&self) -> (usize, usize) {
        let mut ranges = self.data_ranges();
        let served: usize = ranges.iter().map(|(start, end)| end - start + 1).sum();

        ranges.sort_unstable();
        let mut covered = 0usize;
        let mut merged: Option<RequestedRange> = None;
        for (start, end) in ranges {
            match merged {
                Some((_, current_end)) if start <= current_end + 1 => {
                    merged = Some((merged.unwrap().0, current_end.max(end)));
                }
                Some((current_start, current_end)) => {
                    covered += current_end - current_start + 1;
                    merged = Some((start, end));
                }
                None => merged = Some((start, end)),
            }
        }
        if let Some((start, end)) = merged {
            covered += end - start + 1;
        }
        (served, covered)
    }
}

struct ResponseConfig {
    bytes: Vec<u8>,
    content_range_offset: usize,
    response_delay: Option<Duration>,
    range_body_suffix: Vec<u8>,
}

fn serve_connection(
    mut stream: std::net::TcpStream,
    config: &ResponseConfig,
    requests: &Mutex<Vec<String>>,
    ranges: &Mutex<Vec<RequestedRange>>,
) {
    // Accepted sockets can inherit the listener's non-blocking mode, which
    // would make the request read fail spuriously under load.
    let _ = stream.set_nonblocking(false);

    let Some((request_line, range, request)) = read_request(&mut stream) else {
        return;
    };
    requests.lock().unwrap().push(request);
    if let Some(range) = range {
        ranges.lock().unwrap().push(range);
    }
    if let Some(delay) = config.response_delay {
        thread::sleep(delay);
    }

    let total = config.bytes.len();
    if request_line.starts_with("HEAD ") {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    if let Some((start, end)) = range {
        let body = &config.bytes[start..=end];
        let response_len = body.len() + config.range_body_suffix.len();
        let reported_start = start.saturating_add(config.content_range_offset);
        let reported_end = end.saturating_add(config.content_range_offset);
        let response = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Length: {response_len}\r\nContent-Range: bytes {reported_start}-{reported_end}/{total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
        let _ = stream.write_all(&config.range_body_suffix);
    } else {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(&config.bytes);
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<ParsedRequest> {
    let mut request = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];

    loop {
        let read = match stream.read(&mut chunk) {
            Ok(read) => read,
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
                ) =>
            {
                continue;
            }
            Err(_) => return None,
        };
        if read == 0 {
            return None;
        }
        request.extend_from_slice(&chunk[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if request.len() >= 16 * 1024 {
            return None;
        }
    }

    let request = String::from_utf8_lossy(&request).into_owned();
    let mut lines = request.lines();
    let request_line = lines.next()?.to_string();
    let mut range = None;
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("range: bytes=") {
            let (start_s, end_s) = value.trim().split_once('-')?;
            let start = start_s.parse().ok()?;
            let end = end_s.parse().ok()?;
            if start > end {
                return None;
            }
            range = Some((start, end));
            break;
        }
    }

    Some((request_line, range, request))
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
