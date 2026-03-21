#![cfg(feature = "local")]

use std::path::{Path, PathBuf};

#[cfg(feature = "cog")]
use std::io::{Read, Write};
#[cfg(feature = "cog")]
use std::net::{SocketAddr, TcpListener};
#[cfg(feature = "cog")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cog")]
use std::sync::Arc;
#[cfg(feature = "cog")]
use std::sync::Mutex;
#[cfg(feature = "cog")]
use std::thread;
#[cfg(feature = "cog")]
use std::time::Duration;

use geotiff_reader::GeoTiffFile;
use ndarray::ArrayD;

#[cfg(feature = "cog")]
use geotiff_reader::cog::{HttpGeoTiffFile, HttpOpenOptions};

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../testdata/interoperability")
        .join(path)
}

#[test]
fn extracts_real_world_geotiff_metadata() {
    let file = GeoTiffFile::open(fixture("gdal/gcore/data/byte.tif")).unwrap();
    assert_eq!(file.epsg(), Some(26711));
    assert_eq!(file.width(), 20);
    assert_eq!(file.height(), 20);
    assert!(file.transform().is_some());

    let mercator = GeoTiffFile::open(fixture("gdal/gcore/data/WGS_1984_Web_Mercator.tif")).unwrap();
    assert_eq!(mercator.width(), 1);
    assert_eq!(mercator.height(), 1);
    assert!(mercator.transform().is_some());
    assert_ne!(mercator.crs().model_type, 0);
}

#[test]
fn normalizes_real_world_pixel_is_point_tiepoints() {
    let point = GeoTiffFile::open(fixture("gdal/gcore/data/byte_point.tif")).unwrap();
    assert_eq!(point.crs().raster_type, 2);

    let tiepoint = point.metadata().tiepoints.first().unwrap();
    let (x, y) = point
        .pixel_to_geo(tiepoint[0] + 0.5, tiepoint[1] + 0.5)
        .unwrap();
    assert!((x - tiepoint[3]).abs() < 1e-9);
    assert!((y - tiepoint[4]).abs() < 1e-9);
}

#[test]
fn discovers_and_reads_real_world_internal_overviews() {
    let file = GeoTiffFile::open(fixture("gdal/gcore/data/byte_with_ovr.tif")).unwrap();
    assert!(file.overview_count() > 0);

    let base: ArrayD<u8> = file.read_raster().unwrap();
    assert_eq!(base.shape(), &[20, 20]);

    let overview: ArrayD<u8> = file.read_overview(0).unwrap();
    assert!(!overview.is_empty());
}

#[test]
fn reads_real_world_signed_and_compressed_geotiffs() {
    let signed = GeoTiffFile::open(fixture("gdal/gdrivers/data/gtiff/int8.tif")).unwrap();
    let raster: ArrayD<i8> = signed.read_raster().unwrap();
    assert_eq!(raster.shape(), &[20, 20]);

    let jpeg = GeoTiffFile::open(fixture("gdal/gcore/data/gtiff/byte_JPEG.tif")).unwrap();
    let raster: ArrayD<u8> = jpeg.read_raster().unwrap();
    assert_eq!(raster.shape(), &[20, 20]);

    let zstd = GeoTiffFile::open(fixture("gdal/gcore/data/byte_zstd.tif")).unwrap();
    let raster: ArrayD<u8> = zstd.read_raster().unwrap();
    assert_eq!(raster.shape(), &[20, 20]);
}

#[test]
fn reads_real_world_cog_locally() {
    let file =
        GeoTiffFile::open(fixture("gdal/gcore/data/cog/byte_little_endian_golden.tif")).unwrap();
    let raster: ArrayD<u8> = file.read_raster().unwrap();
    assert_eq!(raster.shape(), &[20, 20]);
}

#[cfg(feature = "cog")]
#[test]
fn opens_real_world_cog_over_http_ranges() {
    let bytes =
        std::fs::read(fixture("gdal/gcore/data/cog/byte_little_endian_golden.tif")).unwrap();
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

    assert_eq!(file.inner().width(), 20);
    assert_eq!(file.inner().height(), 20);
    let raster: ArrayD<u8> = file
        .inner()
        .read_raster()
        .unwrap_or_else(|err| panic!("{err}; served ranges: {:?}", server.served_ranges()));
    assert_eq!(raster.shape(), &[20, 20]);
}

#[cfg(feature = "cog")]
type ServedRanges = Arc<Mutex<Vec<Option<(usize, usize)>>>>;

#[cfg(feature = "cog")]
struct TestServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    served_ranges: ServedRanges,
    handle: Option<thread::JoinHandle<()>>,
}

#[cfg(feature = "cog")]
impl TestServer {
    fn start(bytes: Vec<u8>) -> Option<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        listener.set_nonblocking(true).ok()?;
        let addr = listener.local_addr().ok()?;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        let served_ranges: ServedRanges = Arc::new(Mutex::new(Vec::new()));
        let served_ranges_worker = served_ranges.clone();

        let handle = thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let Some((request_line, range)) = read_request(&mut stream) else {
                            continue;
                        };
                        served_ranges_worker.lock().unwrap().push(range);

                        if request_line.starts_with("HEAD ") {
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                bytes.len()
                            );
                            let _ = stream.write_all(response.as_bytes());
                            continue;
                        }

                        if let Some((start, end)) = range {
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
            served_ranges,
            handle: Some(handle),
        })
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn served_ranges(&self) -> Vec<Option<(usize, usize)>> {
        self.served_ranges.lock().unwrap().clone()
    }
}

#[cfg(feature = "cog")]
fn read_request(stream: &mut std::net::TcpStream) -> Option<(String, Option<(usize, usize)>)> {
    let mut request = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];

    loop {
        let read = stream.read(&mut chunk).ok()?;
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

    let request = String::from_utf8_lossy(&request);
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

    Some((request_line, range))
}

#[cfg(feature = "cog")]
impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(self.addr);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
