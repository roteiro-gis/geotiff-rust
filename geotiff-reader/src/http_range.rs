//! Content-Range parsing and validation shared by the blocking and async
//! HTTP range sources.

use std::io::ErrorKind;
use std::time::Duration;

use reqwest::StatusCode;

use crate::{Error, Result};

const RETRY_BASE_DELAY: Duration = Duration::from_millis(100);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(2);

/// Whether a failed request may succeed if it's repeated.
///
/// Object stores routinely shed load with 429 and 503, and long-lived
/// connections drop, while anything else, such as 403 or 404, won't change.
/// Timeouts aren't retried either: they've already spent the time the caller
/// allowed, and repeating them would turn `request_timeout` into a multiple.
pub(crate) fn is_transient(error: &Error) -> bool {
    match error {
        Error::Http(error) => match error.status() {
            Some(status) => matches!(
                status,
                StatusCode::REQUEST_TIMEOUT
                    | StatusCode::TOO_MANY_REQUESTS
                    | StatusCode::INTERNAL_SERVER_ERROR
                    | StatusCode::BAD_GATEWAY
                    | StatusCode::SERVICE_UNAVAILABLE
                    | StatusCode::GATEWAY_TIMEOUT
            ),
            None => !error.is_timeout() && (error.is_connect() || error.is_body()),
        },
        Error::Io(error, _) => error.kind() != ErrorKind::TimedOut,
        _ => false,
    }
}

/// Exponential backoff before retry number `attempt`, counting from zero.
pub(crate) fn retry_delay(attempt: u32) -> Duration {
    RETRY_BASE_DELAY
        .saturating_mul(1u32.checked_shl(attempt).unwrap_or(u32::MAX))
        .min(RETRY_MAX_DELAY)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParsedContentRange {
    pub start: u64,
    pub end: u64,
    pub total: Option<u64>,
}

pub(crate) fn parse_content_range(content_range: &str) -> Option<ParsedContentRange> {
    let (unit, range_and_total) = content_range.trim().split_once(' ')?;
    if !unit.eq_ignore_ascii_case("bytes") {
        return None;
    }
    let (range, total) = range_and_total.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    if start > end {
        return None;
    }
    let total = if total == "*" {
        None
    } else {
        let total = total.parse().ok()?;
        if end >= total {
            return None;
        }
        Some(total)
    };
    Some(ParsedContentRange { start, end, total })
}

pub(crate) fn validate_content_range_header(
    content_range: Option<&str>,
    url: &str,
    expected_start: u64,
    expected_end: u64,
    expected_total: Option<u64>,
) -> Result<()> {
    let content_range = content_range
        .ok_or_else(|| Error::Other(format!("missing Content-Range header for {url}")))?;
    let parsed = parse_content_range(content_range).ok_or_else(|| {
        Error::Other(format!(
            "unable to parse Content-Range for {url}: {content_range}"
        ))
    })?;
    if parsed.start != expected_start || parsed.end != expected_end {
        return Err(Error::Other(format!(
            "unexpected Content-Range for {url}: expected bytes {expected_start}-{expected_end}, got bytes {}-{}",
            parsed.start, parsed.end
        )));
    }
    if let (Some(actual_total), Some(expected_total)) = (parsed.total, expected_total) {
        if actual_total != expected_total {
            return Err(Error::Other(format!(
                "unexpected Content-Range total for {url}: expected {expected_total}, got {actual_total}"
            )));
        }
    }
    Ok(())
}

/// Parse an object length out of a `bytes 0-0/<total>` probe response.
pub(crate) fn probe_total_from_content_range(content_range: &str, url: &str) -> Result<u64> {
    let parsed = parse_content_range(content_range).ok_or_else(|| {
        Error::Other(format!(
            "unable to parse object size from Content-Range: {content_range}"
        ))
    })?;
    if parsed.start != 0 || parsed.end != 0 {
        return Err(Error::Other(format!(
            "unexpected Content-Range for {url}: expected bytes 0-0, got bytes {}-{}",
            parsed.start, parsed.end
        )));
    }
    parsed.total.ok_or_else(|| {
        Error::Other(format!(
            "missing object size in Content-Range for {url}: bytes {}-{}/*",
            parsed.start, parsed.end
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;
    use std::time::Duration;

    use super::{
        is_transient, parse_content_range, probe_total_from_content_range, retry_delay, Error,
    };

    #[test]
    fn retry_delay_backs_off_up_to_a_limit() {
        assert_eq!(retry_delay(0), Duration::from_millis(100));
        assert_eq!(retry_delay(1), Duration::from_millis(200));
        assert_eq!(retry_delay(2), Duration::from_millis(400));
        assert_eq!(retry_delay(5), Duration::from_secs(2));
        assert_eq!(retry_delay(u32::MAX), Duration::from_secs(2));
    }

    #[test]
    fn non_network_errors_are_not_transient() {
        assert!(!is_transient(&Error::Other("bad Content-Range".into())));
        assert!(!is_transient(&Error::NotGeoTiff));
    }

    #[test]
    fn interrupted_reads_are_transient_but_timeouts_are_not() {
        let io_error = |kind| Error::Io(std::io::Error::from(kind), "range".into());

        assert!(is_transient(&io_error(ErrorKind::ConnectionReset)));
        assert!(!is_transient(&io_error(ErrorKind::TimedOut)));
    }

    #[test]
    fn parses_content_range_start_end_and_total() {
        let parsed = parse_content_range("bytes 12-34/100").unwrap();
        assert_eq!(parsed.start, 12);
        assert_eq!(parsed.end, 34);
        assert_eq!(parsed.total, Some(100));
        assert_eq!(parse_content_range("bytes 34-12/100"), None);
        assert_eq!(parse_content_range("items 12-34/100"), None);
    }

    #[test]
    fn probes_total_length_from_content_range() {
        assert_eq!(
            probe_total_from_content_range("bytes 0-0/12345", "http://example").unwrap(),
            12345
        );
        assert!(probe_total_from_content_range("bytes 1-1/5", "http://example").is_err());
    }
}
