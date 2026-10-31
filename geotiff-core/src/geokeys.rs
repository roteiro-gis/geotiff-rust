//! GeoKey directory parsing and construction (TIFF tag 34735).
//!
//! The GeoKey directory is stored as a TIFF SHORT array with the structure:
//! - Header: KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys
//! - Entries: KeyID, TIFFTagLocation, Count, ValueOffset (repeated)
//!
//! GeoKeys reference values either inline (location=0), from the
//! GeoDoubleParams tag (34736), or from the GeoAsciiParams tag (34737).

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

// Well-known GeoKey IDs.
pub const GT_MODEL_TYPE: u16 = 1024;
pub const GT_RASTER_TYPE: u16 = 1025;
pub const GT_CITATION: u16 = 1026;
pub const GEODETIC_CRS_TYPE: u16 = 2048;
pub const GEOGRAPHIC_TYPE: u16 = GEODETIC_CRS_TYPE;
pub const GEODETIC_CITATION: u16 = 2049;
pub const GEOG_CITATION: u16 = 2049;
pub const GEODETIC_DATUM: u16 = 2050;
pub const GEOG_GEODETIC_DATUM: u16 = 2050;
pub const GEOG_ANGULAR_UNITS: u16 = 2054;
pub const PROJECTED_CRS_TYPE: u16 = 3072;
pub const PROJECTED_CS_TYPE: u16 = 3072;
pub const PROJ_CITATION: u16 = 3073;
pub const PROJECTION: u16 = 3074;
pub const PROJ_COORD_TRANS: u16 = 3075;
pub const PROJ_LINEAR_UNITS: u16 = 3076;
pub const VERTICAL_CITATION: u16 = 4097;
pub const VERTICAL_CS_TYPE: u16 = 4096;
pub const VERTICAL_DATUM: u16 = 4098;
pub const VERTICAL_UNITS: u16 = 4099;
const GEO_KEY_DIRECTORY_TAG: u16 = 34735;
const GEO_DOUBLE_PARAMS_TAG: u16 = 34736;
const GEO_ASCII_PARAMS_TAG: u16 = 34737;
pub const GEO_KEY_DIRECTORY_VERSION: u16 = 1;
pub const GEO_KEY_REVISION: u16 = 1;
pub const GEO_KEY_MINOR_REVISION_1_0: u16 = 0;
pub const GEO_KEY_MINOR_REVISION_1_1: u16 = 1;

/// Error returned when a GeoKey directory cannot be represented in the
/// GeoTIFF SHORT-based key directory format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeoKeySerializeError {
    /// KeyDirectoryVersion must be 1.
    InvalidDirectoryVersion { version: u16 },
    /// KeyRevision must be 1.
    InvalidMajorRevision { major_revision: u16 },
    /// The GeoKey directory minor revision must be 0 or 1.
    InvalidMinorRevision { minor_revision: u16 },
    /// A key ID may occur only once in a directory.
    DuplicateKey { key_id: u16 },
    /// GeoAsciiParams values must be delimiter-free ASCII.
    InvalidAsciiValue { key_id: u16 },
    /// The GeoKey directory header stores the key count as a SHORT.
    TooManyKeys { count: usize },
    /// A key references more parameter values than fit in a SHORT count.
    ValueCountTooLarge { key_id: u16, tag: u16, count: usize },
    /// A key's parameter start offset does not fit in a SHORT value offset.
    ParameterOffsetTooLarge {
        key_id: u16,
        tag: u16,
        offset: usize,
    },
}

impl fmt::Display for GeoKeySerializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDirectoryVersion { version } => write!(
                f,
                "GeoKey directory version {version} is invalid; expected 1"
            ),
            Self::InvalidMajorRevision { major_revision } => write!(
                f,
                "GeoKey directory major revision {major_revision} is invalid; expected 1"
            ),
            Self::InvalidMinorRevision { minor_revision } => write!(
                f,
                "GeoKey directory minor revision {minor_revision} is invalid; expected 0 or 1"
            ),
            Self::DuplicateKey { key_id } => {
                write!(f, "GeoKey directory contains duplicate key ID {key_id}")
            }
            Self::InvalidAsciiValue { key_id } => write!(
                f,
                "GeoKey {key_id} ASCII value must contain only ASCII characters and no '|' delimiter"
            ),
            Self::TooManyKeys { count } => {
                write!(
                    f,
                    "GeoKey directory contains {count} keys, exceeding u16::MAX"
                )
            }
            Self::ValueCountTooLarge { key_id, tag, count } => write!(
                f,
                "GeoKey {key_id} references {count} values in tag {tag}, exceeding u16::MAX"
            ),
            Self::ParameterOffsetTooLarge {
                key_id,
                tag,
                offset,
            } => write!(
                f,
                "GeoKey {key_id} parameter offset {offset} in tag {tag} exceeds u16::MAX"
            ),
        }
    }
}

impl Error for GeoKeySerializeError {}

/// A parsed GeoKey entry.
#[derive(Debug, Clone)]
pub struct GeoKey {
    pub id: u16,
    pub value: GeoKeyValue,
}

/// The value of a GeoKey.
#[derive(Debug, Clone)]
pub enum GeoKeyValue {
    /// Short value stored inline.
    Short(u16),
    /// Double value(s) from GeoDoubleParams.
    Double(Vec<f64>),
    /// ASCII string from GeoAsciiParams.
    Ascii(String),
}

/// Parsed GeoKey directory.
#[derive(Debug, Clone)]
pub struct GeoKeyDirectory {
    pub version: u16,
    pub major_revision: u16,
    pub minor_revision: u16,
    pub keys: Vec<GeoKey>,
}

impl GeoKeyDirectory {
    /// Create an empty directory with a GeoTIFF 1.1 writer header.
    pub fn new() -> Self {
        Self {
            version: GEO_KEY_DIRECTORY_VERSION,
            major_revision: GEO_KEY_REVISION,
            minor_revision: GEO_KEY_MINOR_REVISION_1_1,
            keys: Vec::new(),
        }
    }

    /// Parse the GeoKey directory from the three GeoTIFF tags.
    ///
    /// - `directory`: contents of tag 34735 (SHORT array)
    /// - `double_params`: contents of tag 34736 (DOUBLE array), may be empty
    /// - `ascii_params`: contents of tag 34737 (ASCII), may be empty
    ///
    /// Returns `None` only when the directory is shorter than its header or
    /// declares an unknown `KeyDirectoryVersion`. Parsing otherwise follows
    /// libgeotiff: later revisions are accepted and normalized to one this
    /// crate can serialize, entries past the end of a short directory are
    /// ignored, a key that can't be resolved is skipped, and the first of any
    /// duplicate keys wins. Malformed metadata in one key shouldn't make the
    /// raster unreadable.
    pub fn parse(directory: &[u16], double_params: &[f64], ascii_params: &str) -> Option<Self> {
        if directory.len() < 4 || directory[0] != GEO_KEY_DIRECTORY_VERSION {
            return None;
        }

        let minor_revision = match directory[2] {
            minor @ (GEO_KEY_MINOR_REVISION_1_0 | GEO_KEY_MINOR_REVISION_1_1) => minor,
            _ => GEO_KEY_MINOR_REVISION_1_1,
        };
        let num_keys = usize::from(directory[3]);

        let mut keys = Vec::new();
        let mut seen_ids = HashSet::new();
        for entry in directory[4..].chunks_exact(4).take(num_keys) {
            let &[id, location, count, value_offset] = entry else {
                continue;
            };
            let Some(value) = parse_key_value(
                location,
                usize::from(count),
                value_offset,
                directory,
                double_params,
                ascii_params,
            ) else {
                continue;
            };
            if seen_ids.insert(id) {
                keys.push(GeoKey { id, value });
            }
        }

        Some(Self {
            version: GEO_KEY_DIRECTORY_VERSION,
            major_revision: GEO_KEY_REVISION,
            minor_revision,
            keys,
        })
    }

    /// Look up a GeoKey by ID.
    pub fn get(&self, id: u16) -> Option<&GeoKey> {
        self.keys.iter().find(|k| k.id == id)
    }

    /// Get a short value for a key.
    pub fn get_short(&self, id: u16) -> Option<u16> {
        self.get(id).and_then(|k| match &k.value {
            GeoKeyValue::Short(v) => Some(*v),
            _ => None,
        })
    }

    /// Get an ASCII value for a key.
    pub fn get_ascii(&self, id: u16) -> Option<&str> {
        self.get(id).and_then(|k| match &k.value {
            GeoKeyValue::Ascii(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Get double value(s) for a key.
    pub fn get_double(&self, id: u16) -> Option<&[f64]> {
        self.get(id).and_then(|k| match &k.value {
            GeoKeyValue::Double(v) => Some(v.as_slice()),
            _ => None,
        })
    }

    /// Insert or replace a GeoKey.
    pub fn set(&mut self, id: u16, value: GeoKeyValue) {
        if let Some(existing) = self.keys.iter_mut().find(|k| k.id == id) {
            existing.value = value;
        } else {
            self.keys.push(GeoKey { id, value });
        }
    }

    /// Remove a GeoKey by ID.
    pub fn remove(&mut self, id: u16) {
        self.keys.retain(|k| k.id != id);
    }

    /// Serialize the directory into the three TIFF tag payloads.
    ///
    /// Returns `(directory_shorts, double_params, ascii_params)`.
    /// Keys are sorted by ID per spec. Short values go inline (location=0),
    /// Double values reference the double_params array (location=34736),
    /// Ascii values reference the ascii_params string (location=34737).
    pub fn serialize(&self) -> Result<(Vec<u16>, Vec<f64>, String), GeoKeySerializeError> {
        if self.version != GEO_KEY_DIRECTORY_VERSION {
            return Err(GeoKeySerializeError::InvalidDirectoryVersion {
                version: self.version,
            });
        }
        if self.major_revision != GEO_KEY_REVISION {
            return Err(GeoKeySerializeError::InvalidMajorRevision {
                major_revision: self.major_revision,
            });
        }

        let mut sorted_keys = self.keys.clone();
        sorted_keys.sort_by_key(|k| k.id);
        if let Some(duplicate) = sorted_keys.windows(2).find(|keys| keys[0].id == keys[1].id) {
            return Err(GeoKeySerializeError::DuplicateKey {
                key_id: duplicate[0].id,
            });
        }
        let key_count =
            u16::try_from(sorted_keys.len()).map_err(|_| GeoKeySerializeError::TooManyKeys {
                count: sorted_keys.len(),
            })?;

        let mut directory = Vec::new();
        let mut double_params = Vec::new();
        let mut ascii_params = String::new();

        let minor_revision = self.serialized_minor_revision()?;

        // Header: version, major_revision, minor_revision, num_keys
        directory.push(self.version);
        directory.push(self.major_revision);
        directory.push(minor_revision);
        directory.push(key_count);

        for key in &sorted_keys {
            directory.push(key.id);
            match &key.value {
                GeoKeyValue::Short(v) => {
                    directory.push(0); // location: inline
                    directory.push(1); // count
                    directory.push(*v); // value
                }
                GeoKeyValue::Double(v) => {
                    let count = checked_u16_len(key.id, GEO_DOUBLE_PARAMS_TAG, v.len())?;
                    let offset =
                        checked_u16_offset(key.id, GEO_DOUBLE_PARAMS_TAG, double_params.len())?;
                    directory.push(GEO_DOUBLE_PARAMS_TAG); // location: GeoDoubleParams
                    directory.push(count);
                    directory.push(offset);
                    double_params.extend_from_slice(v);
                }
                GeoKeyValue::Ascii(s) => {
                    if !s.is_ascii() || s.contains('|') {
                        return Err(GeoKeySerializeError::InvalidAsciiValue { key_id: key.id });
                    }
                    let ascii_with_pipe = format!("{}|", s);
                    let count =
                        checked_u16_len(key.id, GEO_ASCII_PARAMS_TAG, ascii_with_pipe.len())?;
                    let offset =
                        checked_u16_offset(key.id, GEO_ASCII_PARAMS_TAG, ascii_params.len())?;
                    directory.push(GEO_ASCII_PARAMS_TAG); // location: GeoAsciiParams
                    directory.push(count);
                    directory.push(offset);
                    ascii_params.push_str(&ascii_with_pipe);
                }
            }
        }

        Ok((directory, double_params, ascii_params))
    }

    fn serialized_minor_revision(&self) -> Result<u16, GeoKeySerializeError> {
        match self.minor_revision {
            GEO_KEY_MINOR_REVISION_1_0 if self.requires_geotiff_1_1() => {
                Ok(GEO_KEY_MINOR_REVISION_1_1)
            }
            GEO_KEY_MINOR_REVISION_1_0 | GEO_KEY_MINOR_REVISION_1_1 => Ok(self.minor_revision),
            minor_revision => Err(GeoKeySerializeError::InvalidMinorRevision { minor_revision }),
        }
    }

    fn requires_geotiff_1_1(&self) -> bool {
        self.keys.iter().any(|key| {
            matches!(
                key.id,
                VERTICAL_CS_TYPE | VERTICAL_CITATION | VERTICAL_DATUM | VERTICAL_UNITS
            )
        })
    }
}

fn parse_key_value(
    location: u16,
    count: usize,
    value_offset: u16,
    directory: &[u16],
    double_params: &[f64],
    ascii_params: &str,
) -> Option<GeoKeyValue> {
    let start = usize::from(value_offset);
    match location {
        0 => Some(GeoKeyValue::Short(value_offset)),
        GEO_KEY_DIRECTORY_TAG if count == 1 => {
            directory.get(start).copied().map(GeoKeyValue::Short)
        }
        GEO_DOUBLE_PARAMS_TAG if count > 0 => double_params
            .get(start..start + count)
            .map(|values| GeoKeyValue::Double(values.to_vec())),
        GEO_ASCII_PARAMS_TAG if count > 0 => {
            // Offsets count bytes, and citations aren't always ASCII, so slice
            // bytes rather than a `str`, which panics off a char boundary.
            let bytes = ascii_params.as_bytes();
            let end = (start + count).min(bytes.len());
            let raw = bytes.get(start..end)?;
            let text = String::from_utf8_lossy(raw);
            Some(GeoKeyValue::Ascii(
                text.trim_end_matches('|')
                    .trim_end_matches('\0')
                    .to_string(),
            ))
        }
        _ => None,
    }
}

fn checked_u16_len(key_id: u16, tag: u16, count: usize) -> Result<u16, GeoKeySerializeError> {
    u16::try_from(count).map_err(|_| GeoKeySerializeError::ValueCountTooLarge {
        key_id,
        tag,
        count,
    })
}

fn checked_u16_offset(key_id: u16, tag: u16, offset: usize) -> Result<u16, GeoKeySerializeError> {
    u16::try_from(offset).map_err(|_| GeoKeySerializeError::ParameterOffsetTooLarge {
        key_id,
        tag,
        offset,
    })
}

impl Default for GeoKeyDirectory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        let mut dir = GeoKeyDirectory::new();
        dir.set(GT_MODEL_TYPE, GeoKeyValue::Short(2));
        dir.set(GEOGRAPHIC_TYPE, GeoKeyValue::Short(4326));
        dir.set(GEOG_CITATION, GeoKeyValue::Ascii("WGS 84".into()));

        let (shorts, doubles, ascii) = dir.serialize().unwrap();
        assert_eq!(shorts[..4], [1, 1, 1, 3]);
        let parsed = GeoKeyDirectory::parse(&shorts, &doubles, &ascii).unwrap();

        assert_eq!(parsed.get_short(GT_MODEL_TYPE), Some(2));
        assert_eq!(parsed.get_short(GEOGRAPHIC_TYPE), Some(4326));
        assert_eq!(parsed.get_ascii(GEOG_CITATION), Some("WGS 84"));
    }

    #[test]
    fn serialize_preserves_legacy_minor_revision_zero_when_compatible() {
        let mut dir = GeoKeyDirectory::new();
        dir.minor_revision = GEO_KEY_MINOR_REVISION_1_0;
        dir.set(GT_MODEL_TYPE, GeoKeyValue::Short(2));

        let (shorts, _, _) = dir.serialize().unwrap();
        assert_eq!(shorts[..4], [1, 1, 0, 1]);
    }

    #[test]
    fn serialize_promotes_vertical_geokeys_to_geotiff_1_1_minor_revision() {
        let mut dir = GeoKeyDirectory::new();
        dir.minor_revision = GEO_KEY_MINOR_REVISION_1_0;
        dir.set(VERTICAL_CS_TYPE, GeoKeyValue::Short(5703));

        let (shorts, _, _) = dir.serialize().unwrap();
        assert_eq!(shorts[..4], [1, 1, 1, 1]);
    }

    #[test]
    fn serialize_rejects_invalid_minor_revision() {
        let mut dir = GeoKeyDirectory::new();
        dir.minor_revision = 2;

        let err = dir.serialize().unwrap_err();
        assert_eq!(
            err,
            GeoKeySerializeError::InvalidMinorRevision { minor_revision: 2 }
        );
    }

    #[test]
    fn parse_rejects_unusable_directories() {
        assert!(GeoKeyDirectory::parse(&[], &[], "").is_none());
        assert!(GeoKeyDirectory::parse(&[1, 1, 1], &[], "").is_none());
        assert!(GeoKeyDirectory::parse(&[2, 1, 1, 0], &[], "").is_none());
    }

    #[test]
    fn parse_normalizes_later_revisions_so_the_directory_stays_serializable() {
        for directory in [[1, 1, 2, 0], [1, 2, 0, 0]] {
            let parsed = GeoKeyDirectory::parse(&directory, &[], "").unwrap();
            assert_eq!(parsed.major_revision, GEO_KEY_REVISION);
            assert!(parsed.serialize().is_ok());
        }
    }

    #[test]
    fn parse_tolerates_padded_and_truncated_directories() {
        let padded = [1, 1, 0, 1, GT_MODEL_TYPE, 0, 1, 2, 0, 0, 0, 0];
        let parsed = GeoKeyDirectory::parse(&padded, &[], "").unwrap();
        assert_eq!(parsed.keys.len(), 1);

        let truncated = [1, 1, 0, 5, GT_MODEL_TYPE, 0, 1, 2, GEODETIC_CRS_TYPE, 0];
        let parsed = GeoKeyDirectory::parse(&truncated, &[], "").unwrap();
        assert_eq!(parsed.keys.len(), 1);
        assert_eq!(parsed.get_short(GT_MODEL_TYPE), Some(2));
    }

    #[test]
    fn parse_keeps_the_first_of_duplicate_keys() {
        let directory = [1, 1, 1, 2, GT_MODEL_TYPE, 0, 1, 1, GT_MODEL_TYPE, 0, 1, 2];
        let parsed = GeoKeyDirectory::parse(&directory, &[], "").unwrap();
        assert_eq!(parsed.keys.len(), 1);
        assert_eq!(parsed.get_short(GT_MODEL_TYPE), Some(1));
    }

    #[test]
    fn parse_reads_inline_values_regardless_of_declared_count() {
        let directory = [1, 1, 1, 1, GT_MODEL_TYPE, 0, 0, 2];
        let parsed = GeoKeyDirectory::parse(&directory, &[], "").unwrap();
        assert_eq!(parsed.get_short(GT_MODEL_TYPE), Some(2));
    }

    #[test]
    fn parse_resolves_values_stored_in_the_directory_itself() {
        let directory = [1, 1, 1, 1, GT_MODEL_TYPE, 34735, 1, 8, 2];
        let parsed = GeoKeyDirectory::parse(&directory, &[], "").unwrap();
        assert_eq!(parsed.get_short(GT_MODEL_TYPE), Some(2));
    }

    #[test]
    fn serialize_rejects_invalid_headers_duplicates_and_non_ascii_values() {
        let mut dir = GeoKeyDirectory::new();
        dir.version = 2;
        assert!(matches!(
            dir.serialize(),
            Err(GeoKeySerializeError::InvalidDirectoryVersion { version: 2 })
        ));

        let mut dir = GeoKeyDirectory::new();
        dir.keys = vec![
            GeoKey {
                id: GT_MODEL_TYPE,
                value: GeoKeyValue::Short(1),
            },
            GeoKey {
                id: GT_MODEL_TYPE,
                value: GeoKeyValue::Short(2),
            },
        ];
        assert!(matches!(
            dir.serialize(),
            Err(GeoKeySerializeError::DuplicateKey {
                key_id: GT_MODEL_TYPE
            })
        ));

        let mut dir = GeoKeyDirectory::new();
        dir.set(GEOG_CITATION, GeoKeyValue::Ascii("WGS 84 | invalid".into()));
        assert!(matches!(
            dir.serialize(),
            Err(GeoKeySerializeError::InvalidAsciiValue {
                key_id: GEOG_CITATION
            })
        ));
    }

    #[test]
    fn set_replaces_existing() {
        let mut dir = GeoKeyDirectory::new();
        dir.set(GT_MODEL_TYPE, GeoKeyValue::Short(1));
        dir.set(GT_MODEL_TYPE, GeoKeyValue::Short(2));
        assert_eq!(dir.get_short(GT_MODEL_TYPE), Some(2));
        assert_eq!(dir.keys.len(), 1);
    }

    #[test]
    fn remove_key() {
        let mut dir = GeoKeyDirectory::new();
        dir.set(GT_MODEL_TYPE, GeoKeyValue::Short(1));
        dir.remove(GT_MODEL_TYPE);
        assert!(dir.get(GT_MODEL_TYPE).is_none());
    }

    #[test]
    fn parse_skips_unresolvable_keys_without_panicking() {
        let model_type = [GT_MODEL_TYPE, 0, 1, 2];
        for bad_key in [
            [GEOG_CITATION, 34737, 1, 500],     // ASCII offset past the end
            [GEOG_CITATION, 34737, 0, 0],       // zero-length ASCII
            [GEOG_ANGULAR_UNITS, 34736, 5, 0],  // DOUBLE range past the end
            [GEOG_ANGULAR_UNITS, 34736, 0, 0],  // zero-length DOUBLE
            [GEOG_ANGULAR_UNITS, 65000, 1, 0],  // unknown value location
            [GEOG_ANGULAR_UNITS, 34735, 3, 4],  // SHORT array has no value type
            [GEOG_ANGULAR_UNITS, 34735, 1, 99], // SHORT offset past the end
        ] {
            let mut directory = vec![1, 1, 1, 2];
            directory.extend_from_slice(&model_type);
            directory.extend_from_slice(&bad_key);

            let parsed = GeoKeyDirectory::parse(&directory, &[1.0], "x|").unwrap();

            assert_eq!(parsed.keys.len(), 1, "bad key {bad_key:?}");
            assert_eq!(parsed.get_short(GT_MODEL_TYPE), Some(2));
        }
    }

    #[test]
    fn parse_reads_citations_that_are_not_ascii_or_not_delimited() {
        let citation = |ascii: &str, count: u16| {
            let directory = [1, 1, 1, 1, GEOG_CITATION, 34737, count, 0];
            GeoKeyDirectory::parse(&directory, &[], ascii)
                .unwrap()
                .get_ascii(GEOG_CITATION)
                .map(String::from)
        };

        assert_eq!(citation("Geográfico|", 12).as_deref(), Some("Geográfico"));
        assert_eq!(citation("WGS 84", 6).as_deref(), Some("WGS 84"));
        // A count running past the end keeps whatever text is there.
        assert_eq!(citation("WGS 84|", 99).as_deref(), Some("WGS 84"));
        // Slicing inside a multi-byte character must not panic.
        assert_eq!(citation("é|", 1).as_deref(), Some("\u{fffd}"));
    }

    #[test]
    fn serialize_rejects_too_many_keys() {
        let mut dir = GeoKeyDirectory::new();
        dir.keys = (0..=u16::MAX as usize)
            .map(|index| GeoKey {
                id: index as u16,
                value: GeoKeyValue::Short(1),
            })
            .collect();

        let err = dir.serialize().unwrap_err();
        assert_eq!(
            err,
            GeoKeySerializeError::TooManyKeys {
                count: u16::MAX as usize + 1
            }
        );
    }

    #[test]
    fn serialize_rejects_oversized_double_value_count() {
        let mut dir = GeoKeyDirectory::new();
        dir.set(
            GT_CITATION,
            GeoKeyValue::Double(vec![1.0; u16::MAX as usize + 1]),
        );

        let err = dir.serialize().unwrap_err();
        assert_eq!(
            err,
            GeoKeySerializeError::ValueCountTooLarge {
                key_id: GT_CITATION,
                tag: GEO_DOUBLE_PARAMS_TAG,
                count: u16::MAX as usize + 1
            }
        );
    }

    #[test]
    fn serialize_rejects_oversized_ascii_parameter_offset() {
        let mut dir = GeoKeyDirectory::new();
        dir.set(
            GEOG_CITATION,
            GeoKeyValue::Ascii("a".repeat(u16::MAX as usize - 1)),
        );
        dir.set(PROJ_CITATION, GeoKeyValue::Ascii("b".to_string()));
        dir.set(VERTICAL_CITATION, GeoKeyValue::Ascii("c".to_string()));

        let err = dir.serialize().unwrap_err();
        assert_eq!(
            err,
            GeoKeySerializeError::ParameterOffsetTooLarge {
                key_id: VERTICAL_CITATION,
                tag: GEO_ASCII_PARAMS_TAG,
                offset: u16::MAX as usize + 2
            }
        );
    }
}
