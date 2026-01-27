//! Serialization formats for rmesh data structures.
//!
//! Supports two formats:
//! - **Binary**: MessagePack + zstd compression with integrity checking (compact, fast)
//! - **JSON**: Pretty-printed JSON with wrapper (clean diffs, human-readable)
//!
//! # Binary Format
//!
//! ```text
//! RMESH--- (8 bytes magic)
//! version  (u16 LE, currently 1)
//! reserved (6 bytes, zeros)
//! struct_name (64 bytes, zero-padded ASCII)
//! uncompressed_len (u64 LE)
//! compressed_len (u64 LE)
//! sha256 (32 bytes)
//! [zstd compressed msgpack data]
//! ```
//!
//! Header is 128 bytes. Reading from other languages is trivial:
//!
//! ```python
//! import msgpack, zstd
//! data = msgpack.loads(zstd.decompress(raw[128:]))
//! ```
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "rmesh_name": "StructName",
//!   "data": { ... }
//! }
//! ```
//!
//! `from_bytes` auto-detects the format based on the first byte.

use anyhow::{Result, anyhow};
use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Near-zero tolerance for validating unit vectors/quaternions during deserialization.
const UNIT_EPSILON: f64 = 1e-10;

/// Serde helper for `Unit<Vector3<f64>>` that validates on deserialization.
///
/// Use with `#[serde(with = "crate::serialize::unit_vector3")]` on fields.
/// Returns a serde error instead of panicking if the vector is zero.
pub mod unit_vector3 {
    use super::*;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Unit<Vector3<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Vector3::<f64>::deserialize(deserializer)?;
        Unit::try_new(v, UNIT_EPSILON)
            .ok_or_else(|| serde::de::Error::custom("unit vector cannot be zero or near-zero"))
    }

    pub fn serialize<S>(unit: &Unit<Vector3<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unit.as_ref().serialize(serializer)
    }
}

/// Serde helper for `UnitQuaternion<f64>` that validates on deserialization.
///
/// Use with `#[serde(with = "crate::serialize::unit_quaternion")]` on fields.
/// Returns a serde error instead of panicking if the quaternion is zero.
pub mod unit_quaternion {
    use super::*;

    #[derive(Serialize, Deserialize)]
    struct QuatComponents {
        w: f64,
        i: f64,
        j: f64,
        k: f64,
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<UnitQuaternion<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let c = QuatComponents::deserialize(deserializer)?;
        let q = Quaternion::new(c.w, c.i, c.j, c.k);
        UnitQuaternion::try_new(q, UNIT_EPSILON)
            .ok_or_else(|| serde::de::Error::custom("quaternion cannot be zero or near-zero"))
    }

    pub fn serialize<S>(q: &UnitQuaternion<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let quat = q.quaternion();
        QuatComponents {
            w: quat.w,
            i: quat.i,
            j: quat.j,
            k: quat.k,
        }
        .serialize(serializer)
    }
}

/// Magic number: "RMESH---"
const MAGIC: [u8; 8] = *b"RMESH---";

/// Current format version
const VERSION: u16 = 1;

/// Fixed struct name field size (zero-padded)
const STRUCT_NAME_LEN: usize = 64;

/// Header for serialized data (128 bytes total)
#[derive(Debug, Clone)]
pub struct SerializationHeader {
    pub version: u16,
    pub struct_name: String,
    pub uncompressed_length: u64,
    pub compressed_length: u64,
    pub sha256_hash: [u8; 32],
}

impl SerializationHeader {
    /// Total size of the header in bytes (128 for nice alignment)
    pub const SIZE: usize = 8 + 2 + 6 + 64 + 8 + 8 + 32; // 128 bytes

    /// Create a new header from components
    pub fn new(
        struct_name: &str,
        uncompressed_length: u64,
        compressed_length: u64,
        sha256_hash: [u8; 32],
    ) -> Result<Self> {
        if struct_name.len() > STRUCT_NAME_LEN {
            return Err(anyhow!(
                "Struct name too long: {} bytes (max {})",
                struct_name.len(),
                STRUCT_NAME_LEN
            ));
        }

        Ok(Self {
            version: VERSION,
            struct_name: struct_name.to_string(),
            uncompressed_length,
            compressed_length,
            sha256_hash,
        })
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::SIZE);

        // Magic (8 bytes)
        bytes.extend_from_slice(&MAGIC);

        // Version (2 bytes)
        bytes.extend_from_slice(&self.version.to_le_bytes());

        // Reserved (6 bytes)
        bytes.extend_from_slice(&[0u8; 6]);

        // Struct name (64 bytes, zero-padded)
        let mut name_bytes = [0u8; STRUCT_NAME_LEN];
        let name_str = self.struct_name.as_bytes();
        name_bytes[..name_str.len()].copy_from_slice(name_str);
        bytes.extend_from_slice(&name_bytes);

        // Uncompressed length (8 bytes)
        bytes.extend_from_slice(&self.uncompressed_length.to_le_bytes());

        // Compressed length (8 bytes)
        bytes.extend_from_slice(&self.compressed_length.to_le_bytes());

        // SHA256 hash (32 bytes)
        bytes.extend_from_slice(&self.sha256_hash);

        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(anyhow!(
                "Insufficient data for header: {} bytes (expected {})",
                data.len(),
                Self::SIZE
            ));
        }

        // Check magic
        if data[0..8] != MAGIC {
            return Err(anyhow!(
                "Invalid magic: expected {:?}, got {:?}",
                MAGIC,
                &data[0..8]
            ));
        }

        // Version
        let version = u16::from_le_bytes(
            data[8..10]
                .try_into()
                .map_err(|_| anyhow!("Invalid header: cannot read version"))?,
        );

        // Skip reserved bytes [10..16]

        // Struct name (trim trailing zeros)
        let name_bytes = &data[16..80];
        let name_len = name_bytes
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |i| i + 1);
        let struct_name = String::from_utf8(name_bytes[..name_len].to_vec())
            .map_err(|e| anyhow!("Invalid UTF-8 in struct name: {e}"))?;

        // Lengths
        let uncompressed_length = u64::from_le_bytes(
            data[80..88]
                .try_into()
                .map_err(|_| anyhow!("Invalid header: cannot read uncompressed length"))?,
        );
        let compressed_length = u64::from_le_bytes(
            data[88..96]
                .try_into()
                .map_err(|_| anyhow!("Invalid header: cannot read compressed length"))?,
        );

        // SHA256 hash
        let sha256_hash: [u8; 32] = data[96..128]
            .try_into()
            .map_err(|_| anyhow!("Invalid header: cannot read SHA256 hash"))?;

        Ok(Self {
            version,
            struct_name,
            uncompressed_length,
            compressed_length,
            sha256_hash,
        })
    }
}

/// Get the short struct name from a full type path
fn get_short_type_name<T: ?Sized>() -> String {
    let full_name = std::any::type_name::<T>();
    // Take everything after the last '::'
    full_name
        .rsplit("::")
        .next()
        .unwrap_or(full_name)
        .to_string()
}

/// Validate header and extract compressed data (shared by sync/async paths)
fn validate_and_extract<'a>(
    data: &'a [u8],
    expected_struct_name: &str,
) -> Result<(SerializationHeader, &'a [u8])> {
    let header = SerializationHeader::from_bytes(data)?;
    if header.struct_name != expected_struct_name {
        return Err(anyhow!(
            "Struct name mismatch: expected '{}', got '{}'",
            expected_struct_name,
            header.struct_name
        ));
    }
    let compressed_data = &data[SerializationHeader::SIZE..];
    if compressed_data.len() != header.compressed_length as usize {
        return Err(anyhow!(
            "Compressed data length mismatch: expected {}, got {}",
            header.compressed_length,
            compressed_data.len()
        ));
    }
    Ok((header, compressed_data))
}

/// Verify decompressed data integrity (shared by sync/async paths)
fn verify_decompressed(decompressed: &[u8], header: &SerializationHeader) -> Result<()> {
    if decompressed.len() != header.uncompressed_length as usize {
        return Err(anyhow!(
            "Uncompressed length mismatch: expected {}, got {}",
            header.uncompressed_length,
            decompressed.len()
        ));
    }
    let computed_hash: [u8; 32] = Sha256::digest(decompressed).into();
    if computed_hash != header.sha256_hash {
        return Err(anyhow!("SHA256 hash mismatch: data corruption detected"));
    }
    Ok(())
}

/// Trait for types that can be serialized with rmesh format
///
/// This is automatically implemented for any type that implements
/// Serialize + Deserialize, using the type name as the struct identifier.
pub trait RmeshSerializable: Serialize + for<'de> Deserialize<'de> + Sized {
    /// Serialize to bytes with rmesh format
    ///
    /// - `compress_level`: zstd compression level (default 3, ignored for JSON)
    /// - `as_json`: if true, output pretty-printed JSON instead of binary
    fn to_bytes(&self, compress_level: Option<i32>, as_json: bool) -> Result<Vec<u8>> {
        let struct_name = get_short_type_name::<Self>();

        if as_json {
            // JSON format: wrap with rmesh_name for type identification
            let data =
                serde_json::to_value(self).map_err(|e| anyhow!("Failed to serialize: {e}"))?;
            let wrapper = serde_json::json!({
                "rmesh_name": struct_name,
                "data": data
            });
            let json = serde_json::to_string_pretty(&wrapper)
                .map_err(|e| anyhow!("Failed to serialize JSON: {e}"))?;
            Ok(json.into_bytes())
        } else {
            // Binary format: MessagePack + zstd
            let serialized =
                rmp_serde::to_vec(self).map_err(|e| anyhow!("Failed to serialize: {e}"))?;

            // Calculate SHA256 hash of uncompressed data
            let mut hasher = Sha256::new();
            hasher.update(&serialized);
            let hash: [u8; 32] = hasher.finalize().into();

            // Compress using zstd
            let compressed = zstd::encode_all(serialized.as_slice(), compress_level.unwrap_or(3))
                .map_err(|e| anyhow!("Compression failed: {e}"))?;

            // Create header
            let header = SerializationHeader::new(
                &struct_name,
                serialized.len() as u64,
                compressed.len() as u64,
                hash,
            )?;

            // Combine header + compressed data
            let mut result = header.to_bytes();
            result.extend_from_slice(&compressed);

            Ok(result)
        }
    }

    /// Deserialize from bytes (auto-detects binary vs JSON format)
    fn from_bytes(data: &[u8]) -> Result<Self> {
        let expected_name = get_short_type_name::<Self>();

        if data.first() == Some(&b'{') {
            // JSON format
            Self::from_json_bytes(data, &expected_name)
        } else {
            // Binary format
            Self::from_binary_bytes(data, &expected_name)
        }
    }

    /// Deserialize from JSON bytes
    fn from_json_bytes(data: &[u8], expected_name: &str) -> Result<Self> {
        let wrapper: Value =
            serde_json::from_slice(data).map_err(|e| anyhow!("Failed to parse JSON: {e}"))?;

        let obj = wrapper
            .as_object()
            .ok_or_else(|| anyhow!("Invalid rmesh JSON: expected object at root"))?;

        let rmesh_name = obj
            .get("rmesh_name")
            .ok_or_else(|| {
                anyhow!("Invalid rmesh JSON: missing required field 'rmesh_name' (expected object with 'rmesh_name' and 'data' fields)")
            })?
            .as_str()
            .ok_or_else(|| anyhow!("Invalid rmesh JSON: 'rmesh_name' must be a string"))?;

        if rmesh_name != expected_name {
            return Err(anyhow!(
                "Struct name mismatch: expected '{}', got '{}'",
                expected_name,
                rmesh_name
            ));
        }

        let data_value = obj.get("data").ok_or_else(|| {
            anyhow!("Invalid rmesh JSON: missing required field 'data' (expected object with 'rmesh_name' and 'data' fields)")
        })?;

        serde_json::from_value(data_value.clone())
            .map_err(|e| anyhow!("Failed to deserialize data: {e}"))
    }

    /// Deserialize from binary bytes
    fn from_binary_bytes(data: &[u8], expected_name: &str) -> Result<Self> {
        let (header, compressed_data) = validate_and_extract(data, expected_name)?;
        let decompressed =
            zstd::decode_all(compressed_data).map_err(|e| anyhow!("Decompression failed: {e}"))?;
        verify_decompressed(&decompressed, &header)?;
        rmp_serde::from_slice(&decompressed).map_err(|e| anyhow!("Failed to deserialize: {e}"))
    }
}

/// Blanket implementation: any type with Serialize + Deserialize gets RmeshSerializable
impl<T> RmeshSerializable for T where T: Serialize + for<'de> Deserialize<'de> {}

/// Deserialize from bytes, returning struct name and raw data.
///
/// Useful for inspecting files without knowing the type ahead of time.
/// Returns (struct_name, raw_data) where raw_data is msgpack for binary or JSON for JSON format.
pub fn from_bytes_generic(data: &[u8]) -> Result<(String, Vec<u8>)> {
    if data.first() == Some(&b'{') {
        // JSON format
        let wrapper: Value =
            serde_json::from_slice(data).map_err(|e| anyhow!("Failed to parse JSON: {e}"))?;

        let obj = wrapper
            .as_object()
            .ok_or_else(|| anyhow!("Invalid rmesh JSON: expected object at root"))?;

        let rmesh_name = obj
            .get("rmesh_name")
            .ok_or_else(|| {
                anyhow!("Invalid rmesh JSON: missing required field 'rmesh_name' (expected object with 'rmesh_name' and 'data' fields)")
            })?
            .as_str()
            .ok_or_else(|| anyhow!("Invalid rmesh JSON: 'rmesh_name' must be a string"))?
            .to_string();

        let data_value = obj.get("data").ok_or_else(|| {
            anyhow!("Invalid rmesh JSON: missing required field 'data' (expected object with 'rmesh_name' and 'data' fields)")
        })?;

        let data_bytes =
            serde_json::to_vec(data_value).map_err(|e| anyhow!("Failed to serialize data: {e}"))?;

        Ok((rmesh_name, data_bytes))
    } else {
        // Binary format
        let header = SerializationHeader::from_bytes(data)?;

        // Extract and decompress the payload
        let compressed_data = &data[SerializationHeader::SIZE..];
        let decompressed =
            zstd::decode_all(compressed_data).map_err(|e| anyhow!("Decompression failed: {e}"))?;

        // Verify hash
        let computed_hash: [u8; 32] = Sha256::digest(&decompressed).into();
        if computed_hash != header.sha256_hash {
            return Err(anyhow!("SHA256 hash mismatch: data corruption detected"));
        }

        Ok((header.struct_name, decompressed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestStruct {
        value: i32,
        name: String,
    }

    #[test]
    fn test_header_serialization() {
        let header = SerializationHeader::new("TestStruct", 1000, 500, [42u8; 32]).unwrap();
        let bytes = header.to_bytes();
        let parsed = SerializationHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.struct_name, parsed.struct_name);
        assert_eq!(header.uncompressed_length, parsed.uncompressed_length);
        assert_eq!(header.compressed_length, parsed.compressed_length);
        assert_eq!(header.sha256_hash, parsed.sha256_hash);
    }

    #[test]
    fn test_rmesh_serializable_round_trip_binary() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, false).unwrap();
        let deserialized: TestStruct = RmeshSerializable::from_bytes(&serialized).unwrap();

        assert_eq!(test_data, deserialized);
    }

    #[test]
    fn test_rmesh_serializable_round_trip_json() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, true).unwrap();

        // Verify it's valid JSON with expected structure
        let json_str = std::str::from_utf8(&serialized).unwrap();
        assert!(json_str.contains("\"rmesh_name\": \"TestStruct\""));
        assert!(json_str.contains("\"data\":"));

        let deserialized: TestStruct = RmeshSerializable::from_bytes(&serialized).unwrap();
        assert_eq!(test_data, deserialized);
    }

    #[test]
    fn test_struct_name_extraction() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, false).unwrap();
        let header = SerializationHeader::from_bytes(&serialized).unwrap();

        // Should extract just "TestStruct", not the full path
        assert_eq!(header.struct_name, "TestStruct");
    }

    #[test]
    fn test_wrong_struct_name_binary() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct OtherStruct {
            value: i32,
        }

        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, false).unwrap();
        let result: anyhow::Result<OtherStruct> = RmeshSerializable::from_bytes(&serialized);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Struct name mismatch")
        );
    }

    #[test]
    fn test_wrong_struct_name_json() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct OtherStruct {
            value: i32,
        }

        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, true).unwrap();
        let result: anyhow::Result<OtherStruct> = RmeshSerializable::from_bytes(&serialized);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Struct name mismatch")
        );
    }

    #[test]
    fn test_corruption_detection() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let mut serialized = test_data.to_bytes(None, false).unwrap();

        // Corrupt one byte in the compressed data
        if serialized.len() > SerializationHeader::SIZE {
            serialized[SerializationHeader::SIZE] ^= 0xFF;
        }

        let result: anyhow::Result<TestStruct> = RmeshSerializable::from_bytes(&serialized);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_missing_rmesh_name() {
        let json = r#"{"data": {"value": 42, "name": "test"}}"#;
        let result: anyhow::Result<TestStruct> = RmeshSerializable::from_bytes(json.as_bytes());

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("rmesh_name"));
        assert!(err.contains("data"));
    }

    #[test]
    fn test_json_missing_data() {
        let json = r#"{"rmesh_name": "TestStruct"}"#;
        let result: anyhow::Result<TestStruct> = RmeshSerializable::from_bytes(json.as_bytes());

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("data"));
    }

    #[test]
    fn test_from_bytes_generic_json() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, true).unwrap();
        let (name, _data) = from_bytes_generic(&serialized).unwrap();

        assert_eq!(name, "TestStruct");
    }

    #[test]
    fn test_from_bytes_generic_binary() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None, false).unwrap();
        let (name, _data) = from_bytes_generic(&serialized).unwrap();

        assert_eq!(name, "TestStruct");
    }
}
