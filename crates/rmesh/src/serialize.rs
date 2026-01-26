//! Binary serialization format for rmesh data structures.
//!
//! Uses MessagePack + zstd compression with integrity checking.
//! Designed to be trivially parsed from other languages.
//!
//! # File Format
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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    ) -> Result<Self, String> {
        if struct_name.len() > STRUCT_NAME_LEN {
            return Err(format!(
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
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < Self::SIZE {
            return Err(format!(
                "Insufficient data for header: {} bytes (expected {})",
                data.len(),
                Self::SIZE
            ));
        }

        // Check magic
        if data[0..8] != MAGIC {
            return Err(format!(
                "Invalid magic: expected {:?}, got {:?}",
                MAGIC,
                &data[0..8]
            ));
        }

        // Version
        let version = u16::from_le_bytes(data[8..10].try_into().unwrap());

        // Skip reserved bytes [10..16]

        // Struct name (trim trailing zeros)
        let name_bytes = &data[16..80];
        let name_len = name_bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        let struct_name = String::from_utf8(name_bytes[..name_len].to_vec())
            .map_err(|e| format!("Invalid UTF-8 in struct name: {e}"))?;

        // Lengths
        let uncompressed_length = u64::from_le_bytes(data[80..88].try_into().unwrap());
        let compressed_length = u64::from_le_bytes(data[88..96].try_into().unwrap());

        // SHA256 hash
        let sha256_hash: [u8; 32] = data[96..128].try_into().unwrap();

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
) -> Result<(SerializationHeader, &'a [u8]), String> {
    let header = SerializationHeader::from_bytes(data)?;
    if header.struct_name != expected_struct_name {
        return Err(format!(
            "Struct name mismatch: expected '{}', got '{}'",
            expected_struct_name, header.struct_name
        ));
    }
    let compressed_data = &data[SerializationHeader::SIZE..];
    if compressed_data.len() != header.compressed_length as usize {
        return Err(format!(
            "Compressed data length mismatch: expected {}, got {}",
            header.compressed_length,
            compressed_data.len()
        ));
    }
    Ok((header, compressed_data))
}

/// Verify decompressed data integrity (shared by sync/async paths)
fn verify_decompressed(decompressed: &[u8], header: &SerializationHeader) -> Result<(), String> {
    if decompressed.len() != header.uncompressed_length as usize {
        return Err(format!(
            "Uncompressed length mismatch: expected {}, got {}",
            header.uncompressed_length,
            decompressed.len()
        ));
    }
    let computed_hash: [u8; 32] = Sha256::digest(decompressed).into();
    if computed_hash != header.sha256_hash {
        return Err("SHA256 hash mismatch: data corruption detected".to_string());
    }
    Ok(())
}

/// Trait for types that can be serialized with rmesh format
///
/// This is automatically implemented for any type that implements
/// Serialize + Deserialize, using the type name as the struct identifier.
pub trait RmeshSerializable: Serialize + for<'de> Deserialize<'de> + Sized {
    /// Serialize to bytes with Tetanus format header
    fn to_bytes(&self, compress_level: Option<i32>) -> Result<Vec<u8>, String> {
        let struct_name = get_short_type_name::<Self>();

        // Serialize using MessagePack
        let serialized =
            rmp_serde::to_vec(self).map_err(|e| format!("Failed to serialize: {e}"))?;

        // Calculate SHA256 hash of uncompressed data
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        let hash: [u8; 32] = hasher.finalize().into();

        // Compress using zstd
        let compressed = zstd::encode_all(
            serialized.as_slice(),
            compress_level.unwrap_or(3),
        )
        .map_err(|e| format!("Compression failed: {e}"))?;

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

    /// Deserialize from bytes with rmesh format header
    fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let (header, compressed_data) = validate_and_extract(data, &get_short_type_name::<Self>())?;
        let decompressed = zstd::decode_all(compressed_data)
            .map_err(|e| format!("Decompression failed: {e}"))?;
        verify_decompressed(&decompressed, &header)?;
        rmp_serde::from_slice(&decompressed).map_err(|e| format!("Failed to deserialize: {e}"))
    }
}

/// Blanket implementation: any type with Serialize + Deserialize gets RmeshSerializable
impl<T> RmeshSerializable for T where T: Serialize + for<'de> Deserialize<'de> {}

/// Deserialize from bytes, returning struct name and raw msgpack data.
///
/// Useful for inspecting files without knowing the type ahead of time.
pub fn from_bytes_generic(data: &[u8]) -> Result<(String, Vec<u8>), String> {
    let header = SerializationHeader::from_bytes(data)?;

    // Extract and decompress the payload
    let compressed_data = &data[SerializationHeader::SIZE..];
    let decompressed = zstd::decode_all(compressed_data)
        .map_err(|e| format!("Decompression failed: {e}"))?;

    // Verify hash
    let computed_hash: [u8; 32] = Sha256::digest(&decompressed).into();
    if computed_hash != header.sha256_hash {
        return Err("SHA256 hash mismatch: data corruption detected".to_string());
    }

    Ok((header.struct_name, decompressed))
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
    fn test_tetanus_serializable_round_trip() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None).unwrap();
        let deserialized: TestStruct = RmeshSerializable::from_bytes(&serialized).unwrap();

        assert_eq!(test_data, deserialized);
    }

    #[test]
    fn test_struct_name_extraction() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None).unwrap();
        let header = SerializationHeader::from_bytes(&serialized).unwrap();

        // Should extract just "TestStruct", not the full path
        assert_eq!(header.struct_name, "TestStruct");
    }

    #[test]
    fn test_wrong_struct_name() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct OtherStruct {
            value: i32,
        }

        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let serialized = test_data.to_bytes(None).unwrap();
        let result: Result<OtherStruct, String> = RmeshSerializable::from_bytes(&serialized);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Struct name mismatch"));
    }

    #[test]
    fn test_corruption_detection() {
        let test_data = TestStruct {
            value: 42,
            name: "test".to_string(),
        };

        let mut serialized = test_data.to_bytes(None).unwrap();

        // Corrupt one byte in the compressed data
        if serialized.len() > SerializationHeader::SIZE {
            serialized[SerializationHeader::SIZE] ^= 0xFF;
        }

        let result: Result<TestStruct, String> = RmeshSerializable::from_bytes(&serialized);
        assert!(result.is_err());
    }
}
