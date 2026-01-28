use std::sync::{Arc, OnceLock};

pub use image::DynamicImage;

/// Lazily-decoded image that stores raw bytes and decodes on demand.
/// This is useful for deferring expensive image decoding until the
/// image is actually needed.
///
/// Uses `Arc<[u8]>` for the raw bytes to allow O(1) cloning - only the
/// reference count is incremented, not the actual bytes copied.
pub struct LazyImage {
    /// The raw image bytes (PNG, JPEG, etc.) - shared via Arc for cheap clones
    bytes: Arc<[u8]>,
    /// Cached decoded image (per-instance, not shared on clone)
    decoded: OnceLock<DynamicImage>,
}

impl LazyImage {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes.into(),
            decoded: OnceLock::new(),
        }
    }

    /// Decode the image, caching the result for future calls.
    /// Returns None if decoding fails.
    pub fn decode(&self) -> Option<&DynamicImage> {
        // Try to get cached value first
        if let Some(img) = self.decoded.get() {
            return Some(img);
        }
        // Try to decode and cache
        if let Ok(img) = image::load_from_memory(&self.bytes) {
            let _ = self.decoded.set(img);
            self.decoded.get()
        } else {
            None
        }
    }

    /// Get the decoded image if already cached, without decoding.
    pub fn get_decoded(&self) -> Option<&DynamicImage> {
        self.decoded.get()
    }

    /// Returns the size of the raw bytes.
    pub fn bytes_len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns a reference to the raw bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl std::fmt::Debug for LazyImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyImage")
            .field("bytes_len", &self.bytes.len())
            .field("decoded", &self.decoded.get().is_some())
            .finish()
    }
}

impl Clone for LazyImage {
    fn clone(&self) -> Self {
        // Clone shares the Arc bytes (O(1)), but decoded cache is per-instance
        Self {
            bytes: Arc::clone(&self.bytes),
            decoded: OnceLock::new(),
        }
    }
}
