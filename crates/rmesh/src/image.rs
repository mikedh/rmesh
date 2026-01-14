use std::sync::OnceLock;

pub use image::DynamicImage;

/// Lazily-decoded image that stores raw bytes and decodes on demand.
/// This is useful for deferring expensive image decoding until the
/// image is actually needed.
pub struct LazyImage {
    /// The raw image bytes (PNG, JPEG, etc.)
    pub bytes: Vec<u8>,
    /// Cached decoded image
    decoded: OnceLock<DynamicImage>,
}

impl LazyImage {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
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
        // Clone only the bytes, not the cached decode
        Self {
            bytes: self.bytes.clone(),
            decoded: OnceLock::new(),
        }
    }
}
