use std::fmt;
use std::sync::OnceLock;

/// A newtype around `OnceLock<T>` for lazily-computed cached values.
///
/// Semantics differ from raw `OnceLock` in two ways:
/// - **Clone** always returns a fresh/empty cache (no stale copies).
/// - **PartialEq** always returns `true` (caches are derived data, not identity).
///
/// Does not implement `Serialize`/`Deserialize`; fields using `Cache<T>`
/// must be annotated with `#[serde(skip)]`.
pub struct Cache<T>(OnceLock<T>);

impl<T> Cache<T> {
    pub fn new() -> Self {
        Self(OnceLock::new())
    }

    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        self.0.get_or_init(f)
    }

    pub fn get(&self) -> Option<&T> {
        self.0.get()
    }
}

impl<T: fmt::Debug> fmt::Debug for Cache<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.get() {
            Some(v) => write!(f, "Cache({:?})", v),
            None => write!(f, "Cache(<not computed>)"),
        }
    }
}

impl<T> Default for Cache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for Cache<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> PartialEq for Cache<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
