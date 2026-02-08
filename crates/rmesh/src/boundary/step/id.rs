//! Type-safe STEP entity ID wrapper.

/// A type-safe reference to a STEP entity.
/// The type parameter `T` indicates what kind of entity this ID points to.
#[derive(Debug)]
pub struct Id<T>(pub usize, std::marker::PhantomData<fn() -> T>);

impl<T> Id<T> {
    pub fn new(i: usize) -> Self {
        Id(i, std::marker::PhantomData)
    }

    pub fn empty() -> Self {
        Id::new(0)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for Id<T> {}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
