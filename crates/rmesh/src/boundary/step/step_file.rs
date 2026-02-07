//! STEP file container and entity lookup.

use rayon::prelude::*;

use super::ap214::Entity;
use super::id::Id;
use super::parse::{find_data_section, into_blocks, strip_flatten};

/// A parsed STEP file containing a vector of entities indexed by ID.
#[derive(Debug)]
pub struct StepFile<'a> {
    pub entities: Vec<Entity<'a>>,
}

impl<'a> StepFile<'a> {
    /// Parses a STEP file from a raw array of bytes.
    /// `data` must be preprocessed by [`strip_flatten`] first.
    pub fn parse(data: &'a [u8]) -> Self {
        let blocks = into_blocks(data);
        let (data_start, data_end) = find_data_section(&blocks);

        let parsed: Vec<(usize, Entity<'a>)> = blocks[data_start..data_end]
            .par_iter()
            .filter_map(|b| {
                parse_entity_decl(b)
                    .or_else(|()| parse_entity_fallback(b))
                    .ok()
            })
            .map(|b| b.1)
            .collect();

        // Build entity vector indexed by ID
        let max_id = parsed.iter().map(|b| b.0).max().unwrap_or(0);
        let mut out: Vec<Entity<'a>> = (0..=max_id).map(|_| Entity::_EmptySlot).collect();

        for p in parsed {
            out[p.0] = p.1;
        }

        Self { entities: out }
    }

    /// Preprocess a STEP file (remove comments/whitespace).
    pub fn preprocess(raw_data: &[u8]) -> Vec<u8> {
        strip_flatten(raw_data)
    }

    /// Get an entity by ID, attempting to cast it to type T.
    pub fn entity<T: FromEntity<'a>>(&'a self, i: Id<T>) -> Option<&'a T> {
        T::try_from_entity(&self.entities[i.0])
    }

    /// Get the number of entities in the file.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if the file is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

impl<'a, T> std::ops::Index<Id<T>> for StepFile<'a> {
    type Output = Entity<'a>;

    fn index(&self, id: Id<T>) -> &Self::Output {
        &self.entities[id.0]
    }
}

/// Trait for extracting a specific entity type from the generic Entity enum.
pub trait FromEntity<'a> {
    fn try_from_entity(e: &'a Entity<'a>) -> Option<&'a Self>;
}

/// Parse a single entity declaration like "#123=ENTITY_NAME(...);"
fn parse_entity_decl(s: &[u8]) -> Result<(&[u8], (usize, Entity<'_>)), ()> {
    let s = std::str::from_utf8(s).map_err(|_| ())?;

    use super::parse::Parse;
    use nom::character::complete::char;
    use nom::combinator::map;
    use nom::sequence::tuple;

    let result = map(
        tuple((Id::<()>::parse, char('='), Entity::parse)),
        |(i, _, e)| (i.0, e),
    )(s);

    match result {
        Ok((remaining, value)) => Ok((remaining.as_bytes(), value)),
        Err(_) => Err(()),
    }
}

/// Fallback parser that just extracts the ID when entity parsing fails.
fn parse_entity_fallback(s: &[u8]) -> Result<(&[u8], (usize, Entity<'_>)), ()> {
    let s = std::str::from_utf8(s).map_err(|_| ())?;

    use super::parse::Parse;
    use nom::combinator::map;

    let result = map(Id::<()>::parse, |i| (i.0, Entity::_FailedToParse))(s);

    match result {
        Ok((remaining, value)) => Ok((remaining.as_bytes(), value)),
        Err(_) => Err(()),
    }
}
