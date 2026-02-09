//! STEP file container and entity lookup.

use rayon::prelude::*;

use super::ap214;
use super::ap214::Entity;
use super::id::Id;
use super::parse::{find_data_section, into_blocks};

/// A parsed STEP file containing a vector of entities indexed by ID.
#[derive(Debug)]
pub struct StepFile<'a> {
    pub entities: Vec<Entity<'a>>,
    /// Scale factor from file units to meters (e.g. 0.0254 for inches).
    pub length_scale: f64,
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

        let length_scale = extract_length_scale(&out);

        Self {
            entities: out,
            length_scale,
        }
    }
}

impl<'a, T> std::ops::Index<Id<T>> for StepFile<'a> {
    type Output = Entity<'a>;

    fn index(&self, id: Id<T>) -> &Self::Output {
        &self.entities[id.0]
    }
}

/// Extract the numeric value from a LENGTH_MEASURE_WITH_UNIT entity.
fn extract_length_mwu_value(entity: &Entity<'_>) -> Option<f64> {
    let raw = match entity {
        Entity::Generic(s) => *s,
        _ => return None,
    };
    // Only match LENGTH_MEASURE_WITH_UNIT (not PLANE_ANGLE etc.)
    let inner = raw
        .find("LENGTH_MEASURE_WITH_UNIT(")
        .map(|i| &raw[i + "LENGTH_MEASURE_WITH_UNIT(".len()..])?;
    let paren = inner.find('(')?;
    let rest = &inner[paren + 1..];
    fast_float::parse_partial::<f64, _>(rest)
        .ok()
        .map(|(v, _)| v)
}

/// Scale factor for an SI_UNIT with name METRE.
fn si_metre_scale(si: &ap214::SiUnit_<'_>) -> Option<f64> {
    if si.name != "METRE" {
        return None;
    }
    Some(match si.prefix {
        Some("MILLI") => 0.001,
        Some("CENTI") => 0.01,
        Some("MICRO") => 1e-6,
        Some("KILO") => 1000.0,
        _ => 1.0,
    })
}

/// Extract the length unit scale factor from parsed STEP entities.
///
/// Returns a scale factor from model units to meters:
/// - 0.001  for millimetres (`SI_UNIT(.MILLI.,.METRE.)`)
/// - 1.0    for metres (`SI_UNIT($,.METRE.)`)
/// - 0.0254 for inches (`CONVERSION_BASED_UNIT('INCH', ...)`)
/// - 1.0    as fallback if no length unit is found
fn extract_length_scale(entities: &[Entity<'_>]) -> f64 {
    // Prefer ConversionBasedUnit (e.g. inches) over bare SiUnit (e.g. metres),
    // since the bare SI metre unit may just be the base unit referenced by a conversion.
    //
    // With single-leaf collapse in parse_complex_mapping:
    // - SI length units become bare Entity::SiUnit (LENGTH_UNIT has no args → filtered)
    // - Conversion length units stay in ComplexEntity (both CBU and LENGTH_UNIT have args)
    let mut si_scale: Option<f64> = None;

    for entity in entities {
        match entity {
            // Bare SiUnit — collapsed from a complex entity like
            // (LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))
            Entity::SiUnit(si) => {
                if si_scale.is_none() {
                    si_scale = si_metre_scale(si);
                }
            }
            // ComplexEntity containing ConversionBasedUnit (+ LengthUnit)
            Entity::ComplexEntity(subs) => {
                for sub in subs {
                    if let Entity::ConversionBasedUnit(cbu) = sub {
                        if let Some(val) =
                            extract_length_mwu_value(&entities[cbu.conversion_factor])
                        {
                            return val;
                        }
                    }
                    // Also check for SiUnit inside complex entities
                    if si_scale.is_none() {
                        if let Entity::SiUnit(si) = sub {
                            si_scale = si_metre_scale(si);
                        }
                    }
                }
            }
            // Bare ConversionBasedUnit (unlikely but handle it)
            Entity::ConversionBasedUnit(cbu) => {
                if let Some(val) = extract_length_mwu_value(&entities[cbu.conversion_factor]) {
                    return val;
                }
            }
            _ => {}
        }
    }

    // Fall back to SI unit scale, or assume metres
    si_scale.unwrap_or(1.0)
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
