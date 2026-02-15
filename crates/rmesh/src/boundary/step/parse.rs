//! STEP file tokenizer and low-level parsing utilities.
//!
//! This module provides a zero-copy parser for STEP (ISO 10303-21) files.

#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::stable_sort_primitive)]

use arrayvec::ArrayVec;
use memchr::{memchr, memchr_iter, memchr2};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::{map, map_res, opt},
    error::{Error, ErrorKind},
    sequence::{delimited, preceded, tuple},
};
use std::collections::{HashMap, HashSet};

use super::id::Id;

pub type IResult<'a, U> = nom::IResult<&'a str, U, Error<&'a str>>;

/// Helper function to generate a `nom` error result
fn nom_err<'a, U>(s: &'a str, kind: ErrorKind) -> IResult<'a, U> {
    Err(nom::Err::Error(Error::new(s, kind)))
}

/// A three-valued logical (true, false, unknown)
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Logical(pub Option<bool>);

/// Trait for types that can be parsed from a STEP string
pub trait Parse<'a> {
    fn parse(s: &'a str) -> IResult<'a, Self>
    where
        Self: Sized;
}

impl Parse<'_> for f64 {
    fn parse(s: &str) -> IResult<'_, Self> {
        match fast_float::parse_partial::<f64, _>(s) {
            Err(_) => nom_err(s, ErrorKind::Float),
            Ok((x, n)) => Ok((&s[n..], x)),
        }
    }
}

impl Parse<'_> for i64 {
    fn parse(s: &str) -> IResult<'_, Self> {
        map_res(
            tuple((opt(char('-')), digit1)),
            |(sign, digits): (Option<char>, &str)| -> Result<i64, <i64 as std::str::FromStr>::Err> {
                let num = str::parse::<i64>(digits)?;
                if sign.is_some() { Ok(-num) } else { Ok(num) }
            },
        )(s)
    }
}

impl<'a> Parse<'a> for &'a str {
    fn parse(s: &'a str) -> IResult<'a, &'a str> {
        alt((
            map(
                delimited(
                    char('\''),
                    opt(nom::bytes::complete::is_not("'")),
                    char('\''),
                ),
                |r| r.unwrap_or(""),
            ),
            // Enum tag like .SOMETHING.
            parse_enum_tag,
            // NUL REF
            map(char('$'), |_| ""),
        ))(s)
    }
}

/// A derived (computed) attribute marker `*` in STEP entity parameters.
#[derive(Debug, Copy, Clone)]
pub struct Derived;

impl Parse<'_> for Derived {
    fn parse(s: &str) -> IResult<'_, Self> {
        // Accept `*` (derived marker), `$` (null), or `#digits` (entity ref).
        // Complex entities may use any of these for inherited attributes.
        alt((
            map(char('*'), |_| Derived),
            map(char('$'), |_| Derived),
            map(preceded(char('#'), digit1), |_| Derived),
        ))(s)
    }
}

/// A raw SELECT-type value from a STEP entity parameter.
/// Handles enum tags (`.UNSPECIFIED.`), typed values (`LENGTH_MEASURE(0.0254)`),
/// quoted strings (`'text'`), null refs (`$`), and derived markers (`*`).
#[derive(Debug, Clone, Copy)]
pub struct Select<'a>(pub &'a str);

impl<'a> Parse<'a> for Select<'a> {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((
            map(<&str>::parse, Select),
            map(char('*'), |_| Select("*")),
            map(parse_select_raw, Select),
        ))(s)
    }
}

/// Captures raw text with balanced parentheses, stopping at `,` or `)` at depth 0.
fn parse_select_raw(s: &str) -> IResult<'_, &str> {
    let mut depth = 0usize;
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' | b',' if depth == 0 => {
                return if i == 0 {
                    nom_err(s, ErrorKind::Alpha)
                } else {
                    Ok((&s[i..], &s[..i]))
                };
            }
            b')' => depth -= 1,
            _ => {}
        }
    }
    if s.is_empty() {
        nom_err(s, ErrorKind::Alpha)
    } else {
        Ok(("", s))
    }
}

impl<'a, T: Parse<'a>> Parse<'a> for Vec<T> {
    fn parse(s: &'a str) -> IResult<'a, Vec<T>> {
        delimited(
            char('('),
            nom::multi::separated_list0(char(','), T::parse),
            char(')'),
        )(s)
    }
}

impl<'a, T: Parse<'a>, const CAP: usize> Parse<'a> for ArrayVec<T, CAP> {
    fn parse(s: &'a str) -> IResult<'a, ArrayVec<T, CAP>> {
        let (mut s, _) = char('(')(s)?;
        let mut out = ArrayVec::new();

        let (s_, o) = match T::parse(s) {
            Err(nom::Err::Error(_)) => return Ok((s, out)),
            e => e?,
        };
        s = s_;
        out.push(o);

        loop {
            let (s_, _) = match char::<&str, Error<&str>>(',')(s) {
                Err(nom::Err::Error(_)) => break,
                e => e?,
            };
            s = s_;
            let (s_, o) = match T::parse(s) {
                Err(nom::Err::Error(_)) => break,
                e => e?,
            };
            s = s_;
            out.push(o);
        }
        let (s, _) = char(')')(s)?;
        Ok((s, out))
    }
}

impl<'a, T: Parse<'a>> Parse<'a> for Option<T> {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((map(char('$'), |_| None), map(T::parse, Some)))(s)
    }
}

impl<'a> Parse<'a> for Logical {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((
            map(tag(".T."), |_| Logical(Some(true))),
            map(tag(".F."), |_| Logical(Some(false))),
            map(tag(".UNKNOWN."), |_| Logical(None)),
        ))(s)
    }
}

impl<'a> Parse<'a> for bool {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((map(tag(".T."), |_| true), map(tag(".F."), |_| false)))(s)
    }
}

impl<'a> Parse<'a> for usize {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((
            map_res(preceded(char('#'), digit1), |s: &str| s.parse::<usize>()),
            // NUL id deserializes to 0
            map(char('$'), |_| 0),
        ))(s)
    }
}

impl<'a, T> Parse<'a> for Id<T> {
    fn parse(s: &str) -> IResult<'_, Self> {
        alt((
            map_res(preceded(char('#'), digit1), |s: &str| {
                s.parse().map(Id::new)
            }),
            // NUL id deserializes to 0
            map(char('$'), |_| Id::empty()),
        ))(s)
    }
}

/// Parse an enum tag like .SOMETHING.
pub fn parse_enum_tag(s: &str) -> IResult<'_, &str> {
    delimited(
        char('.'),
        nom::bytes::complete::take_while(|c: char| {
            c == '_' || c.is_ascii_uppercase() || c.is_ascii_digit()
        }),
        char('.'),
    )(s)
}

/// Check if we need to advance to next chunk
fn check_str<'a>(s: &'a str, i: &mut usize, strs: &[&'a str]) -> &'a str {
    if s.is_empty() {
        *i += 1;
        strs.get(*i).copied().unwrap_or("")
    } else {
        s
    }
}

/// Parse a single attribute from a parameter list
pub fn param_from_chunks<'a, T: Parse<'a>>(
    last: bool,
    s: &'a str,
    i: &mut usize,
    strs: &[&'a str],
) -> IResult<'a, T> {
    let s = check_str(s, i, strs);
    let (s, out) = T::parse(s)?;
    let s = check_str(s, i, strs);
    let (s, _) = char(if last { ')' } else { ',' })(s)?;
    Ok((check_str(s, i, strs), out))
}

/// Preprocesses a STEP file, removing comments and whitespace.
pub fn strip_flatten(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        match data[i] {
            b'/' if i + 1 < data.len() && data[i + 1] == b'*' => {
                // Block comment
                for j in memchr_iter(b'/', &data[i + 2..]) {
                    if data[i + j + 1] == b'*' {
                        i += j + 2;
                        break;
                    }
                }
            }
            c if c.is_ascii_whitespace() => (),
            c => out.push(c),
        }
        i += 1;
    }
    out
}

/// Splits a preprocessed STEP file into individual entity blocks.
pub fn into_blocks(data: &[u8]) -> Vec<&[u8]> {
    let mut blocks = Vec::new();
    let mut i = 0;
    let mut start = 0;
    while i < data.len() {
        let next = memchr2(b'\'', b';', &data[i..]).unwrap_or(data.len() - i);
        if i + next >= data.len() {
            break;
        }
        match data[i + next] {
            b'\'' => {
                // Skip over quoted blocks
                if let Some(quote_end) = memchr(b'\'', &data[i + next + 1..]) {
                    i += next + quote_end + 2;
                } else {
                    break;
                }
            }
            b';' => {
                blocks.push(&data[start..=(i + next)]);
                i += next + 1;
                start = i;
            }
            _ => unreachable!(),
        }
    }
    blocks
}

/// Find the DATA section boundaries in a STEP file
pub fn find_data_section(blocks: &[&[u8]]) -> (usize, usize) {
    let data_start = blocks.iter().position(|b| *b == b"DATA;").unwrap_or(0) + 1;
    let data_end = blocks
        .iter()
        .skip(data_start)
        .position(|b| *b == b"ENDSEC;")
        .unwrap_or(0)
        + data_start;
    (data_start, data_end)
}

use super::ap214::{Entity, superclasses_of};
use memchr::memchr3;

/// Parse a complex entity mapping like `(ENTITY1()ENTITY2()...)`.
/// Complex entities are used when an instance satisfies multiple type constraints.
pub fn parse_complex_mapping(s: &str) -> IResult<'_, Entity<'_>> {
    // Map from sub-entity name to its argument string
    let mut subentities: HashMap<&str, &str> = HashMap::new();
    // Map from sub-entity name to the str slice with name + open paren
    let mut name_tags: HashMap<&str, &str> = HashMap::new();

    let bstr = s.as_bytes();
    let mut depth = 0;
    let mut index = 0;
    let mut args_start = 0;
    let mut name: &str = "";

    loop {
        let next = match memchr3(b'(', b')', b'\'', &bstr[index..]) {
            Some(i) => i,
            None => return nom_err(s, ErrorKind::Alt),
        };
        match bstr[index + next] {
            b'(' => {
                if depth == 1 {
                    let name_slice = &bstr[index..(index + next)];
                    name = std::str::from_utf8(name_slice).expect("Could not convert to name");
                    args_start = index + next + 1;
                    let name_tag_slice = &bstr[index..(index + next + 1)];
                    let name_tag =
                        std::str::from_utf8(name_tag_slice).expect("Could not convert tag");
                    name_tags.insert(name, name_tag);
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 1 {
                    let arg_slice = &bstr[args_start..(index + next)];
                    let args = std::str::from_utf8(arg_slice).expect("Could not convert args");
                    subentities.insert(name, args);
                } else if depth == 0 {
                    break;
                }
            }
            b'\'' => {
                // Skip quoted strings
                if let Some(j) = memchr(b'\'', &bstr[(index + next + 1)..]) {
                    index += j + 1;
                } else {
                    return nom_err(s, ErrorKind::Char);
                }
            }
            c => unreachable!("Invalid char: {}", c),
        }
        index += next + 1;
    }

    // Filter to leaf types (not parents of other items)
    let mut potential_leafs: HashSet<&str> = subentities.keys().copied().collect();
    for k in subentities.keys() {
        for sup in superclasses_of(k) {
            potential_leafs.remove(sup);
        }
    }
    // Remove leafs with no arguments
    potential_leafs.retain(|k| !subentities[k].is_empty());

    // Sort for determinism
    let mut potential_leafs: Vec<&str> = potential_leafs.into_iter().collect();
    potential_leafs.sort();

    // Build and parse leaf entities
    let mut leaf_entities = Vec::with_capacity(potential_leafs.len());
    for leaf in potential_leafs {
        let mut chain = vec![leaf];
        loop {
            let sup = superclasses_of(chain.last().unwrap());
            match sup.len() {
                0 => break,
                1 => chain.push(sup[0]),
                _ => return nom_err(s, ErrorKind::LengthValue),
            }
        }

        let mut new_decl: Vec<&str> = vec![name_tags.get(leaf).unwrap()];
        for c in chain.iter().rev() {
            if !subentities[c].is_empty() {
                new_decl.push(subentities[c]);
                new_decl.push(if *c == leaf { ")" } else { "," });
            }
        }
        leaf_entities.push(Entity::parse_chunks(&new_decl)?.1);
    }

    if leaf_entities.len() == 1 {
        Ok(("", leaf_entities.pop().unwrap()))
    } else {
        Ok(("", Entity::ComplexEntity(leaf_entities)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_f64() {
        let (remaining, value) = f64::parse("123.456,").unwrap();
        assert!((value - 123.456).abs() < 1e-10);
        assert_eq!(remaining, ",");

        let (remaining, value) = f64::parse("-1.5E-3)").unwrap();
        assert!((value - (-0.0015)).abs() < 1e-10);
        assert_eq!(remaining, ")");
    }

    #[test]
    fn test_parse_i64() {
        let (remaining, value) = i64::parse("42,").unwrap();
        assert_eq!(value, 42);
        assert_eq!(remaining, ",");

        let (remaining, value) = i64::parse("-100)").unwrap();
        assert_eq!(value, -100);
        assert_eq!(remaining, ")");
    }

    #[test]
    fn test_parse_string() {
        let (remaining, value) = <&str>::parse("'hello',").unwrap();
        assert_eq!(value, "hello");
        assert_eq!(remaining, ",");

        let (remaining, value) = <&str>::parse("$,").unwrap();
        assert_eq!(value, "");
        assert_eq!(remaining, ",");
    }

    #[test]
    fn test_parse_vec() {
        let (remaining, value) = <Vec<f64>>::parse("(1.0,2.0,3.0),").unwrap();
        assert_eq!(value, vec![1.0, 2.0, 3.0]);
        assert_eq!(remaining, ",");
    }

    #[test]
    fn test_parse_id() {
        let (remaining, value) = <Id<()>>::parse("#123,").unwrap();
        assert_eq!(value.0, 123);
        assert_eq!(remaining, ",");

        let (remaining, value) = <Id<()>>::parse("$,").unwrap();
        assert_eq!(value.0, 0);
        assert_eq!(remaining, ",");
    }

    #[test]
    fn test_parse_usize_ref() {
        let (remaining, value) = usize::parse("#123,").unwrap();
        assert_eq!(value, 123);
        assert_eq!(remaining, ",");

        let (remaining, value) = usize::parse("$,").unwrap();
        assert_eq!(value, 0);
        assert_eq!(remaining, ",");
    }

    #[test]
    fn test_parse_bool() {
        let (remaining, value) = bool::parse(".T.,").unwrap();
        assert!(value);
        assert_eq!(remaining, ",");

        let (remaining, value) = bool::parse(".F.)").unwrap();
        assert!(!value);
        assert_eq!(remaining, ")");
    }

    #[test]
    fn test_parse_select() {
        // Enum tag
        let (r, v) = Select::parse(".UNSPECIFIED.,").unwrap();
        assert_eq!(v.0, "UNSPECIFIED");
        assert_eq!(r, ",");

        // Function-call syntax (the key new case)
        let (r, v) = Select::parse("LENGTH_MEASURE(0.0254),#6050)").unwrap();
        assert_eq!(v.0, "LENGTH_MEASURE(0.0254)");
        assert_eq!(r, ",#6050)");

        // Null ref
        let (r, v) = Select::parse("$,").unwrap();
        assert_eq!(v.0, "");
        assert_eq!(r, ",");

        // Derived marker
        let (r, v) = Select::parse("*,").unwrap();
        assert_eq!(v.0, "*");
        assert_eq!(r, ",");

        // Quoted string
        let (r, v) = Select::parse("'hello',").unwrap();
        assert_eq!(v.0, "hello");
        assert_eq!(r, ",");

        // Nested parens
        let (r, v) = Select::parse("A(B(1),C(2)),next").unwrap();
        assert_eq!(v.0, "A(B(1),C(2))");
        assert_eq!(r, ",next");
    }

    #[test]
    fn test_parse_derived() {
        // Standard derived marker
        let (r, _) = Derived::parse("*,").unwrap();
        assert_eq!(r, ",");

        // Entity reference (common in complex entities)
        let (r, _) = Derived::parse("#5184,").unwrap();
        assert_eq!(r, ",");

        // Null
        let (r, _) = Derived::parse("$,").unwrap();
        assert_eq!(r, ",");
    }

    #[test]
    fn test_strip_flatten() {
        let input = b"/* comment */ DATA;\n  #1=TEST('hello');\nENDSEC;";
        let output = strip_flatten(input);
        assert_eq!(output, b"DATA;#1=TEST('hello');ENDSEC;");
    }

    #[test]
    fn test_into_blocks() {
        let input = b"DATA;#1=TEST('hello');ENDSEC;";
        let blocks = into_blocks(input);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], b"DATA;");
        assert_eq!(blocks[1], b"#1=TEST('hello');");
        assert_eq!(blocks[2], b"ENDSEC;");
    }
}
