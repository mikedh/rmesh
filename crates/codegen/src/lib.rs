#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::panic)]
#![allow(dead_code)]
#![allow(mismatched_lifetime_syntaxes)]

mod generator;
mod parse;

pub use generator::generate;
pub use parse::{parse_express, strip_comments_and_lower};
