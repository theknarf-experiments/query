//! SQL Parser using Chumsky
//!
//! This crate parses token streams into an AST.

pub mod ast;
pub mod parser;

pub use ast::*;
pub use parser::{parse, ParseError, ParseResult};
