//! SQL Parser using Chumsky
//!
//! This crate provides a two-phase SQL parsing pipeline:
//! 1. Lexing: Converting raw SQL text into tokens
//! 2. Parsing: Converting tokens into an Abstract Syntax Tree (AST)

pub mod ast;
pub mod lexer;
pub mod parser;

// Re-export AST types
pub use ast::*;

// Re-export lexer types
pub use lexer::{FloatBits, Keyword, LexError, LexResult, Span, Token, lexer};

// Re-export parser types
pub use parser::{ParseError, ParseResult, parse};
