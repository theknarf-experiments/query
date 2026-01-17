//! Parser for Datalog programs
//!
//! This crate implements a parser combinator-based parser using the Chumsky library.
//! It parses Datalog programs from text into an AST (Abstract Syntax Tree).
//!
//! # Supported Syntax
//!
//! - **Facts**: `parent(john, mary).`
//! - **Rules**: `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`
//! - **Constraints**: `:- unsafe(X).`
//! - **Choice Rules**: `{ selected(X) : item(X) } 2.`
//! - **Constants**: `#const max_items = 10.`
//! - **Ranges**: `cell(1..width, 1..height).`
//! - **Built-ins**: Arithmetic (`X + Y = Z`) and comparisons (`X > 5`)
//!
//! # Example
//!
//! ```ignore
//! use datalog_parser::{parse_program, SrcId};
//!
//! let program_text = "parent(john, mary). ancestor(X, Z) :- parent(X, Z).";
//! let program = parse_program(program_text, SrcId::empty()).expect("Parse error");
//! ```

pub mod ast;
mod parser;
mod span;
mod src;
mod token;

// Re-export AST types at crate root for convenience
pub use ast::{
    Atom, ComparisonLiteral, ComparisonOp, Constraint, Fact, Literal, Program, Query, Rule,
    Statement, Symbol, Term, Value,
};
pub use parser::{parse_program, parse_query, ParseError};
pub use span::Span;
pub use src::SrcId;
pub use token::{Keyword, LexError, Token};
