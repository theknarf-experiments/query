//! Datalog Planner
//!
//! This crate converts parsed Datalog AST into a planned IR ready for evaluation.
//! It performs safety checking, stratification analysis, and produces an optimized
//! representation for the evaluator.

mod convert;
pub mod ir;
pub mod safety;
pub mod stratification;

// Re-export the main planning function
pub use convert::{plan_program, PlanError};

// Re-export IR types for use by datalog-eval
// These are the primary types that downstream crates should use
pub use ir::{
    Atom, BuiltIn, Comparison, ComparisonOp, Literal, PlannedConstraint, PlannedProgram,
    PlannedQuery, PlannedRule, PlannedStratum, Symbol, Term, Value,
};

// Provide simpler aliases for common types
pub type Rule = PlannedRule;
pub type Constraint = PlannedConstraint;
pub type Query = PlannedQuery;

// Re-export safety
pub use safety::{check_program_safety, check_rule_safety, SafetyError};

// Re-export stratification
pub use stratification::{stratify, Stratification, StratificationError};

// Re-export parser's parse function, SrcId, and Statement for parsing
pub use datalog_parser::{parse_program, SrcId, Statement};

// Re-export parser AST types needed for working with parsed facts
// These are prefixed with Ast to distinguish from IR types
pub use datalog_parser::{Atom as AstAtom, Fact as AstFact, Term as AstTerm, Value as AstValue};
