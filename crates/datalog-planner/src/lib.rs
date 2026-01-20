pub mod safety;
pub mod stratification;

// Re-export parser types needed by datalog-eval
pub use datalog_parser::{
    Atom, ComparisonOp, Constraint, Literal, Query, Rule, Symbol, Term, Value,
};

// Re-export safety
pub use safety::{check_program_safety, check_rule_safety, SafetyError};

// Re-export stratification
pub use stratification::{stratify, Stratification, StratificationError};
