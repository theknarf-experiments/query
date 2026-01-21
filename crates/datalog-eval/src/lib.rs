pub mod builtins;
pub mod datalog_context;
pub mod datalog_unification;
pub mod delta_tracker;
pub mod evaluation;
pub mod grounding;
pub mod query;

// Re-export unification types
pub use datalog_unification::{unify, unify_atoms, Substitution};

// Re-export builtins
pub use builtins::*;

// Re-export delta tracker
pub use delta_tracker::DeltaTracker;

// Re-export grounding
pub use grounding::*;

// Re-export evaluation
pub use evaluation::*;

// Re-export query
pub use query::*;

// Re-export datalog context types
pub use datalog_context::{
    sql_value_to_term, DatalogContext, InsertError, InsertOutcome, PredicateSchema,
};
