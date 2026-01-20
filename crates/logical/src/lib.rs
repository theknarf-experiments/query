//! Logical Layer
//!
//! This crate provides the logical abstraction layer between storage and the rest
//! of the system. It re-exports storage types and provides Datalog integration.

// Datalog adapter modules
pub mod datalog_constants;
pub mod datalog_context;
pub mod datalog_unification;
pub mod delta_tracker;

// Re-export everything from storage
pub use storage::*;

// Re-export Datalog types
pub use datalog_constants::ConstantEnv;
pub use datalog_context::{
    atom_to_row, create_derived_schema, ensure_derived_table, row_to_atom, DatalogContext,
    InsertError, InsertOutcome, PredicateSchema,
};
pub use datalog_unification::{unify, unify_atoms, Substitution};
pub use delta_tracker::DeltaTracker;
