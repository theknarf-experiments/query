//! Logical Layer
//!
//! This crate provides the logical abstraction layer between storage and the rest
//! of the system. It re-exports storage types and provides Datalog integration,
//! as well as the Runtime trait for trigger function execution.

// Datalog adapter modules
pub mod datalog_constants;
pub mod datalog_context;
pub mod datalog_unification;
pub mod delta_tracker;

// Runtime module for trigger execution
pub mod runtime;

// Trigger-aware operations
pub mod operations;

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

// Re-export Runtime types
pub use runtime::{NoOpRuntime, Runtime, RuntimeError, TriggerContext, TriggerResult};

// Re-export trigger-aware operations
pub use operations::{
    delete_with_triggers, insert_with_triggers, update_with_triggers, OperationError,
    OperationResult,
};
