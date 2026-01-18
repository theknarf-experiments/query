//! SQL Storage Engine Trait and Implementations
//!
//! This crate defines the storage engine interface and provides implementations.
//! It also provides Datalog fact storage capabilities.

mod engine;
mod memory;
mod value;

// Datalog storage modules (from datalog-core)
pub mod datalog_constants;
pub mod datalog_context;
pub mod datalog_unification;
pub mod delta_tracker;

pub use engine::{
    ColumnSchema, DataType, ForeignKeyRef, IndexInfo, ReferentialAction, Row, StorageEngine,
    StorageError, StorageResult, TableConstraint, TableSchema,
};
pub use memory::MemoryEngine;
pub use value::{DateValue, TimeValue, TimestampValue, Value};

// Re-export Datalog storage types
pub use datalog_constants::ConstantEnv;
pub use datalog_context::{
    atom_to_row, create_derived_schema, ensure_derived_table, row_to_atom, DatalogContext,
    InsertError, InsertOutcome, PredicateSchema,
};
// Backwards compatibility alias
#[allow(deprecated)]
pub use datalog_context::FactDatabase;
pub use datalog_unification::{unify, unify_atoms, Substitution};
pub use delta_tracker::DeltaTracker;
