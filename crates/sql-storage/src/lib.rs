//! SQL Storage Engine Trait and Implementations
//!
//! This crate defines the storage engine interface and provides implementations.
//! It also provides Datalog fact storage capabilities.

mod engine;
mod memory;
mod value;

// Datalog storage modules (from datalog-core)
pub mod datalog_constants;
pub mod datalog_database;
pub mod datalog_unification;

pub use engine::{
    ColumnSchema, DataType, ForeignKeyRef, ReferentialAction, Row, StorageEngine, StorageError,
    StorageResult, TableConstraint, TableSchema,
};
pub use memory::MemoryEngine;
pub use value::{DateValue, TimeValue, TimestampValue, Value};

// Re-export Datalog storage types
pub use datalog_constants::ConstantEnv;
pub use datalog_database::{FactDatabase, InsertError, PredicateSchema};
pub use datalog_unification::{unify, unify_atoms, Substitution};
