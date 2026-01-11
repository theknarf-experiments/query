//! SQL Storage Engine Trait and Implementations
//!
//! This crate defines the storage engine interface and provides implementations.

mod engine;
mod memory;
mod value;

pub use engine::{
    ColumnSchema, DataType, ForeignKeyRef, ReferentialAction, Row, StorageEngine, StorageError,
    StorageResult, TableConstraint, TableSchema,
};
pub use memory::MemoryEngine;
pub use value::Value;
