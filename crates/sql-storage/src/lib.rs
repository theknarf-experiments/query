//! SQL Storage Engine Trait and Implementations
//!
//! This crate defines the storage engine interface and provides implementations.

mod engine;
mod memory;
mod value;

pub use engine::{
    ColumnSchema, DataType, Row, StorageEngine, StorageError, StorageResult, TableSchema,
};
pub use memory::MemoryEngine;
pub use value::Value;
