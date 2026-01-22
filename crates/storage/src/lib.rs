//! Storage Engine Trait and Implementations
//!
//! This crate defines the storage engine interface and provides implementations.

mod engine;
mod io_types;
mod memory;
pub mod metadata;
mod value;

pub use engine::{
    ColumnSchema, DataType, ForeignKeyRef, IndexInfo, ReferentialAction, Row, StorageEngine,
    StorageError, StorageResult, TableConstraint, TableSchema,
};
pub use io_types::{ExportData, ImportData};
pub use memory::MemoryEngine;
pub use metadata::{
    FunctionDef, TriggerDef, TriggerEvent, TriggerTiming, events_from_json, events_to_json,
};
pub use value::{DateValue, JsonParseError, JsonValue, TimeValue, TimestampValue, Value};
