//! Logical Layer
//!
//! This crate provides the logical abstraction layer between storage and the rest
//! of the system. It re-exports storage types and provides Datalog integration,
//! as well as the Runtime trait for trigger function execution.

// Datalog adapter module
pub mod datalog_context;

// Runtime module for trigger execution
pub mod runtime;

// Trigger-aware operations
pub mod operations;

// Re-export everything from storage
pub use storage::*;

// Re-export Datalog types
pub use datalog_context::{sql_value_to_term, DatalogContext, InsertError, PredicateSchema};

// Re-export Runtime types
pub use runtime::{NoOpRuntime, Runtime, RuntimeError, TriggerContext, TriggerResult};

// Re-export trigger-aware operations
pub use operations::{delete, insert, update, OperationError};
