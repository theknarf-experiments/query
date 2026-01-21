//! Logical Layer
//!
//! This crate provides the logical abstraction layer between storage and the rest
//! of the system. It re-exports storage types and provides the Runtime trait for
//! trigger function execution, as well as trigger-aware operations.

// Runtime module for trigger execution
pub mod runtime;

// Trigger-aware operations
pub mod operations;

// Re-export everything from storage
pub use storage::*;

// Re-export Runtime types
pub use runtime::{NoOpRuntime, Runtime, RuntimeError, TriggerContext, TriggerResult};

// Re-export trigger-aware operations
pub use operations::{delete, insert, update, OperationError};
