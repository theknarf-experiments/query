//! Logical Layer
//!
//! This crate provides the logical abstraction layer between storage and the rest
//! of the system. It re-exports storage types and will contain additional logic
//! in the future.

// Re-export everything from storage
pub use storage::*;
