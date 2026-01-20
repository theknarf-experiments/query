//! SQL Engine
//!
//! This crate provides the SQL execution engine including query execution,
//! transactions, triggers, views, and procedures. It also integrates Datalog
//! query support.

pub mod datalog;
mod executor;
mod runtime;

pub use executor::{Engine, ExecError, ExecResult, QueryResult};
pub use runtime::SqlRuntime;

// Re-export Datalog types
pub use datalog::DatalogError;
