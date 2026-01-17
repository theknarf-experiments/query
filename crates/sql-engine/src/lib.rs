//! SQL Engine
//!
//! This crate ties together all the SQL components into a complete database engine.
//! It supports both SQL and Datalog queries against the same underlying tables.

pub mod datalog;
mod executor;
pub mod io;

pub use executor::{Engine, ExecError, ExecResult, QueryResult};
pub use io::{
    export_csv, export_json, import_csv, import_json, ExportData, ExportError, ImportData,
    ImportError,
};
pub use sql_parser;
pub use sql_planner;
pub use sql_storage;

// Re-export Datalog types
pub use datalog::DatalogError;
