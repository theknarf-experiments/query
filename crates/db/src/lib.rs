//! Database Engine
//!
//! This is the top-level crate that ties together all database components.
//! It supports both SQL and Datalog queries against the same underlying tables.
//!
//! # Example
//!
//! ```ignore
//! use db::Engine;
//!
//! let mut engine = Engine::new();
//! engine.execute("CREATE TABLE users (id INT, name TEXT)");
//! engine.execute("INSERT INTO users VALUES (1, 'alice')");
//!
//! let result = engine.execute("SELECT * FROM users")?;
//! ```

pub mod io;

// Re-export sql-engine as the main interface
pub use sql_engine::{datalog, Engine, ExecError, ExecResult, QueryResult};

// Re-export I/O functionality
pub use io::{export_csv, export_json, import_csv, import_json, ExportError, ImportError};
pub use sql_storage::{ExportData, ImportData};

// Re-export underlying crates for advanced usage
pub use sql_parser;
pub use sql_planner;
pub use sql_storage;
