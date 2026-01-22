//! Query Engine
//!
//! An embeddable query engine supporting both SQL and Datalog.
//! Designed to be integrated into applications that need to provide
//! query capabilities over their own data.
//!
//! # Features
//!
//! - **SQL support**: Full SQL query parsing and execution
//! - **Datalog support**: Logic programming queries with recursion and negation
//! - **Pluggable storage**: Bring your own storage backend via the `StorageEngine` trait
//! - **In-memory engine included**: `MemoryEngine` for quick prototyping
//!
//! # Example
//!
//! ```ignore
//! use query::Engine;
//!
//! let mut engine = Engine::new();
//! engine.execute("CREATE TABLE users (id INT, name TEXT)");
//! engine.execute("INSERT INTO users VALUES (1, 'alice')");
//!
//! let result = engine.execute("SELECT * FROM users")?;
//! ```

pub mod io;

// Re-export sql-engine as the main interface
pub use sql_engine::{datalog, Engine, ExecError, ExecResult, QueryResult, SqlRuntime};

// Re-export I/O functionality
pub use io::{export_csv, export_json, import_csv, import_json, ExportError, ImportError};
pub use logical::{ExportData, ImportData};

// Re-export underlying crates for advanced usage
pub use logical;
pub use sql_parser;
pub use sql_planner;
