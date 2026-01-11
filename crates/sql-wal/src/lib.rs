//! Write-Ahead Log (WAL) for durability
//!
//! The WAL ensures that database operations are durable by writing
//! them to disk before applying them to the database.

mod wal;

pub use wal::{LogEntry, Operation, Wal, WalError, WalResult};
