//! Storage Engine trait definition

use crate::Value;

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Storage engine errors
#[derive(Debug, Clone, PartialEq)]
pub enum StorageError {
    TableNotFound(String),
    TableAlreadyExists(String),
    ColumnNotFound(String),
    TypeMismatch { expected: String, found: String },
    ConstraintViolation(String),
}

/// Schema for a table column
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
}

/// Data types supported by the storage engine
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Int,
    Float,
    Text,
    Bool,
}

/// Schema for a table
#[derive(Debug, Clone, PartialEq)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnSchema>,
}

/// A row of data
pub type Row = Vec<Value>;

/// Trait for pluggable storage engines
pub trait StorageEngine: Send + Sync {
    /// Create a new table
    fn create_table(&mut self, schema: TableSchema) -> StorageResult<()>;

    /// Drop a table
    fn drop_table(&mut self, name: &str) -> StorageResult<()>;

    /// Get table schema
    fn get_schema(&self, name: &str) -> StorageResult<&TableSchema>;

    /// Insert a row into a table
    fn insert(&mut self, table: &str, row: Row) -> StorageResult<()>;

    /// Scan all rows from a table
    fn scan(&self, table: &str) -> StorageResult<Vec<Row>>;

    /// Delete rows matching a predicate
    fn delete<F>(&mut self, table: &str, predicate: F) -> StorageResult<usize>
    where
        F: Fn(&Row) -> bool;
}
