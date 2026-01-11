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
    pub unique: bool,
    pub default: Option<crate::Value>,
    pub references: Option<ForeignKeyRef>,
}

/// Foreign key reference information
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignKeyRef {
    pub table: String,
    pub column: String,
    pub on_delete: ReferentialAction,
    pub on_update: ReferentialAction,
}

/// Actions for referential integrity
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReferentialAction {
    #[default]
    NoAction,
    Cascade,
    SetNull,
    SetDefault,
    Restrict,
}

/// Data types supported by the storage engine
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Int,
    Float,
    Text,
    Bool,
}

/// Table constraint
#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    PrimaryKey {
        columns: Vec<String>,
    },
    ForeignKey {
        columns: Vec<String>,
        references_table: String,
        references_columns: Vec<String>,
        on_delete: ReferentialAction,
        on_update: ReferentialAction,
    },
    Unique {
        columns: Vec<String>,
    },
}

/// Schema for a table
#[derive(Debug, Clone, PartialEq)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnSchema>,
    pub constraints: Vec<TableConstraint>,
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

    /// Get all table names
    fn table_names(&self) -> Vec<String>;

    /// Insert a row into a table
    fn insert(&mut self, table: &str, row: Row) -> StorageResult<()>;

    /// Scan all rows from a table
    fn scan(&self, table: &str) -> StorageResult<Vec<Row>>;

    /// Delete rows matching a predicate
    fn delete<F>(&mut self, table: &str, predicate: F) -> StorageResult<usize>
    where
        F: Fn(&Row) -> bool;

    /// Update rows matching a predicate
    fn update<F, U>(&mut self, table: &str, predicate: F, updater: U) -> StorageResult<usize>
    where
        F: Fn(&Row) -> bool,
        U: Fn(&mut Row);

    /// Add a column to a table
    fn add_column(
        &mut self,
        table: &str,
        column: ColumnSchema,
        default: Value,
    ) -> StorageResult<()>;

    /// Drop a column from a table
    fn drop_column(&mut self, table: &str, column: &str) -> StorageResult<()>;

    /// Rename a column
    fn rename_column(&mut self, table: &str, old_name: &str, new_name: &str) -> StorageResult<()>;

    /// Rename a table
    fn rename_table(&mut self, old_name: &str, new_name: &str) -> StorageResult<()>;
}
