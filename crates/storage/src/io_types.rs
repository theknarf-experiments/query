//! I/O data types for import/export operations
//!
//! These types are used to transfer data between the storage engine
//! and external formats (CSV, JSON, etc.)

use crate::Value;

/// Data for export operations (columns + rows)
#[derive(Debug, Clone)]
pub struct ExportData {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Data from import operations (columns + rows)
#[derive(Debug, Clone)]
pub struct ImportData {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}
