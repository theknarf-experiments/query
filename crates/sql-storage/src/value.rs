//! SQL Value types

use serde::{Deserialize, Serialize};

/// A runtime SQL value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl Value {
    /// Check if the value is null
    pub const fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}
