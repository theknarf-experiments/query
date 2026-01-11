//! In-memory storage engine implementation

use crate::engine::{Row, StorageEngine, StorageError, StorageResult, TableSchema};
use std::collections::HashMap;

/// In-memory storage engine
#[derive(Debug, Default)]
pub struct MemoryEngine {
    tables: HashMap<String, TableData>,
}

#[derive(Debug)]
struct TableData {
    schema: TableSchema,
    rows: Vec<Row>,
}

impl MemoryEngine {
    /// Create a new in-memory storage engine
    pub fn new() -> Self {
        Self::default()
    }
}

impl StorageEngine for MemoryEngine {
    fn create_table(&mut self, schema: TableSchema) -> StorageResult<()> {
        if self.tables.contains_key(&schema.name) {
            return Err(StorageError::TableAlreadyExists(schema.name.clone()));
        }
        let name = schema.name.clone();
        self.tables.insert(
            name,
            TableData {
                schema,
                rows: Vec::new(),
            },
        );
        Ok(())
    }

    fn drop_table(&mut self, name: &str) -> StorageResult<()> {
        self.tables
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| StorageError::TableNotFound(name.to_string()))
    }

    fn get_schema(&self, name: &str) -> StorageResult<&TableSchema> {
        self.tables
            .get(name)
            .map(|t| &t.schema)
            .ok_or_else(|| StorageError::TableNotFound(name.to_string()))
    }

    fn insert(&mut self, table: &str, row: Row) -> StorageResult<()> {
        let table_data = self
            .tables
            .get_mut(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;
        table_data.rows.push(row);
        Ok(())
    }

    fn scan(&self, table: &str) -> StorageResult<Vec<Row>> {
        self.tables
            .get(table)
            .map(|t| t.rows.clone())
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))
    }

    fn delete<F>(&mut self, table: &str, predicate: F) -> StorageResult<usize>
    where
        F: Fn(&Row) -> bool,
    {
        let table_data = self
            .tables
            .get_mut(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;

        let original_len = table_data.rows.len();
        table_data.rows.retain(|row| !predicate(row));
        Ok(original_len - table_data.rows.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ColumnSchema, DataType};
    use crate::Value;

    fn create_users_schema() -> TableSchema {
        TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: Vec::new(),
        }
    }

    #[test]
    fn test_create_table() {
        let mut engine = MemoryEngine::new();
        let schema = create_users_schema();
        assert!(engine.create_table(schema).is_ok());
    }

    #[test]
    fn test_create_duplicate_table() {
        let mut engine = MemoryEngine::new();
        let schema = create_users_schema();
        engine.create_table(schema.clone()).unwrap();
        assert_eq!(
            engine.create_table(schema),
            Err(StorageError::TableAlreadyExists("users".to_string()))
        );
    }

    #[test]
    fn test_drop_table() {
        let mut engine = MemoryEngine::new();
        let schema = create_users_schema();
        engine.create_table(schema).unwrap();
        assert!(engine.drop_table("users").is_ok());
        assert_eq!(
            engine.get_schema("users"),
            Err(StorageError::TableNotFound("users".to_string()))
        );
    }

    #[test]
    fn test_insert_and_scan() {
        let mut engine = MemoryEngine::new();
        let schema = create_users_schema();
        engine.create_table(schema).unwrap();

        let row1 = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let row2 = vec![Value::Int(2), Value::Text("Bob".to_string())];

        engine.insert("users", row1.clone()).unwrap();
        engine.insert("users", row2.clone()).unwrap();

        let rows = engine.scan("users").unwrap();
        assert_eq!(rows, vec![row1, row2]);
    }

    #[test]
    fn test_delete() {
        let mut engine = MemoryEngine::new();
        let schema = create_users_schema();
        engine.create_table(schema).unwrap();

        engine
            .insert(
                "users",
                vec![Value::Int(1), Value::Text("Alice".to_string())],
            )
            .unwrap();
        engine
            .insert("users", vec![Value::Int(2), Value::Text("Bob".to_string())])
            .unwrap();
        engine
            .insert(
                "users",
                vec![Value::Int(3), Value::Text("Charlie".to_string())],
            )
            .unwrap();

        let deleted = engine
            .delete("users", |row| matches!(&row[0], Value::Int(id) if *id == 2))
            .unwrap();

        assert_eq!(deleted, 1);
        let rows = engine.scan("users").unwrap();
        assert_eq!(rows.len(), 2);
    }
}
