//! In-memory storage engine implementation

use crate::engine::{ColumnSchema, Row, StorageEngine, StorageError, StorageResult, TableSchema};
use std::collections::HashMap;

/// In-memory storage engine
#[derive(Debug, Default, Clone)]
pub struct MemoryEngine {
    tables: HashMap<String, TableData>,
}

#[derive(Debug, Clone)]
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

    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    fn update<F, U>(&mut self, table: &str, predicate: F, updater: U) -> StorageResult<usize>
    where
        F: Fn(&Row) -> bool,
        U: Fn(&mut Row),
    {
        let table_data = self
            .tables
            .get_mut(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;

        let mut count = 0;
        for row in &mut table_data.rows {
            if predicate(row) {
                updater(row);
                count += 1;
            }
        }
        Ok(count)
    }

    fn add_column(
        &mut self,
        table: &str,
        column: ColumnSchema,
        default: crate::Value,
    ) -> StorageResult<()> {
        let table_data = self
            .tables
            .get_mut(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;

        // Add column to schema
        table_data.schema.columns.push(column);

        // Add default value to all existing rows
        for row in &mut table_data.rows {
            row.push(default.clone());
        }

        Ok(())
    }

    fn drop_column(&mut self, table: &str, column: &str) -> StorageResult<()> {
        let table_data = self
            .tables
            .get_mut(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;

        // Find column index
        let col_idx = table_data
            .schema
            .columns
            .iter()
            .position(|c| c.name == column)
            .ok_or_else(|| StorageError::ColumnNotFound(column.to_string()))?;

        // Remove column from schema
        table_data.schema.columns.remove(col_idx);

        // Remove value from all existing rows
        for row in &mut table_data.rows {
            row.remove(col_idx);
        }

        Ok(())
    }

    fn rename_column(&mut self, table: &str, old_name: &str, new_name: &str) -> StorageResult<()> {
        let table_data = self
            .tables
            .get_mut(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;

        // Find column and rename
        let col = table_data
            .schema
            .columns
            .iter_mut()
            .find(|c| c.name == old_name)
            .ok_or_else(|| StorageError::ColumnNotFound(old_name.to_string()))?;

        col.name = new_name.to_string();
        Ok(())
    }

    fn rename_table(&mut self, old_name: &str, new_name: &str) -> StorageResult<()> {
        let mut table_data = self
            .tables
            .remove(old_name)
            .ok_or_else(|| StorageError::TableNotFound(old_name.to_string()))?;

        if self.tables.contains_key(new_name) {
            // Restore old table and return error
            self.tables.insert(old_name.to_string(), table_data);
            return Err(StorageError::TableAlreadyExists(new_name.to_string()));
        }

        table_data.schema.name = new_name.to_string();
        self.tables.insert(new_name.to_string(), table_data);
        Ok(())
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
