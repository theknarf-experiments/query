//! In-memory storage engine implementation

use crate::engine::{ColumnSchema, Row, StorageEngine, StorageError, StorageResult, TableSchema};
use crate::Value;
use std::collections::HashMap;

/// In-memory storage engine
#[derive(Debug, Default, Clone)]
pub struct MemoryEngine {
    tables: HashMap<String, TableData>,
    indexes: HashMap<String, IndexData>,
}

#[derive(Debug, Clone)]
struct TableData {
    schema: TableSchema,
    rows: Vec<Row>,
}

/// Index structure - maps values to row indices
#[derive(Debug, Clone)]
struct IndexData {
    table: String,
    column: String,
    column_idx: usize,
    /// Map from value hash to list of row indices
    entries: HashMap<u64, Vec<usize>>,
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
        let row_idx = table_data.rows.len();
        table_data.rows.push(row.clone());

        // Update indexes for this table
        for index in self.indexes.values_mut() {
            if index.table == table {
                if let Some(val) = row.get(index.column_idx) {
                    let hash = value_hash(val);
                    index.entries.entry(hash).or_default().push(row_idx);
                }
            }
        }

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
        let deleted_count = original_len - table_data.rows.len();

        // Rebuild indexes for this table since row positions have shifted
        if deleted_count > 0 {
            let table_data = self.tables.get(table).unwrap();
            for index in self.indexes.values_mut() {
                if index.table == table {
                    index.entries.clear();
                    for (row_idx, row) in table_data.rows.iter().enumerate() {
                        if let Some(val) = row.get(index.column_idx) {
                            let hash = value_hash(val);
                            index.entries.entry(hash).or_default().push(row_idx);
                        }
                    }
                }
            }
        }

        Ok(deleted_count)
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

    fn create_index(&mut self, table: &str, column: &str, name: &str) -> StorageResult<()> {
        // Check if index already exists
        if self.indexes.contains_key(name) {
            return Err(StorageError::IndexAlreadyExists(name.to_string()));
        }

        // Get table and find column index
        let table_data = self
            .tables
            .get(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;

        let column_idx = table_data
            .schema
            .columns
            .iter()
            .position(|c| c.name == column)
            .ok_or_else(|| StorageError::ColumnNotFound(column.to_string()))?;

        // Build the index entries from existing rows
        let mut entries: HashMap<u64, Vec<usize>> = HashMap::new();
        for (row_idx, row) in table_data.rows.iter().enumerate() {
            if let Some(val) = row.get(column_idx) {
                let hash = value_hash(val);
                entries.entry(hash).or_default().push(row_idx);
            }
        }

        self.indexes.insert(
            name.to_string(),
            IndexData {
                table: table.to_string(),
                column: column.to_string(),
                column_idx,
                entries,
            },
        );

        Ok(())
    }

    fn drop_index(&mut self, name: &str) -> StorageResult<()> {
        self.indexes
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| StorageError::IndexNotFound(name.to_string()))
    }

    fn index_lookup(&self, table: &str, column: &str, value: &Value) -> Option<Vec<usize>> {
        // Find index for this table/column
        for index in self.indexes.values() {
            if index.table == table && index.column == column {
                let hash = value_hash(value);
                return index.entries.get(&hash).cloned();
            }
        }
        None
    }

    fn has_index(&self, table: &str, column: &str) -> bool {
        self.indexes
            .values()
            .any(|idx| idx.table == table && idx.column == column)
    }

    fn get_rows_by_indices(&self, table: &str, indices: &[usize]) -> StorageResult<Vec<Row>> {
        let table_data = self
            .tables
            .get(table)
            .ok_or_else(|| StorageError::TableNotFound(table.to_string()))?;
        Ok(indices
            .iter()
            .filter_map(|&idx| table_data.rows.get(idx).cloned())
            .collect())
    }
}

/// Hash a Value for index lookup
fn value_hash(v: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    match v {
        Value::Null => 0u8.hash(&mut hasher),
        Value::Bool(b) => {
            1u8.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        Value::Int(i) => {
            2u8.hash(&mut hasher);
            i.hash(&mut hasher);
        }
        Value::Float(f) => {
            3u8.hash(&mut hasher);
            f.to_bits().hash(&mut hasher);
        }
        Value::Text(s) => {
            4u8.hash(&mut hasher);
            s.hash(&mut hasher);
        }
        Value::Date(d) => {
            5u8.hash(&mut hasher);
            d.year.hash(&mut hasher);
            d.month.hash(&mut hasher);
            d.day.hash(&mut hasher);
        }
        Value::Time(t) => {
            6u8.hash(&mut hasher);
            t.hour.hash(&mut hasher);
            t.minute.hash(&mut hasher);
            t.second.hash(&mut hasher);
            t.microsecond.hash(&mut hasher);
        }
        Value::Timestamp(ts) => {
            7u8.hash(&mut hasher);
            ts.date.year.hash(&mut hasher);
            ts.date.month.hash(&mut hasher);
            ts.date.day.hash(&mut hasher);
            ts.time.hour.hash(&mut hasher);
            ts.time.minute.hash(&mut hasher);
            ts.time.second.hash(&mut hasher);
            ts.time.microsecond.hash(&mut hasher);
        }
    }
    hasher.finish()
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

    #[test]
    fn test_delete_rebuilds_index() {
        // This test verifies that DELETE correctly rebuilds indexes.
        // After deleting a row, the index entries should point to correct positions.
        let mut engine = MemoryEngine::new();
        let schema = create_users_schema();
        engine.create_table(schema).unwrap();

        // Insert three rows: Alice (0), Bob (1), Charlie (2)
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

        // Create index on name column
        engine.create_index("users", "name", "idx_name").unwrap();

        // Verify index works before delete
        let alice_indices = engine
            .index_lookup("users", "name", &Value::Text("Alice".to_string()))
            .unwrap();
        let alice_rows = engine.get_rows_by_indices("users", &alice_indices).unwrap();
        assert_eq!(alice_rows.len(), 1);
        assert_eq!(alice_rows[0][1], Value::Text("Alice".to_string()));

        let charlie_indices = engine
            .index_lookup("users", "name", &Value::Text("Charlie".to_string()))
            .unwrap();
        let charlie_rows = engine
            .get_rows_by_indices("users", &charlie_indices)
            .unwrap();
        assert_eq!(charlie_rows.len(), 1);
        assert_eq!(charlie_rows[0][1], Value::Text("Charlie".to_string()));

        // Delete Bob (middle row) - this shifts Charlie from index 2 to index 1
        let deleted = engine
            .delete("users", |row| matches!(&row[0], Value::Int(id) if *id == 2))
            .unwrap();
        assert_eq!(deleted, 1);

        // Verify scan still works (rows are correct)
        let all_rows = engine.scan("users").unwrap();
        assert_eq!(all_rows.len(), 2);
        assert_eq!(all_rows[0][1], Value::Text("Alice".to_string()));
        assert_eq!(all_rows[1][1], Value::Text("Charlie".to_string()));

        // Now the bug: Charlie's index still points to position 2, but Charlie is now at position 1
        // Index lookup for Charlie returns [2], but row[2] doesn't exist!
        let charlie_indices_after = engine
            .index_lookup("users", "name", &Value::Text("Charlie".to_string()))
            .unwrap();
        let charlie_rows_after = engine
            .get_rows_by_indices("users", &charlie_indices_after)
            .unwrap();

        // Index lookup should correctly return Charlie's row after delete
        assert_eq!(
            charlie_rows_after.len(),
            1,
            "Index should return exactly one row for Charlie"
        );
        assert_eq!(
            charlie_rows_after[0][1],
            Value::Text("Charlie".to_string()),
            "Index lookup for Charlie should return Charlie's row, not nothing or wrong data"
        );
    }
}
