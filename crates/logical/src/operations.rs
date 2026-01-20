//! Trigger-aware database operations
//!
//! This module provides wrapper functions for insert, update, and delete operations
//! that automatically fire triggers before and after the operation.

use crate::runtime::{
    Runtime, RuntimeError, TriggerContext, TriggerResult, DEFAULT_MAX_TRIGGER_DEPTH,
};
use storage::{Row, StorageEngine, StorageError, TriggerEvent, TriggerTiming};

/// Error type for triggered operations
#[derive(Debug, Clone, PartialEq)]
pub enum OperationError {
    /// Storage error
    Storage(StorageError),
    /// Runtime error during trigger execution
    Runtime(RuntimeError),
    /// Trigger aborted the operation
    TriggerAbort(String),
    /// Trigger depth exceeded (infinite recursion prevention)
    TriggerDepthExceeded { depth: u32, max_depth: u32 },
}

impl From<StorageError> for OperationError {
    fn from(err: StorageError) -> Self {
        OperationError::Storage(err)
    }
}

impl From<RuntimeError> for OperationError {
    fn from(err: RuntimeError) -> Self {
        OperationError::Runtime(err)
    }
}

impl std::fmt::Display for OperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationError::Storage(e) => write!(f, "Storage error: {:?}", e),
            OperationError::Runtime(e) => write!(f, "Runtime error: {}", e),
            OperationError::TriggerAbort(msg) => write!(f, "Trigger aborted: {}", msg),
            OperationError::TriggerDepthExceeded { depth, max_depth } => {
                write!(
                    f,
                    "Trigger depth exceeded: {} > {} (possible infinite recursion)",
                    depth, max_depth
                )
            }
        }
    }
}

/// Result type for triggered operations
pub type OperationResult<T> = Result<T, OperationError>;

/// Insert a row with trigger support (convenience wrapper with default depth)
pub fn insert<S: StorageEngine, R: Runtime<S>>(
    storage: &mut S,
    runtime: &R,
    table: &str,
    row: Row,
) -> OperationResult<bool> {
    insert_with_depth(storage, runtime, table, row, 1, DEFAULT_MAX_TRIGGER_DEPTH)
}

/// Insert a row with trigger support and explicit depth tracking
///
/// Executes BEFORE INSERT triggers, performs the insert, then executes AFTER INSERT triggers.
/// BEFORE triggers can modify the row or abort the operation.
///
/// # Arguments
/// * `storage` - The storage engine
/// * `runtime` - The trigger runtime
/// * `table` - Table name to insert into
/// * `row` - The row to insert
/// * `depth` - Current trigger depth (starts at 1)
/// * `max_depth` - Maximum trigger depth (use DEFAULT_MAX_TRIGGER_DEPTH)
///
/// # Returns
/// * `Ok(true)` - Row was inserted
/// * `Ok(false)` - Row was skipped (BEFORE trigger returned Skip)
/// * `Err(...)` - Error occurred
pub fn insert_with_depth<S: StorageEngine, R: Runtime<S>>(
    storage: &mut S,
    runtime: &R,
    table: &str,
    row: Row,
    depth: u32,
    max_depth: u32,
) -> OperationResult<bool> {
    // Check depth limit
    if depth > max_depth {
        return Err(OperationError::TriggerDepthExceeded { depth, max_depth });
    }

    // Get column names for the trigger context
    let schema = storage.get_schema(table)?;
    let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

    // Get trigger function names upfront (clone to avoid borrow issues)
    let before_trigger_funcs: Vec<String> = storage
        .get_triggers_for_table(table, TriggerEvent::Insert, TriggerTiming::Before)
        .iter()
        .map(|t| t.function_name.clone())
        .collect();

    // Execute BEFORE INSERT triggers
    let mut current_row = row;

    for function_name in &before_trigger_funcs {
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table,
            old_row: None,
            new_row: Some(&current_row),
            column_names: &column_names,
            depth,
            max_depth,
        };

        match runtime.execute_trigger_function(function_name, context, storage)? {
            TriggerResult::Proceed(Some(modified_row)) => {
                current_row = modified_row;
            }
            TriggerResult::Proceed(None) => {
                // Continue with unmodified row
            }
            TriggerResult::Skip => {
                return Ok(false);
            }
            TriggerResult::Abort(msg) => {
                return Err(OperationError::TriggerAbort(msg));
            }
        }
    }

    // Perform the actual insert
    storage.insert(table, current_row.clone())?;

    // Get AFTER trigger function names
    let after_trigger_funcs: Vec<String> = storage
        .get_triggers_for_table(table, TriggerEvent::Insert, TriggerTiming::After)
        .iter()
        .map(|t| t.function_name.clone())
        .collect();

    // Execute AFTER INSERT triggers
    for function_name in &after_trigger_funcs {
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::After,
            table,
            old_row: None,
            new_row: Some(&current_row),
            column_names: &column_names,
            depth,
            max_depth,
        };

        match runtime.execute_trigger_function(function_name, context, storage)? {
            TriggerResult::Abort(msg) => {
                return Err(OperationError::TriggerAbort(msg));
            }
            _ => {
                // AFTER triggers can't modify or skip - proceed
            }
        }
    }

    Ok(true)
}

/// Delete rows with trigger support (convenience wrapper with default depth)
pub fn delete<S, R, F>(
    storage: &mut S,
    runtime: &R,
    table: &str,
    predicate: F,
) -> OperationResult<usize>
where
    S: StorageEngine,
    R: Runtime<S>,
    F: Fn(&Row) -> bool,
{
    delete_with_depth(
        storage,
        runtime,
        table,
        predicate,
        1,
        DEFAULT_MAX_TRIGGER_DEPTH,
    )
}

/// Delete rows with trigger support and explicit depth tracking
///
/// Executes BEFORE DELETE triggers for each matching row, performs the delete,
/// then executes AFTER DELETE triggers.
///
/// # Returns
/// Number of rows deleted (excludes rows skipped by BEFORE triggers)
pub fn delete_with_depth<S, R, F>(
    storage: &mut S,
    runtime: &R,
    table: &str,
    predicate: F,
    depth: u32,
    max_depth: u32,
) -> OperationResult<usize>
where
    S: StorageEngine,
    R: Runtime<S>,
    F: Fn(&Row) -> bool,
{
    // Check depth limit
    if depth > max_depth {
        return Err(OperationError::TriggerDepthExceeded { depth, max_depth });
    }

    // Get column names for the trigger context
    let schema = storage.get_schema(table)?;
    let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

    // Get all rows and find which ones match
    let all_rows = storage.scan(table)?;
    let matching_rows: Vec<(usize, Row)> = all_rows
        .into_iter()
        .enumerate()
        .filter(|(_, row)| predicate(row))
        .collect();

    // Get trigger function names upfront (clone to avoid borrow issues)
    let before_trigger_funcs: Vec<String> = storage
        .get_triggers_for_table(table, TriggerEvent::Delete, TriggerTiming::Before)
        .iter()
        .map(|t| t.function_name.clone())
        .collect();

    // Track which rows to actually delete (after BEFORE triggers)
    let mut rows_to_delete: Vec<usize> = Vec::new();
    let mut deleted_rows: Vec<Row> = Vec::new();

    // Execute BEFORE DELETE triggers for each matching row
    for (idx, row) in &matching_rows {
        let mut should_delete = true;

        for function_name in &before_trigger_funcs {
            let context = TriggerContext {
                event: TriggerEvent::Delete,
                timing: TriggerTiming::Before,
                table,
                old_row: Some(row),
                new_row: None,
                column_names: &column_names,
                depth,
                max_depth,
            };

            match runtime.execute_trigger_function(function_name, context, storage)? {
                TriggerResult::Skip => {
                    should_delete = false;
                    break;
                }
                TriggerResult::Abort(msg) => {
                    return Err(OperationError::TriggerAbort(msg));
                }
                TriggerResult::Proceed(_) => {
                    // Continue
                }
            }
        }

        if should_delete {
            rows_to_delete.push(*idx);
            deleted_rows.push(row.clone());
        }
    }

    // Perform the actual deletes
    // We delete rows that are in deleted_rows
    storage.delete(table, |row| deleted_rows.iter().any(|r| r == row))?;

    // Get AFTER trigger function names
    let after_trigger_funcs: Vec<String> = storage
        .get_triggers_for_table(table, TriggerEvent::Delete, TriggerTiming::After)
        .iter()
        .map(|t| t.function_name.clone())
        .collect();

    // Execute AFTER DELETE triggers for each deleted row
    for row in &deleted_rows {
        for function_name in &after_trigger_funcs {
            let context = TriggerContext {
                event: TriggerEvent::Delete,
                timing: TriggerTiming::After,
                table,
                old_row: Some(row),
                new_row: None,
                column_names: &column_names,
                depth,
                max_depth,
            };

            match runtime.execute_trigger_function(function_name, context, storage)? {
                TriggerResult::Abort(msg) => {
                    return Err(OperationError::TriggerAbort(msg));
                }
                _ => {
                    // AFTER triggers can't modify or skip
                }
            }
        }
    }

    // Return the number of rows we intended to delete (not storage's count which may differ)
    Ok(rows_to_delete.len())
}

/// Update rows with trigger support (convenience wrapper with default depth)
pub fn update<S, R, F, U>(
    storage: &mut S,
    runtime: &R,
    table: &str,
    predicate: F,
    updater: U,
) -> OperationResult<usize>
where
    S: StorageEngine,
    R: Runtime<S>,
    F: Fn(&Row) -> bool,
    U: Fn(&mut Row),
{
    update_with_depth(
        storage,
        runtime,
        table,
        predicate,
        updater,
        1,
        DEFAULT_MAX_TRIGGER_DEPTH,
    )
}

/// Update rows with trigger support and explicit depth tracking
///
/// Executes BEFORE UPDATE triggers for each matching row, performs the update,
/// then executes AFTER UPDATE triggers.
///
/// # Returns
/// Number of rows updated (excludes rows skipped by BEFORE triggers)
pub fn update_with_depth<S, R, F, U>(
    storage: &mut S,
    runtime: &R,
    table: &str,
    predicate: F,
    updater: U,
    depth: u32,
    max_depth: u32,
) -> OperationResult<usize>
where
    S: StorageEngine,
    R: Runtime<S>,
    F: Fn(&Row) -> bool,
    U: Fn(&mut Row),
{
    // Check depth limit
    if depth > max_depth {
        return Err(OperationError::TriggerDepthExceeded { depth, max_depth });
    }

    // Get column names for the trigger context
    let schema = storage.get_schema(table)?;
    let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

    // Get all rows and find which ones match
    let all_rows = storage.scan(table)?;
    let matching_rows: Vec<(usize, Row)> = all_rows
        .into_iter()
        .enumerate()
        .filter(|(_, row)| predicate(row))
        .collect();

    // Get trigger function names upfront (clone to avoid borrow issues)
    let before_trigger_funcs: Vec<String> = storage
        .get_triggers_for_table(table, TriggerEvent::Update, TriggerTiming::Before)
        .iter()
        .map(|t| t.function_name.clone())
        .collect();

    // Track updates to apply: (index, old_row, new_row)
    let mut updates: Vec<(usize, Row, Row)> = Vec::new();

    // Execute BEFORE UPDATE triggers for each matching row
    for (idx, old_row) in &matching_rows {
        // Compute the updated row
        let mut new_row = old_row.clone();
        updater(&mut new_row);

        let mut should_update = true;
        let mut current_new_row = new_row;

        for function_name in &before_trigger_funcs {
            let context = TriggerContext {
                event: TriggerEvent::Update,
                timing: TriggerTiming::Before,
                table,
                old_row: Some(old_row),
                new_row: Some(&current_new_row),
                column_names: &column_names,
                depth,
                max_depth,
            };

            match runtime.execute_trigger_function(function_name, context, storage)? {
                TriggerResult::Proceed(Some(modified_row)) => {
                    current_new_row = modified_row;
                }
                TriggerResult::Proceed(None) => {
                    // Continue with unmodified row
                }
                TriggerResult::Skip => {
                    should_update = false;
                    break;
                }
                TriggerResult::Abort(msg) => {
                    return Err(OperationError::TriggerAbort(msg));
                }
            }
        }

        if should_update {
            updates.push((*idx, old_row.clone(), current_new_row));
        }
    }

    // Perform the actual updates
    // We match rows by comparing to old_row values in updates
    storage.update(
        table,
        |row| updates.iter().any(|(_, old, _)| old == row),
        |row| {
            // Find the matching update and apply the new row
            if let Some((_, _, new_row)) = updates.iter().find(|(_, old, _)| old == row) {
                *row = new_row.clone();
            }
        },
    )?;

    // Get AFTER trigger function names
    let after_trigger_funcs: Vec<String> = storage
        .get_triggers_for_table(table, TriggerEvent::Update, TriggerTiming::After)
        .iter()
        .map(|t| t.function_name.clone())
        .collect();

    // Execute AFTER UPDATE triggers for each updated row
    for (_, old_row, new_row) in &updates {
        for function_name in &after_trigger_funcs {
            let context = TriggerContext {
                event: TriggerEvent::Update,
                timing: TriggerTiming::After,
                table,
                old_row: Some(old_row),
                new_row: Some(new_row),
                column_names: &column_names,
                depth,
                max_depth,
            };

            match runtime.execute_trigger_function(function_name, context, storage)? {
                TriggerResult::Abort(msg) => {
                    return Err(OperationError::TriggerAbort(msg));
                }
                _ => {
                    // AFTER triggers can't modify or skip
                }
            }
        }
    }

    Ok(updates.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::NoOpRuntime;
    use storage::{ColumnSchema, DataType, MemoryEngine, TableSchema, Value};

    fn create_test_table(storage: &mut MemoryEngine) {
        let schema = TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: true,
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
            constraints: vec![],
        };
        storage.create_table(schema).unwrap();
    }

    #[test]
    fn test_insert_with_no_triggers() {
        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);
        let runtime = NoOpRuntime;

        let row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let result = insert(&mut storage, &runtime, "users", row);

        assert_eq!(result, Ok(true));

        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[0][1], Value::Text("Alice".to_string()));
    }

    #[test]
    fn test_delete_with_no_triggers() {
        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);
        let runtime = NoOpRuntime;

        // Insert some rows
        storage
            .insert(
                "users",
                vec![Value::Int(1), Value::Text("Alice".to_string())],
            )
            .unwrap();
        storage
            .insert("users", vec![Value::Int(2), Value::Text("Bob".to_string())])
            .unwrap();

        // Delete one row
        let result = delete(&mut storage, &runtime, "users", |row| {
            row[0] == Value::Int(1)
        });

        assert_eq!(result, Ok(1));

        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(2));
    }

    #[test]
    fn test_update_with_no_triggers() {
        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);
        let runtime = NoOpRuntime;

        // Insert a row
        storage
            .insert(
                "users",
                vec![Value::Int(1), Value::Text("Alice".to_string())],
            )
            .unwrap();

        // Update the row
        let result = update(
            &mut storage,
            &runtime,
            "users",
            |row| row[0] == Value::Int(1),
            |row| row[1] = Value::Text("Alicia".to_string()),
        );

        assert_eq!(result, Ok(1));

        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][1], Value::Text("Alicia".to_string()));
    }

    #[test]
    fn test_depth_exceeded() {
        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);
        let runtime = NoOpRuntime;

        let row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        // Try with depth already at max
        let result = insert_with_depth(&mut storage, &runtime, "users", row, 5, 3);

        assert_eq!(
            result,
            Err(OperationError::TriggerDepthExceeded {
                depth: 5,
                max_depth: 3
            })
        );

        // No rows should be inserted
        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 0);
    }

    // Test with a custom runtime that modifies rows
    struct ModifyingRuntime;

    impl<S: StorageEngine> Runtime<S> for ModifyingRuntime {
        fn execute_trigger_function(
            &self,
            _function_name: &str,
            context: TriggerContext,
            _storage: &mut S,
        ) -> Result<TriggerResult, RuntimeError> {
            // For BEFORE INSERT, modify the name to uppercase
            if context.event == TriggerEvent::Insert && context.timing == TriggerTiming::Before {
                if let Some(row) = context.new_row {
                    let mut modified = row.clone();
                    if let Value::Text(name) = &modified[1] {
                        modified[1] = Value::Text(name.to_uppercase());
                    }
                    return Ok(TriggerResult::Proceed(Some(modified)));
                }
            }
            Ok(TriggerResult::Proceed(None))
        }
    }

    #[test]
    fn test_before_insert_trigger_modifies_row() {
        use storage::TriggerDef;

        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);

        // Create a trigger
        storage
            .create_trigger(TriggerDef {
                name: "uppercase_name".to_string(),
                table_name: "users".to_string(),
                timing: TriggerTiming::Before,
                events: vec![TriggerEvent::Insert],
                function_name: "uppercase_func".to_string(),
            })
            .unwrap();

        let runtime = ModifyingRuntime;

        let row = vec![Value::Int(1), Value::Text("alice".to_string())];
        let result = insert(&mut storage, &runtime, "users", row);

        assert_eq!(result, Ok(true));

        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        // Name should be uppercased by the trigger
        assert_eq!(rows[0][1], Value::Text("ALICE".to_string()));
    }

    // Test with a runtime that skips rows
    struct SkippingRuntime;

    impl<S: StorageEngine> Runtime<S> for SkippingRuntime {
        fn execute_trigger_function(
            &self,
            _function_name: &str,
            context: TriggerContext,
            _storage: &mut S,
        ) -> Result<TriggerResult, RuntimeError> {
            // Skip rows with id > 5
            if context.timing == TriggerTiming::Before {
                if let Some(row) = context.new_row.or(context.old_row) {
                    if let Value::Int(id) = row[0] {
                        if id > 5 {
                            return Ok(TriggerResult::Skip);
                        }
                    }
                }
            }
            Ok(TriggerResult::Proceed(None))
        }
    }

    #[test]
    fn test_before_insert_trigger_skips_row() {
        use storage::TriggerDef;

        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);

        storage
            .create_trigger(TriggerDef {
                name: "skip_high_ids".to_string(),
                table_name: "users".to_string(),
                timing: TriggerTiming::Before,
                events: vec![TriggerEvent::Insert],
                function_name: "skip_func".to_string(),
            })
            .unwrap();

        let runtime = SkippingRuntime;

        // This should be inserted
        let row1 = vec![Value::Int(3), Value::Text("Alice".to_string())];
        let result1 = insert(&mut storage, &runtime, "users", row1);
        assert_eq!(result1, Ok(true));

        // This should be skipped
        let row2 = vec![Value::Int(10), Value::Text("Bob".to_string())];
        let result2 = insert(&mut storage, &runtime, "users", row2);
        assert_eq!(result2, Ok(false));

        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(3));
    }

    // Test with a runtime that aborts
    struct AbortingRuntime;

    impl<S: StorageEngine> Runtime<S> for AbortingRuntime {
        fn execute_trigger_function(
            &self,
            _function_name: &str,
            _context: TriggerContext,
            _storage: &mut S,
        ) -> Result<TriggerResult, RuntimeError> {
            Ok(TriggerResult::Abort("Operation not allowed".to_string()))
        }
    }

    #[test]
    fn test_before_insert_trigger_aborts() {
        use storage::TriggerDef;

        let mut storage = MemoryEngine::new();
        create_test_table(&mut storage);

        storage
            .create_trigger(TriggerDef {
                name: "abort_all".to_string(),
                table_name: "users".to_string(),
                timing: TriggerTiming::Before,
                events: vec![TriggerEvent::Insert],
                function_name: "abort_func".to_string(),
            })
            .unwrap();

        let runtime = AbortingRuntime;

        let row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let result = insert(&mut storage, &runtime, "users", row);

        assert_eq!(
            result,
            Err(OperationError::TriggerAbort(
                "Operation not allowed".to_string()
            ))
        );

        // No rows should be inserted
        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 0);
    }
}
