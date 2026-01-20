//! SQL Runtime implementation for trigger function execution
//!
//! This module provides the SqlRuntime which can execute trigger functions
//! written in SQL. Functions have access to OLD and NEW pseudo-tables.

use logical::{Runtime, RuntimeError, StorageEngine, TriggerContext, TriggerResult, Value};

/// SQL Runtime for executing trigger functions
///
/// The runtime executes functions stored in the database and provides
/// access to OLD and NEW rows during trigger execution.
pub struct SqlRuntime;

impl SqlRuntime {
    /// Create a new SQL runtime
    pub fn new() -> Self {
        Self
    }

    /// Execute a function body with the given context
    ///
    /// Function body format (simple DSL for now):
    /// - `RETURN NEW` - Return the NEW row unchanged (for BEFORE INSERT/UPDATE)
    /// - `RETURN NULL` - Skip the row (for BEFORE triggers)
    /// - `RETURN OLD` - Return the OLD row (for BEFORE UPDATE)
    /// - `SET NEW.<column> = <value>; RETURN NEW` - Modify and return NEW
    /// - `RAISE ERROR '<message>'` - Abort with error
    ///
    /// More complex SQL execution can be added later.
    fn execute_function_body(
        &self,
        body: &str,
        context: &TriggerContext,
    ) -> Result<TriggerResult, RuntimeError> {
        let body = body.trim();

        // Handle RAISE ERROR
        if body.to_uppercase().starts_with("RAISE ERROR") {
            let msg = extract_string_literal(body, "RAISE ERROR")
                .unwrap_or_else(|| "Trigger error".to_string());
            return Ok(TriggerResult::Abort(msg));
        }

        // Handle RETURN NULL (skip)
        if body.to_uppercase() == "RETURN NULL" || body.to_uppercase() == "RETURN NULL;" {
            return Ok(TriggerResult::Skip);
        }

        // Handle RETURN NEW
        if body.to_uppercase() == "RETURN NEW" || body.to_uppercase() == "RETURN NEW;" {
            return Ok(TriggerResult::Proceed(context.new_row.cloned()));
        }

        // Handle RETURN OLD
        if body.to_uppercase() == "RETURN OLD" || body.to_uppercase() == "RETURN OLD;" {
            return Ok(TriggerResult::Proceed(context.old_row.cloned()));
        }

        // Handle SET NEW.<column> = <value>; RETURN NEW
        if body.to_uppercase().contains("SET NEW.") {
            return self.execute_set_statements(body, context);
        }

        // Default: proceed without modification
        Ok(TriggerResult::Proceed(None))
    }

    /// Execute SET statements that modify the NEW row
    fn execute_set_statements(
        &self,
        body: &str,
        context: &TriggerContext,
    ) -> Result<TriggerResult, RuntimeError> {
        let mut new_row = context
            .new_row
            .cloned()
            .ok_or_else(|| RuntimeError::ExecutionError("No NEW row available".to_string()))?;

        // Split by semicolons and process each statement
        for stmt in body.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }

            let upper = stmt.to_uppercase();

            if upper.starts_with("SET NEW.") {
                // Parse: SET NEW.<column> = <value>
                let rest = &stmt[8..]; // Skip "SET NEW."
                if let Some(eq_pos) = rest.find('=') {
                    let column = rest[..eq_pos].trim();
                    let value_str = rest[eq_pos + 1..].trim();

                    // Find column index
                    let col_idx = context
                        .column_names
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(column))
                        .ok_or_else(|| {
                            RuntimeError::ExecutionError(format!("Column not found: {}", column))
                        })?;

                    // Parse value
                    let value = parse_literal_value(value_str, context)?;
                    new_row[col_idx] = value;
                }
            } else if upper == "RETURN NEW" {
                return Ok(TriggerResult::Proceed(Some(new_row)));
            } else if upper == "RETURN NULL" {
                return Ok(TriggerResult::Skip);
            } else if upper.starts_with("RAISE ERROR") {
                let msg = extract_string_literal(stmt, "RAISE ERROR")
                    .unwrap_or_else(|| "Trigger error".to_string());
                return Ok(TriggerResult::Abort(msg));
            }
        }

        // If we processed SET statements but no explicit RETURN, return the modified row
        Ok(TriggerResult::Proceed(Some(new_row)))
    }
}

impl Default for SqlRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: StorageEngine> Runtime<S> for SqlRuntime {
    fn execute_trigger_function(
        &self,
        function_name: &str,
        context: TriggerContext,
        storage: &S,
    ) -> Result<TriggerResult, RuntimeError> {
        // Look up the function in storage
        let func = storage
            .get_function(function_name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(function_name.to_string()))?;

        // Execute the function body
        self.execute_function_body(&func.body, &context)
    }
}

/// Extract a string literal after a keyword
fn extract_string_literal(text: &str, keyword: &str) -> Option<String> {
    let upper = text.to_uppercase();
    let keyword_upper = keyword.to_uppercase();

    if let Some(pos) = upper.find(&keyword_upper) {
        let rest = &text[pos + keyword.len()..].trim();
        // Look for quoted string
        if rest.starts_with('\'') {
            if let Some(end) = rest[1..].find('\'') {
                return Some(rest[1..end + 1].to_string());
            }
        }
    }
    None
}

/// Parse a literal value from a string
fn parse_literal_value(s: &str, context: &TriggerContext) -> Result<Value, RuntimeError> {
    let s = s.trim();

    // NULL
    if s.to_uppercase() == "NULL" {
        return Ok(Value::Null);
    }

    // Boolean
    if s.to_uppercase() == "TRUE" {
        return Ok(Value::Bool(true));
    }
    if s.to_uppercase() == "FALSE" {
        return Ok(Value::Bool(false));
    }

    // String literal
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        return Ok(Value::Text(s[1..s.len() - 1].to_string()));
    }

    // NEW.<column> reference
    if s.to_uppercase().starts_with("NEW.") {
        let col_name = &s[4..];
        if let Some(new_row) = context.new_row {
            if let Some(idx) = context
                .column_names
                .iter()
                .position(|c| c.eq_ignore_ascii_case(col_name))
            {
                return Ok(new_row.get(idx).cloned().unwrap_or(Value::Null));
            }
        }
        return Ok(Value::Null);
    }

    // OLD.<column> reference
    if s.to_uppercase().starts_with("OLD.") {
        let col_name = &s[4..];
        if let Some(old_row) = context.old_row {
            if let Some(idx) = context
                .column_names
                .iter()
                .position(|c| c.eq_ignore_ascii_case(col_name))
            {
                return Ok(old_row.get(idx).cloned().unwrap_or(Value::Null));
            }
        }
        return Ok(Value::Null);
    }

    // Integer
    if let Ok(n) = s.parse::<i64>() {
        return Ok(Value::Int(n));
    }

    // Float
    if let Ok(f) = s.parse::<f64>() {
        return Ok(Value::Float(f));
    }

    // Unknown - treat as text
    Ok(Value::Text(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use logical::{
        ColumnSchema, DataType, FunctionDef, MemoryEngine, TableSchema, TriggerEvent, TriggerTiming,
    };

    fn create_test_storage() -> MemoryEngine {
        let mut storage = MemoryEngine::new();

        // Create a test table
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

        storage
    }

    #[test]
    fn test_return_new() {
        let mut storage = create_test_storage();
        storage
            .create_function(FunctionDef {
                name: "return_new".to_string(),
                params: "[]".to_string(),
                body: "RETURN NEW".to_string(),
                language: "sql".to_string(),
            })
            .unwrap();

        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
        };

        let result = runtime.execute_trigger_function("return_new", context, &storage);
        assert_eq!(result, Ok(TriggerResult::Proceed(Some(new_row))));
    }

    #[test]
    fn test_return_null_skips() {
        let mut storage = create_test_storage();
        storage
            .create_function(FunctionDef {
                name: "skip_row".to_string(),
                params: "[]".to_string(),
                body: "RETURN NULL".to_string(),
                language: "sql".to_string(),
            })
            .unwrap();

        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
        };

        let result = runtime.execute_trigger_function("skip_row", context, &storage);
        assert_eq!(result, Ok(TriggerResult::Skip));
    }

    #[test]
    fn test_raise_error() {
        let mut storage = create_test_storage();
        storage
            .create_function(FunctionDef {
                name: "abort_func".to_string(),
                params: "[]".to_string(),
                body: "RAISE ERROR 'Not allowed'".to_string(),
                language: "sql".to_string(),
            })
            .unwrap();

        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
        };

        let result = runtime.execute_trigger_function("abort_func", context, &storage);
        assert_eq!(result, Ok(TriggerResult::Abort("Not allowed".to_string())));
    }

    #[test]
    fn test_set_new_column() {
        let mut storage = create_test_storage();
        storage
            .create_function(FunctionDef {
                name: "uppercase_name".to_string(),
                params: "[]".to_string(),
                body: "SET NEW.name = 'UPPERCASE'; RETURN NEW".to_string(),
                language: "sql".to_string(),
            })
            .unwrap();

        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
        };

        let result = runtime.execute_trigger_function("uppercase_name", context, &storage);
        assert_eq!(
            result,
            Ok(TriggerResult::Proceed(Some(vec![
                Value::Int(1),
                Value::Text("UPPERCASE".to_string())
            ])))
        );
    }

    #[test]
    fn test_function_not_found() {
        let storage = create_test_storage();
        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
        };

        let result = runtime.execute_trigger_function("nonexistent", context, &storage);
        assert_eq!(
            result,
            Err(RuntimeError::FunctionNotFound("nonexistent".to_string()))
        );
    }

    #[test]
    fn test_set_column_to_integer() {
        let mut storage = create_test_storage();
        storage
            .create_function(FunctionDef {
                name: "set_id".to_string(),
                params: "[]".to_string(),
                body: "SET NEW.id = 999; RETURN NEW".to_string(),
                language: "sql".to_string(),
            })
            .unwrap();

        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
        };

        let result = runtime.execute_trigger_function("set_id", context, &storage);
        assert_eq!(
            result,
            Ok(TriggerResult::Proceed(Some(vec![
                Value::Int(999),
                Value::Text("alice".to_string())
            ])))
        );
    }
}
