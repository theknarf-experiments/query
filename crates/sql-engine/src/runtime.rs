//! SQL Runtime implementation for trigger function execution
//!
//! This module provides the SqlRuntime which can execute trigger functions
//! written in SQL. Functions have access to OLD and NEW pseudo-tables.
//!
//! Supported statements in trigger functions:
//! - `RETURN NEW` - Return the NEW row unchanged
//! - `RETURN NULL` - Skip the row (BEFORE triggers only)
//! - `RETURN OLD` - Return the OLD row
//! - `SET NEW.<column> = <value>` - Modify column in NEW row
//! - `RAISE ERROR '<message>'` - Abort with error
//! - `INSERT INTO <table> VALUES (...)` - Insert into another table

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
    /// Function body format:
    /// - `RETURN NEW` - Return the NEW row unchanged (for BEFORE INSERT/UPDATE)
    /// - `RETURN NULL` - Skip the row (for BEFORE triggers)
    /// - `RETURN OLD` - Return the OLD row (for BEFORE UPDATE)
    /// - `SET NEW.<column> = <value>; RETURN NEW` - Modify and return NEW
    /// - `RAISE ERROR '<message>'` - Abort with error
    /// - `INSERT INTO <table> VALUES (...)` - Insert into another table
    fn execute_function_body<S: StorageEngine>(
        &self,
        body: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<TriggerResult, RuntimeError> {
        let body = body.trim();

        // Handle RAISE ERROR first (highest priority)
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

        // Handle multi-statement bodies (separated by semicolons)
        // Execute each statement in order
        let mut current_row = context.new_row.cloned();

        for stmt in body.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }

            let upper = stmt.to_uppercase();

            if upper.starts_with("SET NEW.") {
                // Parse: SET NEW.<column> = <value>
                if let Some(row) = &mut current_row {
                    self.execute_single_set(stmt, context, row)?;
                }
            } else if upper.starts_with("INSERT INTO") {
                // Execute INSERT INTO statement
                self.execute_insert_into(stmt, context, storage)?;
            } else if upper == "RETURN NEW" {
                return Ok(TriggerResult::Proceed(current_row));
            } else if upper == "RETURN NULL" {
                return Ok(TriggerResult::Skip);
            } else if upper == "RETURN OLD" {
                return Ok(TriggerResult::Proceed(context.old_row.cloned()));
            } else if upper.starts_with("RAISE ERROR") {
                let msg = extract_string_literal(stmt, "RAISE ERROR")
                    .unwrap_or_else(|| "Trigger error".to_string());
                return Ok(TriggerResult::Abort(msg));
            } else if upper.starts_with("IF NOT EXISTS") {
                // Parse: IF NOT EXISTS (SELECT 1 FROM table WHERE col = val) THEN RAISE ERROR '...'
                if let Some(abort_msg) = self.execute_if_not_exists(stmt, context, storage)? {
                    return Ok(TriggerResult::Abort(abort_msg));
                }
            } else if upper.starts_with("IF EXISTS EXCLUDING OLD") {
                // Parse: IF EXISTS EXCLUDING OLD (SELECT ...) THEN RAISE ERROR '...'
                // This excludes rows matching OLD values from the check
                if let Some(abort_msg) =
                    self.execute_if_exists_excluding_old(stmt, context, storage)?
                {
                    return Ok(TriggerResult::Abort(abort_msg));
                }
            } else if upper.starts_with("IF EXISTS") {
                // Parse: IF EXISTS (SELECT 1 FROM table WHERE col = val) THEN RAISE ERROR '...'
                if let Some(abort_msg) = self.execute_if_exists(stmt, context, storage)? {
                    return Ok(TriggerResult::Abort(abort_msg));
                }
            } else if upper.starts_with("DELETE FROM") {
                // Execute DELETE FROM statement
                self.execute_delete_from(stmt, context, storage)?;
            } else if upper.starts_with("UPDATE ") {
                // Execute UPDATE statement
                self.execute_update(stmt, context, storage)?;
            }
            // Ignore unknown statements for now
        }

        // If we processed statements but no explicit RETURN, return the (possibly modified) row
        Ok(TriggerResult::Proceed(current_row))
    }

    /// Execute a single SET NEW.<column> = <value> statement
    fn execute_single_set(
        &self,
        stmt: &str,
        context: &TriggerContext,
        row: &mut [Value],
    ) -> Result<(), RuntimeError> {
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
            row[col_idx] = value;
        }
        Ok(())
    }

    /// Execute an INSERT INTO statement
    fn execute_insert_into<S: StorageEngine>(
        &self,
        stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<(), RuntimeError> {
        // Parse: INSERT INTO <table> VALUES (...)
        // Simple parser for: INSERT INTO table_name VALUES (val1, val2, ...)
        let upper = stmt.to_uppercase();

        // Find table name
        let after_into = if let Some(pos) = upper.find("INSERT INTO") {
            &stmt[pos + 12..].trim_start()
        } else {
            return Err(RuntimeError::ExecutionError(
                "Invalid INSERT INTO syntax".to_string(),
            ));
        };

        // Find where VALUES starts
        let values_pos = after_into.to_uppercase().find("VALUES");
        let table_name = if let Some(pos) = values_pos {
            after_into[..pos].trim()
        } else {
            return Err(RuntimeError::ExecutionError(
                "Missing VALUES in INSERT INTO".to_string(),
            ));
        };

        // Extract values between parentheses
        let values_part = &after_into[values_pos.unwrap_or(0) + 6..].trim_start();
        let values = parse_values_list(values_part, context)?;

        // Insert into storage (without triggering recursive triggers to avoid infinite loops)
        // For now, we do a direct insert to avoid recursion
        storage
            .insert(table_name, values)
            .map_err(|e| RuntimeError::ExecutionError(format!("Insert failed: {:?}", e)))?;

        Ok(())
    }

    /// Execute an IF NOT EXISTS check for FK validation
    /// Format: IF NOT EXISTS (SELECT 1 FROM table WHERE col = val) THEN RAISE ERROR 'msg'
    fn execute_if_not_exists<S: StorageEngine>(
        &self,
        stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<Option<String>, RuntimeError> {
        // Parse: IF NOT EXISTS (SELECT ...) THEN RAISE ERROR '...'
        let upper = stmt.to_uppercase();

        // Find the SELECT part between parentheses
        let select_start = stmt.find('(').ok_or_else(|| {
            RuntimeError::ExecutionError("Missing opening parenthesis in IF NOT EXISTS".to_string())
        })?;
        let select_end = stmt.find(')').ok_or_else(|| {
            RuntimeError::ExecutionError("Missing closing parenthesis in IF NOT EXISTS".to_string())
        })?;

        let select_stmt = &stmt[select_start + 1..select_end];
        let exists = self.check_exists(select_stmt, context, storage)?;

        if !exists {
            // Find THEN RAISE ERROR '...'
            if let Some(then_pos) = upper.find("THEN RAISE ERROR") {
                let after_then = &stmt[then_pos + 16..];
                let msg = extract_string_literal(&format!("'{}", after_then), "'")
                    .unwrap_or_else(|| "Foreign key constraint violation".to_string());
                return Ok(Some(msg));
            }
        }

        Ok(None)
    }

    /// Execute an IF EXISTS check for RESTRICT constraint
    /// Format: IF EXISTS (SELECT 1 FROM table WHERE col = val) THEN RAISE ERROR 'msg'
    fn execute_if_exists<S: StorageEngine>(
        &self,
        stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<Option<String>, RuntimeError> {
        // Parse: IF EXISTS (SELECT ...) THEN RAISE ERROR '...'
        let upper = stmt.to_uppercase();

        // Find the SELECT part between parentheses
        let select_start = stmt.find('(').ok_or_else(|| {
            RuntimeError::ExecutionError("Missing opening parenthesis in IF EXISTS".to_string())
        })?;
        let select_end = stmt.find(')').ok_or_else(|| {
            RuntimeError::ExecutionError("Missing closing parenthesis in IF EXISTS".to_string())
        })?;

        let select_stmt = &stmt[select_start + 1..select_end];
        let exists = self.check_exists(select_stmt, context, storage)?;

        if exists {
            // Find THEN RAISE ERROR '...'
            if let Some(then_pos) = upper.find("THEN RAISE ERROR") {
                let after_then = &stmt[then_pos + 16..];
                let msg = extract_string_literal(&format!("'{}", after_then), "'")
                    .unwrap_or_else(|| "Constraint violation".to_string());
                return Ok(Some(msg));
            }
        }

        Ok(None)
    }

    /// Execute IF EXISTS EXCLUDING OLD check for unique constraint on UPDATE
    /// Format: IF EXISTS EXCLUDING OLD (SELECT 1 FROM table WHERE col = val) THEN RAISE ERROR 'msg'
    /// This excludes rows that match ALL of the OLD row's values in the specified columns
    fn execute_if_exists_excluding_old<S: StorageEngine>(
        &self,
        stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<Option<String>, RuntimeError> {
        let upper = stmt.to_uppercase();

        // Find the SELECT part between parentheses
        let select_start = stmt.find('(').ok_or_else(|| {
            RuntimeError::ExecutionError(
                "Missing opening parenthesis in IF EXISTS EXCLUDING OLD".to_string(),
            )
        })?;
        let select_end = stmt.find(')').ok_or_else(|| {
            RuntimeError::ExecutionError(
                "Missing closing parenthesis in IF EXISTS EXCLUDING OLD".to_string(),
            )
        })?;

        let select_stmt = &stmt[select_start + 1..select_end];
        let exists = self.check_exists_excluding_old(select_stmt, context, storage)?;

        if exists {
            // Find THEN RAISE ERROR '...'
            if let Some(then_pos) = upper.find("THEN RAISE ERROR") {
                let after_then = &stmt[then_pos + 16..];
                let msg = extract_string_literal(&format!("'{}", after_then), "'")
                    .unwrap_or_else(|| "Constraint violation".to_string());
                return Ok(Some(msg));
            }
        }

        Ok(None)
    }

    /// Check if a SELECT query returns any rows, excluding the OLD row
    /// Format: SELECT 1 FROM table WHERE col = val
    fn check_exists_excluding_old<S: StorageEngine>(
        &self,
        select_stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<bool, RuntimeError> {
        let upper = select_stmt.to_uppercase();

        // Parse: SELECT 1 FROM table WHERE col = val
        let from_pos = upper
            .find("FROM")
            .ok_or_else(|| RuntimeError::ExecutionError("Missing FROM in SELECT".to_string()))?;

        let after_from = &select_stmt[from_pos + 4..].trim_start();
        let where_pos = after_from.to_uppercase().find("WHERE");

        let table_name = if let Some(pos) = where_pos {
            after_from[..pos].trim()
        } else {
            after_from.trim()
        };

        // Get rows from table
        let rows = storage
            .scan(table_name)
            .map_err(|e| RuntimeError::ExecutionError(format!("Scan failed: {:?}", e)))?;

        // If no WHERE clause, just check if table has any rows (excluding OLD)
        if where_pos.is_none() {
            if let Some(old_row) = context.old_row {
                // Exclude old row - count rows that don't match old_row
                return Ok(rows.iter().any(|row| row != old_row));
            }
            return Ok(!rows.is_empty());
        }

        // Parse WHERE clause
        let where_clause = &after_from[where_pos.unwrap() + 5..].trim();

        // Get schema for column lookup
        let schema = storage
            .get_schema(table_name)
            .map_err(|e| RuntimeError::ExecutionError(format!("Schema not found: {:?}", e)))?;

        // Parse all WHERE conditions (support AND)
        let conditions = parse_where_conditions(where_clause, context)?;

        // Check if any row matches (excluding the OLD row)
        let old_row = context.old_row;

        Ok(rows.iter().any(|row| {
            // Skip if this row matches the OLD row exactly
            if let Some(old) = old_row
                && row == old
            {
                return false;
            }

            // Check if this row matches all WHERE conditions
            conditions.iter().all(|(col, val)| {
                if let Some(col_idx) = schema
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(col))
                {
                    row.get(col_idx)
                        .map(|v| values_equal(v, val))
                        .unwrap_or(false)
                } else {
                    false
                }
            })
        }))
    }

    /// Check if a SELECT query returns any rows
    /// Format: SELECT 1 FROM table WHERE col = val
    fn check_exists<S: StorageEngine>(
        &self,
        select_stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<bool, RuntimeError> {
        let upper = select_stmt.to_uppercase();

        // Parse: SELECT 1 FROM table WHERE col = val
        let from_pos = upper
            .find("FROM")
            .ok_or_else(|| RuntimeError::ExecutionError("Missing FROM in SELECT".to_string()))?;

        let after_from = &select_stmt[from_pos + 4..].trim_start();
        let where_pos = after_from.to_uppercase().find("WHERE");

        let table_name = if let Some(pos) = where_pos {
            after_from[..pos].trim()
        } else {
            after_from.trim()
        };

        // Get rows from table
        let rows = storage
            .scan(table_name)
            .map_err(|e| RuntimeError::ExecutionError(format!("Scan failed: {:?}", e)))?;

        // If no WHERE clause, just check if table has any rows
        if where_pos.is_none() {
            return Ok(!rows.is_empty());
        }

        // Parse WHERE clause
        let where_clause = &after_from[where_pos.unwrap() + 5..].trim();
        let (col, val) = parse_simple_where(where_clause, context)?;

        // Get schema for column lookup
        let schema = storage
            .get_schema(table_name)
            .map_err(|e| RuntimeError::ExecutionError(format!("Schema not found: {:?}", e)))?;

        let col_idx = schema
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(&col))
            .ok_or_else(|| RuntimeError::ExecutionError(format!("Column not found: {}", col)))?;

        // Check if any row matches
        Ok(rows.iter().any(|row| {
            row.get(col_idx)
                .map(|v| values_equal(v, &val))
                .unwrap_or(false)
        }))
    }

    /// Execute a DELETE FROM statement
    /// Format: DELETE FROM table WHERE col = val
    fn execute_delete_from<S: StorageEngine>(
        &self,
        stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<(), RuntimeError> {
        let upper = stmt.to_uppercase();

        // Parse: DELETE FROM table WHERE col = val
        let from_pos = upper
            .find("FROM")
            .ok_or_else(|| RuntimeError::ExecutionError("Missing FROM in DELETE".to_string()))?;

        let after_from = &stmt[from_pos + 4..].trim_start();
        let where_pos = after_from
            .to_uppercase()
            .find("WHERE")
            .ok_or_else(|| RuntimeError::ExecutionError("Missing WHERE in DELETE".to_string()))?;

        let table_name = after_from[..where_pos].trim();
        let where_clause = &after_from[where_pos + 5..].trim();
        let (col, val) = parse_simple_where(where_clause, context)?;

        // Get schema for column lookup
        let schema = storage
            .get_schema(table_name)
            .map_err(|e| RuntimeError::ExecutionError(format!("Schema not found: {:?}", e)))?;

        let col_idx = schema
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(&col))
            .ok_or_else(|| RuntimeError::ExecutionError(format!("Column not found: {}", col)))?;

        // Delete matching rows
        storage
            .delete(table_name, |row| {
                row.get(col_idx)
                    .map(|v| values_equal(v, &val))
                    .unwrap_or(false)
            })
            .map_err(|e| RuntimeError::ExecutionError(format!("Delete failed: {:?}", e)))?;

        Ok(())
    }

    /// Execute an UPDATE statement
    /// Format: UPDATE table SET col = val WHERE col2 = val2
    fn execute_update<S: StorageEngine>(
        &self,
        stmt: &str,
        context: &TriggerContext,
        storage: &mut S,
    ) -> Result<(), RuntimeError> {
        let upper = stmt.to_uppercase();

        // Parse: UPDATE table SET col = val WHERE col2 = val2
        let set_pos = upper
            .find("SET")
            .ok_or_else(|| RuntimeError::ExecutionError("Missing SET in UPDATE".to_string()))?;

        let table_name = stmt[7..set_pos].trim(); // Skip "UPDATE "

        let after_set = &stmt[set_pos + 3..].trim_start();
        let where_pos = after_set
            .to_uppercase()
            .find("WHERE")
            .ok_or_else(|| RuntimeError::ExecutionError("Missing WHERE in UPDATE".to_string()))?;

        // Parse SET clause: col = val
        let set_clause = &after_set[..where_pos].trim();
        let (set_col, set_val) = parse_simple_assignment(set_clause, context)?;

        // Parse WHERE clause: col2 = val2
        let where_clause = &after_set[where_pos + 5..].trim();
        let (where_col, where_val) = parse_simple_where(where_clause, context)?;

        // Get schema for column lookup
        let schema = storage
            .get_schema(table_name)
            .map_err(|e| RuntimeError::ExecutionError(format!("Schema not found: {:?}", e)))?;

        let set_col_idx = schema
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(&set_col))
            .ok_or_else(|| {
                RuntimeError::ExecutionError(format!("Column not found: {}", set_col))
            })?;

        let where_col_idx = schema
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(&where_col))
            .ok_or_else(|| {
                RuntimeError::ExecutionError(format!("Column not found: {}", where_col))
            })?;

        // Update matching rows
        storage
            .update(
                table_name,
                |row| {
                    row.get(where_col_idx)
                        .map(|v| values_equal(v, &where_val))
                        .unwrap_or(false)
                },
                |row| {
                    if let Some(val) = row.get_mut(set_col_idx) {
                        *val = set_val.clone();
                    }
                },
            )
            .map_err(|e| RuntimeError::ExecutionError(format!("Update failed: {:?}", e)))?;

        Ok(())
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
        storage: &mut S,
    ) -> Result<TriggerResult, RuntimeError> {
        // Look up the function in storage and clone the body to release the borrow
        let func_body = storage
            .get_function(function_name)
            .map(|f| f.body.clone())
            .ok_or_else(|| RuntimeError::FunctionNotFound(function_name.to_string()))?;

        // Execute the function body
        self.execute_function_body(&func_body, &context, storage)
    }
}

/// Extract a string literal after a keyword
fn extract_string_literal(text: &str, keyword: &str) -> Option<String> {
    let upper = text.to_uppercase();
    let keyword_upper = keyword.to_uppercase();

    if let Some(pos) = upper.find(&keyword_upper) {
        let rest = &text[pos + keyword.len()..].trim();
        // Look for quoted string
        if let Some(stripped) = rest.strip_prefix('\'')
            && let Some(end) = stripped.find('\'')
        {
            return Some(stripped[..end].to_string());
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
        if let Some(new_row) = context.new_row
            && let Some(idx) = context
                .column_names
                .iter()
                .position(|c| c.eq_ignore_ascii_case(col_name))
        {
            return Ok(new_row.get(idx).cloned().unwrap_or(Value::Null));
        }
        return Ok(Value::Null);
    }

    // OLD.<column> reference
    if s.to_uppercase().starts_with("OLD.") {
        let col_name = &s[4..];
        if let Some(old_row) = context.old_row
            && let Some(idx) = context
                .column_names
                .iter()
                .position(|c| c.eq_ignore_ascii_case(col_name))
        {
            return Ok(old_row.get(idx).cloned().unwrap_or(Value::Null));
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

/// Parse a VALUES list like (val1, val2, ...) into a vector of Values
fn parse_values_list(s: &str, context: &TriggerContext) -> Result<Vec<Value>, RuntimeError> {
    let s = s.trim();

    // Find the opening and closing parentheses
    let start = s.find('(').ok_or_else(|| {
        RuntimeError::ExecutionError("Missing opening parenthesis in VALUES".to_string())
    })?;
    let end = s.rfind(')').ok_or_else(|| {
        RuntimeError::ExecutionError("Missing closing parenthesis in VALUES".to_string())
    })?;

    if end <= start {
        return Err(RuntimeError::ExecutionError(
            "Invalid VALUES syntax".to_string(),
        ));
    }

    let inner = &s[start + 1..end];

    // Split by commas, but be careful of commas inside strings
    let mut values = Vec::new();
    let mut current = String::new();
    let mut in_string = false;

    for ch in inner.chars() {
        if ch == '\'' {
            in_string = !in_string;
            current.push(ch);
        } else if ch == ',' && !in_string {
            let val = parse_literal_value(current.trim(), context)?;
            values.push(val);
            current.clear();
        } else {
            current.push(ch);
        }
    }

    // Don't forget the last value
    if !current.trim().is_empty() {
        let val = parse_literal_value(current.trim(), context)?;
        values.push(val);
    }

    Ok(values)
}

/// Parse a simple WHERE clause: col = val
fn parse_simple_where(
    clause: &str,
    context: &TriggerContext,
) -> Result<(String, Value), RuntimeError> {
    let clause = clause.trim();

    // Find the equals sign
    let eq_pos = clause
        .find('=')
        .ok_or_else(|| RuntimeError::ExecutionError("Missing = in WHERE clause".to_string()))?;

    let col = clause[..eq_pos].trim().to_string();
    let val_str = clause[eq_pos + 1..].trim();
    let val = parse_literal_value(val_str, context)?;

    Ok((col, val))
}

/// Parse a simple assignment: col = val
fn parse_simple_assignment(
    clause: &str,
    context: &TriggerContext,
) -> Result<(String, Value), RuntimeError> {
    // Same logic as parse_simple_where
    parse_simple_where(clause, context)
}

/// Parse WHERE clause with multiple AND conditions
/// Format: col1 = val1 AND col2 = val2 AND ...
fn parse_where_conditions(
    clause: &str,
    context: &TriggerContext,
) -> Result<Vec<(String, Value)>, RuntimeError> {
    let clause = clause.trim();
    let upper = clause.to_uppercase();

    // Split by AND (case-insensitive)
    let mut conditions = Vec::new();
    let mut start = 0;

    loop {
        // Find next AND
        let and_pos = upper[start..].find(" AND ");

        let end = if let Some(pos) = and_pos {
            start + pos
        } else {
            clause.len()
        };

        // Parse this condition
        let condition = clause[start..end].trim();
        if !condition.is_empty() {
            let (col, val) = parse_simple_where(condition, context)?;
            conditions.push((col, val));
        }

        if and_pos.is_none() {
            break;
        }

        start = end + 5; // Skip " AND "
    }

    Ok(conditions)
}

/// Check if two values are equal
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Date(x), Value::Date(y)) => x == y,
        (Value::Time(x), Value::Time(y)) => x == y,
        (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
        _ => false,
    }
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
            depth: 1,
            max_depth: 3,
        };

        let result = runtime.execute_trigger_function("return_new", context, &mut storage);
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
            depth: 1,
            max_depth: 3,
        };

        let result = runtime.execute_trigger_function("skip_row", context, &mut storage);
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
            depth: 1,
            max_depth: 3,
        };

        let result = runtime.execute_trigger_function("abort_func", context, &mut storage);
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
            depth: 1,
            max_depth: 3,
        };

        let result = runtime.execute_trigger_function("uppercase_name", context, &mut storage);
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
        let mut storage = create_test_storage();
        let runtime = SqlRuntime::new();
        let new_row = vec![Value::Int(1), Value::Text("Alice".to_string())];
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "users",
            old_row: None,
            new_row: Some(&new_row),
            column_names: &["id".to_string(), "name".to_string()],
            depth: 1,
            max_depth: 3,
        };

        let result = runtime.execute_trigger_function("nonexistent", context, &mut storage);
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
            depth: 1,
            max_depth: 3,
        };

        let result = runtime.execute_trigger_function("set_id", context, &mut storage);
        assert_eq!(
            result,
            Ok(TriggerResult::Proceed(Some(vec![
                Value::Int(999),
                Value::Text("alice".to_string())
            ])))
        );
    }
}
