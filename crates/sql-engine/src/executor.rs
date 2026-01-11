//! Query executor - executes logical plans against storage

use sql_parser::{
    AggregateFunc, Assignment, BinaryOp, ColumnDef, DataType, Expr, ForeignKeyRef as ParserFKRef,
    JoinType, ReferentialAction as ParserRefAction, TriggerAction, TriggerEvent, TriggerTiming,
    UnaryOp,
};
use sql_planner::LogicalPlan;
use sql_storage::{
    ColumnSchema, DataType as StorageDataType, ForeignKeyRef, MemoryEngine,
    ReferentialAction as StorageRefAction, Row, StorageEngine, StorageError, TableSchema, Value,
};

/// Result type for execution operations
pub type ExecResult = Result<QueryResult, ExecError>;

/// Execution error
#[derive(Debug, Clone, PartialEq)]
pub enum ExecError {
    /// Storage error
    Storage(StorageError),
    /// Table not found
    TableNotFound(String),
    /// Column not found
    ColumnNotFound(String),
    /// Type error
    TypeError(String),
    /// Invalid expression
    InvalidExpression(String),
    /// Transaction error - no active transaction
    NoActiveTransaction,
    /// Transaction error - transaction already active
    TransactionAlreadyActive,
    /// Savepoint not found
    SavepointNotFound(String),
    /// Foreign key constraint violation
    ForeignKeyViolation {
        table: String,
        column: String,
        references_table: String,
        references_column: String,
    },
    /// Trigger error (RAISE ERROR)
    TriggerError(String),
    /// Trigger not found
    TriggerNotFound(String),
    /// Trigger already exists
    TriggerAlreadyExists(String),
}

impl From<StorageError> for ExecError {
    fn from(err: StorageError) -> Self {
        ExecError::Storage(err)
    }
}

/// Result of a query execution
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    /// SELECT result with column names and rows
    Select {
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
    },
    /// Number of rows affected (INSERT, UPDATE, DELETE)
    RowsAffected(usize),
    /// DDL success (CREATE TABLE, DROP TABLE)
    Success,
    /// Transaction operation completed
    TransactionStarted,
    /// Transaction committed
    TransactionCommitted,
    /// Transaction rolled back
    TransactionRolledBack,
    /// Savepoint created
    SavepointCreated(String),
    /// Savepoint released
    SavepointReleased(String),
    /// Rolled back to savepoint
    RolledBackToSavepoint(String),
}

/// Transaction state
#[derive(Debug, Clone, Default)]
struct TransactionState {
    /// Is a transaction currently active?
    active: bool,
    /// Savepoints (name -> snapshot index)
    savepoints: Vec<(String, usize)>,
    /// Snapshots of storage state for rollback
    snapshots: Vec<MemoryEngine>,
}

/// Stored trigger definition
#[derive(Debug, Clone)]
struct Trigger {
    name: String,
    timing: TriggerTiming,
    event: TriggerEvent,
    table: String,
    body: Vec<TriggerAction>,
}

/// Database engine that executes queries
pub struct Engine {
    storage: MemoryEngine,
    transaction: TransactionState,
    triggers: Vec<Trigger>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Create a new database engine
    pub fn new() -> Self {
        Self {
            storage: MemoryEngine::new(),
            transaction: TransactionState::default(),
            triggers: Vec::new(),
        }
    }

    /// Execute a SQL string
    pub fn execute(&mut self, sql: &str) -> ExecResult {
        let stmt = sql_parser::parse(sql)
            .map_err(|_| ExecError::InvalidExpression("Parse error".to_string()))?;
        let plan = sql_planner::plan(stmt)
            .map_err(|_| ExecError::InvalidExpression("Planning error".to_string()))?;
        self.execute_plan(plan)
    }

    /// Execute a logical plan
    fn execute_plan(&mut self, plan: LogicalPlan) -> ExecResult {
        match plan {
            LogicalPlan::CreateTable { name, columns } => {
                self.execute_create_table(&name, &columns)
            }
            LogicalPlan::Insert {
                table,
                columns,
                values,
            } => self.execute_insert(&table, columns.as_deref(), &values),
            LogicalPlan::Update {
                table,
                assignments,
                where_clause,
            } => self.execute_update(&table, &assignments, where_clause.as_ref()),
            LogicalPlan::Delete {
                table,
                where_clause,
            } => self.execute_delete(&table, where_clause.as_ref()),
            // Transaction operations
            LogicalPlan::Begin => self.begin_transaction(),
            LogicalPlan::Commit => self.commit_transaction(),
            LogicalPlan::Rollback => self.rollback_transaction(),
            LogicalPlan::Savepoint { name } => self.create_savepoint(&name),
            LogicalPlan::ReleaseSavepoint { name } => self.release_savepoint(&name),
            LogicalPlan::RollbackTo { name } => self.rollback_to_savepoint(&name),
            // Trigger operations
            LogicalPlan::CreateTrigger {
                name,
                timing,
                event,
                table,
                body,
            } => self.create_trigger(&name, timing, event, &table, body),
            LogicalPlan::DropTrigger { name } => self.drop_trigger(&name),
            _ => self.execute_query(plan),
        }
    }

    /// Execute a CREATE TABLE
    fn execute_create_table(&mut self, name: &str, columns: &[ColumnDef]) -> ExecResult {
        let schema = TableSchema {
            name: name.to_string(),
            columns: columns
                .iter()
                .map(|c| ColumnSchema {
                    name: c.name.clone(),
                    data_type: convert_data_type(&c.data_type),
                    nullable: c.nullable,
                    primary_key: c.primary_key,
                    unique: c.unique,
                    default: c.default.as_ref().map(eval_literal),
                    references: c.references.as_ref().map(convert_fk_ref),
                })
                .collect(),
            constraints: Vec::new(), // TODO: Handle table-level constraints
        };
        self.storage.create_table(schema)?;
        Ok(QueryResult::Success)
    }

    /// Execute an INSERT
    fn execute_insert(
        &mut self,
        table: &str,
        _columns: Option<&[String]>,
        values: &[Vec<Expr>],
    ) -> ExecResult {
        let schema = self.storage.get_schema(table)?.clone();
        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

        let mut count = 0;
        for value_row in values {
            let mut row: Row = value_row.iter().map(eval_literal).collect();

            // Fire BEFORE INSERT triggers
            self.fire_triggers(
                table,
                &TriggerEvent::Insert,
                &TriggerTiming::Before,
                &mut row,
                &column_names,
            )?;

            self.storage.insert(table, row.clone())?;

            // Fire AFTER INSERT triggers
            self.fire_triggers(
                table,
                &TriggerEvent::Insert,
                &TriggerTiming::After,
                &mut row,
                &column_names,
            )?;

            count += 1;
        }
        Ok(QueryResult::RowsAffected(count))
    }

    /// Execute an UPDATE
    fn execute_update(
        &mut self,
        table: &str,
        assignments: &[Assignment],
        where_clause: Option<&Expr>,
    ) -> ExecResult {
        let schema = self.storage.get_schema(table)?;
        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

        // Get all rows
        let rows = self.storage.scan(table)?;

        // Find rows to update and their new values
        let mut updates: Vec<(usize, Row)> = Vec::new();
        for (idx, row) in rows.iter().enumerate() {
            let should_update = match where_clause {
                Some(predicate) => eval_predicate(predicate, row, &column_names),
                None => true,
            };
            if should_update {
                // Apply assignments to create new row
                let mut new_row = row.clone();
                for assignment in assignments {
                    if let Some(col_idx) = column_names.iter().position(|c| c == &assignment.column)
                    {
                        new_row[col_idx] = eval_expr(&assignment.value, row, &column_names);
                    }
                }
                updates.push((idx, new_row));
            }
        }

        // Delete old rows and insert new ones
        // For simplicity, delete all matching rows then insert updated versions
        let count = updates.len();
        if count > 0 {
            // Delete matching rows
            let _ = self.storage.delete(table, |row| match where_clause {
                Some(predicate) => eval_predicate(predicate, row, &column_names),
                None => true,
            })?;

            // Insert updated rows
            for (_, new_row) in updates {
                self.storage.insert(table, new_row)?;
            }
        }

        Ok(QueryResult::RowsAffected(count))
    }

    /// Execute a DELETE with referential integrity support
    fn execute_delete(&mut self, table: &str, where_clause: Option<&Expr>) -> ExecResult {
        let schema = self.storage.get_schema(table)?.clone();
        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

        // Find rows to delete
        let all_rows = self.storage.scan(table)?;
        let rows_to_delete: Vec<Row> = all_rows
            .into_iter()
            .filter(|row| match where_clause {
                Some(predicate) => eval_predicate(predicate, row, &column_names),
                None => true,
            })
            .collect();

        if rows_to_delete.is_empty() {
            return Ok(QueryResult::RowsAffected(0));
        }

        // Find primary key columns
        let pk_columns: Vec<(usize, &str)> = schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.primary_key)
            .map(|(i, c)| (i, c.name.as_str()))
            .collect();

        // Get deleted key values (for checking foreign key references)
        let deleted_values: Vec<Vec<(&str, &Value)>> = rows_to_delete
            .iter()
            .map(|row| {
                pk_columns
                    .iter()
                    .filter_map(|(idx, col_name)| row.get(*idx).map(|v| (*col_name, v)))
                    .collect()
            })
            .collect();

        // Handle referential integrity for each referencing table
        self.handle_cascade_delete(table, &deleted_values)?;

        // Now delete the rows from the main table
        let deleted = self.storage.delete(table, |row| match where_clause {
            Some(predicate) => eval_predicate(predicate, row, &column_names),
            None => true,
        })?;

        Ok(QueryResult::RowsAffected(deleted))
    }

    /// Handle cascade delete for foreign key references
    fn handle_cascade_delete(
        &mut self,
        referenced_table: &str,
        deleted_values: &[Vec<(&str, &Value)>],
    ) -> Result<(), ExecError> {
        // Find all tables that reference this table
        let table_names = self.storage.table_names();
        let mut fk_refs: Vec<(String, String, String, StorageRefAction)> = Vec::new();

        for tbl_name in &table_names {
            if let Ok(schema) = self.storage.get_schema(tbl_name) {
                for col in &schema.columns {
                    if let Some(ref fk) = col.references {
                        if fk.table == referenced_table {
                            fk_refs.push((
                                tbl_name.clone(),
                                col.name.clone(),
                                fk.column.clone(),
                                fk.on_delete.clone(),
                            ));
                        }
                    }
                }
            }
        }

        // Process each foreign key reference
        for (child_table, child_col, parent_col, action) in fk_refs {
            let child_schema = self.storage.get_schema(&child_table)?.clone();
            let child_col_idx = child_schema
                .columns
                .iter()
                .position(|c| c.name == child_col)
                .ok_or_else(|| ExecError::ColumnNotFound(child_col.clone()))?;

            // Check for matching values in the child table
            for deleted_row in deleted_values {
                if let Some((_, deleted_val)) =
                    deleted_row.iter().find(|(col, _)| *col == parent_col)
                {
                    match action {
                        StorageRefAction::Cascade => {
                            // Recursively delete matching rows in child table
                            self.storage.delete(&child_table, |row| {
                                row.get(child_col_idx)
                                    .map(|v| values_equal(v, deleted_val))
                                    .unwrap_or(false)
                            })?;
                        }
                        StorageRefAction::SetNull => {
                            // Set the foreign key column to NULL
                            self.storage.update(
                                &child_table,
                                |row| {
                                    row.get(child_col_idx)
                                        .map(|v| values_equal(v, deleted_val))
                                        .unwrap_or(false)
                                },
                                |row| {
                                    if let Some(val) = row.get_mut(child_col_idx) {
                                        *val = Value::Null;
                                    }
                                },
                            )?;
                        }
                        StorageRefAction::SetDefault => {
                            // Set the foreign key column to its default value
                            let default_val = child_schema.columns[child_col_idx]
                                .default
                                .clone()
                                .unwrap_or(Value::Null);
                            self.storage.update(
                                &child_table,
                                |row| {
                                    row.get(child_col_idx)
                                        .map(|v| values_equal(v, deleted_val))
                                        .unwrap_or(false)
                                },
                                |row| {
                                    if let Some(val) = row.get_mut(child_col_idx) {
                                        *val = default_val.clone();
                                    }
                                },
                            )?;
                        }
                        StorageRefAction::Restrict | StorageRefAction::NoAction => {
                            // Check if any child rows reference this value
                            let child_rows = self.storage.scan(&child_table)?;
                            let has_reference = child_rows.iter().any(|row| {
                                row.get(child_col_idx)
                                    .map(|v| values_equal(v, deleted_val))
                                    .unwrap_or(false)
                            });

                            if has_reference {
                                return Err(ExecError::ForeignKeyViolation {
                                    table: child_table.clone(),
                                    column: child_col.clone(),
                                    references_table: referenced_table.to_string(),
                                    references_column: parent_col.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Begin a new transaction
    fn begin_transaction(&mut self) -> ExecResult {
        if self.transaction.active {
            return Err(ExecError::TransactionAlreadyActive);
        }
        // Take a snapshot of the current storage state
        self.transaction.snapshots.push(self.storage.clone());
        self.transaction.active = true;
        self.transaction.savepoints.clear();
        Ok(QueryResult::TransactionStarted)
    }

    /// Commit the current transaction
    fn commit_transaction(&mut self) -> ExecResult {
        if !self.transaction.active {
            return Err(ExecError::NoActiveTransaction);
        }
        // Discard snapshots - changes are permanent
        self.transaction.snapshots.clear();
        self.transaction.savepoints.clear();
        self.transaction.active = false;
        Ok(QueryResult::TransactionCommitted)
    }

    /// Rollback the current transaction
    fn rollback_transaction(&mut self) -> ExecResult {
        if !self.transaction.active {
            return Err(ExecError::NoActiveTransaction);
        }
        // Restore from the initial snapshot
        if let Some(snapshot) = self.transaction.snapshots.first().cloned() {
            self.storage = snapshot;
        }
        self.transaction.snapshots.clear();
        self.transaction.savepoints.clear();
        self.transaction.active = false;
        Ok(QueryResult::TransactionRolledBack)
    }

    /// Create a savepoint
    fn create_savepoint(&mut self, name: &str) -> ExecResult {
        if !self.transaction.active {
            return Err(ExecError::NoActiveTransaction);
        }
        // Take a snapshot and record the savepoint
        let snapshot_idx = self.transaction.snapshots.len();
        self.transaction.snapshots.push(self.storage.clone());
        self.transaction
            .savepoints
            .push((name.to_string(), snapshot_idx));
        Ok(QueryResult::SavepointCreated(name.to_string()))
    }

    /// Release a savepoint
    fn release_savepoint(&mut self, name: &str) -> ExecResult {
        if !self.transaction.active {
            return Err(ExecError::NoActiveTransaction);
        }
        // Find and remove the savepoint (but keep the snapshot for now)
        let pos = self
            .transaction
            .savepoints
            .iter()
            .position(|(n, _)| n == name);
        match pos {
            Some(idx) => {
                self.transaction.savepoints.remove(idx);
                Ok(QueryResult::SavepointReleased(name.to_string()))
            }
            None => Err(ExecError::SavepointNotFound(name.to_string())),
        }
    }

    /// Rollback to a savepoint
    fn rollback_to_savepoint(&mut self, name: &str) -> ExecResult {
        if !self.transaction.active {
            return Err(ExecError::NoActiveTransaction);
        }
        // Find the savepoint
        let pos = self
            .transaction
            .savepoints
            .iter()
            .position(|(n, _)| n == name);
        match pos {
            Some(idx) => {
                let (_, snapshot_idx) = self.transaction.savepoints[idx].clone();
                // Restore from the savepoint's snapshot
                if let Some(snapshot) = self.transaction.snapshots.get(snapshot_idx).cloned() {
                    self.storage = snapshot;
                }
                // Remove all savepoints after this one
                self.transaction.savepoints.truncate(idx + 1);
                // Remove all snapshots after the savepoint's snapshot
                self.transaction.snapshots.truncate(snapshot_idx + 1);
                Ok(QueryResult::RolledBackToSavepoint(name.to_string()))
            }
            None => Err(ExecError::SavepointNotFound(name.to_string())),
        }
    }

    /// Create a trigger
    fn create_trigger(
        &mut self,
        name: &str,
        timing: TriggerTiming,
        event: TriggerEvent,
        table: &str,
        body: Vec<TriggerAction>,
    ) -> ExecResult {
        // Check if trigger already exists
        if self.triggers.iter().any(|t| t.name == name) {
            return Err(ExecError::TriggerAlreadyExists(name.to_string()));
        }

        self.triggers.push(Trigger {
            name: name.to_string(),
            timing,
            event,
            table: table.to_string(),
            body,
        });

        Ok(QueryResult::Success)
    }

    /// Drop a trigger
    fn drop_trigger(&mut self, name: &str) -> ExecResult {
        let pos = self.triggers.iter().position(|t| t.name == name);
        match pos {
            Some(idx) => {
                self.triggers.remove(idx);
                Ok(QueryResult::Success)
            }
            None => Err(ExecError::TriggerNotFound(name.to_string())),
        }
    }

    /// Fire triggers for a table and event
    fn fire_triggers(
        &self,
        table: &str,
        event: &TriggerEvent,
        timing: &TriggerTiming,
        row: &mut Row,
        column_names: &[String],
    ) -> Result<(), ExecError> {
        for trigger in &self.triggers {
            if trigger.table == table && &trigger.event == event && &trigger.timing == timing {
                for action in &trigger.body {
                    match action {
                        TriggerAction::SetColumn { column, value } => {
                            if let Some(idx) = column_names.iter().position(|c| c == column) {
                                row[idx] = eval_expr(value, row, column_names);
                            }
                        }
                        TriggerAction::RaiseError(msg) => {
                            return Err(ExecError::TriggerError(msg.clone()));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute a SELECT query
    fn execute_query(&self, plan: LogicalPlan) -> ExecResult {
        match plan {
            LogicalPlan::Scan { table } => {
                let schema = self.storage.get_schema(&table)?;
                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();
                let rows = self.storage.scan(&table)?;
                Ok(QueryResult::Select {
                    columns: column_names,
                    rows,
                })
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
            } => {
                let left_result = self.execute_query(*left)?;
                let right_result = self.execute_query(*right)?;

                match (left_result, right_result) {
                    (
                        QueryResult::Select {
                            columns: left_cols,
                            rows: left_rows,
                        },
                        QueryResult::Select {
                            columns: right_cols,
                            rows: right_rows,
                        },
                    ) => {
                        // Combine column names
                        let mut combined_cols = left_cols.clone();
                        combined_cols.extend(right_cols.clone());

                        let mut result_rows: Vec<Vec<Value>> = Vec::new();

                        match join_type {
                            JoinType::Inner => {
                                // INNER JOIN: only rows where ON condition matches
                                for left_row in &left_rows {
                                    for right_row in &right_rows {
                                        let combined_row = combine_rows(left_row, right_row);
                                        if check_join_condition(&on, &combined_row, &combined_cols)
                                        {
                                            result_rows.push(combined_row);
                                        }
                                    }
                                }
                            }
                            JoinType::Left => {
                                // LEFT JOIN: all left rows, matching right rows or NULLs
                                for left_row in &left_rows {
                                    let mut matched = false;
                                    for right_row in &right_rows {
                                        let combined_row = combine_rows(left_row, right_row);
                                        if check_join_condition(&on, &combined_row, &combined_cols)
                                        {
                                            result_rows.push(combined_row);
                                            matched = true;
                                        }
                                    }
                                    if !matched {
                                        // Add left row with NULLs for right columns
                                        let null_row: Vec<Value> =
                                            vec![Value::Null; right_cols.len()];
                                        result_rows.push(combine_rows(left_row, &null_row));
                                    }
                                }
                            }
                            JoinType::Right => {
                                // RIGHT JOIN: all right rows, matching left rows or NULLs
                                for right_row in &right_rows {
                                    let mut matched = false;
                                    for left_row in &left_rows {
                                        let combined_row = combine_rows(left_row, right_row);
                                        if check_join_condition(&on, &combined_row, &combined_cols)
                                        {
                                            result_rows.push(combined_row);
                                            matched = true;
                                        }
                                    }
                                    if !matched {
                                        // Add right row with NULLs for left columns
                                        let null_row: Vec<Value> =
                                            vec![Value::Null; left_cols.len()];
                                        result_rows.push(combine_rows(&null_row, right_row));
                                    }
                                }
                            }
                            JoinType::Full => {
                                // FULL OUTER JOIN: all rows from both sides
                                let mut left_matched: Vec<bool> = vec![false; left_rows.len()];
                                let mut right_matched: Vec<bool> = vec![false; right_rows.len()];

                                for (li, left_row) in left_rows.iter().enumerate() {
                                    for (ri, right_row) in right_rows.iter().enumerate() {
                                        let combined_row = combine_rows(left_row, right_row);
                                        if check_join_condition(&on, &combined_row, &combined_cols)
                                        {
                                            result_rows.push(combined_row);
                                            left_matched[li] = true;
                                            right_matched[ri] = true;
                                        }
                                    }
                                }

                                // Add unmatched left rows
                                for (li, left_row) in left_rows.iter().enumerate() {
                                    if !left_matched[li] {
                                        let null_row: Vec<Value> =
                                            vec![Value::Null; right_cols.len()];
                                        result_rows.push(combine_rows(left_row, &null_row));
                                    }
                                }

                                // Add unmatched right rows
                                for (ri, right_row) in right_rows.iter().enumerate() {
                                    if !right_matched[ri] {
                                        let null_row: Vec<Value> =
                                            vec![Value::Null; left_cols.len()];
                                        result_rows.push(combine_rows(&null_row, right_row));
                                    }
                                }
                            }
                            JoinType::Cross => {
                                // CROSS JOIN: cartesian product
                                for left_row in &left_rows {
                                    for right_row in &right_rows {
                                        result_rows.push(combine_rows(left_row, right_row));
                                    }
                                }
                            }
                        }

                        Ok(QueryResult::Select {
                            columns: combined_cols,
                            rows: result_rows,
                        })
                    }
                    _ => Err(ExecError::InvalidExpression(
                        "Join requires Select inputs".to_string(),
                    )),
                }
            }
            LogicalPlan::Filter { input, predicate } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        let filtered: Vec<Vec<Value>> = rows
                            .into_iter()
                            .filter(|row| eval_predicate(&predicate, row, &columns))
                            .collect();
                        Ok(QueryResult::Select {
                            columns,
                            rows: filtered,
                        })
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::Projection { input, exprs } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        // Handle SELECT *
                        if exprs.len() == 1 {
                            if let (Expr::Column(c), _) = &exprs[0] {
                                if c == "*" {
                                    return Ok(QueryResult::Select { columns, rows });
                                }
                            }
                        }

                        // Check if any expression is an aggregate
                        let has_aggregate = exprs.iter().any(|(expr, _)| is_aggregate(expr));

                        let new_columns: Vec<String> = exprs
                            .iter()
                            .enumerate()
                            .map(|(i, (expr, alias))| {
                                alias.clone().unwrap_or_else(|| match expr {
                                    Expr::Column(c) => c.clone(),
                                    Expr::Aggregate { func, .. } => {
                                        format!("{:?}", func).to_lowercase()
                                    }
                                    _ => format!("col{}", i),
                                })
                            })
                            .collect();

                        if has_aggregate {
                            // Aggregate all rows into a single result row
                            let aggregated_row: Vec<Value> = exprs
                                .iter()
                                .map(|(expr, _)| eval_aggregate(expr, &rows, &columns))
                                .collect();

                            Ok(QueryResult::Select {
                                columns: new_columns,
                                rows: vec![aggregated_row],
                            })
                        } else {
                            let new_rows: Vec<Vec<Value>> = rows
                                .iter()
                                .map(|row| {
                                    exprs
                                        .iter()
                                        .map(|(expr, _)| eval_expr(expr, row, &columns))
                                        .collect()
                                })
                                .collect();

                            Ok(QueryResult::Select {
                                columns: new_columns,
                                rows: new_rows,
                            })
                        }
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::Sort { input, order_by } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, mut rows } => {
                        // Sort by first order_by expression
                        if let Some(ob) = order_by.first() {
                            rows.sort_by(|a, b| {
                                let val_a = eval_expr(&ob.expr, a, &columns);
                                let val_b = eval_expr(&ob.expr, b, &columns);
                                let cmp = compare_values(&val_a, &val_b);
                                if ob.desc {
                                    cmp.reverse()
                                } else {
                                    cmp
                                }
                            });
                        }
                        Ok(QueryResult::Select { columns, rows })
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        let limited: Vec<Vec<Value>> =
                            rows.into_iter().skip(offset).take(limit).collect();
                        Ok(QueryResult::Select {
                            columns,
                            rows: limited,
                        })
                    }
                    other => Ok(other),
                }
            }
            _ => Err(ExecError::InvalidExpression("Unsupported plan".to_string())),
        }
    }
}

/// Convert parser DataType to storage DataType
fn convert_data_type(dt: &DataType) -> StorageDataType {
    match dt {
        DataType::Int => StorageDataType::Int,
        DataType::Float => StorageDataType::Float,
        DataType::Text => StorageDataType::Text,
        DataType::Bool => StorageDataType::Bool,
    }
}

/// Convert parser foreign key ref to storage format
fn convert_fk_ref(fk: &ParserFKRef) -> ForeignKeyRef {
    ForeignKeyRef {
        table: fk.table.clone(),
        column: fk.column.clone(),
        on_delete: convert_ref_action(&fk.on_delete),
        on_update: convert_ref_action(&fk.on_update),
    }
}

/// Convert parser referential action to storage format
fn convert_ref_action(action: &ParserRefAction) -> StorageRefAction {
    match action {
        ParserRefAction::NoAction => StorageRefAction::NoAction,
        ParserRefAction::Cascade => StorageRefAction::Cascade,
        ParserRefAction::SetNull => StorageRefAction::SetNull,
        ParserRefAction::SetDefault => StorageRefAction::SetDefault,
        ParserRefAction::Restrict => StorageRefAction::Restrict,
    }
}

/// Evaluate a literal expression to a Value
fn eval_literal(expr: &Expr) -> Value {
    match expr {
        Expr::Integer(n) => Value::Int(*n),
        Expr::Float(f) => Value::Float(*f),
        Expr::String(s) => Value::Text(s.clone()),
        Expr::Boolean(b) => Value::Bool(*b),
        Expr::Null => Value::Null,
        Expr::UnaryOp {
            op: UnaryOp::Neg,
            expr,
        } => match eval_literal(expr) {
            Value::Int(n) => Value::Int(-n),
            Value::Float(f) => Value::Float(-f),
            other => other,
        },
        _ => Value::Null,
    }
}

/// Evaluate an expression against a row
fn eval_expr(expr: &Expr, row: &[Value], columns: &[String]) -> Value {
    match expr {
        Expr::Column(name) => {
            if let Some(idx) = columns.iter().position(|c| c == name) {
                row.get(idx).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        Expr::Integer(n) => Value::Int(*n),
        Expr::Float(f) => Value::Float(*f),
        Expr::String(s) => Value::Text(s.clone()),
        Expr::Boolean(b) => Value::Bool(*b),
        Expr::Null => Value::Null,
        Expr::UnaryOp { op, expr } => {
            let val = eval_expr(expr, row, columns);
            match op {
                UnaryOp::Neg => match val {
                    Value::Int(n) => Value::Int(-n),
                    Value::Float(f) => Value::Float(-f),
                    _ => Value::Null,
                },
                UnaryOp::Not => match val {
                    Value::Bool(b) => Value::Bool(!b),
                    _ => Value::Null,
                },
            }
        }
        Expr::BinaryOp { left, op, right } => {
            let l = eval_expr(left, row, columns);
            let r = eval_expr(right, row, columns);
            eval_binary_op(&l, op, &r)
        }
        // Aggregates should not be evaluated per-row; they're handled by eval_aggregate
        Expr::Aggregate { .. } => Value::Null,
    }
}

/// Evaluate a binary operation
fn eval_binary_op(left: &Value, op: &BinaryOp, right: &Value) -> Value {
    match op {
        BinaryOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Mod => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a % b),
            _ => Value::Null,
        },
        BinaryOp::Eq => Value::Bool(values_equal(left, right)),
        BinaryOp::NotEq => Value::Bool(!values_equal(left, right)),
        BinaryOp::Lt => Value::Bool(matches!(
            compare_values(left, right),
            std::cmp::Ordering::Less
        )),
        BinaryOp::Gt => Value::Bool(matches!(
            compare_values(left, right),
            std::cmp::Ordering::Greater
        )),
        BinaryOp::LtEq => Value::Bool(!matches!(
            compare_values(left, right),
            std::cmp::Ordering::Greater
        )),
        BinaryOp::GtEq => Value::Bool(!matches!(
            compare_values(left, right),
            std::cmp::Ordering::Less
        )),
        BinaryOp::And => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a && *b),
            _ => Value::Null,
        },
        BinaryOp::Or => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a || *b),
            _ => Value::Null,
        },
    }
}

/// Check if two values are equal
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

/// Compare two values
fn compare_values(left: &Value, right: &Value) -> std::cmp::Ordering {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::Text(a), Value::Text(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    }
}

/// Evaluate a predicate expression
fn eval_predicate(expr: &Expr, row: &[Value], columns: &[String]) -> bool {
    match eval_expr(expr, row, columns) {
        Value::Bool(b) => b,
        _ => false,
    }
}

/// Combine two rows into one
fn combine_rows(left: &[Value], right: &[Value]) -> Vec<Value> {
    let mut combined = left.to_vec();
    combined.extend(right.iter().cloned());
    combined
}

/// Check if JOIN condition is satisfied
fn check_join_condition(on: &Option<Expr>, row: &[Value], columns: &[String]) -> bool {
    match on {
        Some(expr) => eval_predicate(expr, row, columns),
        None => true, // No ON clause (e.g., CROSS JOIN)
    }
}

/// Check if an expression is an aggregate function
fn is_aggregate(expr: &Expr) -> bool {
    matches!(expr, Expr::Aggregate { .. })
}

/// Evaluate an aggregate expression over all rows
fn eval_aggregate(expr: &Expr, rows: &[Vec<Value>], columns: &[String]) -> Value {
    match expr {
        Expr::Aggregate { func, arg } => {
            let values: Vec<Value> = rows
                .iter()
                .map(|row| eval_expr(arg, row, columns))
                .collect();

            match func {
                AggregateFunc::Count => {
                    // COUNT(*) counts all rows, COUNT(col) counts non-NULL values
                    match arg.as_ref() {
                        Expr::Column(c) if c == "*" => Value::Int(rows.len() as i64),
                        _ => {
                            let count = values.iter().filter(|v| !matches!(v, Value::Null)).count();
                            Value::Int(count as i64)
                        }
                    }
                }
                AggregateFunc::Sum => {
                    let sum: f64 = values
                        .iter()
                        .filter_map(|v| match v {
                            Value::Int(n) => Some(*n as f64),
                            Value::Float(f) => Some(*f),
                            _ => None,
                        })
                        .sum();
                    // Return Int if all inputs were Int, otherwise Float
                    if values
                        .iter()
                        .all(|v| matches!(v, Value::Int(_) | Value::Null))
                    {
                        Value::Int(sum as i64)
                    } else {
                        Value::Float(sum)
                    }
                }
                AggregateFunc::Avg => {
                    let nums: Vec<f64> = values
                        .iter()
                        .filter_map(|v| match v {
                            Value::Int(n) => Some(*n as f64),
                            Value::Float(f) => Some(*f),
                            _ => None,
                        })
                        .collect();
                    if nums.is_empty() {
                        Value::Null
                    } else {
                        Value::Float(nums.iter().sum::<f64>() / nums.len() as f64)
                    }
                }
                AggregateFunc::Min => values
                    .into_iter()
                    .filter(|v| !matches!(v, Value::Null))
                    .min_by(compare_values)
                    .unwrap_or(Value::Null),
                AggregateFunc::Max => values
                    .into_iter()
                    .filter(|v| !matches!(v, Value::Null))
                    .max_by(compare_values)
                    .unwrap_or(Value::Null),
            }
        }
        // For non-aggregate expressions, just return the first row's value (or NULL if empty)
        _ => rows
            .first()
            .map(|row| eval_expr(expr, row, columns))
            .unwrap_or(Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_and_insert() {
        let mut engine = Engine::new();

        let result = engine.execute("CREATE TABLE users (id INT, name TEXT)");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), QueryResult::Success);

        let result = engine.execute("INSERT INTO users (id, name) VALUES (1, 'alice')");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));
    }

    #[test]
    fn test_select_all() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_with_where() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users WHERE id = 1");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_with_order_by() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (3, 'charlie')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users ORDER BY id");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
                assert_eq!(rows[1][0], Value::Int(2));
                assert_eq!(rows[2][0], Value::Int(3));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_with_limit() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (3, 'charlie')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users LIMIT 2");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_projection() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();

        let result = engine.execute("SELECT name FROM users");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["name"]);
                assert_eq!(rows[0][0], Value::Text("alice".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_arithmetic_expression() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE nums (a INT, b INT)").unwrap();
        engine
            .execute("INSERT INTO nums (a, b) VALUES (10, 3)")
            .unwrap();

        let result = engine.execute("SELECT a + b AS total FROM nums");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["total"]);
                assert_eq!(rows[0][0], Value::Int(13));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_comparison_operators() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (5)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (10)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (15)").unwrap();

        let result = engine.execute("SELECT * FROM t WHERE x > 5");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }

        let result = engine.execute("SELECT * FROM t WHERE x >= 10");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_update() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        let result = engine.execute("UPDATE users SET name = 'alicia' WHERE id = 1");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));

        // Verify the update
        let result = engine.execute("SELECT name FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("alicia".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_update_multiple_rows() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT, y INT)").unwrap();
        engine
            .execute("INSERT INTO t (x, y) VALUES (1, 10)")
            .unwrap();
        engine
            .execute("INSERT INTO t (x, y) VALUES (2, 20)")
            .unwrap();
        engine
            .execute("INSERT INTO t (x, y) VALUES (3, 30)")
            .unwrap();

        let result = engine.execute("UPDATE t SET y = 100 WHERE x > 1");
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(2));

        let result = engine.execute("SELECT * FROM t WHERE y = 100");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_delete() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        let result = engine.execute("DELETE FROM users WHERE id = 1");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));

        // Verify the delete
        let result = engine.execute("SELECT * FROM users");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_delete_all() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (3)").unwrap();

        let result = engine.execute("DELETE FROM t");
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(3));

        let result = engine.execute("SELECT * FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 0);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_inner_join() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("CREATE TABLE orders (id INT, user_id INT, item TEXT)")
            .unwrap();

        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, item) VALUES (1, 1, 'book')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, item) VALUES (2, 1, 'pen')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, item) VALUES (3, 3, 'notebook')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users JOIN orders ON id = user_id");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                // Should have columns from both tables
                assert_eq!(columns.len(), 5);
                // Alice has 2 orders, Bob has 0 (user_id 3 doesn't match anyone)
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_left_join() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("CREATE TABLE orders (id INT, user_id INT)")
            .unwrap();

        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id) VALUES (1, 1)")
            .unwrap();

        let result = engine.execute("SELECT * FROM users LEFT JOIN orders ON id = user_id");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                // Alice matches, Bob doesn't but still included with NULLs
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_cross_join() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE a (x INT)").unwrap();
        engine.execute("CREATE TABLE b (y INT)").unwrap();

        engine.execute("INSERT INTO a (x) VALUES (1)").unwrap();
        engine.execute("INSERT INTO a (x) VALUES (2)").unwrap();
        engine.execute("INSERT INTO b (y) VALUES (10)").unwrap();
        engine.execute("INSERT INTO b (y) VALUES (20)").unwrap();
        engine.execute("INSERT INTO b (y) VALUES (30)").unwrap();

        let result = engine.execute("SELECT * FROM a CROSS JOIN b");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns.len(), 2);
                // Cartesian product: 2 * 3 = 6 rows
                assert_eq!(rows.len(), 6);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_count_star() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (3, 'charlie')")
            .unwrap();

        let result = engine.execute("SELECT COUNT(*) FROM users");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns.len(), 1);
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(3));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_sum() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE nums (x INT)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (10)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (20)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (30)").unwrap();

        let result = engine.execute("SELECT SUM(x) FROM nums");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(60));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_avg() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE nums (x INT)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (10)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (20)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (30)").unwrap();

        let result = engine.execute("SELECT AVG(x) FROM nums");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Float(20.0));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_min_max() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE nums (x INT)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (15)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (5)").unwrap();
        engine.execute("INSERT INTO nums (x) VALUES (25)").unwrap();

        let result = engine.execute("SELECT MIN(x), MAX(x) FROM nums");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(5));
                assert_eq!(rows[0][1], Value::Int(25));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_count_with_where() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, active INT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, active) VALUES (1, 1)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, active) VALUES (2, 0)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, active) VALUES (3, 1)")
            .unwrap();

        let result = engine.execute("SELECT COUNT(*) FROM users WHERE active = 1");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(2));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_begin_commit() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();

        // Begin transaction
        let result = engine.execute("BEGIN");
        assert_eq!(result.unwrap(), QueryResult::TransactionStarted);

        // Insert some data
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();

        // Commit transaction
        let result = engine.execute("COMMIT");
        assert_eq!(result.unwrap(), QueryResult::TransactionCommitted);

        // Data should still be there
        let result = engine.execute("SELECT * FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_begin_rollback() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();

        // Begin transaction
        engine.execute("BEGIN").unwrap();

        // Insert more data
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (3)").unwrap();

        // Verify data is there during transaction
        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(3));
            }
            _ => panic!("Expected Select result"),
        }

        // Rollback transaction
        let result = engine.execute("ROLLBACK");
        assert_eq!(result.unwrap(), QueryResult::TransactionRolledBack);

        // Only original data should remain
        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_savepoint() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();

        // Begin transaction
        engine.execute("BEGIN").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();

        // Create savepoint
        let result = engine.execute("SAVEPOINT sp1");
        assert_eq!(
            result.unwrap(),
            QueryResult::SavepointCreated("sp1".to_string())
        );

        // Insert more data
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();

        // Rollback to savepoint
        let result = engine.execute("ROLLBACK TO sp1");
        assert_eq!(
            result.unwrap(),
            QueryResult::RolledBackToSavepoint("sp1".to_string())
        );

        // Only data before savepoint should remain
        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }

        // Commit
        engine.execute("COMMIT").unwrap();

        // Data should persist
        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_release_savepoint() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();

        engine.execute("BEGIN").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();
        engine.execute("SAVEPOINT sp1").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();

        // Release savepoint
        let result = engine.execute("RELEASE SAVEPOINT sp1");
        assert_eq!(
            result.unwrap(),
            QueryResult::SavepointReleased("sp1".to_string())
        );

        // Trying to rollback to released savepoint should fail
        let result = engine.execute("ROLLBACK TO sp1");
        assert_eq!(
            result.unwrap_err(),
            ExecError::SavepointNotFound("sp1".to_string())
        );

        engine.execute("COMMIT").unwrap();
    }

    #[test]
    fn test_nested_savepoints() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();

        engine.execute("BEGIN").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();
        engine.execute("SAVEPOINT sp1").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();
        engine.execute("SAVEPOINT sp2").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (3)").unwrap();

        // Rollback to first savepoint (should remove sp2)
        engine.execute("ROLLBACK TO sp1").unwrap();

        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }

        // sp2 should no longer exist
        let result = engine.execute("ROLLBACK TO sp2");
        assert_eq!(
            result.unwrap_err(),
            ExecError::SavepointNotFound("sp2".to_string())
        );

        engine.execute("COMMIT").unwrap();
    }

    #[test]
    fn test_transaction_error_no_active() {
        let mut engine = Engine::new();

        // Commit without begin should fail
        let result = engine.execute("COMMIT");
        assert_eq!(result.unwrap_err(), ExecError::NoActiveTransaction);

        // Rollback without begin should fail
        let result = engine.execute("ROLLBACK");
        assert_eq!(result.unwrap_err(), ExecError::NoActiveTransaction);

        // Savepoint without begin should fail
        let result = engine.execute("SAVEPOINT sp1");
        assert_eq!(result.unwrap_err(), ExecError::NoActiveTransaction);
    }

    #[test]
    fn test_transaction_already_active() {
        let mut engine = Engine::new();

        engine.execute("BEGIN").unwrap();

        // Second BEGIN should fail
        let result = engine.execute("BEGIN");
        assert_eq!(result.unwrap_err(), ExecError::TransactionAlreadyActive);

        engine.execute("COMMIT").unwrap();
    }

    #[test]
    fn test_update_in_transaction_rollback() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();

        engine.execute("BEGIN").unwrap();
        engine
            .execute("UPDATE users SET name = 'bob' WHERE id = 1")
            .unwrap();

        // Verify update during transaction
        let result = engine.execute("SELECT name FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("bob".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Rollback
        engine.execute("ROLLBACK").unwrap();

        // Original value should be restored
        let result = engine.execute("SELECT name FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("alice".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_delete_in_transaction_rollback() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (2)").unwrap();

        engine.execute("BEGIN").unwrap();
        engine.execute("DELETE FROM t WHERE x = 1").unwrap();

        // Verify delete during transaction
        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }

        // Rollback
        engine.execute("ROLLBACK").unwrap();

        // Deleted row should be restored
        let result = engine.execute("SELECT COUNT(*) FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(2));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_cascade_delete() {
        let mut engine = Engine::new();

        // Create parent table
        engine
            .execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();

        // Create child table with CASCADE delete
        engine
            .execute(
                "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id) ON DELETE CASCADE, item TEXT)",
            )
            .unwrap();

        // Insert data
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, item) VALUES (1, 1, 'book')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, item) VALUES (2, 1, 'pen')")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, item) VALUES (3, 2, 'notebook')")
            .unwrap();

        // Delete user 1 - should cascade delete their orders
        let result = engine.execute("DELETE FROM users WHERE id = 1");
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));

        // Verify orders for user 1 are deleted
        let result = engine.execute("SELECT COUNT(*) FROM orders");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1)); // Only bob's order remains
            }
            _ => panic!("Expected Select result"),
        }

        // Verify only bob's order remains
        let result = engine.execute("SELECT item FROM orders");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("notebook".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_set_null_on_delete() {
        let mut engine = Engine::new();

        // Create parent table
        engine
            .execute("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT)")
            .unwrap();

        // Create child table with SET NULL on delete
        engine
            .execute(
                "CREATE TABLE employees (id INT PRIMARY KEY, dept_id INT REFERENCES departments(id) ON DELETE SET NULL, name TEXT)",
            )
            .unwrap();

        // Insert data
        engine
            .execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
            .unwrap();
        engine
            .execute("INSERT INTO departments (id, name) VALUES (2, 'Sales')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, dept_id, name) VALUES (1, 1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, dept_id, name) VALUES (2, 1, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, dept_id, name) VALUES (3, 2, 'charlie')")
            .unwrap();

        // Delete Engineering department - should set dept_id to NULL for alice and bob
        let result = engine.execute("DELETE FROM departments WHERE id = 1");
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));

        // Verify employees still exist but with NULL dept_id
        let result = engine.execute("SELECT COUNT(*) FROM employees");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(3)); // All employees still exist
            }
            _ => panic!("Expected Select result"),
        }

        // Verify alice's dept_id is NULL
        let result = engine.execute("SELECT dept_id FROM employees WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Null);
            }
            _ => panic!("Expected Select result"),
        }

        // Charlie's dept should still be 2
        let result = engine.execute("SELECT dept_id FROM employees WHERE id = 3");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(2));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_restrict_on_delete() {
        let mut engine = Engine::new();

        // Create parent table
        engine
            .execute("CREATE TABLE categories (id INT PRIMARY KEY, name TEXT)")
            .unwrap();

        // Create child table with RESTRICT on delete
        engine
            .execute(
                "CREATE TABLE products (id INT PRIMARY KEY, category_id INT REFERENCES categories(id) ON DELETE RESTRICT, name TEXT)",
            )
            .unwrap();

        // Insert data
        engine
            .execute("INSERT INTO categories (id, name) VALUES (1, 'Electronics')")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, category_id, name) VALUES (1, 1, 'Phone')")
            .unwrap();

        // Try to delete category - should fail due to RESTRICT
        let result = engine.execute("DELETE FROM categories WHERE id = 1");
        assert!(matches!(
            result.unwrap_err(),
            ExecError::ForeignKeyViolation { .. }
        ));

        // Verify category still exists
        let result = engine.execute("SELECT COUNT(*) FROM categories");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_no_fk_reference_delete() {
        let mut engine = Engine::new();

        // Create tables without foreign key
        engine
            .execute("CREATE TABLE parent (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        engine
            .execute("CREATE TABLE child (id INT PRIMARY KEY, parent_id INT, name TEXT)")
            .unwrap();

        // Insert data
        engine
            .execute("INSERT INTO parent (id, name) VALUES (1, 'p1')")
            .unwrap();
        engine
            .execute("INSERT INTO child (id, parent_id, name) VALUES (1, 1, 'c1')")
            .unwrap();

        // Delete parent - should work since there's no FK constraint
        let result = engine.execute("DELETE FROM parent WHERE id = 1");
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));

        // Child should still exist
        let result = engine.execute("SELECT COUNT(*) FROM child");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_create_and_drop_trigger() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE t (id INT, name TEXT)")
            .unwrap();

        // Create a trigger
        let result = engine.execute(
            "CREATE TRIGGER set_name BEFORE INSERT ON t FOR EACH ROW SET name = 'default'",
        );
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Try to create duplicate trigger
        let result = engine
            .execute("CREATE TRIGGER set_name BEFORE INSERT ON t FOR EACH ROW SET name = 'other'");
        assert_eq!(
            result.unwrap_err(),
            ExecError::TriggerAlreadyExists("set_name".to_string())
        );

        // Drop the trigger
        let result = engine.execute("DROP TRIGGER set_name");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Try to drop non-existent trigger
        let result = engine.execute("DROP TRIGGER nonexistent");
        assert_eq!(
            result.unwrap_err(),
            ExecError::TriggerNotFound("nonexistent".to_string())
        );
    }

    #[test]
    fn test_before_insert_trigger_set_column() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT, status TEXT)")
            .unwrap();

        // Create a BEFORE INSERT trigger that sets a default status
        engine
            .execute(
                "CREATE TRIGGER set_status BEFORE INSERT ON users FOR EACH ROW SET status = 'active'",
            )
            .unwrap();

        // Insert without status - trigger should set it
        engine
            .execute("INSERT INTO users (id, name, status) VALUES (1, 'alice', 'pending')")
            .unwrap();

        // The status should be overwritten by the trigger
        let result = engine.execute("SELECT status FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("active".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_before_insert_trigger_raise_error() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE orders (id INT, amount INT)")
            .unwrap();

        // Create a trigger that prevents inserts (for testing RAISE ERROR)
        engine
            .execute(
                "CREATE TRIGGER no_inserts BEFORE INSERT ON orders FOR EACH ROW RAISE ERROR 'Inserts not allowed'",
            )
            .unwrap();

        // Try to insert - should fail
        let result = engine.execute("INSERT INTO orders (id, amount) VALUES (1, 100)");
        assert_eq!(
            result.unwrap_err(),
            ExecError::TriggerError("Inserts not allowed".to_string())
        );

        // Verify no data was inserted
        let result = engine.execute("SELECT COUNT(*) FROM orders");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(0));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_multiple_triggers_same_table() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE items (id INT, name TEXT, created TEXT)")
            .unwrap();

        // Create multiple triggers
        engine
            .execute(
                "CREATE TRIGGER set_name BEFORE INSERT ON items FOR EACH ROW SET name = 'item'",
            )
            .unwrap();
        engine
            .execute(
                "CREATE TRIGGER set_created BEFORE INSERT ON items FOR EACH ROW SET created = 'now'",
            )
            .unwrap();

        // Insert - both triggers should fire
        engine
            .execute("INSERT INTO items (id, name, created) VALUES (1, 'original', 'old')")
            .unwrap();

        let result = engine.execute("SELECT name, created FROM items WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("item".to_string()));
                assert_eq!(rows[0][1], Value::Text("now".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }
}
