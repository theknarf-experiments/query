//! Query executor - executes logical plans against storage

use std::collections::HashMap;

use logical::{
    delete, insert, update, ColumnSchema, DataType as StorageDataType, ExportData, ForeignKeyRef,
    FunctionDef, ImportData, JsonValue, MemoryEngine, OperationError,
    ReferentialAction as StorageRefAction, Row, StorageEngine, StorageError,
    TableConstraint as StorageTableConstraint, TableSchema, TriggerDef,
    TriggerEvent as StorageTriggerEvent, TriggerTiming as StorageTriggerTiming, Value,
};

use crate::runtime::SqlRuntime;
use sql_parser::{
    AggregateFunc, AlterAction, Assignment, BinaryOp, ColumnDef, Cte, DataType, Expr,
    ForeignKeyRef as ParserFKRef, JoinType, OrderBy, ProcedureStatement,
    ReferentialAction as ParserRefAction, SetOperator, Statement,
    TableConstraint as ParserTableConstraint, TriggerAction, TriggerActionType, TriggerEvent,
    TriggerTiming, UnaryOp, WindowFunc,
};
use sql_planner::LogicalPlan;

/// CTE context - stores materialized CTE results during query execution
#[derive(Default, Clone)]
struct CteContext {
    /// Maps CTE name to (columns, rows)
    ctes: HashMap<String, (Vec<String>, Vec<Vec<Value>>)>,
}

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
    /// Procedure not found
    ProcedureNotFound(String),
    /// Procedure already exists
    ProcedureAlreadyExists(String),
    /// Procedure parameter count mismatch
    ProcedureArgCountMismatch { expected: usize, got: usize },
    /// Trigger depth exceeded (infinite recursion prevention)
    TriggerDepthExceeded { depth: u32, max_depth: u32 },
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

/// Stored view definition
#[derive(Debug, Clone)]
struct ViewDefinition {
    columns: Option<Vec<String>>,
    query: sql_planner::LogicalPlan,
}

/// Stored procedure definition
#[derive(Debug, Clone)]
struct ProcedureDefinition {
    params: Vec<sql_parser::ProcedureParam>,
    body: Vec<sql_parser::ProcedureStatement>,
}

/// Foreign key trigger info for creating implicit triggers
struct FkTriggerInfo {
    child_table: String,
    child_column: String,
    parent_table: String,
    parent_column: String,
    on_delete: StorageRefAction,
    on_update: StorageRefAction,
}

/// Unique constraint trigger info for creating implicit triggers
struct UniqueConstraintInfo {
    table: String,
    columns: Vec<String>,
    is_primary_key: bool,
}

/// Database engine that executes queries
pub struct Engine {
    storage: MemoryEngine,
    transaction: TransactionState,
    views: HashMap<String, ViewDefinition>,
    procedures: HashMap<String, ProcedureDefinition>,
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
            views: HashMap::new(),
            procedures: HashMap::new(),
        }
    }

    /// Export a table to ExportData format for CSV/JSON export
    pub fn export_table(&self, table: &str) -> Result<ExportData, ExecError> {
        let schema = self.storage.get_schema(table)?;
        let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let rows = self.storage.scan(table)?;

        Ok(ExportData { columns, rows })
    }

    /// Import data into a table from ImportData
    pub fn import_into_table(
        &mut self,
        table: &str,
        data: &ImportData,
    ) -> Result<usize, ExecError> {
        let schema = self.storage.get_schema(table)?.clone();
        let table_columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

        // Map import columns to table columns
        let column_mapping: Vec<Option<usize>> = if data.columns.is_empty() {
            // If no headers, assume columns are in order
            (0..table_columns.len()).map(Some).collect()
        } else {
            data.columns
                .iter()
                .map(|c| table_columns.iter().position(|tc| tc == c))
                .collect()
        };

        let mut count = 0;
        for row in &data.rows {
            // Build a row matching the table schema
            let mut new_row: Row = vec![Value::Null; table_columns.len()];
            for (i, value) in row.iter().enumerate() {
                if let Some(Some(target_idx)) = column_mapping.get(i) {
                    new_row[*target_idx] = value.clone();
                }
            }

            // Use trigger-aware insert from logical layer
            let runtime = SqlRuntime::new();
            match insert(&mut self.storage, &runtime, table, new_row.clone()) {
                Ok(true) => {}         // Row inserted
                Ok(false) => continue, // Row skipped by BEFORE trigger
                Err(logical::OperationError::TriggerAbort(msg)) => {
                    return Err(ExecError::TriggerError(msg));
                }
                Err(logical::OperationError::Storage(e)) => {
                    return Err(ExecError::Storage(e));
                }
                Err(logical::OperationError::Runtime(e)) => {
                    return Err(ExecError::InvalidExpression(e.to_string()));
                }
                Err(logical::OperationError::TriggerDepthExceeded { depth, max_depth }) => {
                    return Err(ExecError::TriggerDepthExceeded { depth, max_depth });
                }
            }

            count += 1;
        }

        Ok(count)
    }

    /// Execute a SQL string
    pub fn execute(&mut self, sql: &str) -> ExecResult {
        let stmt = sql_parser::parse(sql)
            .map_err(|_| ExecError::InvalidExpression("Parse error".to_string()))?;
        let plan = sql_planner::plan(stmt)
            .map_err(|_| ExecError::InvalidExpression("Planning error".to_string()))?;
        self.execute_plan(plan)
    }

    /// Execute a Datalog program against the database
    ///
    /// Datalog queries can access all existing SQL tables as predicates.
    /// For example, if you have a table `parent(parent, child)`, you can
    /// query it with `?- parent(X, Y).`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut engine = Engine::new();
    /// engine.execute("CREATE TABLE parent (parent TEXT, child TEXT)");
    /// engine.execute("INSERT INTO parent VALUES ('john', 'mary')");
    ///
    /// // Find transitive closure
    /// let result = engine.execute_datalog(r#"
    ///     ancestor(X, Y) :- parent(X, Y).
    ///     ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
    ///     ?- ancestor(X, Y).
    /// "#)?;
    /// ```
    pub fn execute_datalog(&mut self, program: &str) -> ExecResult {
        crate::datalog::execute_datalog_program(&mut self.storage, program).map_err(ExecError::from)
    }

    /// Execute a logical plan
    fn execute_plan(&mut self, plan: LogicalPlan) -> ExecResult {
        match plan {
            LogicalPlan::CreateTable {
                name,
                columns,
                constraints,
            } => self.execute_create_table(&name, &columns, &constraints),
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
            // Function operations
            LogicalPlan::CreateFunction {
                name,
                body,
                language,
            } => self.create_function(&name, &body, &language),
            LogicalPlan::DropFunction { name } => self.drop_function(&name),
            // Trigger operations
            LogicalPlan::CreateTrigger {
                name,
                timing,
                events,
                table,
                action,
            } => self.create_trigger(&name, timing, events, &table, action),
            LogicalPlan::DropTrigger { name } => self.drop_trigger(&name),
            // DDL operations
            LogicalPlan::DropTable { name } => {
                self.storage.drop_table(&name)?;
                Ok(QueryResult::Success)
            }
            LogicalPlan::AlterTable { table, action } => self.execute_alter_table(&table, action),
            // Index operations
            LogicalPlan::CreateIndex {
                name,
                table,
                columns,
            } => {
                self.storage
                    .create_composite_index(&table, &columns, &name, false)?;
                Ok(QueryResult::Success)
            }
            LogicalPlan::DropIndex { name } => {
                self.storage.drop_index(&name)?;
                Ok(QueryResult::Success)
            }
            // View operations
            LogicalPlan::CreateView {
                name,
                columns,
                query,
            } => {
                self.views.insert(
                    name,
                    ViewDefinition {
                        columns,
                        query: *query,
                    },
                );
                Ok(QueryResult::Success)
            }
            LogicalPlan::DropView { name } => {
                self.views.remove(&name);
                Ok(QueryResult::Success)
            }
            LogicalPlan::CreateProcedure { name, params, body } => {
                self.procedures
                    .insert(name, ProcedureDefinition { params, body });
                Ok(QueryResult::Success)
            }
            LogicalPlan::DropProcedure { name } => {
                if self.procedures.remove(&name).is_none() {
                    return Err(ExecError::ProcedureNotFound(name));
                }
                Ok(QueryResult::Success)
            }
            LogicalPlan::CallProcedure { name, args } => self.execute_procedure(&name, args),
            _ => self.execute_query(plan),
        }
    }

    /// Execute an ALTER TABLE statement
    fn execute_alter_table(&mut self, table: &str, action: AlterAction) -> ExecResult {
        match action {
            AlterAction::AddColumn(col_def) => {
                let default_val = col_def
                    .default
                    .as_ref()
                    .map(eval_literal)
                    .unwrap_or(Value::Null);
                let col_schema = ColumnSchema {
                    name: col_def.name.clone(),
                    data_type: convert_data_type(&col_def.data_type),
                    nullable: col_def.nullable,
                    primary_key: col_def.primary_key,
                    unique: col_def.unique,
                    default: col_def.default.as_ref().map(eval_literal),
                    references: col_def.references.as_ref().map(convert_fk_ref),
                };
                self.storage.add_column(table, col_schema, default_val)?;
                Ok(QueryResult::Success)
            }
            AlterAction::DropColumn(col_name) => {
                self.storage.drop_column(table, &col_name)?;
                Ok(QueryResult::Success)
            }
            AlterAction::RenameColumn { old_name, new_name } => {
                self.storage.rename_column(table, &old_name, &new_name)?;
                Ok(QueryResult::Success)
            }
            AlterAction::RenameTable(new_name) => {
                self.storage.rename_table(table, &new_name)?;
                Ok(QueryResult::Success)
            }
        }
    }

    /// Execute a stored procedure
    fn execute_procedure(&mut self, name: &str, args: Vec<Expr>) -> ExecResult {
        // Look up the procedure
        let procedure = self
            .procedures
            .get(name)
            .ok_or_else(|| ExecError::ProcedureNotFound(name.to_string()))?
            .clone();

        // Check argument count
        if args.len() != procedure.params.len() {
            return Err(ExecError::ProcedureArgCountMismatch {
                expected: procedure.params.len(),
                got: args.len(),
            });
        }

        // Build parameter bindings (name -> value)
        let mut bindings: HashMap<String, Value> = HashMap::new();
        for (param, arg) in procedure.params.iter().zip(args.iter()) {
            let value = eval_literal(arg);
            bindings.insert(param.name.clone(), value);
        }

        // Execute each statement in the procedure body
        let mut last_result = QueryResult::Success;
        for stmt in &procedure.body {
            last_result = self.execute_procedure_statement(stmt, &bindings)?;
        }

        Ok(last_result)
    }

    /// Execute a single statement within a procedure body
    fn execute_procedure_statement(
        &mut self,
        stmt: &ProcedureStatement,
        bindings: &HashMap<String, Value>,
    ) -> ExecResult {
        match stmt {
            ProcedureStatement::Sql(sql_stmt) => {
                // Substitute parameters in the statement and execute
                let substituted = substitute_params_in_statement(sql_stmt, bindings);
                let plan = sql_planner::plan(substituted)
                    .map_err(|_| ExecError::InvalidExpression("Planning error".to_string()))?;
                self.execute_plan(plan)
            }
            ProcedureStatement::Declare { .. } => {
                // Variable declarations are no-ops at runtime (already handled via bindings)
                Ok(QueryResult::Success)
            }
            ProcedureStatement::SetVar { .. } => {
                // SET @var = expr - for now just ignore (simplified implementation)
                Ok(QueryResult::Success)
            }
            ProcedureStatement::Return(_) => {
                // RETURN - procedure ends (simplified - just return success)
                Ok(QueryResult::Success)
            }
        }
    }

    /// Execute a CREATE TABLE
    fn execute_create_table(
        &mut self,
        name: &str,
        columns: &[ColumnDef],
        constraints: &[ParserTableConstraint],
    ) -> ExecResult {
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
            constraints: constraints.iter().map(convert_table_constraint).collect(),
        };

        // Collect FK references before creating table
        let fk_refs: Vec<_> = columns
            .iter()
            .filter_map(|c| {
                c.references.as_ref().map(|fk| FkTriggerInfo {
                    child_table: name.to_string(),
                    child_column: c.name.clone(),
                    parent_table: fk.table.clone(),
                    parent_column: fk.column.clone(),
                    on_delete: convert_ref_action(&fk.on_delete),
                    on_update: convert_ref_action(&fk.on_update),
                })
            })
            .collect();

        // Collect unique constraints (column-level)
        let mut unique_constraints: Vec<UniqueConstraintInfo> = columns
            .iter()
            .filter(|c| c.unique || c.primary_key)
            .map(|c| UniqueConstraintInfo {
                table: name.to_string(),
                columns: vec![c.name.clone()],
                is_primary_key: c.primary_key,
            })
            .collect();

        // Collect unique constraints (table-level)
        for constraint in constraints {
            match constraint {
                ParserTableConstraint::PrimaryKey { columns: cols, .. } => {
                    unique_constraints.push(UniqueConstraintInfo {
                        table: name.to_string(),
                        columns: cols.clone(),
                        is_primary_key: true,
                    });
                }
                ParserTableConstraint::Unique { columns: cols, .. } => {
                    unique_constraints.push(UniqueConstraintInfo {
                        table: name.to_string(),
                        columns: cols.clone(),
                        is_primary_key: false,
                    });
                }
                _ => {}
            }
        }

        self.storage.create_table(schema)?;

        // Create implicit triggers for FK constraints
        for fk in fk_refs {
            self.create_fk_triggers(&fk)?;
        }

        // Create implicit triggers for unique/primary key constraints
        for uc in unique_constraints {
            self.create_unique_triggers(&uc)?;
        }

        Ok(QueryResult::Success)
    }

    /// Create implicit triggers for a unique constraint
    fn create_unique_triggers(&mut self, uc: &UniqueConstraintInfo) -> Result<(), ExecError> {
        // Create BEFORE INSERT trigger to check uniqueness
        self.create_unique_insert_trigger(uc)?;

        // Create BEFORE UPDATE trigger to check uniqueness
        self.create_unique_update_trigger(uc)?;

        Ok(())
    }

    /// Create BEFORE INSERT trigger for unique constraint
    fn create_unique_insert_trigger(&mut self, uc: &UniqueConstraintInfo) -> Result<(), ExecError> {
        let cols_str = uc.columns.join("_");
        let constraint_type = if uc.is_primary_key { "pk" } else { "unique" };
        let func_name = format!("__{}_insert_{}_{}__", constraint_type, uc.table, cols_str);
        let trigger_name = format!("__{}_insert_{}_{}__", constraint_type, uc.table, cols_str);

        // Build the WHERE clause for checking existing rows
        let where_parts: Vec<String> = uc
            .columns
            .iter()
            .map(|col| format!("{} = NEW.{}", col, col))
            .collect();
        let where_clause = where_parts.join(" AND ");

        // Build the constraint description for error message
        let constraint_desc = if uc.is_primary_key {
            format!("PRIMARY KEY ({})", uc.columns.join(", "))
        } else {
            format!("UNIQUE ({})", uc.columns.join(", "))
        };

        // Function body: IF EXISTS (SELECT 1 FROM table WHERE cols = NEW.cols) THEN RAISE ERROR
        let body = format!(
            "IF EXISTS (SELECT 1 FROM {} WHERE {}) THEN RAISE ERROR '{}'; RETURN NEW",
            uc.table, where_clause, constraint_desc
        );

        // Create function
        self.storage.create_function(FunctionDef {
            name: func_name.clone(),
            params: "[]".to_string(),
            body,
            language: "sql".to_string(),
        })?;

        // Create trigger
        self.storage.create_trigger(TriggerDef {
            name: trigger_name,
            table_name: uc.table.clone(),
            timing: StorageTriggerTiming::Before,
            events: vec![StorageTriggerEvent::Insert],
            function_name: func_name,
        })?;

        Ok(())
    }

    /// Create BEFORE UPDATE trigger for unique constraint
    fn create_unique_update_trigger(&mut self, uc: &UniqueConstraintInfo) -> Result<(), ExecError> {
        let cols_str = uc.columns.join("_");
        let constraint_type = if uc.is_primary_key { "pk" } else { "unique" };
        let func_name = format!("__{}_update_{}_{}__", constraint_type, uc.table, cols_str);
        let trigger_name = format!("__{}_update_{}_{}__", constraint_type, uc.table, cols_str);

        // Build the WHERE clause for checking existing rows
        let where_parts: Vec<String> = uc
            .columns
            .iter()
            .map(|col| format!("{} = NEW.{}", col, col))
            .collect();
        let where_clause = where_parts.join(" AND ");

        // Build the constraint description for error message
        let constraint_desc = if uc.is_primary_key {
            format!("PRIMARY KEY ({})", uc.columns.join(", "))
        } else {
            format!("UNIQUE ({})", uc.columns.join(", "))
        };

        // Function body: check if any OTHER row has the same values
        // Use IF EXISTS EXCLUDING OLD to skip the row being updated
        let body = format!(
            "IF EXISTS EXCLUDING OLD (SELECT 1 FROM {} WHERE {}) THEN RAISE ERROR '{}'; RETURN NEW",
            uc.table, where_clause, constraint_desc
        );

        // Create function
        self.storage.create_function(FunctionDef {
            name: func_name.clone(),
            params: "[]".to_string(),
            body,
            language: "sql".to_string(),
        })?;

        // Create trigger
        self.storage.create_trigger(TriggerDef {
            name: trigger_name,
            table_name: uc.table.clone(),
            timing: StorageTriggerTiming::Before,
            events: vec![StorageTriggerEvent::Update],
            function_name: func_name,
        })?;

        Ok(())
    }

    /// Create implicit triggers for a foreign key constraint
    fn create_fk_triggers(&mut self, fk: &FkTriggerInfo) -> Result<(), ExecError> {
        // 1. Create BEFORE INSERT trigger on child table to validate parent exists
        self.create_fk_insert_validation_trigger(fk)?;

        // 2. Create BEFORE UPDATE trigger on child table to validate new parent exists
        self.create_fk_update_validation_trigger(fk)?;

        // 3. Create AFTER DELETE trigger on parent table based on ON DELETE action
        self.create_fk_on_delete_trigger(fk)?;

        // 4. Create AFTER UPDATE trigger on parent table based on ON UPDATE action
        self.create_fk_on_update_trigger(fk)?;

        Ok(())
    }

    /// Create BEFORE INSERT trigger to validate FK constraint
    fn create_fk_insert_validation_trigger(&mut self, fk: &FkTriggerInfo) -> Result<(), ExecError> {
        let func_name = format!(
            "__fk_validate_insert_{}_{}__",
            fk.child_table, fk.child_column
        );
        let trigger_name = format!("__fk_insert_{}_{}__", fk.child_table, fk.child_column);

        // Function body: IF NOT EXISTS (SELECT 1 FROM parent WHERE id = NEW.fk_col) THEN RAISE ERROR '...'
        let body = format!(
            "IF NOT EXISTS (SELECT 1 FROM {} WHERE {} = NEW.{}) THEN RAISE ERROR 'Foreign key constraint violation: {} references {}({})'; RETURN NEW",
            fk.parent_table,
            fk.parent_column,
            fk.child_column,
            fk.child_column,
            fk.parent_table,
            fk.parent_column
        );

        // Create function
        self.storage.create_function(FunctionDef {
            name: func_name.clone(),
            params: "[]".to_string(),
            body,
            language: "sql".to_string(),
        })?;

        // Create trigger
        self.storage.create_trigger(TriggerDef {
            name: trigger_name,
            table_name: fk.child_table.clone(),
            timing: StorageTriggerTiming::Before,
            events: vec![StorageTriggerEvent::Insert],
            function_name: func_name,
        })?;

        Ok(())
    }

    /// Create BEFORE UPDATE trigger to validate FK constraint when FK column changes
    fn create_fk_update_validation_trigger(&mut self, fk: &FkTriggerInfo) -> Result<(), ExecError> {
        let func_name = format!(
            "__fk_validate_update_{}_{}__",
            fk.child_table, fk.child_column
        );
        let trigger_name = format!("__fk_update_{}_{}__", fk.child_table, fk.child_column);

        // Function body: check if new parent exists
        let body = format!(
            "IF NOT EXISTS (SELECT 1 FROM {} WHERE {} = NEW.{}) THEN RAISE ERROR 'Foreign key constraint violation: {} references {}({})'; RETURN NEW",
            fk.parent_table,
            fk.parent_column,
            fk.child_column,
            fk.child_column,
            fk.parent_table,
            fk.parent_column
        );

        // Create function
        self.storage.create_function(FunctionDef {
            name: func_name.clone(),
            params: "[]".to_string(),
            body,
            language: "sql".to_string(),
        })?;

        // Create trigger
        self.storage.create_trigger(TriggerDef {
            name: trigger_name,
            table_name: fk.child_table.clone(),
            timing: StorageTriggerTiming::Before,
            events: vec![StorageTriggerEvent::Update],
            function_name: func_name,
        })?;

        Ok(())
    }

    /// Create DELETE trigger on parent table based on ON DELETE action
    /// - RESTRICT/NoAction: BEFORE DELETE trigger to check references exist (abort if so)
    /// - CASCADE/SetNull/SetDefault: AFTER DELETE trigger to perform the action
    fn create_fk_on_delete_trigger(&mut self, fk: &FkTriggerInfo) -> Result<(), ExecError> {
        let (body, timing) = match fk.on_delete {
            StorageRefAction::Cascade => {
                // DELETE FROM child WHERE fk_col = OLD.pk_col
                (
                    format!(
                        "DELETE FROM {} WHERE {} = OLD.{}; RETURN OLD",
                        fk.child_table, fk.child_column, fk.parent_column
                    ),
                    StorageTriggerTiming::After,
                )
            }
            StorageRefAction::SetNull => {
                // UPDATE child SET fk_col = NULL WHERE fk_col = OLD.pk_col
                (
                    format!(
                        "UPDATE {} SET {} = NULL WHERE {} = OLD.{}; RETURN OLD",
                        fk.child_table, fk.child_column, fk.child_column, fk.parent_column
                    ),
                    StorageTriggerTiming::After,
                )
            }
            StorageRefAction::SetDefault => {
                // For now, treat SetDefault as SetNull (proper implementation would need default value)
                (
                    format!(
                        "UPDATE {} SET {} = NULL WHERE {} = OLD.{}; RETURN OLD",
                        fk.child_table, fk.child_column, fk.child_column, fk.parent_column
                    ),
                    StorageTriggerTiming::After,
                )
            }
            StorageRefAction::Restrict | StorageRefAction::NoAction => {
                // BEFORE DELETE: check if references exist and abort if so
                (
                    format!(
                        "IF EXISTS (SELECT 1 FROM {} WHERE {} = OLD.{}) THEN RAISE ERROR 'Cannot delete: referenced by {}({})'; RETURN OLD",
                        fk.child_table, fk.child_column, fk.parent_column,
                        fk.child_table, fk.child_column
                    ),
                    StorageTriggerTiming::Before,
                )
            }
        };

        let func_name = format!(
            "__fk_on_delete_{}_{}_{}__",
            fk.parent_table, fk.child_table, fk.child_column
        );
        let trigger_name = format!(
            "__fk_delete_{}_{}_{}__",
            fk.parent_table, fk.child_table, fk.child_column
        );

        // Create function
        self.storage.create_function(FunctionDef {
            name: func_name.clone(),
            params: "[]".to_string(),
            body,
            language: "sql".to_string(),
        })?;

        // Create trigger on PARENT table
        self.storage.create_trigger(TriggerDef {
            name: trigger_name,
            table_name: fk.parent_table.clone(),
            timing,
            events: vec![StorageTriggerEvent::Delete],
            function_name: func_name,
        })?;

        Ok(())
    }

    /// Create UPDATE trigger on parent table based on ON UPDATE action
    /// - RESTRICT/NoAction: BEFORE UPDATE trigger to check references exist (abort if so)
    /// - CASCADE/SetNull/SetDefault: AFTER UPDATE trigger to perform the action
    fn create_fk_on_update_trigger(&mut self, fk: &FkTriggerInfo) -> Result<(), ExecError> {
        let (body, timing) = match fk.on_update {
            StorageRefAction::Cascade => {
                // UPDATE child SET fk_col = NEW.pk_col WHERE fk_col = OLD.pk_col
                (
                    format!(
                        "UPDATE {} SET {} = NEW.{} WHERE {} = OLD.{}; RETURN NEW",
                        fk.child_table,
                        fk.child_column,
                        fk.parent_column,
                        fk.child_column,
                        fk.parent_column
                    ),
                    StorageTriggerTiming::After,
                )
            }
            StorageRefAction::SetNull => {
                // UPDATE child SET fk_col = NULL WHERE fk_col = OLD.pk_col
                (
                    format!(
                        "UPDATE {} SET {} = NULL WHERE {} = OLD.{}; RETURN NEW",
                        fk.child_table, fk.child_column, fk.child_column, fk.parent_column
                    ),
                    StorageTriggerTiming::After,
                )
            }
            StorageRefAction::SetDefault => {
                // For now, treat SetDefault as SetNull
                (
                    format!(
                        "UPDATE {} SET {} = NULL WHERE {} = OLD.{}; RETURN NEW",
                        fk.child_table, fk.child_column, fk.child_column, fk.parent_column
                    ),
                    StorageTriggerTiming::After,
                )
            }
            StorageRefAction::Restrict | StorageRefAction::NoAction => {
                // BEFORE UPDATE: check if references exist and abort if so
                (
                    format!(
                        "IF EXISTS (SELECT 1 FROM {} WHERE {} = OLD.{}) THEN RAISE ERROR 'Cannot update: referenced by {}({})'; RETURN NEW",
                        fk.child_table, fk.child_column, fk.parent_column,
                        fk.child_table, fk.child_column
                    ),
                    StorageTriggerTiming::Before,
                )
            }
        };

        let func_name = format!(
            "__fk_on_update_{}_{}_{}__",
            fk.parent_table, fk.child_table, fk.child_column
        );
        let trigger_name = format!(
            "__fk_update_{}_{}_{}__",
            fk.parent_table, fk.child_table, fk.child_column
        );

        // Create function
        self.storage.create_function(FunctionDef {
            name: func_name.clone(),
            params: "[]".to_string(),
            body,
            language: "sql".to_string(),
        })?;

        // Create trigger on PARENT table
        self.storage.create_trigger(TriggerDef {
            name: trigger_name,
            table_name: fk.parent_table.clone(),
            timing,
            events: vec![StorageTriggerEvent::Update],
            function_name: func_name,
        })?;

        Ok(())
    }

    /// Execute an INSERT
    fn execute_insert(
        &mut self,
        table: &str,
        _columns: Option<&[String]>,
        values: &[Vec<Expr>],
    ) -> ExecResult {
        let runtime = SqlRuntime::new();
        let mut count = 0;

        for value_row in values {
            let row: Row = value_row.iter().map(eval_literal).collect();

            // Use trigger-aware insert
            match insert(&mut self.storage, &runtime, table, row) {
                Ok(true) => count += 1,
                Ok(false) => {
                    // Row was skipped by BEFORE trigger
                }
                Err(OperationError::Storage(e)) => return Err(ExecError::Storage(e)),
                Err(OperationError::Runtime(e)) => {
                    return Err(ExecError::InvalidExpression(format!(
                        "Trigger error: {}",
                        e
                    )));
                }
                Err(OperationError::TriggerAbort(msg)) => {
                    return Err(ExecError::InvalidExpression(format!(
                        "Trigger aborted: {}",
                        msg
                    )));
                }
                Err(OperationError::TriggerDepthExceeded { depth, max_depth }) => {
                    return Err(ExecError::TriggerDepthExceeded { depth, max_depth });
                }
            }
        }
        Ok(QueryResult::RowsAffected(count))
    }

    /// Execute an UPDATE with trigger-based referential integrity support
    fn execute_update(
        &mut self,
        table: &str,
        assignments: &[Assignment],
        where_clause: Option<&Expr>,
    ) -> ExecResult {
        let schema = self.storage.get_schema(table)?;
        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let column_names_clone = column_names.clone();
        let assignments_clone: Vec<_> = assignments
            .iter()
            .map(|a| (a.column.clone(), a.value.clone()))
            .collect();

        let runtime = SqlRuntime::new();

        // Use trigger-aware update that handles FK constraints through triggers
        let result = update(
            &mut self.storage,
            &runtime,
            table,
            |row| match where_clause {
                Some(predicate) => eval_predicate(predicate, row, &column_names),
                None => true,
            },
            |row| {
                // Apply assignments to the row
                for (col_name, value_expr) in &assignments_clone {
                    if let Some(col_idx) = column_names_clone.iter().position(|c| c == col_name) {
                        row[col_idx] = eval_expr(value_expr, row, &column_names_clone);
                    }
                }
            },
        );

        match result {
            Ok(count) => Ok(QueryResult::RowsAffected(count)),
            Err(OperationError::Storage(e)) => Err(ExecError::Storage(e)),
            Err(OperationError::Runtime(e)) => Err(ExecError::InvalidExpression(format!(
                "Trigger error: {}",
                e
            ))),
            Err(OperationError::TriggerAbort(msg)) => {
                // FK constraint violation from trigger
                Err(ExecError::InvalidExpression(format!(
                    "Constraint violation: {}",
                    msg
                )))
            }
            Err(OperationError::TriggerDepthExceeded { depth, max_depth }) => {
                Err(ExecError::TriggerDepthExceeded { depth, max_depth })
            }
        }
    }

    /// Execute a DELETE with trigger-based referential integrity support
    fn execute_delete(&mut self, table: &str, where_clause: Option<&Expr>) -> ExecResult {
        let schema = self.storage.get_schema(table)?.clone();
        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

        let runtime = SqlRuntime::new();

        // Use trigger-aware delete that handles FK constraints through AFTER DELETE triggers
        let deleted = delete(
            &mut self.storage,
            &runtime,
            table,
            |row| match where_clause {
                Some(predicate) => eval_predicate(predicate, row, &column_names),
                None => true,
            },
        );

        match deleted {
            Ok(count) => Ok(QueryResult::RowsAffected(count)),
            Err(OperationError::Storage(e)) => Err(ExecError::Storage(e)),
            Err(OperationError::Runtime(e)) => Err(ExecError::InvalidExpression(format!(
                "Trigger error: {}",
                e
            ))),
            Err(OperationError::TriggerAbort(msg)) => {
                // FK constraint violation from trigger
                Err(ExecError::InvalidExpression(format!(
                    "Constraint violation: {}",
                    msg
                )))
            }
            Err(OperationError::TriggerDepthExceeded { depth, max_depth }) => {
                Err(ExecError::TriggerDepthExceeded { depth, max_depth })
            }
        }
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

    /// Create a function (stored in metadata)
    fn create_function(&mut self, name: &str, body: &str, language: &str) -> ExecResult {
        use logical::FunctionDef;

        let func = FunctionDef {
            name: name.to_string(),
            params: "[]".to_string(), // No params for trigger functions
            body: body.to_string(),
            language: language.to_string(),
        };

        self.storage.create_function(func).map_err(|e| match e {
            logical::StorageError::FunctionAlreadyExists(n) => {
                ExecError::InvalidExpression(format!("Function already exists: {}", n))
            }
            _ => ExecError::Storage(e),
        })?;

        Ok(QueryResult::Success)
    }

    /// Drop a function
    fn drop_function(&mut self, name: &str) -> ExecResult {
        self.storage.drop_function(name).map_err(|e| match e {
            logical::StorageError::FunctionNotFound(n) => {
                ExecError::InvalidExpression(format!("Function not found: {}", n))
            }
            _ => ExecError::Storage(e),
        })?;

        Ok(QueryResult::Success)
    }

    /// Create a trigger (stored in metadata)
    fn create_trigger(
        &mut self,
        name: &str,
        timing: TriggerTiming,
        events: Vec<TriggerEvent>,
        table: &str,
        action: TriggerActionType,
    ) -> ExecResult {
        use logical::{
            TriggerDef, TriggerEvent as StorageTriggerEvent, TriggerTiming as StorageTriggerTiming,
        };

        // Convert to storage types
        let storage_timing = match timing {
            TriggerTiming::Before => StorageTriggerTiming::Before,
            TriggerTiming::After => StorageTriggerTiming::After,
        };

        let storage_events: Vec<StorageTriggerEvent> = events
            .iter()
            .map(|e| match e {
                TriggerEvent::Insert => StorageTriggerEvent::Insert,
                TriggerEvent::Update => StorageTriggerEvent::Update,
                TriggerEvent::Delete => StorageTriggerEvent::Delete,
            })
            .collect();

        // Get function name from action
        let function_name = match &action {
            TriggerActionType::ExecuteFunction(name) => name.clone(),
            TriggerActionType::InlineActions(actions) => {
                // For legacy inline actions, create an implicit function
                let func_name = format!("__trigger_{}__", name);
                let body = convert_inline_actions_to_body(actions);
                self.create_function(&func_name, &body, "sql")?;
                func_name
            }
        };

        let trigger = TriggerDef {
            name: name.to_string(),
            table_name: table.to_string(),
            timing: storage_timing,
            events: storage_events,
            function_name,
        };

        self.storage.create_trigger(trigger).map_err(|e| match e {
            logical::StorageError::TriggerAlreadyExists(n) => ExecError::TriggerAlreadyExists(n),
            _ => ExecError::Storage(e),
        })?;

        Ok(QueryResult::Success)
    }

    /// Drop a trigger
    fn drop_trigger(&mut self, name: &str) -> ExecResult {
        self.storage.drop_trigger(name).map_err(|e| match e {
            logical::StorageError::TriggerNotFound(n) => ExecError::TriggerNotFound(n),
            _ => ExecError::Storage(e),
        })?;

        Ok(QueryResult::Success)
    }

    /// Execute a SELECT query
    fn execute_query(&self, plan: LogicalPlan) -> ExecResult {
        self.execute_query_with_cte(plan, &CteContext::default())
    }

    /// Execute a CTE and return its result as (columns, rows)
    fn execute_cte(
        &self,
        cte: &Cte,
        ctx: &CteContext,
    ) -> Result<(Vec<String>, Vec<Vec<Value>>), ExecError> {
        // Plan and execute the CTE query based on its type
        let plan = self.plan_cte_query(&cte.query)?;

        let result = self.execute_query_with_cte(plan, ctx)?;

        match result {
            QueryResult::Select { columns, rows } => {
                // If the CTE has explicit column names, use those
                let final_columns = if let Some(ref cte_columns) = cte.columns {
                    cte_columns.clone()
                } else {
                    columns
                };
                Ok((final_columns, rows))
            }
            _ => Err(ExecError::InvalidExpression(
                "CTE query must be a SELECT".to_string(),
            )),
        }
    }

    /// Plan a CTE query (handles both simple SELECT and set operations)
    fn plan_cte_query(&self, query: &sql_parser::CteQuery) -> Result<LogicalPlan, ExecError> {
        match query {
            sql_parser::CteQuery::Select(select) => {
                sql_planner::plan(Statement::Select(select.clone())).map_err(|_| {
                    ExecError::InvalidExpression("Failed to plan CTE SELECT".to_string())
                })
            }
            sql_parser::CteQuery::SetOp(set_op) => {
                // Convert CteSetOperation to a LogicalPlan SetOperation
                self.plan_cte_set_op(set_op)
            }
        }
    }

    /// Plan a CTE set operation (UNION/INTERSECT/EXCEPT within a CTE)
    fn plan_cte_set_op(
        &self,
        set_op: &sql_parser::CteSetOperation,
    ) -> Result<LogicalPlan, ExecError> {
        let left = self.plan_cte_query(&set_op.left)?;
        let right = self.plan_cte_query(&set_op.right)?;

        Ok(LogicalPlan::SetOperation {
            left: Box::new(left),
            right: Box::new(right),
            op: set_op.op.clone(),
            all: set_op.all,
        })
    }

    /// Execute a SELECT query with CTE context
    fn execute_query_with_cte(&self, plan: LogicalPlan, cte_ctx: &CteContext) -> ExecResult {
        match plan {
            LogicalPlan::WithCte {
                ctes,
                recursive: _recursive,
                input,
            } => {
                // Materialize each CTE
                let mut ctx = cte_ctx.clone();
                for cte in ctes {
                    let cte_result = self.execute_cte(&cte, &ctx)?;
                    ctx.ctes.insert(cte.name.clone(), cte_result);
                }
                // Execute the main query with the CTE context
                self.execute_query_with_cte(*input, &ctx)
            }
            LogicalPlan::Scan { table } => {
                // Handle special virtual tables
                if table == "dual" || table == "__empty__" {
                    // "dual" is a virtual single-row table for SELECT without FROM
                    // "__empty__" is used for empty base cases
                    let rows = if table == "dual" {
                        vec![vec![Value::Null]] // Single row with a dummy value
                    } else {
                        vec![] // Empty table
                    };
                    return Ok(QueryResult::Select {
                        columns: vec!["DUMMY".to_string()],
                        rows,
                    });
                }

                // Check if this is a CTE reference first
                if let Some((columns, rows)) = cte_ctx.ctes.get(&table) {
                    return Ok(QueryResult::Select {
                        columns: columns.clone(),
                        rows: rows.clone(),
                    });
                }

                // Check if this is a view reference
                if let Some(view_def) = self.views.get(&table).cloned() {
                    let result = self.execute_query_with_cte(view_def.query.clone(), cte_ctx)?;
                    // Apply column renaming if view has explicit columns
                    if let QueryResult::Select { columns, rows } = result {
                        let final_columns = if let Some(view_cols) = view_def.columns {
                            view_cols
                        } else {
                            columns
                        };
                        return Ok(QueryResult::Select {
                            columns: final_columns,
                            rows,
                        });
                    }
                    return Ok(result);
                }

                let schema = self.storage.get_schema(&table)?;
                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();
                let rows = self.storage.scan(&table)?;
                Ok(QueryResult::Select {
                    columns: column_names,
                    rows,
                })
            }
            LogicalPlan::IndexScan {
                table,
                column,
                value,
            } => {
                let schema = self.storage.get_schema(&table)?;
                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();

                // Evaluate the value expression to get the lookup value
                let lookup_value = eval_literal(&value);

                // Try index lookup
                if let Some(indices) = self.storage.index_lookup(&table, &column, &lookup_value) {
                    let rows = self.storage.get_rows_by_indices(&table, &indices)?;
                    Ok(QueryResult::Select {
                        columns: column_names,
                        rows,
                    })
                } else {
                    // Fall back to full scan if index lookup fails
                    let rows = self.storage.scan(&table)?;
                    Ok(QueryResult::Select {
                        columns: column_names,
                        rows,
                    })
                }
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
            } => {
                let left_result = self.execute_query_with_cte(*left, cte_ctx)?;
                let right_result = self.execute_query_with_cte(*right, cte_ctx)?;

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
                // Check if we can use an index for this filter
                if let Some((table_name, indexed_rows, remaining_predicate)) =
                    self.try_index_optimization(&input, &predicate)
                {
                    let schema = self.storage.get_schema(&table_name)?;
                    let column_names: Vec<String> =
                        schema.columns.iter().map(|c| c.name.clone()).collect();

                    // If we got indexed rows, filter by any remaining predicates
                    let filtered = match remaining_predicate {
                        Some(pred) => indexed_rows
                            .into_iter()
                            .filter(|row| {
                                self.eval_predicate_with_subquery(&pred, row, &column_names)
                            })
                            .collect(),
                        None => indexed_rows,
                    };

                    return Ok(QueryResult::Select {
                        columns: column_names,
                        rows: filtered,
                    });
                }

                // Fall back to regular filtering
                let result = self.execute_query_with_cte(*input, cte_ctx)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        let filtered: Vec<Vec<Value>> = rows
                            .into_iter()
                            .filter(|row| {
                                self.eval_predicate_with_subquery(&predicate, row, &columns)
                            })
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
                let result = self.execute_query_with_cte(*input, cte_ctx)?;
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

                        // Check if any expression is an aggregate or window function
                        let has_aggregate = exprs.iter().any(|(expr, _)| is_aggregate(expr));
                        let has_window = exprs.iter().any(|(expr, _)| is_window_function(expr));

                        let new_columns: Vec<String> = exprs
                            .iter()
                            .enumerate()
                            .map(|(i, (expr, alias))| {
                                alias.clone().unwrap_or_else(|| match expr {
                                    Expr::Column(c) => c.clone(),
                                    Expr::Aggregate { func, .. } => {
                                        format!("{:?}", func).to_lowercase()
                                    }
                                    Expr::WindowFunction { func, .. } => {
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
                        } else if has_window {
                            // Evaluate window functions
                            let new_rows = eval_window_functions(&exprs, &rows, &columns);
                            Ok(QueryResult::Select {
                                columns: new_columns,
                                rows: new_rows,
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
                let result = self.execute_query_with_cte(*input, cte_ctx)?;
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
                let result = self.execute_query_with_cte(*input, cte_ctx)?;
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
            LogicalPlan::Distinct { input } => {
                let result = self.execute_query_with_cte(*input, cte_ctx)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        // Remove duplicate rows using a HashSet-like approach
                        // Since Row contains Value which has f64, we need a custom comparison
                        let mut unique_rows: Vec<Vec<Value>> = Vec::new();
                        for row in rows {
                            if !unique_rows.iter().any(|r| r == &row) {
                                unique_rows.push(row);
                            }
                        }
                        Ok(QueryResult::Select {
                            columns,
                            rows: unique_rows,
                        })
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::SetOperation {
                left,
                right,
                op,
                all,
            } => {
                let left_result = self.execute_query_with_cte(*left, cte_ctx)?;
                let right_result = self.execute_query_with_cte(*right, cte_ctx)?;

                match (left_result, right_result) {
                    (
                        QueryResult::Select {
                            columns: left_cols,
                            rows: left_rows,
                        },
                        QueryResult::Select {
                            columns: _right_cols,
                            rows: right_rows,
                        },
                    ) => {
                        let result_rows = match op {
                            SetOperator::Union => {
                                // Combine the rows
                                let mut combined_rows = left_rows;
                                combined_rows.extend(right_rows);

                                // If not UNION ALL, remove duplicates
                                if !all {
                                    let mut unique_rows: Vec<Vec<Value>> = Vec::new();
                                    for row in combined_rows {
                                        if !unique_rows.iter().any(|r| r == &row) {
                                            unique_rows.push(row);
                                        }
                                    }
                                    combined_rows = unique_rows;
                                }
                                combined_rows
                            }
                            SetOperator::Intersect => {
                                // Return rows that exist in both sets
                                let mut result: Vec<Vec<Value>> = Vec::new();
                                for left_row in &left_rows {
                                    if right_rows.iter().any(|r| r == left_row)
                                        && (all || !result.iter().any(|r| r == left_row))
                                    {
                                        result.push(left_row.clone());
                                    }
                                }
                                result
                            }
                            SetOperator::Except => {
                                // Return rows from left that don't exist in right
                                let mut result: Vec<Vec<Value>> = Vec::new();
                                for left_row in &left_rows {
                                    if !right_rows.iter().any(|r| r == left_row)
                                        && (all || !result.iter().any(|r| r == left_row))
                                    {
                                        result.push(left_row.clone());
                                    }
                                }
                                result
                            }
                        };

                        Ok(QueryResult::Select {
                            columns: left_cols,
                            rows: result_rows,
                        })
                    }
                    _ => Err(ExecError::InvalidExpression(
                        "Set operation requires SELECT inputs".to_string(),
                    )),
                }
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
                having,
            } => {
                let result = self.execute_query_with_cte(*input, cte_ctx)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        // Group the rows
                        let groups = group_rows(&rows, &group_by, &columns);

                        // Compute aggregates for each group
                        let new_columns: Vec<String> = aggregates
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

                        let mut result_rows: Vec<Vec<Value>> = Vec::new();

                        for (_key, group_rows) in &groups {
                            // Compute aggregates for this group
                            let row_values: Vec<Value> = aggregates
                                .iter()
                                .map(|(expr, _)| {
                                    if is_aggregate(expr) {
                                        eval_aggregate(expr, group_rows, &columns)
                                    } else {
                                        // For non-aggregate columns in GROUP BY,
                                        // just take the first row's value
                                        group_rows
                                            .first()
                                            .map(|r| eval_expr(expr, r, &columns))
                                            .unwrap_or(Value::Null)
                                    }
                                })
                                .collect();

                            // Apply HAVING filter if present
                            if let Some(ref having_expr) = having {
                                // Evaluate having with aggregate values
                                let having_result =
                                    eval_having(having_expr, group_rows, &columns, &new_columns);
                                if !having_result {
                                    continue;
                                }
                            }

                            result_rows.push(row_values);
                        }

                        Ok(QueryResult::Select {
                            columns: new_columns,
                            rows: result_rows,
                        })
                    }
                    other => Ok(other),
                }
            }
            // Recursive query evaluation (semi-naive)
            LogicalPlan::Recursive {
                name,
                columns,
                base,
                step,
            } => {
                // Execute base case
                let base_result = self.execute_query_with_cte(*base, cte_ctx)?;
                let (base_cols, base_rows) = match base_result {
                    QueryResult::Select { columns, rows } => (columns, rows),
                    _ => {
                        return Err(ExecError::InvalidExpression(
                            "Recursive base must be a query".to_string(),
                        ))
                    }
                };

                // Initialize result with base case
                let mut result_rows = base_rows.clone();
                let mut delta = base_rows;

                // Create extended CTE context with the recursive relation
                let mut recursive_ctx = cte_ctx.clone();

                // Iterate until fixpoint
                loop {
                    if delta.is_empty() {
                        break;
                    }

                    // Bind delta to the recursive relation for this iteration
                    recursive_ctx
                        .ctes
                        .insert(name.clone(), (columns.clone(), delta.clone()));

                    // Execute step
                    let step_result =
                        self.execute_query_with_cte((*step).clone(), &recursive_ctx)?;
                    let new_rows = match step_result {
                        QueryResult::Select { rows, .. } => rows,
                        _ => {
                            return Err(ExecError::InvalidExpression(
                                "Recursive step must be a query".to_string(),
                            ))
                        }
                    };

                    // Delta = new rows not already in result
                    let mut new_delta = Vec::new();
                    for row in new_rows {
                        if !result_rows.contains(&row) && !new_delta.contains(&row) {
                            new_delta.push(row);
                        }
                    }

                    // Add new delta to result
                    result_rows.extend(new_delta.clone());
                    delta = new_delta;
                }

                Ok(QueryResult::Select {
                    columns: if base_cols.is_empty() {
                        columns
                    } else {
                        base_cols
                    },
                    rows: result_rows,
                })
            }
            // Stratified evaluation for negation
            LogicalPlan::Stratify { strata } => {
                // Execute each stratum in order
                let mut final_result = QueryResult::Select {
                    columns: vec![],
                    rows: vec![],
                };

                for stratum in strata {
                    final_result = self.execute_query_with_cte(stratum, cte_ctx)?;
                }

                Ok(final_result)
            }
            // Reference to recursive relation (resolved via CTE context)
            LogicalPlan::RecursiveRef { name } => {
                if let Some((columns, rows)) = cte_ctx.ctes.get(&name) {
                    Ok(QueryResult::Select {
                        columns: columns.clone(),
                        rows: rows.clone(),
                    })
                } else {
                    Err(ExecError::TableNotFound(format!(
                        "Recursive relation '{}' not found",
                        name
                    )))
                }
            }
            // WITH RECURSIVE CTE - compute recursive result and bind as CTE
            LogicalPlan::WithRecursiveCte {
                name,
                columns,
                base,
                step,
                pre_ctes,
                input,
            } => {
                // First, materialize any pre-CTEs
                let mut ctx = cte_ctx.clone();
                for cte in pre_ctes {
                    let cte_result = self.execute_cte(&cte, &ctx)?;
                    ctx.ctes.insert(cte.name.clone(), cte_result);
                }

                // Execute recursive CTE using semi-naive evaluation
                // Execute base case
                let base_result = self.execute_query_with_cte(*base, &ctx)?;
                let (base_cols, base_rows) = match base_result {
                    QueryResult::Select { columns, rows } => (columns, rows),
                    _ => {
                        return Err(ExecError::InvalidExpression(
                            "Recursive base must be a query".to_string(),
                        ))
                    }
                };

                // Use explicit column names if provided, otherwise use base columns
                let cte_columns = columns.unwrap_or(base_cols);

                // Initialize result with base case
                let mut result_rows = base_rows.clone();
                let mut delta = base_rows;

                // Create extended CTE context with the recursive relation
                let mut recursive_ctx = ctx.clone();

                // Iterate until fixpoint
                loop {
                    if delta.is_empty() {
                        break;
                    }

                    // Bind current delta as the recursive relation
                    recursive_ctx
                        .ctes
                        .insert(name.clone(), (cte_columns.clone(), delta.clone()));

                    // Execute step
                    let step_result =
                        self.execute_query_with_cte((*step).clone(), &recursive_ctx)?;
                    let new_rows = match step_result {
                        QueryResult::Select { rows, .. } => rows,
                        _ => {
                            return Err(ExecError::InvalidExpression(
                                "Recursive step must be a query".to_string(),
                            ))
                        }
                    };

                    // Compute delta: new rows not already in result
                    delta = new_rows
                        .into_iter()
                        .filter(|row| !result_rows.contains(row))
                        .collect();

                    // Add new rows to result
                    result_rows.extend(delta.clone());
                }

                // Bind the final result as a CTE and execute the main query
                recursive_ctx
                    .ctes
                    .insert(name.clone(), (cte_columns.clone(), result_rows));
                self.execute_query_with_cte(*input, &recursive_ctx)
            }
            _ => Err(ExecError::InvalidExpression("Unsupported plan".to_string())),
        }
    }

    /// Evaluate a predicate with subquery support
    fn eval_predicate_with_subquery(&self, expr: &Expr, row: &[Value], columns: &[String]) -> bool {
        match self.eval_expr_with_subquery(expr, row, columns) {
            Value::Bool(b) => b,
            _ => false,
        }
    }

    /// Try to use an index for a Filter(Scan) operation.
    /// Returns Some((table_name, rows, remaining_predicate)) if successful.
    fn try_index_optimization(
        &self,
        input: &LogicalPlan,
        predicate: &Expr,
    ) -> Option<(String, Vec<Vec<Value>>, Option<Expr>)> {
        // Check if input is a simple Scan
        let table = match input {
            LogicalPlan::Scan { table } => table.clone(),
            _ => return None,
        };

        // Extract equality conditions from the predicate
        let (index_cond, remaining) = self.extract_index_conditions(&table, predicate);

        // If we found an indexable condition, use it
        if let Some((column, value)) = index_cond {
            let lookup_value = eval_literal(&value);
            if let Some(indices) = self.storage.index_lookup(&table, &column, &lookup_value) {
                if let Ok(rows) = self.storage.get_rows_by_indices(&table, &indices) {
                    return Some((table, rows, remaining));
                }
            }
        }

        None
    }

    /// Extract an equality condition that can use an index from a predicate.
    /// Returns (indexable_condition, remaining_predicate)
    fn extract_index_conditions(
        &self,
        table: &str,
        predicate: &Expr,
    ) -> (Option<(String, Expr)>, Option<Expr>) {
        match predicate {
            // Simple equality: column = literal
            Expr::BinaryOp {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                // Check if left is column and right is literal, and column has index
                if let Expr::Column(col) = left.as_ref() {
                    if is_literal(right) && self.storage.has_index(table, col) {
                        return (Some((col.clone(), *right.clone())), None);
                    }
                }
                // Check if right is column and left is literal
                if let Expr::Column(col) = right.as_ref() {
                    if is_literal(left) && self.storage.has_index(table, col) {
                        return (Some((col.clone(), *left.clone())), None);
                    }
                }
                (None, Some(predicate.clone()))
            }
            // AND: try to find an indexable condition in either side
            Expr::BinaryOp {
                left,
                op: BinaryOp::And,
                right,
            } => {
                // Try left side first
                let (left_idx, left_rem) = self.extract_index_conditions(table, left);
                if left_idx.is_some() {
                    // Combine remaining with right side
                    let remaining = match left_rem {
                        Some(rem) => Some(Expr::BinaryOp {
                            left: Box::new(rem),
                            op: BinaryOp::And,
                            right: right.clone(),
                        }),
                        None => Some(*right.clone()),
                    };
                    return (left_idx, remaining);
                }

                // Try right side
                let (right_idx, right_rem) = self.extract_index_conditions(table, right);
                if right_idx.is_some() {
                    // Combine remaining with left side
                    let remaining = match right_rem {
                        Some(rem) => Some(Expr::BinaryOp {
                            left: left.clone(),
                            op: BinaryOp::And,
                            right: Box::new(rem),
                        }),
                        None => Some(*left.clone()),
                    };
                    return (right_idx, remaining);
                }

                (None, Some(predicate.clone()))
            }
            _ => (None, Some(predicate.clone())),
        }
    }

    /// Evaluate an expression with subquery support
    fn eval_expr_with_subquery(&self, expr: &Expr, row: &[Value], columns: &[String]) -> Value {
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
            Expr::UnaryOp { op, expr: inner } => {
                let val = self.eval_expr_with_subquery(inner, row, columns);
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
                let l = self.eval_expr_with_subquery(left, row, columns);
                let r = self.eval_expr_with_subquery(right, row, columns);
                eval_binary_op(&l, op, &r)
            }
            Expr::Aggregate { .. } => Value::Null,
            Expr::Subquery(select) => {
                // Execute the subquery and return the first value
                if let Ok(plan) = sql_planner::plan(Statement::Select(select.clone())) {
                    // Clone self to avoid borrow issues - subqueries use same storage
                    if let Ok(QueryResult::Select { rows, .. }) = self.execute_subquery(&plan) {
                        if let Some(first_row) = rows.first() {
                            if let Some(first_val) = first_row.first() {
                                return first_val.clone();
                            }
                        }
                    }
                }
                Value::Null
            }
            Expr::InSubquery {
                expr,
                subquery,
                negated,
            } => {
                let val = self.eval_expr_with_subquery(expr, row, columns);
                if let Ok(plan) = sql_planner::plan(Statement::Select(subquery.clone())) {
                    if let Ok(QueryResult::Select { rows, .. }) = self.execute_subquery(&plan) {
                        // Check if val is in any of the subquery result rows
                        let found = rows
                            .iter()
                            .any(|r| r.first().map(|v| values_equal(v, &val)).unwrap_or(false));
                        return Value::Bool(if *negated { !found } else { found });
                    }
                }
                Value::Bool(false)
            }
            Expr::Exists(select) => {
                if let Ok(plan) = sql_planner::plan(Statement::Select(select.clone())) {
                    if let Ok(QueryResult::Select { rows, .. }) = self.execute_subquery(&plan) {
                        return Value::Bool(!rows.is_empty());
                    }
                }
                Value::Bool(false)
            }
            Expr::Like {
                expr,
                pattern,
                negated,
            } => {
                let val = self.eval_expr_with_subquery(expr, row, columns);
                let pat = self.eval_expr_with_subquery(pattern, row, columns);
                let result = match (val, pat) {
                    (Value::Text(s), Value::Text(p)) => match_like_pattern(&s, &p),
                    _ => false,
                };
                Value::Bool(if *negated { !result } else { result })
            }
            Expr::IsNull { expr, negated } => {
                let val = self.eval_expr_with_subquery(expr, row, columns);
                let is_null = matches!(val, Value::Null);
                Value::Bool(if *negated { !is_null } else { is_null })
            }
            Expr::Case {
                operand,
                when_clauses,
                else_result,
            } => {
                match operand {
                    Some(op) => {
                        let op_val = self.eval_expr_with_subquery(op, row, columns);
                        for (when_expr, then_expr) in when_clauses {
                            let when_val = self.eval_expr_with_subquery(when_expr, row, columns);
                            if values_equal(&op_val, &when_val) {
                                return self.eval_expr_with_subquery(then_expr, row, columns);
                            }
                        }
                    }
                    None => {
                        for (when_expr, then_expr) in when_clauses {
                            let cond = self.eval_expr_with_subquery(when_expr, row, columns);
                            if matches!(cond, Value::Bool(true)) {
                                return self.eval_expr_with_subquery(then_expr, row, columns);
                            }
                        }
                    }
                }
                match else_result {
                    Some(else_expr) => self.eval_expr_with_subquery(else_expr, row, columns),
                    None => Value::Null,
                }
            }
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let val = self.eval_expr_with_subquery(expr, row, columns);
                let low_val = self.eval_expr_with_subquery(low, row, columns);
                let high_val = self.eval_expr_with_subquery(high, row, columns);
                let in_range = !matches!(compare_values(&val, &low_val), std::cmp::Ordering::Less)
                    && !matches!(compare_values(&val, &high_val), std::cmp::Ordering::Greater);
                Value::Bool(if *negated { !in_range } else { in_range })
            }
            Expr::WindowFunction { .. } => {
                // Window functions are evaluated during query execution, not at expression level
                // Return NULL here as a placeholder - actual evaluation happens in execute_query
                Value::Null
            }
            Expr::Function { name, args } => {
                let evaluated_args: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr_with_subquery(a, row, columns))
                    .collect();
                eval_function(name, &evaluated_args)
            }
        }
    }

    /// Execute a subquery (read-only)
    fn execute_subquery(&self, plan: &LogicalPlan) -> ExecResult {
        // Clone the plan to avoid borrow issues
        let plan = plan.clone();

        // Create a temporary mutable self for query execution
        // Note: subqueries are read-only, so this is safe
        let temp_engine = Engine {
            storage: self.storage.clone(),
            transaction: TransactionState::default(),
            views: self.views.clone(),
            procedures: self.procedures.clone(),
        };
        temp_engine.execute_query(plan)
    }
}

/// Group rows by the GROUP BY expressions
fn group_rows(
    rows: &[Vec<Value>],
    group_by: &[Expr],
    columns: &[String],
) -> Vec<(Vec<Value>, Vec<Vec<Value>>)> {
    use std::collections::HashMap;

    if group_by.is_empty() {
        // No GROUP BY - all rows are one group
        return vec![(Vec::new(), rows.to_vec())];
    }

    let mut groups: HashMap<Vec<u64>, Vec<Vec<Value>>> = HashMap::new();

    for row in rows {
        // Compute the group key
        let key: Vec<Value> = group_by
            .iter()
            .map(|e| eval_expr(e, row, columns))
            .collect();

        // Hash the key for the HashMap
        let hash_key: Vec<u64> = key.iter().map(value_hash).collect();

        groups.entry(hash_key).or_default().push(row.clone());
    }

    // Convert to Vec to maintain iteration order
    groups
        .into_values()
        .map(|v| {
            // Reconstruct the actual key values from the first row
            let key: Vec<Value> = if v.is_empty() {
                Vec::new()
            } else {
                group_by
                    .iter()
                    .map(|e| eval_expr(e, &v[0], columns))
                    .collect()
            };
            (key, v)
        })
        .collect()
}

/// Hash a Value for grouping
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
        Value::Json(j) => {
            8u8.hash(&mut hasher);
            j.to_string().hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Evaluate HAVING clause with aggregate support
fn eval_having(
    expr: &Expr,
    group_rows: &[Vec<Value>],
    input_columns: &[String],
    _output_columns: &[String],
) -> bool {
    match eval_having_expr(expr, group_rows, input_columns) {
        Value::Bool(b) => b,
        _ => false,
    }
}

/// Evaluate an expression in HAVING context (where aggregates are computed over the group)
fn eval_having_expr(expr: &Expr, group_rows: &[Vec<Value>], columns: &[String]) -> Value {
    match expr {
        Expr::Aggregate { .. } => eval_aggregate(expr, group_rows, columns),
        Expr::BinaryOp { left, op, right } => {
            let l = eval_having_expr(left, group_rows, columns);
            let r = eval_having_expr(right, group_rows, columns);
            eval_binary_op(&l, op, &r)
        }
        Expr::UnaryOp { op, expr: inner } => {
            let val = eval_having_expr(inner, group_rows, columns);
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
        Expr::Column(name) => {
            // For non-aggregate columns in HAVING, use the first row's value
            group_rows
                .first()
                .and_then(|row| {
                    columns
                        .iter()
                        .position(|c| c == name)
                        .map(|idx| row.get(idx).cloned().unwrap_or(Value::Null))
                })
                .unwrap_or(Value::Null)
        }
        Expr::Integer(n) => Value::Int(*n),
        Expr::Float(f) => Value::Float(*f),
        Expr::String(s) => Value::Text(s.clone()),
        Expr::Boolean(b) => Value::Bool(*b),
        Expr::Null => Value::Null,
        // Subquery expressions need to be evaluated with context
        Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists(_) => Value::Null,
        Expr::Like {
            expr,
            pattern,
            negated,
        } => {
            let val = eval_having_expr(expr, group_rows, columns);
            let pat = eval_having_expr(pattern, group_rows, columns);
            let result = match (val, pat) {
                (Value::Text(s), Value::Text(p)) => match_like_pattern(&s, &p),
                _ => false,
            };
            Value::Bool(if *negated { !result } else { result })
        }
        Expr::IsNull { expr, negated } => {
            let val = eval_having_expr(expr, group_rows, columns);
            let is_null = matches!(val, Value::Null);
            Value::Bool(if *negated { !is_null } else { is_null })
        }
        Expr::Case {
            operand,
            when_clauses,
            else_result,
        } => {
            match operand {
                Some(op) => {
                    let op_val = eval_having_expr(op, group_rows, columns);
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = eval_having_expr(when_expr, group_rows, columns);
                        if values_equal(&op_val, &when_val) {
                            return eval_having_expr(then_expr, group_rows, columns);
                        }
                    }
                }
                None => {
                    for (when_expr, then_expr) in when_clauses {
                        let cond = eval_having_expr(when_expr, group_rows, columns);
                        if matches!(cond, Value::Bool(true)) {
                            return eval_having_expr(then_expr, group_rows, columns);
                        }
                    }
                }
            }
            match else_result {
                Some(else_expr) => eval_having_expr(else_expr, group_rows, columns),
                None => Value::Null,
            }
        }
        Expr::Between {
            expr,
            low,
            high,
            negated,
        } => {
            let val = eval_having_expr(expr, group_rows, columns);
            let low_val = eval_having_expr(low, group_rows, columns);
            let high_val = eval_having_expr(high, group_rows, columns);
            let in_range = !matches!(compare_values(&val, &low_val), std::cmp::Ordering::Less)
                && !matches!(compare_values(&val, &high_val), std::cmp::Ordering::Greater);
            Value::Bool(if *negated { !in_range } else { in_range })
        }
        Expr::WindowFunction { .. } => {
            // Window functions are evaluated during query execution, not in HAVING clause
            Value::Null
        }
        Expr::Function { name, args } => {
            let evaluated_args: Vec<Value> = args
                .iter()
                .map(|a| eval_having_expr(a, group_rows, columns))
                .collect();
            eval_function(name, &evaluated_args)
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
        DataType::Date => StorageDataType::Date,
        DataType::Time => StorageDataType::Time,
        DataType::Timestamp => StorageDataType::Timestamp,
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

/// Convert parser table constraint to storage format
fn convert_table_constraint(constraint: &ParserTableConstraint) -> StorageTableConstraint {
    match constraint {
        ParserTableConstraint::PrimaryKey { columns, .. } => StorageTableConstraint::PrimaryKey {
            columns: columns.clone(),
        },
        ParserTableConstraint::ForeignKey {
            columns,
            references_table,
            references_columns,
            on_delete,
            on_update,
            ..
        } => StorageTableConstraint::ForeignKey {
            columns: columns.clone(),
            references_table: references_table.clone(),
            references_columns: references_columns.clone(),
            on_delete: convert_ref_action(on_delete),
            on_update: convert_ref_action(on_update),
        },
        ParserTableConstraint::Unique { columns, .. } => StorageTableConstraint::Unique {
            columns: columns.clone(),
        },
        ParserTableConstraint::Check { .. } => {
            // CHECK constraints are not yet enforced at storage level
            // For now, we skip them (they're validated at query time)
            StorageTableConstraint::Unique {
                columns: Vec::new(),
            }
        }
    }
}

/// Substitute procedure parameters in a statement
fn substitute_params_in_statement(
    stmt: &Statement,
    bindings: &HashMap<String, Value>,
) -> Statement {
    match stmt {
        Statement::Insert(insert) => Statement::Insert(sql_parser::InsertStatement {
            table: insert.table.clone(),
            columns: insert.columns.clone(),
            values: insert
                .values
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| substitute_params_in_expr(e, bindings))
                        .collect()
                })
                .collect(),
        }),
        Statement::Update(update) => Statement::Update(sql_parser::UpdateStatement {
            table: update.table.clone(),
            assignments: update
                .assignments
                .iter()
                .map(|a| sql_parser::Assignment {
                    column: a.column.clone(),
                    value: substitute_params_in_expr(&a.value, bindings),
                })
                .collect(),
            where_clause: update
                .where_clause
                .as_ref()
                .map(|e| substitute_params_in_expr(e, bindings)),
        }),
        Statement::Delete(delete) => Statement::Delete(sql_parser::DeleteStatement {
            table: delete.table.clone(),
            where_clause: delete
                .where_clause
                .as_ref()
                .map(|e| substitute_params_in_expr(e, bindings)),
        }),
        Statement::Select(select) => {
            Statement::Select(Box::new(substitute_params_in_select(select, bindings)))
        }
        // Other statements don't need parameter substitution
        other => other.clone(),
    }
}

/// Substitute procedure parameters in a SELECT statement
fn substitute_params_in_select(
    select: &sql_parser::SelectStatement,
    bindings: &HashMap<String, Value>,
) -> sql_parser::SelectStatement {
    sql_parser::SelectStatement {
        with_clause: select.with_clause.clone(),
        distinct: select.distinct,
        columns: select
            .columns
            .iter()
            .map(|col| match col {
                sql_parser::SelectColumn::Star => sql_parser::SelectColumn::Star,
                sql_parser::SelectColumn::Expr { expr, alias } => sql_parser::SelectColumn::Expr {
                    expr: substitute_params_in_expr(expr, bindings),
                    alias: alias.clone(),
                },
            })
            .collect(),
        from: select.from.clone(),
        joins: select.joins.clone(),
        where_clause: select
            .where_clause
            .as_ref()
            .map(|e| substitute_params_in_expr(e, bindings)),
        group_by: select
            .group_by
            .iter()
            .map(|e| substitute_params_in_expr(e, bindings))
            .collect(),
        having: select
            .having
            .as_ref()
            .map(|e| substitute_params_in_expr(e, bindings)),
        order_by: select.order_by.clone(),
        limit: select.limit.clone(),
        offset: select.offset.clone(),
    }
}

/// Substitute procedure parameters in an expression
fn substitute_params_in_expr(expr: &Expr, bindings: &HashMap<String, Value>) -> Expr {
    match expr {
        Expr::Column(name) => {
            // Check if this column name matches a parameter binding
            if let Some(value) = bindings.get(name) {
                value_to_expr(value)
            } else {
                Expr::Column(name.clone())
            }
        }
        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left: Box::new(substitute_params_in_expr(left, bindings)),
            op: op.clone(),
            right: Box::new(substitute_params_in_expr(right, bindings)),
        },
        Expr::UnaryOp { op, expr } => Expr::UnaryOp {
            op: op.clone(),
            expr: Box::new(substitute_params_in_expr(expr, bindings)),
        },
        // For other expression types, just clone them
        _ => expr.clone(),
    }
}

/// Convert a Value back to an Expr (for parameter substitution)
fn value_to_expr(value: &Value) -> Expr {
    match value {
        Value::Int(n) => Expr::Integer(*n),
        Value::Float(f) => Expr::Float(*f),
        Value::Text(s) => Expr::String(s.clone()),
        Value::Bool(b) => Expr::Boolean(*b),
        Value::Null => Expr::Null,
        Value::Date(d) => Expr::String(d.to_string()),
        Value::Time(t) => Expr::String(t.to_string()),
        Value::Timestamp(ts) => Expr::String(ts.to_string()),
        Value::Json(j) => Expr::String(j.to_string()),
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

/// Check if an expression is a literal value (for index optimization)
fn is_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(_) | Expr::Float(_) | Expr::String(_) | Expr::Boolean(_) | Expr::Null => true,
        Expr::UnaryOp {
            op: UnaryOp::Neg,
            expr,
        } => is_literal(expr),
        _ => false,
    }
}

/// Convert inline trigger actions to a function body string
fn convert_inline_actions_to_body(actions: &[TriggerAction]) -> String {
    let mut parts = Vec::new();

    for action in actions {
        match action {
            TriggerAction::SetColumn { column, value } => {
                // Convert expression to a simple string representation
                // For now, we only support literals in inline actions
                let value_str = expr_to_literal_string(value);
                parts.push(format!("SET NEW.{} = {}", column, value_str));
            }
            TriggerAction::RaiseError(msg) => {
                parts.push(format!("RAISE ERROR '{}'", msg.replace('\'', "''")));
            }
        }
    }

    // If we have SET statements, add RETURN NEW at the end
    if parts.iter().any(|p| p.starts_with("SET NEW.")) {
        parts.push("RETURN NEW".to_string());
    }

    parts.join("; ")
}

/// Convert an expression to a literal string for trigger function body
fn expr_to_literal_string(expr: &Expr) -> String {
    match expr {
        Expr::Integer(n) => n.to_string(),
        Expr::Float(f) => f.to_string(),
        Expr::String(s) => format!("'{}'", s.replace('\'', "''")),
        Expr::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        Expr::Null => "NULL".to_string(),
        Expr::Column(name) => format!("NEW.{}", name), // Assume column refs are NEW
        _ => "NULL".to_string(), // Complex expressions not supported in inline triggers
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
        // Subquery expressions need to be evaluated with engine context
        Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists(_) => Value::Null,
        // LIKE pattern matching
        Expr::Like {
            expr,
            pattern,
            negated,
        } => {
            let val = eval_expr(expr, row, columns);
            let pat = eval_expr(pattern, row, columns);
            let result = match (val, pat) {
                (Value::Text(s), Value::Text(p)) => match_like_pattern(&s, &p),
                _ => false,
            };
            Value::Bool(if *negated { !result } else { result })
        }
        // IS NULL / IS NOT NULL
        Expr::IsNull { expr, negated } => {
            let val = eval_expr(expr, row, columns);
            let is_null = matches!(val, Value::Null);
            Value::Bool(if *negated { !is_null } else { is_null })
        }
        // CASE WHEN expression
        Expr::Case {
            operand,
            when_clauses,
            else_result,
        } => {
            match operand {
                Some(op) => {
                    // Simple CASE: CASE expr WHEN val THEN result ...
                    let op_val = eval_expr(op, row, columns);
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = eval_expr(when_expr, row, columns);
                        if values_equal(&op_val, &when_val) {
                            return eval_expr(then_expr, row, columns);
                        }
                    }
                }
                None => {
                    // Searched CASE: CASE WHEN cond THEN result ...
                    for (when_expr, then_expr) in when_clauses {
                        let cond = eval_expr(when_expr, row, columns);
                        if matches!(cond, Value::Bool(true)) {
                            return eval_expr(then_expr, row, columns);
                        }
                    }
                }
            }
            // Return ELSE result or NULL
            match else_result {
                Some(else_expr) => eval_expr(else_expr, row, columns),
                None => Value::Null,
            }
        }
        // BETWEEN expression
        Expr::Between {
            expr,
            low,
            high,
            negated,
        } => {
            let val = eval_expr(expr, row, columns);
            let low_val = eval_expr(low, row, columns);
            let high_val = eval_expr(high, row, columns);
            let in_range = !matches!(compare_values(&val, &low_val), std::cmp::Ordering::Less)
                && !matches!(compare_values(&val, &high_val), std::cmp::Ordering::Greater);
            Value::Bool(if *negated { !in_range } else { in_range })
        }
        Expr::WindowFunction { .. } => {
            // Window functions are evaluated during query execution, not at expression level
            Value::Null
        }
        Expr::Function { name, args } => {
            let evaluated_args: Vec<Value> =
                args.iter().map(|a| eval_expr(a, row, columns)).collect();
            eval_function(name, &evaluated_args)
        }
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

/// Match a SQL LIKE pattern using dynamic programming
/// % matches any sequence of characters
/// _ matches any single character
fn match_like_pattern(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();
    let s_len = s_chars.len();
    let p_len = p_chars.len();

    // dp[i][j] = true if s[0..i] matches pattern[0..j]
    let mut dp = vec![vec![false; p_len + 1]; s_len + 1];
    dp[0][0] = true;

    // Handle patterns starting with %
    for j in 1..=p_len {
        if p_chars[j - 1] == '%' {
            dp[0][j] = dp[0][j - 1];
        }
    }

    for i in 1..=s_len {
        for j in 1..=p_len {
            let pc = p_chars[j - 1];
            let sc = s_chars[i - 1];

            if pc == '%' {
                // % can match zero chars (dp[i][j-1]) or one more char (dp[i-1][j])
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if pc == '_' || pc == sc {
                // _ matches any single char, or exact match
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[s_len][p_len]
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

/// Check if an expression is a window function
fn is_window_function(expr: &Expr) -> bool {
    matches!(expr, Expr::WindowFunction { .. })
}

/// Evaluate window functions over all rows
fn eval_window_functions(
    exprs: &[(Expr, Option<String>)],
    rows: &[Vec<Value>],
    columns: &[String],
) -> Vec<Vec<Value>> {
    if rows.is_empty() {
        return Vec::new();
    }

    // For each row, we need to calculate all expressions including window functions
    rows.iter()
        .enumerate()
        .map(|(row_idx, row)| {
            exprs
                .iter()
                .map(|(expr, _)| {
                    if let Expr::WindowFunction {
                        func,
                        partition_by,
                        order_by,
                    } = expr
                    {
                        eval_single_window_function(
                            func,
                            partition_by,
                            order_by,
                            row_idx,
                            rows,
                            columns,
                        )
                    } else {
                        eval_expr(expr, row, columns)
                    }
                })
                .collect()
        })
        .collect()
}

/// Evaluate a single window function for a specific row
fn eval_single_window_function(
    func: &WindowFunc,
    partition_by: &[Expr],
    order_by: &[OrderBy],
    row_idx: usize,
    rows: &[Vec<Value>],
    columns: &[String],
) -> Value {
    let current_row = &rows[row_idx];

    // Get partition values for current row
    let current_partition: Vec<Value> = partition_by
        .iter()
        .map(|e| eval_expr(e, current_row, columns))
        .collect();

    // Find all rows in the same partition
    let partition_rows: Vec<(usize, &Vec<Value>)> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            let partition_values: Vec<Value> = partition_by
                .iter()
                .map(|e| eval_expr(e, r, columns))
                .collect();
            partition_values == current_partition
        })
        .collect();

    // Sort partition rows by ORDER BY if specified
    let mut sorted_indices: Vec<usize> = partition_rows.iter().map(|(i, _)| *i).collect();
    if !order_by.is_empty() {
        sorted_indices.sort_by(|&a, &b| {
            for ob in order_by {
                let val_a = eval_expr(&ob.expr, &rows[a], columns);
                let val_b = eval_expr(&ob.expr, &rows[b], columns);
                let cmp = compare_values(&val_a, &val_b);
                let cmp = if ob.desc { cmp.reverse() } else { cmp };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    // Find position of current row in sorted partition
    let position = sorted_indices
        .iter()
        .position(|&i| i == row_idx)
        .unwrap_or(0);

    match func {
        WindowFunc::RowNumber => {
            // ROW_NUMBER: sequential number starting from 1
            Value::Int((position + 1) as i64)
        }
        WindowFunc::Rank => {
            // RANK: rank with gaps for ties
            if position == 0 {
                Value::Int(1)
            } else {
                // Get current row's order values
                let current_order_values: Vec<Value> = order_by
                    .iter()
                    .map(|ob| eval_expr(&ob.expr, &rows[row_idx], columns))
                    .collect();

                // Count rows with different (smaller) order values before current row
                let rows_before = sorted_indices[..position]
                    .iter()
                    .filter(|&&idx| {
                        let other_order_values: Vec<Value> = order_by
                            .iter()
                            .map(|ob| eval_expr(&ob.expr, &rows[idx], columns))
                            .collect();
                        other_order_values != current_order_values
                    })
                    .count();

                Value::Int((rows_before + 1) as i64)
            }
        }
        WindowFunc::DenseRank => {
            // DENSE_RANK: rank without gaps for ties
            if position == 0 {
                Value::Int(1)
            } else {
                let current_order_values: Vec<Value> = order_by
                    .iter()
                    .map(|ob| eval_expr(&ob.expr, &rows[row_idx], columns))
                    .collect();

                // Count distinct order value combinations before current row
                let mut seen_values: Vec<Vec<Value>> = Vec::new();
                for &idx in &sorted_indices[..position] {
                    let other_order_values: Vec<Value> = order_by
                        .iter()
                        .map(|ob| eval_expr(&ob.expr, &rows[idx], columns))
                        .collect();
                    if other_order_values != current_order_values
                        && !seen_values.contains(&other_order_values)
                    {
                        seen_values.push(other_order_values);
                    }
                }

                Value::Int((seen_values.len() + 1) as i64)
            }
        }
    }
}

/// Evaluate a scalar function call
fn eval_function(name: &str, args: &[Value]) -> Value {
    match name.to_lowercase().as_str() {
        // JSON functions
        "json_extract" => eval_json_extract(args),
        "json_array_length" => eval_json_array_length(args),
        "json_type" => eval_json_type(args),
        "json_valid" => eval_json_valid(args),

        // String functions
        "upper" => eval_upper(args),
        "lower" => eval_lower(args),
        "length" => eval_length(args),
        "concat" => eval_concat(args),
        "substr" | "substring" => eval_substr(args),
        "trim" => eval_trim(args),
        "ltrim" => eval_ltrim(args),
        "rtrim" => eval_rtrim(args),
        "replace" => eval_replace(args),

        // Numeric functions
        "abs" => eval_abs(args),
        "round" => eval_round(args),
        "ceil" | "ceiling" => eval_ceil(args),
        "floor" => eval_floor(args),

        // Null handling
        "coalesce" => eval_coalesce(args),
        "nullif" => eval_nullif(args),
        "ifnull" => eval_ifnull(args),

        // Unknown function
        _ => Value::Null,
    }
}

// ===== JSON Functions =====

fn eval_json_extract(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let json_value = match &args[0] {
        Value::Json(j) => j.clone(),
        Value::Text(s) => match JsonValue::parse(s) {
            Ok(j) => j,
            Err(_) => return Value::Null,
        },
        _ => return Value::Null,
    };

    let path = match &args[1] {
        Value::Text(s) => s.as_str(),
        _ => return Value::Null,
    };

    match json_value.extract(path) {
        Some(extracted) => json_value_to_sql(extracted),
        None => Value::Null,
    }
}

fn eval_json_array_length(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    let json_value = match &args[0] {
        Value::Json(j) => j.clone(),
        Value::Text(s) => match JsonValue::parse(s) {
            Ok(j) => j,
            Err(_) => return Value::Null,
        },
        _ => return Value::Null,
    };

    // If path provided, extract first
    let target = if args.len() >= 2 {
        match &args[1] {
            Value::Text(path) => match json_value.extract(path) {
                Some(j) => j.clone(),
                None => return Value::Null,
            },
            _ => json_value,
        }
    } else {
        json_value
    };

    match target.array_length() {
        Some(len) => Value::Int(len as i64),
        None => Value::Null,
    }
}

fn eval_json_type(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    let json_value = match &args[0] {
        Value::Json(j) => j.clone(),
        Value::Text(s) => match JsonValue::parse(s) {
            Ok(j) => j,
            Err(_) => return Value::Null,
        },
        _ => return Value::Null,
    };

    // If path provided, extract first
    let target = if args.len() >= 2 {
        match &args[1] {
            Value::Text(path) => match json_value.extract(path) {
                Some(j) => j.clone(),
                None => return Value::Null,
            },
            _ => json_value,
        }
    } else {
        json_value
    };

    let type_name = match target {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Array(_) => "array",
        JsonValue::Object(_) => "object",
    };

    Value::Text(type_name.to_string())
}

fn eval_json_valid(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }

    match &args[0] {
        Value::Json(_) => Value::Bool(true),
        Value::Text(s) => Value::Bool(JsonValue::parse(s).is_ok()),
        _ => Value::Bool(false),
    }
}

/// Convert JsonValue to SQL Value
fn json_value_to_sql(json: &JsonValue) -> Value {
    match json {
        JsonValue::Null => Value::Null,
        JsonValue::Bool(b) => Value::Bool(*b),
        JsonValue::Number(n) => {
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                Value::Int(*n as i64)
            } else {
                Value::Float(*n)
            }
        }
        JsonValue::String(s) => Value::Text(s.clone()),
        // Arrays and objects are returned as JSON
        JsonValue::Array(_) | JsonValue::Object(_) => Value::Json(json.clone()),
    }
}

// ===== String Functions =====

fn eval_upper(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Text(s)) => Value::Text(s.to_uppercase()),
        _ => Value::Null,
    }
}

fn eval_lower(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Text(s)) => Value::Text(s.to_lowercase()),
        _ => Value::Null,
    }
}

fn eval_length(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Text(s)) => Value::Int(s.len() as i64),
        _ => Value::Null,
    }
}

fn eval_concat(args: &[Value]) -> Value {
    let mut result = String::new();
    for arg in args {
        match arg {
            Value::Text(s) => result.push_str(s),
            Value::Int(n) => result.push_str(&n.to_string()),
            Value::Float(f) => result.push_str(&f.to_string()),
            Value::Bool(b) => result.push_str(if *b { "true" } else { "false" }),
            Value::Null => {} // NULL is skipped in concat
            _ => {}
        }
    }
    Value::Text(result)
}

fn eval_substr(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let s = match &args[0] {
        Value::Text(s) => s,
        _ => return Value::Null,
    };

    let start = match &args[1] {
        Value::Int(n) => (*n as usize).saturating_sub(1), // SQL is 1-indexed
        _ => return Value::Null,
    };

    let len = if args.len() >= 3 {
        match &args[2] {
            Value::Int(n) => Some(*n as usize),
            _ => return Value::Null,
        }
    } else {
        None
    };

    let chars: Vec<char> = s.chars().collect();
    if start >= chars.len() {
        return Value::Text(String::new());
    }

    let result: String = match len {
        Some(l) => chars[start..].iter().take(l).collect(),
        None => chars[start..].iter().collect(),
    };

    Value::Text(result)
}

fn eval_trim(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Text(s)) => Value::Text(s.trim().to_string()),
        _ => Value::Null,
    }
}

fn eval_ltrim(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Text(s)) => Value::Text(s.trim_start().to_string()),
        _ => Value::Null,
    }
}

fn eval_rtrim(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Text(s)) => Value::Text(s.trim_end().to_string()),
        _ => Value::Null,
    }
}

fn eval_replace(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    match (&args[0], &args[1], &args[2]) {
        (Value::Text(s), Value::Text(from), Value::Text(to)) => Value::Text(s.replace(from, to)),
        _ => Value::Null,
    }
}

// ===== Numeric Functions =====

fn eval_abs(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Int(n)) => Value::Int(n.abs()),
        Some(Value::Float(f)) => Value::Float(f.abs()),
        _ => Value::Null,
    }
}

fn eval_round(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    let decimals = if args.len() >= 2 {
        match &args[1] {
            Value::Int(n) => *n as i32,
            _ => 0,
        }
    } else {
        0
    };

    match &args[0] {
        Value::Int(n) => Value::Int(*n),
        Value::Float(f) => {
            let multiplier = 10_f64.powi(decimals);
            Value::Float((f * multiplier).round() / multiplier)
        }
        _ => Value::Null,
    }
}

fn eval_ceil(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Int(n)) => Value::Int(*n),
        Some(Value::Float(f)) => Value::Int(f.ceil() as i64),
        _ => Value::Null,
    }
}

fn eval_floor(args: &[Value]) -> Value {
    match args.first() {
        Some(Value::Int(n)) => Value::Int(*n),
        Some(Value::Float(f)) => Value::Int(f.floor() as i64),
        _ => Value::Null,
    }
}

// ===== Null Handling Functions =====

fn eval_coalesce(args: &[Value]) -> Value {
    for arg in args {
        if !arg.is_null() {
            return arg.clone();
        }
    }
    Value::Null
}

fn eval_nullif(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    if values_equal(&args[0], &args[1]) {
        Value::Null
    } else {
        args[0].clone()
    }
}

fn eval_ifnull(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    if args[0].is_null() {
        args[1].clone()
    } else {
        args[0].clone()
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
    fn test_unique_column_constraint_sql() {
        let mut engine = Engine::new();

        // Create table with UNIQUE constraint on email column
        let result = engine.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE)");
        assert!(result.is_ok());

        // First insert should succeed
        let result =
            engine.execute("INSERT INTO users (id, email) VALUES (1, 'alice@example.com')");
        assert!(result.is_ok());

        // Second insert with different email should succeed
        let result = engine.execute("INSERT INTO users (id, email) VALUES (2, 'bob@example.com')");
        assert!(result.is_ok());

        // Third insert with duplicate email should fail
        let result =
            engine.execute("INSERT INTO users (id, email) VALUES (3, 'alice@example.com')");
        assert!(
            result.is_err(),
            "Should fail with UNIQUE constraint violation"
        );
    }

    #[test]
    fn test_unique_table_constraint_sql() {
        let mut engine = Engine::new();

        // Create table with multi-column UNIQUE constraint
        let result = engine.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, product_id INT, UNIQUE(user_id, product_id))",
        );
        assert!(result.is_ok());

        // Insert should succeed
        let result =
            engine.execute("INSERT INTO orders (id, user_id, product_id) VALUES (1, 100, 1)");
        assert!(result.is_ok());

        // Same user, different product should succeed
        let result =
            engine.execute("INSERT INTO orders (id, user_id, product_id) VALUES (2, 100, 2)");
        assert!(result.is_ok());

        // Same (user_id, product_id) should fail
        let result =
            engine.execute("INSERT INTO orders (id, user_id, product_id) VALUES (3, 100, 1)");
        assert!(
            result.is_err(),
            "Should fail with UNIQUE constraint violation on (user_id, product_id)"
        );
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
        // The error now comes from a trigger-based FK constraint
        let result = engine.execute("DELETE FROM categories WHERE id = 1");
        let err = result.unwrap_err();
        match &err {
            ExecError::InvalidExpression(msg) => {
                assert!(
                    msg.contains("Cannot delete") || msg.contains("referenced by"),
                    "Expected FK constraint error, got: {}",
                    msg
                );
            }
            _ => panic!(
                "Expected InvalidExpression error with FK constraint message, got: {:?}",
                err
            ),
        }

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

        // Try to create duplicate trigger - will fail because function already exists
        let result = engine
            .execute("CREATE TRIGGER set_name BEFORE INSERT ON t FOR EACH ROW SET name = 'other'");
        // We get a function exists error because inline trigger creates a function named __trigger_<name>__
        assert!(result.is_err());

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

        // Try to insert - should fail with trigger abort message
        let result = engine.execute("INSERT INTO orders (id, amount) VALUES (1, 100)");
        assert_eq!(
            result.unwrap_err(),
            ExecError::InvalidExpression("Trigger aborted: Inserts not allowed".to_string())
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

    #[test]
    fn test_drop_table() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t (x INT)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (1)").unwrap();

        let result = engine.execute("DROP TABLE t");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Table should no longer exist
        let result = engine.execute("SELECT * FROM t");
        assert!(result.is_err());
    }

    #[test]
    fn test_alter_table_add_column() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE users (id INT)").unwrap();
        engine.execute("INSERT INTO users (id) VALUES (1)").unwrap();
        engine.execute("INSERT INTO users (id) VALUES (2)").unwrap();

        // Add a new column
        let result = engine.execute("ALTER TABLE users ADD COLUMN name TEXT");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Existing rows should have NULL for the new column
        let result = engine.execute("SELECT id, name FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, columns } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows[0][0], Value::Int(1));
                assert_eq!(rows[0][1], Value::Null);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_alter_table_drop_column() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT, email TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name, email) VALUES (1, 'alice', 'a@example.com')")
            .unwrap();

        // Drop a column
        let result = engine.execute("ALTER TABLE users DROP COLUMN email");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Column should no longer exist
        let result = engine.execute("SELECT * FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { columns, .. } => {
                assert_eq!(columns.len(), 2);
                assert!(!columns.contains(&"email".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_alter_table_rename_column() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();

        // Rename column
        let result = engine.execute("ALTER TABLE users RENAME COLUMN name TO full_name");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Column should have new name
        let result = engine.execute("SELECT * FROM users WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert!(columns.contains(&"full_name".to_string()));
                assert!(!columns.contains(&"name".to_string()));
                assert_eq!(rows[0][1], Value::Text("alice".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_alter_table_rename_table() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE old_name (id INT)").unwrap();
        engine
            .execute("INSERT INTO old_name (id) VALUES (1)")
            .unwrap();

        // Rename table
        let result = engine.execute("ALTER TABLE old_name RENAME TO new_name");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Old name should not work
        let result = engine.execute("SELECT * FROM old_name");
        assert!(result.is_err());

        // New name should work
        let result = engine.execute("SELECT * FROM new_name");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_group_by_single_column() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE orders (id INT, category TEXT, amount INT)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, category, amount) VALUES (1, 'electronics', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, category, amount) VALUES (2, 'electronics', 200)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, category, amount) VALUES (3, 'books', 50)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, category, amount) VALUES (4, 'books', 75)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, category, amount) VALUES (5, 'clothing', 150)")
            .unwrap();

        // Group by category and sum amounts
        let result = engine.execute("SELECT category, SUM(amount) FROM orders GROUP BY category");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["category", "sum"]);
                assert_eq!(rows.len(), 3); // 3 categories

                // Find each category's total
                for row in &rows {
                    match &row[0] {
                        Value::Text(cat) if cat == "electronics" => {
                            assert_eq!(row[1], Value::Int(300));
                        }
                        Value::Text(cat) if cat == "books" => {
                            assert_eq!(row[1], Value::Int(125));
                        }
                        Value::Text(cat) if cat == "clothing" => {
                            assert_eq!(row[1], Value::Int(150));
                        }
                        _ => {}
                    }
                }
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_group_by_with_count() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, country TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, country) VALUES (1, 'USA')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, country) VALUES (2, 'USA')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, country) VALUES (3, 'UK')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, country) VALUES (4, 'Germany')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, country) VALUES (5, 'USA')")
            .unwrap();

        // Count users per country
        let result = engine.execute("SELECT country, COUNT(*) FROM users GROUP BY country");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 3); // 3 countries

                for row in &rows {
                    match &row[0] {
                        Value::Text(country) if country == "USA" => {
                            assert_eq!(row[1], Value::Int(3));
                        }
                        Value::Text(country) if country == "UK" => {
                            assert_eq!(row[1], Value::Int(1));
                        }
                        Value::Text(country) if country == "Germany" => {
                            assert_eq!(row[1], Value::Int(1));
                        }
                        _ => {}
                    }
                }
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_group_by_with_having() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE sales (id INT, product TEXT, amount INT)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, product, amount) VALUES (1, 'A', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, product, amount) VALUES (2, 'A', 200)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, product, amount) VALUES (3, 'B', 50)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, product, amount) VALUES (4, 'C', 300)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, product, amount) VALUES (5, 'C', 400)")
            .unwrap();

        // Only return products with total > 200
        let result = engine.execute(
            "SELECT product, SUM(amount) FROM sales GROUP BY product HAVING SUM(amount) > 200",
        );
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                // A = 300, B = 50, C = 700 -> A and C should be returned
                assert_eq!(rows.len(), 2);

                let products: Vec<&str> = rows
                    .iter()
                    .filter_map(|r| match &r[0] {
                        Value::Text(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(products.contains(&"A"));
                assert!(products.contains(&"C"));
                assert!(!products.contains(&"B"));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_group_by_with_avg() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE scores (id INT, subject TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (id, subject, score) VALUES (1, 'Math', 90)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (id, subject, score) VALUES (2, 'Math', 80)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (id, subject, score) VALUES (3, 'Science', 70)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (id, subject, score) VALUES (4, 'Science', 90)")
            .unwrap();

        // Average score per subject
        let result = engine.execute("SELECT subject, AVG(score) FROM scores GROUP BY subject");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);

                for row in &rows {
                    match &row[0] {
                        Value::Text(subject) if subject == "Math" => {
                            assert_eq!(row[1], Value::Float(85.0));
                        }
                        Value::Text(subject) if subject == "Science" => {
                            assert_eq!(row[1], Value::Float(80.0));
                        }
                        _ => {}
                    }
                }
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_group_by_with_min_max() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE products (id INT, category TEXT, price INT)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, category, price) VALUES (1, 'A', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, category, price) VALUES (2, 'A', 200)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, category, price) VALUES (3, 'B', 50)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, category, price) VALUES (4, 'B', 150)")
            .unwrap();

        // Min and max price per category
        let result = engine
            .execute("SELECT category, MIN(price), MAX(price) FROM products GROUP BY category");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);

                for row in &rows {
                    match &row[0] {
                        Value::Text(cat) if cat == "A" => {
                            assert_eq!(row[1], Value::Int(100)); // MIN
                            assert_eq!(row[2], Value::Int(200)); // MAX
                        }
                        Value::Text(cat) if cat == "B" => {
                            assert_eq!(row[1], Value::Int(50)); // MIN
                            assert_eq!(row[2], Value::Int(150)); // MAX
                        }
                        _ => {}
                    }
                }
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_in_subquery() {
        let mut engine = Engine::new();

        // Create and populate parent table
        engine
            .execute("CREATE TABLE departments (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
            .unwrap();
        engine
            .execute("INSERT INTO departments (id, name) VALUES (2, 'Sales')")
            .unwrap();
        engine
            .execute("INSERT INTO departments (id, name) VALUES (3, 'Marketing')")
            .unwrap();

        // Create and populate child table
        engine
            .execute("CREATE TABLE employees (id INT, name TEXT, dept_id INT)")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept_id) VALUES (1, 'Alice', 1)")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept_id) VALUES (2, 'Bob', 2)")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept_id) VALUES (3, 'Charlie', 4)")
            .unwrap(); // dept_id 4 doesn't exist

        // Select employees in existing departments using IN subquery
        let result = engine
            .execute("SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments)");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                // Alice (dept 1) and Bob (dept 2) should be returned, not Charlie (dept 4)
                assert_eq!(rows.len(), 2);
                let names: Vec<&str> = rows
                    .iter()
                    .filter_map(|r| match &r[0] {
                        Value::Text(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(names.contains(&"Alice"));
                assert!(names.contains(&"Bob"));
                assert!(!names.contains(&"Charlie"));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_not_in_subquery() {
        let mut engine = Engine::new();

        // Create tables
        engine.execute("CREATE TABLE active_ids (id INT)").unwrap();
        engine
            .execute("INSERT INTO active_ids (id) VALUES (1)")
            .unwrap();
        engine
            .execute("INSERT INTO active_ids (id) VALUES (3)")
            .unwrap();

        engine
            .execute("CREATE TABLE items (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO items (id, name) VALUES (1, 'A')")
            .unwrap();
        engine
            .execute("INSERT INTO items (id, name) VALUES (2, 'B')")
            .unwrap();
        engine
            .execute("INSERT INTO items (id, name) VALUES (3, 'C')")
            .unwrap();

        // Select items NOT IN active_ids
        let result =
            engine.execute("SELECT name FROM items WHERE id NOT IN (SELECT id FROM active_ids)");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                // Only item 2 ('B') should be returned
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("B".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_exists_subquery() {
        let mut engine = Engine::new();

        // Create tables
        engine
            .execute("CREATE TABLE orders (id INT, customer_id INT)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, customer_id) VALUES (1, 100)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, customer_id) VALUES (2, 101)")
            .unwrap();

        engine
            .execute("CREATE TABLE customers (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO customers (id, name) VALUES (100, 'Alice')")
            .unwrap();
        engine
            .execute("INSERT INTO customers (id, name) VALUES (101, 'Bob')")
            .unwrap();
        engine
            .execute("INSERT INTO customers (id, name) VALUES (102, 'Charlie')")
            .unwrap();

        // EXISTS with subquery that returns results
        let result =
            engine.execute("SELECT name FROM customers WHERE EXISTS (SELECT * FROM orders)");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                // EXISTS returns true for all rows since orders has data
                assert_eq!(rows.len(), 3);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_exists_empty_subquery() {
        let mut engine = Engine::new();

        // Create tables
        engine.execute("CREATE TABLE empty_table (id INT)").unwrap();

        engine
            .execute("CREATE TABLE data (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO data (id, name) VALUES (1, 'Test')")
            .unwrap();

        // EXISTS with empty subquery
        let result =
            engine.execute("SELECT name FROM data WHERE EXISTS (SELECT * FROM empty_table)");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                // EXISTS returns false since empty_table has no data
                assert_eq!(rows.len(), 0);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_date_type() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE events (id INT, event_date DATE, name TEXT)")
            .unwrap();

        // Get schema and verify DATE type
        let schema = engine.storage.get_schema("events").unwrap();
        assert_eq!(schema.columns[1].data_type, logical::DataType::Date);
    }

    #[test]
    fn test_timestamp_type() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE logs (id INT, created_at TIMESTAMP, message TEXT)")
            .unwrap();

        // Get schema and verify TIMESTAMP type
        let schema = engine.storage.get_schema("logs").unwrap();
        assert_eq!(schema.columns[1].data_type, logical::DataType::Timestamp);
    }

    #[test]
    fn test_time_type() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE schedules (id INT, start_time TIME)")
            .unwrap();

        // Get schema and verify TIME type
        let schema = engine.storage.get_schema("schedules").unwrap();
        assert_eq!(schema.columns[1].data_type, logical::DataType::Time);
    }

    #[test]
    fn test_create_index() {
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

        // Create index
        let result = engine.execute("CREATE INDEX idx_users_id ON users (id)");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Data should still be queryable
        let result = engine.execute("SELECT * FROM users");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_drop_index() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("CREATE INDEX idx_users_id ON users (id)")
            .unwrap();

        // Drop index
        let result = engine.execute("DROP INDEX idx_users_id");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Dropping again should fail
        let result = engine.execute("DROP INDEX idx_users_id");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_composite_index() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE orders (id INT, user_id INT, product_id INT)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, product_id) VALUES (1, 100, 1)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, product_id) VALUES (2, 100, 2)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (id, user_id, product_id) VALUES (3, 200, 1)")
            .unwrap();

        // Create composite index
        let result =
            engine.execute("CREATE INDEX idx_user_product ON orders (user_id, product_id)");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Data should still be queryable
        let result = engine.execute("SELECT * FROM orders WHERE user_id = 100");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_index_lookup() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        // Insert before index creation
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        // Create index on existing data
        engine
            .execute("CREATE INDEX idx_users_id ON users (id)")
            .unwrap();

        // Insert after index creation (should update index)
        engine
            .execute("INSERT INTO users (id, name) VALUES (3, 'charlie')")
            .unwrap();

        // Verify index exists and can lookup values
        let lookup = engine.storage.index_lookup("users", "id", &Value::Int(1));
        assert!(lookup.is_some());
        let indices = lookup.unwrap();
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_create_index_errors() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("CREATE INDEX idx_users_id ON users (id)")
            .unwrap();

        // Duplicate index name should fail
        let result = engine.execute("CREATE INDEX idx_users_id ON users (name)");
        assert!(result.is_err());

        // Index on non-existent table should fail
        let result = engine.execute("CREATE INDEX idx_foo ON nonexistent (col)");
        assert!(result.is_err());

        // Index on non-existent column should fail
        let result = engine.execute("CREATE INDEX idx_bar ON users (nonexistent)");
        assert!(result.is_err());
    }

    #[test]
    fn test_index_used_in_query() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT, email TEXT)")
            .unwrap();

        // Insert many rows
        for i in 0..100 {
            engine
                .execute(&format!(
                    "INSERT INTO users (id, name, email) VALUES ({}, 'user{}', 'user{}@test.com')",
                    i, i, i
                ))
                .unwrap();
        }

        // Create index on id column
        engine
            .execute("CREATE INDEX idx_users_id ON users (id)")
            .unwrap();

        // Query using the indexed column - should use index
        let result = engine.execute("SELECT name FROM users WHERE id = 50");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("user50".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Query using index with additional conditions (compound predicate)
        let result = engine.execute("SELECT name FROM users WHERE id = 25 AND name = 'user25'");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("user25".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Query where index value doesn't exist
        let result = engine.execute("SELECT name FROM users WHERE id = 999");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 0);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_distinct() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE items (category TEXT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO items (category, name) VALUES ('fruit', 'apple')")
            .unwrap();
        engine
            .execute("INSERT INTO items (category, name) VALUES ('fruit', 'banana')")
            .unwrap();
        engine
            .execute("INSERT INTO items (category, name) VALUES ('veggie', 'carrot')")
            .unwrap();
        engine
            .execute("INSERT INTO items (category, name) VALUES ('fruit', 'apple')")
            .unwrap(); // duplicate

        // Without DISTINCT - should have 4 rows
        let result = engine.execute("SELECT category FROM items");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 4);
            }
            _ => panic!("Expected Select result"),
        }

        // With DISTINCT - should have 2 unique categories
        let result = engine.execute("SELECT DISTINCT category FROM items");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_distinct_multiple_columns() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE orders (product TEXT, qty INT)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (product, qty) VALUES ('apple', 10)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (product, qty) VALUES ('apple', 10)")
            .unwrap(); // duplicate
        engine
            .execute("INSERT INTO orders (product, qty) VALUES ('apple', 20)")
            .unwrap();
        engine
            .execute("INSERT INTO orders (product, qty) VALUES ('banana', 10)")
            .unwrap();

        // With DISTINCT on multiple columns - should have 3 unique combinations
        let result = engine.execute("SELECT DISTINCT product, qty FROM orders");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 3);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_like_pattern() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE users (name TEXT)").unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('john')")
            .unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('johnny')")
            .unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('alice')")
            .unwrap();

        // Match starting with 'john'
        let result = engine.execute("SELECT * FROM users WHERE name LIKE 'john%'");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2); // john, johnny
            }
            _ => panic!("Expected Select result"),
        }

        // Match ending with 'ice'
        let result = engine.execute("SELECT * FROM users WHERE name LIKE '%ice'");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1); // alice
            }
            _ => panic!("Expected Select result"),
        }

        // Match with underscore (single char)
        let result = engine.execute("SELECT * FROM users WHERE name LIKE 'b_b'");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1); // bob
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_not_like() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE users (name TEXT)").unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('john')")
            .unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (name) VALUES ('alice')")
            .unwrap();

        // NOT LIKE - exclude john
        let result = engine.execute("SELECT * FROM users WHERE name NOT LIKE 'john'");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2); // bob, alice
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_is_null() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, email TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, email) VALUES (1, 'a@example.com')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, email) VALUES (2, NULL)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, email) VALUES (3, 'b@example.com')")
            .unwrap();

        // Find rows where email IS NULL
        let result = engine.execute("SELECT * FROM users WHERE email IS NULL");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(2));
            }
            _ => panic!("Expected Select result"),
        }

        // Find rows where email IS NOT NULL
        let result = engine.execute("SELECT * FROM users WHERE email IS NOT NULL");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_case_when_searched() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('alice', 95)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('bob', 72)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('charlie', 45)")
            .unwrap();

        // Searched CASE: CASE WHEN cond THEN result ...
        let result = engine.execute(
            "SELECT name, CASE WHEN score >= 90 THEN 'A' WHEN score >= 70 THEN 'B' ELSE 'C' END FROM scores"
        );
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0][1], Value::Text("A".to_string())); // alice: 95 -> A
                assert_eq!(rows[1][1], Value::Text("B".to_string())); // bob: 72 -> B
                assert_eq!(rows[2][1], Value::Text("C".to_string())); // charlie: 45 -> C
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_case_when_no_else() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE items (status TEXT)").unwrap();
        engine
            .execute("INSERT INTO items (status) VALUES ('active')")
            .unwrap();
        engine
            .execute("INSERT INTO items (status) VALUES ('pending')")
            .unwrap();

        // CASE without ELSE returns NULL when no match
        let result = engine.execute("SELECT CASE WHEN status = 'active' THEN 'yes' END FROM items");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("yes".to_string()));
                assert_eq!(rows[1][0], Value::Null);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_between() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE products (name TEXT, price INT)")
            .unwrap();
        engine
            .execute("INSERT INTO products (name, price) VALUES ('cheap', 10)")
            .unwrap();
        engine
            .execute("INSERT INTO products (name, price) VALUES ('medium', 50)")
            .unwrap();
        engine
            .execute("INSERT INTO products (name, price) VALUES ('expensive', 100)")
            .unwrap();

        // Test BETWEEN
        let result = engine.execute("SELECT name FROM products WHERE price BETWEEN 20 AND 80");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("medium".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Test NOT BETWEEN
        let result = engine
            .execute("SELECT name FROM products WHERE price NOT BETWEEN 20 AND 80 ORDER BY price");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], Value::Text("cheap".to_string()));
                assert_eq!(rows[1][0], Value::Text("expensive".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Test BETWEEN is inclusive on both ends
        let result = engine
            .execute("SELECT name FROM products WHERE price BETWEEN 10 AND 50 ORDER BY price");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], Value::Text("cheap".to_string()));
                assert_eq!(rows[1][0], Value::Text("medium".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_union() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE cats (name TEXT)").unwrap();
        engine.execute("CREATE TABLE dogs (name TEXT)").unwrap();

        engine
            .execute("INSERT INTO cats (name) VALUES ('whiskers')")
            .unwrap();
        engine
            .execute("INSERT INTO cats (name) VALUES ('fluffy')")
            .unwrap();
        engine
            .execute("INSERT INTO dogs (name) VALUES ('rex')")
            .unwrap();
        engine
            .execute("INSERT INTO dogs (name) VALUES ('buddy')")
            .unwrap();

        // Basic UNION
        let result = engine.execute("SELECT name FROM cats UNION SELECT name FROM dogs");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 4);
                let names: Vec<_> = rows.iter().map(|r| r[0].clone()).collect();
                assert!(names.contains(&Value::Text("whiskers".to_string())));
                assert!(names.contains(&Value::Text("fluffy".to_string())));
                assert!(names.contains(&Value::Text("rex".to_string())));
                assert!(names.contains(&Value::Text("buddy".to_string())));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_union_removes_duplicates() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t1 (val INT)").unwrap();
        engine.execute("CREATE TABLE t2 (val INT)").unwrap();

        engine.execute("INSERT INTO t1 (val) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t1 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (3)").unwrap();

        // UNION removes duplicates (2 appears in both tables)
        let result = engine.execute("SELECT val FROM t1 UNION SELECT val FROM t2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 3); // 1, 2, 3 (not 1, 2, 2, 3)
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_union_all_keeps_duplicates() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t1 (val INT)").unwrap();
        engine.execute("CREATE TABLE t2 (val INT)").unwrap();

        engine.execute("INSERT INTO t1 (val) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t1 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (3)").unwrap();

        // UNION ALL keeps all rows including duplicates
        let result = engine.execute("SELECT val FROM t1 UNION ALL SELECT val FROM t2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 4); // 1, 2, 2, 3
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_chained_unions() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t1 (val INT)").unwrap();
        engine.execute("CREATE TABLE t2 (val INT)").unwrap();
        engine.execute("CREATE TABLE t3 (val INT)").unwrap();

        engine.execute("INSERT INTO t1 (val) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t3 (val) VALUES (3)").unwrap();

        // Multiple UNIONs
        let result =
            engine.execute("SELECT val FROM t1 UNION SELECT val FROM t2 UNION SELECT val FROM t3");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 3);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_intersect() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t1 (val INT)").unwrap();
        engine.execute("CREATE TABLE t2 (val INT)").unwrap();

        engine.execute("INSERT INTO t1 (val) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t1 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t1 (val) VALUES (3)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (3)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (4)").unwrap();

        // INTERSECT returns rows that exist in both
        let result = engine.execute("SELECT val FROM t1 INTERSECT SELECT val FROM t2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2); // 2 and 3
                let vals: Vec<_> = rows.iter().map(|r| r[0].clone()).collect();
                assert!(vals.contains(&Value::Int(2)));
                assert!(vals.contains(&Value::Int(3)));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_except() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t1 (val INT)").unwrap();
        engine.execute("CREATE TABLE t2 (val INT)").unwrap();

        engine.execute("INSERT INTO t1 (val) VALUES (1)").unwrap();
        engine.execute("INSERT INTO t1 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t1 (val) VALUES (3)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (2)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (3)").unwrap();
        engine.execute("INSERT INTO t2 (val) VALUES (4)").unwrap();

        // EXCEPT returns rows in first set but not in second
        let result = engine.execute("SELECT val FROM t1 EXCEPT SELECT val FROM t2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1); // Only 1
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }

        // EXCEPT the other way
        let result = engine.execute("SELECT val FROM t2 EXCEPT SELECT val FROM t1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1); // Only 4
                assert_eq!(rows[0][0], Value::Int(4));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_row_number() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE sales (id INT, amount INT)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, amount) VALUES (1, 100)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, amount) VALUES (2, 200)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (id, amount) VALUES (3, 150)")
            .unwrap();

        // ROW_NUMBER() OVER (ORDER BY amount)
        let result = engine.execute("SELECT id, ROW_NUMBER() OVER (ORDER BY amount) FROM sales");
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns.len(), 2);
                assert_eq!(rows.len(), 3);
                // Sorted by amount: 100, 150, 200 => ids: 1, 3, 2
                // Row numbers should be 1, 2, 3 based on original row order
                // Row 0 (id=1, amount=100) is position 0 in sorted order => row_number = 1
                // Row 1 (id=2, amount=200) is position 2 in sorted order => row_number = 3
                // Row 2 (id=3, amount=150) is position 1 in sorted order => row_number = 2
                assert_eq!(rows[0][1], Value::Int(1)); // id=1, amount=100 is first
                assert_eq!(rows[1][1], Value::Int(3)); // id=2, amount=200 is third
                assert_eq!(rows[2][1], Value::Int(2)); // id=3, amount=150 is second
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_row_number_with_partition() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE sales (dept TEXT, emp TEXT, amount INT)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (dept, emp, amount) VALUES ('A', 'Alice', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (dept, emp, amount) VALUES ('A', 'Bob', 200)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (dept, emp, amount) VALUES ('B', 'Charlie', 150)")
            .unwrap();
        engine
            .execute("INSERT INTO sales (dept, emp, amount) VALUES ('B', 'Diana', 250)")
            .unwrap();

        // ROW_NUMBER() OVER (PARTITION BY dept ORDER BY amount)
        let result = engine.execute(
            "SELECT dept, emp, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY amount) FROM sales",
        );
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 4);
                // Within dept A: Alice (100) = 1, Bob (200) = 2
                // Within dept B: Charlie (150) = 1, Diana (250) = 2
                assert_eq!(rows[0][2], Value::Int(1)); // Alice is #1 in dept A
                assert_eq!(rows[1][2], Value::Int(2)); // Bob is #2 in dept A
                assert_eq!(rows[2][2], Value::Int(1)); // Charlie is #1 in dept B
                assert_eq!(rows[3][2], Value::Int(2)); // Diana is #2 in dept B
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_rank_with_ties() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Alice', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Bob', 90)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Charlie', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Diana', 80)")
            .unwrap();

        // RANK() should give same rank to ties and skip ranks
        // Sorted by score: 80, 90, 100, 100 => ranks: 1, 2, 3, 3 (4 is skipped)
        let result = engine.execute("SELECT name, RANK() OVER (ORDER BY score) FROM scores");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 4);
                // Original order: Alice(100), Bob(90), Charlie(100), Diana(80)
                // In sorted order: Diana(80)=1, Bob(90)=2, Alice(100)=3, Charlie(100)=3
                assert_eq!(rows[0][1], Value::Int(3)); // Alice score=100, rank=3
                assert_eq!(rows[1][1], Value::Int(2)); // Bob score=90, rank=2
                assert_eq!(rows[2][1], Value::Int(3)); // Charlie score=100, rank=3
                assert_eq!(rows[3][1], Value::Int(1)); // Diana score=80, rank=1
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_dense_rank() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Alice', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Bob', 90)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Charlie', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO scores (name, score) VALUES ('Diana', 80)")
            .unwrap();

        // DENSE_RANK() gives same rank to ties but doesn't skip ranks
        // Sorted by score: 80, 90, 100, 100 => dense_ranks: 1, 2, 3, 3
        let result = engine.execute("SELECT name, DENSE_RANK() OVER (ORDER BY score) FROM scores");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 4);
                // Original order: Alice(100), Bob(90), Charlie(100), Diana(80)
                // In sorted order: Diana(80)=1, Bob(90)=2, Alice(100)=3, Charlie(100)=3
                assert_eq!(rows[0][1], Value::Int(3)); // Alice score=100, dense_rank=3
                assert_eq!(rows[1][1], Value::Int(2)); // Bob score=90, dense_rank=2
                assert_eq!(rows[2][1], Value::Int(3)); // Charlie score=100, dense_rank=3
                assert_eq!(rows[3][1], Value::Int(1)); // Diana score=80, dense_rank=1
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_simple_cte() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept) VALUES (1, 'Alice', 'Engineering')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept) VALUES (2, 'Bob', 'Sales')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept) VALUES (3, 'Charlie', 'Engineering')")
            .unwrap();

        // Simple CTE
        let result = engine.execute(
            "WITH engineers AS (SELECT id, name FROM employees WHERE dept = 'Engineering') \
             SELECT * FROM engineers",
        );
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][1], Value::Text("Alice".to_string()));
                assert_eq!(rows[1][1], Value::Text("Charlie".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_cte_with_column_names() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE numbers (val INT)").unwrap();
        engine
            .execute("INSERT INTO numbers (val) VALUES (1)")
            .unwrap();
        engine
            .execute("INSERT INTO numbers (val) VALUES (2)")
            .unwrap();
        engine
            .execute("INSERT INTO numbers (val) VALUES (3)")
            .unwrap();

        // CTE with explicit column names
        let result = engine.execute(
            "WITH doubled (value) AS (SELECT val FROM numbers) \
             SELECT value FROM doubled",
        );
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["value"]);
                assert_eq!(rows.len(), 3);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_multiple_ctes() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE products (id INT, name TEXT, price INT)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 100)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 200)")
            .unwrap();
        engine
            .execute("INSERT INTO products (id, name, price) VALUES (3, 'Thing', 50)")
            .unwrap();

        // Multiple CTEs
        let result = engine.execute(
            "WITH cheap AS (SELECT * FROM products WHERE price < 150), \
                  expensive AS (SELECT * FROM products WHERE price >= 150) \
             SELECT name FROM cheap",
        );
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
                // Widget (100) and Thing (50) are cheap
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_simple_view() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept) VALUES (1, 'Alice', 'Engineering')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept) VALUES (2, 'Bob', 'Sales')")
            .unwrap();
        engine
            .execute("INSERT INTO employees (id, name, dept) VALUES (3, 'Charlie', 'Engineering')")
            .unwrap();

        // Create a view
        let result = engine.execute(
            "CREATE VIEW engineers AS SELECT id, name FROM employees WHERE dept = 'Engineering'",
        );
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Query the view
        let result = engine.execute("SELECT * FROM engineers");
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][1], Value::Text("Alice".to_string()));
                assert_eq!(rows[1][1], Value::Text("Charlie".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_view_with_columns() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE numbers (val INT)").unwrap();
        engine
            .execute("INSERT INTO numbers (val) VALUES (1)")
            .unwrap();
        engine
            .execute("INSERT INTO numbers (val) VALUES (2)")
            .unwrap();

        // Create a view with explicit column names
        let result = engine.execute("CREATE VIEW doubled (value) AS SELECT val FROM numbers");
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Query the view - column should be renamed
        let result = engine.execute("SELECT value FROM doubled");
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["value"]);
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_drop_view() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t (val INT)").unwrap();
        engine.execute("INSERT INTO t (val) VALUES (1)").unwrap();

        // Create and then drop a view
        engine
            .execute("CREATE VIEW v AS SELECT val FROM t")
            .unwrap();
        let result = engine.execute("SELECT * FROM v");
        assert!(result.is_ok());

        engine.execute("DROP VIEW v").unwrap();
        // View should no longer exist - query will fail
        let result = engine.execute("SELECT * FROM v");
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_procedure() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE logs (id INT, message TEXT)")
            .unwrap();

        // Create a simple procedure that inserts a log entry
        let result = engine.execute(
            "CREATE PROCEDURE add_log (msg TEXT) AS BEGIN INSERT INTO logs (id, message) VALUES (1, msg) END",
        );
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Call the procedure
        let result = engine.execute("CALL add_log ('Hello World')");
        assert!(result.is_ok());

        // Verify the insert happened
        let result = engine.execute("SELECT * FROM logs");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][1], Value::Text("Hello World".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_procedure_with_multiple_params() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE users (id INT, name TEXT, age INT)")
            .unwrap();

        // Create a procedure with multiple parameters
        let result = engine.execute(
            "CREATE PROCEDURE add_user (user_id INT, user_name TEXT, user_age INT) AS BEGIN INSERT INTO users (id, name, age) VALUES (user_id, user_name, user_age) END",
        );
        assert_eq!(result.unwrap(), QueryResult::Success);

        // Call the procedure
        engine.execute("CALL add_user (1, 'Alice', 30)").unwrap();
        engine.execute("CALL add_user (2, 'Bob', 25)").unwrap();

        // Verify the inserts
        let result = engine.execute("SELECT * FROM users ORDER BY id");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], Value::Int(1));
                assert_eq!(rows[0][1], Value::Text("Alice".to_string()));
                assert_eq!(rows[0][2], Value::Int(30));
                assert_eq!(rows[1][0], Value::Int(2));
                assert_eq!(rows[1][1], Value::Text("Bob".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_drop_procedure() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t (val INT)").unwrap();

        // Create a procedure
        engine
            .execute("CREATE PROCEDURE do_nothing AS BEGIN SELECT * FROM t END")
            .unwrap();

        // Call should work
        let result = engine.execute("CALL do_nothing");
        assert!(result.is_ok());

        // Drop the procedure
        engine.execute("DROP PROCEDURE do_nothing").unwrap();

        // Call should fail
        let result = engine.execute("CALL do_nothing");
        assert!(matches!(result, Err(ExecError::ProcedureNotFound(_))));
    }

    #[test]
    fn test_procedure_arg_count_mismatch() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t (val INT)").unwrap();

        // Create a procedure with one parameter
        engine
            .execute(
                "CREATE PROCEDURE needs_one (x INT) AS BEGIN INSERT INTO t (val) VALUES (x) END",
            )
            .unwrap();

        // Call with wrong number of arguments
        let result = engine.execute("CALL needs_one");
        assert!(matches!(
            result,
            Err(ExecError::ProcedureArgCountMismatch {
                expected: 1,
                got: 0
            })
        ));

        let result = engine.execute("CALL needs_one (1, 2)");
        assert!(matches!(
            result,
            Err(ExecError::ProcedureArgCountMismatch {
                expected: 1,
                got: 2
            })
        ));
    }

    #[test]
    fn test_exec_keyword() {
        let mut engine = Engine::new();

        engine.execute("CREATE TABLE t (val INT)").unwrap();

        engine
            .execute(
                "CREATE PROCEDURE insert_val (v INT) AS BEGIN INSERT INTO t (val) VALUES (v) END",
            )
            .unwrap();

        // Use EXEC instead of CALL
        let result = engine.execute("EXEC insert_val (42)");
        assert!(result.is_ok());

        let result = engine.execute("SELECT * FROM t");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(42));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_json_extract_function() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE json_data (id INT, data TEXT)")
            .unwrap();
        engine
            .execute(
                r#"INSERT INTO json_data (id, data) VALUES (1, '{"name": "alice", "age": 30}')"#,
            )
            .unwrap();
        engine
            .execute(r#"INSERT INTO json_data (id, data) VALUES (2, '{"name": "bob", "nested": {"city": "NYC"}}')"#)
            .unwrap();

        // Extract simple field
        let result =
            engine.execute(r#"SELECT json_extract(data, '$.name') FROM json_data WHERE id = 1"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("alice".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Extract nested field
        let result = engine
            .execute(r#"SELECT json_extract(data, '$.nested.city') FROM json_data WHERE id = 2"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("NYC".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Extract numeric field (integers are returned as Int)
        let result =
            engine.execute(r#"SELECT json_extract(data, '$.age') FROM json_data WHERE id = 1"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(30));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_json_array_length_function() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE arrays (id INT, data TEXT)")
            .unwrap();
        engine
            .execute(r#"INSERT INTO arrays (id, data) VALUES (1, '[1, 2, 3, 4, 5]')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO arrays (id, data) VALUES (2, '{"items": [10, 20]}')"#)
            .unwrap();

        // Top-level array length
        let result = engine.execute(r#"SELECT json_array_length(data) FROM arrays WHERE id = 1"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(5));
            }
            _ => panic!("Expected Select result"),
        }

        // Nested array length
        let result =
            engine.execute(r#"SELECT json_array_length(data, '$.items') FROM arrays WHERE id = 2"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(2));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_json_type_function() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE types (id INT, data TEXT)")
            .unwrap();
        engine
            .execute(r#"INSERT INTO types (id, data) VALUES (1, '"hello"')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO types (id, data) VALUES (2, '42')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO types (id, data) VALUES (3, 'true')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO types (id, data) VALUES (4, 'null')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO types (id, data) VALUES (5, '[1, 2, 3]')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO types (id, data) VALUES (6, '{"key": "value"}')"#)
            .unwrap();

        // String type
        let result = engine.execute(r#"SELECT json_type(data) FROM types WHERE id = 1"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("string".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Number type
        let result = engine.execute(r#"SELECT json_type(data) FROM types WHERE id = 2"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("number".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Boolean type
        let result = engine.execute(r#"SELECT json_type(data) FROM types WHERE id = 3"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("boolean".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Null type
        let result = engine.execute(r#"SELECT json_type(data) FROM types WHERE id = 4"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("null".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Array type
        let result = engine.execute(r#"SELECT json_type(data) FROM types WHERE id = 5"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("array".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // Object type
        let result = engine.execute(r#"SELECT json_type(data) FROM types WHERE id = 6"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("object".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_json_valid_function() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE validation (id INT, data TEXT)")
            .unwrap();
        engine
            .execute(r#"INSERT INTO validation (id, data) VALUES (1, '{"valid": true}')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO validation (id, data) VALUES (2, 'not valid json')"#)
            .unwrap();

        // Valid JSON
        let result = engine.execute(r#"SELECT json_valid(data) FROM validation WHERE id = 1"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Bool(true));
            }
            _ => panic!("Expected Select result"),
        }

        // Invalid JSON
        let result = engine.execute(r#"SELECT json_valid(data) FROM validation WHERE id = 2"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Bool(false));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_string_functions() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE strings (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO strings (id, name) VALUES (1, 'Hello World')")
            .unwrap();

        // UPPER
        let result = engine.execute("SELECT upper(name) FROM strings WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("HELLO WORLD".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // LOWER
        let result = engine.execute("SELECT lower(name) FROM strings WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("hello world".to_string()));
            }
            _ => panic!("Expected Select result"),
        }

        // LENGTH
        let result = engine.execute("SELECT length(name) FROM strings WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(11));
            }
            _ => panic!("Expected Select result"),
        }

        // TRIM
        engine
            .execute("INSERT INTO strings (id, name) VALUES (2, '  padded  ')")
            .unwrap();
        let result = engine.execute("SELECT trim(name) FROM strings WHERE id = 2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("padded".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_numeric_functions() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE numbers (id INT, val FLOAT)")
            .unwrap();
        engine
            .execute("INSERT INTO numbers (id, val) VALUES (1, -3.7)")
            .unwrap();

        // ABS
        let result = engine.execute("SELECT abs(val) FROM numbers WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Float(3.7));
            }
            _ => panic!("Expected Select result"),
        }

        // ROUND
        let result = engine.execute("SELECT round(val) FROM numbers WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Float(-4.0));
            }
            _ => panic!("Expected Select result"),
        }

        // CEIL/CEILING (returns Int)
        let result = engine.execute("SELECT ceil(val) FROM numbers WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(-3));
            }
            _ => panic!("Expected Select result"),
        }

        // FLOOR (returns Int)
        let result = engine.execute("SELECT floor(val) FROM numbers WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(-4));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_coalesce_and_nullif_functions() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE nulls (id INT, a INT, b INT)")
            .unwrap();
        engine
            .execute("INSERT INTO nulls (id, a, b) VALUES (1, NULL, 10)")
            .unwrap();
        engine
            .execute("INSERT INTO nulls (id, a, b) VALUES (2, 5, 10)")
            .unwrap();

        // COALESCE - returns first non-null
        let result = engine.execute("SELECT coalesce(a, b) FROM nulls WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(10));
            }
            _ => panic!("Expected Select result"),
        }

        // COALESCE - returns first when non-null
        let result = engine.execute("SELECT coalesce(a, b) FROM nulls WHERE id = 2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(5));
            }
            _ => panic!("Expected Select result"),
        }

        // NULLIF - returns null when equal
        let result = engine.execute("SELECT nullif(b, 10) FROM nulls WHERE id = 1");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Null);
            }
            _ => panic!("Expected Select result"),
        }

        // NULLIF - returns first when not equal
        let result = engine.execute("SELECT nullif(a, 10) FROM nulls WHERE id = 2");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(5));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_function_in_where_clause() {
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE items (id INT, data TEXT)")
            .unwrap();
        engine
            .execute(r#"INSERT INTO items (id, data) VALUES (1, '{"status": "active"}')"#)
            .unwrap();
        engine
            .execute(r#"INSERT INTO items (id, data) VALUES (2, '{"status": "inactive"}')"#)
            .unwrap();

        // Use json_extract in WHERE clause
        let result = engine
            .execute(r#"SELECT id FROM items WHERE json_extract(data, '$.status') = 'active'"#);
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }
}
