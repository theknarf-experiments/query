//! Query plan types

use crate::ir::{
    AlterAction, Assignment, ColumnDef, Cte, Expr, JoinType, OrderBy, ProcedureParam,
    ProcedureStatement, SetOperator, TableConstraint, TriggerActionType, TriggerEvent,
    TriggerTiming,
};

/// A logical query plan node
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Scan a table
    Scan { table: String },
    /// Common Table Expression (WITH clause)
    WithCte {
        /// The CTE definitions
        ctes: Vec<Cte>,
        /// Whether this is a recursive CTE
        recursive: bool,
        /// The main query
        input: Box<LogicalPlan>,
    },
    /// Index scan - lookup rows using an index
    IndexScan {
        table: String,
        column: String,
        value: Expr,
    },
    /// Join two inputs
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        on: Option<Expr>,
    },
    /// Filter rows
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    /// Project columns/expressions
    Projection {
        input: Box<LogicalPlan>,
        exprs: Vec<(Expr, Option<String>)>,
    },
    /// Aggregate rows by grouping
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<Expr>,
        aggregates: Vec<(Expr, Option<String>)>,
        having: Option<Expr>,
    },
    /// Sort rows
    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<OrderBy>,
    },
    /// Limit rows
    Limit {
        input: Box<LogicalPlan>,
        limit: usize,
        offset: usize,
    },
    /// Set operation (UNION, INTERSECT, EXCEPT)
    SetOperation {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        op: SetOperator,
        all: bool,
    },
    /// Remove duplicate rows
    Distinct { input: Box<LogicalPlan> },
    /// Insert rows into a table
    Insert {
        table: String,
        columns: Option<Vec<String>>,
        values: Vec<Vec<Expr>>,
    },
    /// Create a new table
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
        constraints: Vec<TableConstraint>,
    },
    /// Update rows in a table
    Update {
        table: String,
        assignments: Vec<Assignment>,
        where_clause: Option<Expr>,
    },
    /// Delete rows from a table
    Delete {
        table: String,
        where_clause: Option<Expr>,
    },
    /// Begin transaction
    Begin,
    /// Commit transaction
    Commit,
    /// Rollback transaction
    Rollback,
    /// Create savepoint
    Savepoint { name: String },
    /// Release savepoint
    ReleaseSavepoint { name: String },
    /// Rollback to savepoint
    RollbackTo { name: String },
    /// Create a function (for triggers)
    CreateFunction {
        name: String,
        body: String,
        language: String,
    },
    /// Drop a function
    DropFunction { name: String },
    /// Create a trigger
    CreateTrigger {
        name: String,
        timing: TriggerTiming,
        events: Vec<TriggerEvent>,
        table: String,
        action: TriggerActionType,
    },
    /// Drop a trigger
    DropTrigger { name: String },
    /// Drop a table
    DropTable { name: String },
    /// Alter a table
    AlterTable { table: String, action: AlterAction },
    /// Create an index (supports composite indexes)
    CreateIndex {
        name: String,
        table: String,
        columns: Vec<String>,
    },
    /// Drop an index
    DropIndex { name: String },
    /// Create a view
    CreateView {
        name: String,
        columns: Option<Vec<String>>,
        query: Box<LogicalPlan>,
    },
    /// Drop a view
    DropView { name: String },
    /// Create a stored procedure
    CreateProcedure {
        name: String,
        params: Vec<ProcedureParam>,
        body: Vec<ProcedureStatement>,
    },
    /// Drop a stored procedure
    DropProcedure { name: String },
    /// Call a stored procedure
    CallProcedure { name: String, args: Vec<Expr> },

    // ===== Recursive query support (for Datalog and SQL recursive CTEs) =====
    /// Recursive fixpoint evaluation
    Recursive {
        name: String,
        columns: Vec<String>,
        base: Box<LogicalPlan>,
        step: Box<LogicalPlan>,
    },

    /// Stratified evaluation for handling negation
    Stratify { strata: Vec<LogicalPlan> },

    /// Reference to a recursive relation (used within Recursive.step)
    RecursiveRef { name: String },

    /// WITH RECURSIVE CTE
    WithRecursiveCte {
        name: String,
        columns: Option<Vec<String>>,
        base: Box<LogicalPlan>,
        step: Box<LogicalPlan>,
        pre_ctes: Vec<Cte>,
        input: Box<LogicalPlan>,
    },
}
