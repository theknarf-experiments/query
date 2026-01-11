//! Query plan types

use sql_parser::{Assignment, Expr, JoinType, OrderBy};

/// A logical query plan node
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Scan a table
    Scan { table: String },
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
        /// The type of set operation
        op: sql_parser::SetOperator,
        /// If true, keep duplicates (UNION ALL, etc.)
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
        columns: Vec<sql_parser::ColumnDef>,
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
    /// Create a trigger
    CreateTrigger {
        name: String,
        timing: sql_parser::TriggerTiming,
        event: sql_parser::TriggerEvent,
        table: String,
        body: Vec<sql_parser::TriggerAction>,
    },
    /// Drop a trigger
    DropTrigger { name: String },
    /// Drop a table
    DropTable { name: String },
    /// Alter a table
    AlterTable {
        table: String,
        action: sql_parser::AlterAction,
    },
    /// Create an index
    CreateIndex {
        name: String,
        table: String,
        column: String,
    },
    /// Drop an index
    DropIndex { name: String },
}
