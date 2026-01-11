//! Query plan types

use sql_parser::{Assignment, Expr, OrderBy};

/// A logical query plan node
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Scan a table
    Scan { table: String },
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
}
