//! Query plan types

/// A logical query plan node
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Scan a table
    Scan {
        table: String,
        projection: Option<Vec<String>>,
    },
    /// Filter rows
    Filter {
        input: Box<LogicalPlan>,
        predicate: sql_parser::Expr,
    },
    /// Project columns
    Projection {
        input: Box<LogicalPlan>,
        exprs: Vec<sql_parser::Expr>,
    },
    /// Sort rows
    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<sql_parser::OrderBy>,
    },
    /// Limit rows
    Limit {
        input: Box<LogicalPlan>,
        limit: usize,
        offset: usize,
    },
}
