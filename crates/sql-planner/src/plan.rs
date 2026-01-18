//! Query plan types

use sql_parser::{Assignment, Cte, Expr, JoinType, OrderBy, ProcedureParam, ProcedureStatement};

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
        constraints: Vec<sql_parser::TableConstraint>,
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
    ///
    /// Evaluates a recursive query using semi-naive evaluation:
    /// 1. Evaluate base case to get initial facts
    /// 2. Repeatedly evaluate step with new facts (delta) until fixpoint
    ///
    /// Used for both Datalog recursive rules and SQL recursive CTEs.
    Recursive {
        /// Name of the recursive relation being computed
        name: String,
        /// Column names for the recursive relation
        columns: Vec<String>,
        /// Base case (non-recursive part)
        base: Box<LogicalPlan>,
        /// Recursive step (may reference `name` for recursive calls)
        step: Box<LogicalPlan>,
    },

    /// Stratified evaluation for handling negation
    ///
    /// Evaluates plans in stratum order. Each stratum is computed to fixpoint
    /// before moving to the next. This ensures correct semantics for negation:
    /// a predicate can only be negated if it's fully computed in a lower stratum.
    Stratify {
        /// Plans to execute in order (each stratum)
        strata: Vec<LogicalPlan>,
    },

    /// Reference to a recursive relation (used within Recursive.step)
    ///
    /// During semi-naive evaluation, this is bound to either the full
    /// relation or the delta (newly derived facts) depending on context.
    RecursiveRef {
        /// Name of the recursive relation to reference
        name: String,
    },

    /// WITH RECURSIVE CTE - computes a recursive CTE and binds it for the main query
    ///
    /// This is a high-level node that combines:
    /// 1. Computing the recursive CTE using semi-naive evaluation
    /// 2. Binding the result as a CTE for the main query
    WithRecursiveCte {
        /// Name of the recursive CTE
        name: String,
        /// Column names for the CTE (optional)
        columns: Option<Vec<String>>,
        /// The base case plan (non-recursive part)
        base: Box<LogicalPlan>,
        /// The recursive step plan (references the CTE name)
        step: Box<LogicalPlan>,
        /// Non-recursive CTEs defined before this one
        pre_ctes: Vec<sql_parser::Cte>,
        /// The main query that uses the CTE
        input: Box<LogicalPlan>,
    },
}
