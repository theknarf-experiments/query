//! Intermediate Representation (IR) types for the query planner
//!
//! These types represent the planner's output - a logical plan that can be
//! executed by the engine. They are intentionally separate from the parser's
//! AST types to decouple parsing from planning/execution.

/// An expression in the logical plan
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference
    Column(String),
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Boolean(bool),
    /// NULL literal
    Null,
    /// Binary operation
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// Unary operation
    UnaryOp { op: UnaryOp, expr: Box<Expr> },
    /// Aggregate function call
    Aggregate { func: AggregateFunc, arg: Box<Expr> },
    /// Subquery
    Subquery(Box<super::LogicalPlan>),
    /// IN expression with subquery
    InSubquery {
        expr: Box<Expr>,
        subquery: Box<super::LogicalPlan>,
        negated: bool,
    },
    /// EXISTS subquery
    Exists(Box<super::LogicalPlan>),
    /// LIKE pattern matching
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    /// IS NULL / IS NOT NULL
    IsNull { expr: Box<Expr>, negated: bool },
    /// CASE WHEN expression
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_result: Option<Box<Expr>>,
    },
    /// BETWEEN expression
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    /// Window function call
    WindowFunction {
        func: WindowFunc,
        partition_by: Vec<Expr>,
        order_by: Vec<OrderBy>,
    },
    /// Scalar function call
    Function { name: String, args: Vec<Expr> },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    // Logical
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Aggregate functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Window functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
}

/// ORDER BY clause element
#[derive(Debug, Clone, PartialEq)]
pub struct OrderBy {
    pub expr: Expr,
    pub desc: bool,
}

/// Type of JOIN operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Set operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOperator {
    Union,
    Intersect,
    Except,
}

/// Column assignment (SET column = value)
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub column: String,
    pub value: Expr,
}

/// Data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int,
    Float,
    Text,
    Bool,
    Date,
    Time,
    Timestamp,
}

/// Column definition for CREATE TABLE
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub default: Option<Expr>,
    pub references: Option<ForeignKeyRef>,
}

/// Foreign key reference
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignKeyRef {
    pub table: String,
    pub column: String,
    pub on_delete: ReferentialAction,
    pub on_update: ReferentialAction,
}

/// Referential action for foreign keys
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferentialAction {
    #[default]
    NoAction,
    Cascade,
    SetNull,
    SetDefault,
    Restrict,
}

/// Table constraint
#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    PrimaryKey {
        name: Option<String>,
        columns: Vec<String>,
    },
    ForeignKey {
        name: Option<String>,
        columns: Vec<String>,
        references_table: String,
        references_columns: Vec<String>,
        on_delete: ReferentialAction,
        on_update: ReferentialAction,
    },
    Unique {
        name: Option<String>,
        columns: Vec<String>,
    },
    Check {
        name: Option<String>,
        expr: Expr,
    },
}

/// A Common Table Expression (CTE)
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    pub name: String,
    pub columns: Option<Vec<String>>,
    pub plan: Box<super::LogicalPlan>,
}

/// When the trigger fires
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
}

/// Event that fires the trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
}

/// Trigger action type
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerActionType {
    /// Execute a named function (PostgreSQL style)
    ExecuteFunction(String),
    /// Inline actions (legacy style)
    InlineActions(Vec<TriggerAction>),
}

/// Action to take when trigger fires
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerAction {
    /// Set a column to an expression
    SetColumn { column: String, value: Expr },
    /// Prevent the operation (for BEFORE triggers)
    RaiseError(String),
}

/// ALTER TABLE action
#[derive(Debug, Clone, PartialEq)]
pub enum AlterAction {
    AddColumn(ColumnDef),
    DropColumn(String),
    RenameColumn { old_name: String, new_name: String },
    RenameTable(String),
}

/// Parameter definition for stored procedures
#[derive(Debug, Clone, PartialEq)]
pub struct ProcedureParam {
    pub name: String,
    pub data_type: DataType,
}

/// Statement inside a procedure body
#[derive(Debug, Clone, PartialEq)]
pub enum ProcedureStatement {
    /// A logical plan to execute
    Plan(Box<super::LogicalPlan>),
    /// Variable declaration
    Declare { name: String, data_type: DataType },
    /// Assignment: SET @var = expr
    SetVar { name: String, value: Expr },
    /// Return statement
    Return(Option<Expr>),
}
