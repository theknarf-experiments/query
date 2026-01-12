//! SQL Abstract Syntax Tree types

/// A SQL statement
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// SELECT statement
    Select(Box<SelectStatement>),
    /// Set operation (UNION, INTERSECT, EXCEPT) of two or more SELECT statements
    SetOperation(SetOperationStatement),
    /// INSERT statement
    Insert(InsertStatement),
    /// UPDATE statement
    Update(UpdateStatement),
    /// DELETE statement
    Delete(DeleteStatement),
    /// CREATE TABLE statement
    CreateTable(CreateTableStatement),
    /// CREATE TRIGGER statement
    CreateTrigger(CreateTriggerStatement),
    /// DROP TRIGGER statement
    DropTrigger(String),
    /// DROP TABLE statement
    DropTable(String),
    /// ALTER TABLE statement
    AlterTable(AlterTableStatement),
    /// CREATE INDEX statement
    CreateIndex(CreateIndexStatement),
    /// DROP INDEX statement
    DropIndex(String),
    /// CREATE VIEW statement
    CreateView(CreateViewStatement),
    /// DROP VIEW statement
    DropView(String),
    /// BEGIN TRANSACTION
    Begin,
    /// COMMIT TRANSACTION
    Commit,
    /// ROLLBACK TRANSACTION
    Rollback,
    /// SAVEPOINT name
    Savepoint(String),
    /// RELEASE SAVEPOINT name
    ReleaseSavepoint(String),
    /// ROLLBACK TO SAVEPOINT name
    RollbackTo(String),
}

/// Set operation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetOperator {
    /// UNION - combines result sets
    Union,
    /// INTERSECT - returns common rows
    Intersect,
    /// EXCEPT - returns rows in first set but not in second
    Except,
}

/// A set operation statement combining multiple SELECT queries
#[derive(Debug, Clone, PartialEq)]
pub struct SetOperationStatement {
    /// The left side of the operation
    pub left: Box<SelectOrSet>,
    /// The right side of the operation
    pub right: Box<SelectOrSet>,
    /// The type of set operation
    pub op: SetOperator,
    /// Whether to keep duplicates (UNION ALL, etc.)
    pub all: bool,
}

/// Either a SELECT statement or a set operation (for chaining)
#[derive(Debug, Clone, PartialEq)]
pub enum SelectOrSet {
    Select(Box<SelectStatement>),
    SetOp(SetOperationStatement),
}

/// A Common Table Expression (CTE)
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    /// Name of the CTE
    pub name: String,
    /// Optional column names
    pub columns: Option<Vec<String>>,
    /// The query defining the CTE
    pub query: Box<SelectStatement>,
}

/// WITH clause containing CTEs
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    /// Whether this is a recursive CTE
    pub recursive: bool,
    /// The CTEs in the WITH clause
    pub ctes: Vec<Cte>,
}

/// A SELECT statement
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    pub with_clause: Option<WithClause>,
    pub distinct: bool,
    pub columns: Vec<SelectColumn>,
    pub from: Option<TableRef>,
    pub joins: Vec<Join>,
    pub where_clause: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderBy>,
    pub limit: Option<Expr>,
    pub offset: Option<Expr>,
}

/// A column in a SELECT statement
#[derive(Debug, Clone, PartialEq)]
pub enum SelectColumn {
    /// All columns (*)
    Star,
    /// A single expression with optional alias
    Expr { expr: Expr, alias: Option<String> },
}

/// A table reference
#[derive(Debug, Clone, PartialEq)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
}

/// A JOIN clause
#[derive(Debug, Clone, PartialEq)]
pub struct Join {
    pub join_type: JoinType,
    pub table: TableRef,
    pub on: Option<Expr>,
}

/// Type of JOIN operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// An expression
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
    /// Subquery - a SELECT statement used as an expression
    Subquery(Box<SelectStatement>),
    /// IN expression with subquery
    InSubquery {
        expr: Box<Expr>,
        subquery: Box<SelectStatement>,
        negated: bool,
    },
    /// EXISTS subquery
    Exists(Box<SelectStatement>),
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
        /// Optional operand for simple CASE (CASE x WHEN ...)
        operand: Option<Box<Expr>>,
        /// WHEN conditions and results
        when_clauses: Vec<(Expr, Expr)>,
        /// ELSE result
        else_result: Option<Box<Expr>>,
    },
    /// BETWEEN expression (expr BETWEEN low AND high)
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
}

/// Window functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
}

/// Aggregate functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// ORDER BY clause element
#[derive(Debug, Clone, PartialEq)]
pub struct OrderBy {
    pub expr: Expr,
    pub desc: bool,
}

/// INSERT statement
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStatement {
    pub table: String,
    pub columns: Option<Vec<String>>,
    pub values: Vec<Vec<Expr>>,
}

/// UPDATE statement
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: Vec<Assignment>,
    pub where_clause: Option<Expr>,
}

/// Column assignment (SET column = value)
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub column: String,
    pub value: Expr,
}

/// DELETE statement
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<Expr>,
}

/// CREATE TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
}

/// Column definition
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

/// Data types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Int,
    Float,
    Text,
    Bool,
    Date,
    Time,
    Timestamp,
}

/// CREATE INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStatement {
    pub name: String,
    pub table: String,
    pub column: String,
}

/// CREATE VIEW statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateViewStatement {
    pub name: String,
    pub columns: Option<Vec<String>>,
    pub query: Box<SelectStatement>,
}

/// CREATE TRIGGER statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTriggerStatement {
    pub name: String,
    pub timing: TriggerTiming,
    pub event: TriggerEvent,
    pub table: String,
    pub body: Vec<TriggerAction>,
}

/// When the trigger fires (BEFORE or AFTER)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
}

/// Event that fires the trigger
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
}

/// Action to take when trigger fires
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerAction {
    /// Set a column to an expression
    SetColumn { column: String, value: Expr },
    /// Prevent the operation (for BEFORE triggers)
    RaiseError(String),
}

/// ALTER TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStatement {
    pub table: String,
    pub action: AlterAction,
}

/// ALTER TABLE action
#[derive(Debug, Clone, PartialEq)]
pub enum AlterAction {
    /// ADD COLUMN
    AddColumn(ColumnDef),
    /// DROP COLUMN
    DropColumn(String),
    /// RENAME COLUMN
    RenameColumn { old_name: String, new_name: String },
    /// RENAME TABLE
    RenameTable(String),
}
