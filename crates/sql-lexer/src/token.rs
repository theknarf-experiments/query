//! Token types for SQL lexer

use std::hash::{Hash, Hasher};

/// Wrapper for f64 that implements Hash and Eq using bit representation
#[derive(Debug, Clone, Copy)]
pub struct FloatBits(pub f64);

impl FloatBits {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    pub fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for FloatBits {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatBits {}

impl Hash for FloatBits {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for FloatBits {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

/// SQL Keywords
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Keyword {
    // DML
    Select,
    Distinct,
    From,
    Where,
    Insert,
    Into,
    Values,
    Update,
    Set,
    Delete,

    // DDL
    Create,
    Drop,
    Alter,
    Table,
    Index,

    // Data types
    Int,
    Integer,
    Text,
    Varchar,
    Bool,
    Boolean,
    Float,
    Double,
    Null,
    Date,
    Timestamp,
    Time,

    // Constraints
    Primary,
    Key,
    Foreign,
    References,
    Unique,
    Not,
    Default,
    Constraint,
    Check,
    Cascade,
    Restrict,
    Action,
    NoAction,

    // Logical operators
    And,
    Or,
    Is,
    In,
    Like,
    Between,

    // Joins
    Join,
    Inner,
    Left,
    Right,
    Outer,
    Full,
    Cross,
    On,

    // Ordering
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Offset,

    // Grouping
    Group,
    Having,

    // Transactions
    Begin,
    Commit,
    Rollback,
    Transaction,
    Savepoint,
    Release,
    To,

    // Aliases
    As,

    // Boolean literals
    True,
    False,

    // Aggregate functions
    Count,
    Sum,
    Avg,
    Min,
    Max,

    // Triggers
    Trigger,
    Before,
    After,
    For,
    Each,
    Row,
    Raise,
    Error,

    // Alter table
    Add,
    Column,
    Rename,

    // Subqueries
    Exists,

    // CASE expressions
    Case,
    When,
    Then,
    Else,
    End,

    // Set operations
    Union,
    All,
}

/// Error when parsing a keyword fails
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseKeywordError;

impl std::fmt::Display for ParseKeywordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown keyword")
    }
}

impl std::error::Error for ParseKeywordError {}

impl std::str::FromStr for Keyword {
    type Err = ParseKeywordError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "SELECT" => Ok(Self::Select),
            "DISTINCT" => Ok(Self::Distinct),
            "FROM" => Ok(Self::From),
            "WHERE" => Ok(Self::Where),
            "INSERT" => Ok(Self::Insert),
            "INTO" => Ok(Self::Into),
            "VALUES" => Ok(Self::Values),
            "UPDATE" => Ok(Self::Update),
            "SET" => Ok(Self::Set),
            "DELETE" => Ok(Self::Delete),
            "CREATE" => Ok(Self::Create),
            "DROP" => Ok(Self::Drop),
            "ALTER" => Ok(Self::Alter),
            "TABLE" => Ok(Self::Table),
            "INDEX" => Ok(Self::Index),
            "INT" => Ok(Self::Int),
            "INTEGER" => Ok(Self::Integer),
            "TEXT" => Ok(Self::Text),
            "VARCHAR" => Ok(Self::Varchar),
            "BOOL" => Ok(Self::Bool),
            "BOOLEAN" => Ok(Self::Boolean),
            "FLOAT" => Ok(Self::Float),
            "DOUBLE" => Ok(Self::Double),
            "NULL" => Ok(Self::Null),
            "DATE" => Ok(Self::Date),
            "TIMESTAMP" => Ok(Self::Timestamp),
            "TIME" => Ok(Self::Time),
            "PRIMARY" => Ok(Self::Primary),
            "KEY" => Ok(Self::Key),
            "FOREIGN" => Ok(Self::Foreign),
            "REFERENCES" => Ok(Self::References),
            "UNIQUE" => Ok(Self::Unique),
            "NOT" => Ok(Self::Not),
            "DEFAULT" => Ok(Self::Default),
            "CONSTRAINT" => Ok(Self::Constraint),
            "CHECK" => Ok(Self::Check),
            "CASCADE" => Ok(Self::Cascade),
            "RESTRICT" => Ok(Self::Restrict),
            "ACTION" => Ok(Self::Action),
            "NO" => Ok(Self::NoAction), // NO ACTION is parsed as NO + ACTION
            "AND" => Ok(Self::And),
            "OR" => Ok(Self::Or),
            "IS" => Ok(Self::Is),
            "IN" => Ok(Self::In),
            "LIKE" => Ok(Self::Like),
            "BETWEEN" => Ok(Self::Between),
            "JOIN" => Ok(Self::Join),
            "INNER" => Ok(Self::Inner),
            "LEFT" => Ok(Self::Left),
            "RIGHT" => Ok(Self::Right),
            "OUTER" => Ok(Self::Outer),
            "FULL" => Ok(Self::Full),
            "CROSS" => Ok(Self::Cross),
            "ON" => Ok(Self::On),
            "ORDER" => Ok(Self::Order),
            "BY" => Ok(Self::By),
            "ASC" => Ok(Self::Asc),
            "DESC" => Ok(Self::Desc),
            "LIMIT" => Ok(Self::Limit),
            "OFFSET" => Ok(Self::Offset),
            "GROUP" => Ok(Self::Group),
            "HAVING" => Ok(Self::Having),
            "BEGIN" => Ok(Self::Begin),
            "COMMIT" => Ok(Self::Commit),
            "ROLLBACK" => Ok(Self::Rollback),
            "TRANSACTION" => Ok(Self::Transaction),
            "SAVEPOINT" => Ok(Self::Savepoint),
            "RELEASE" => Ok(Self::Release),
            "TO" => Ok(Self::To),
            "AS" => Ok(Self::As),
            "TRUE" => Ok(Self::True),
            "FALSE" => Ok(Self::False),
            "COUNT" => Ok(Self::Count),
            "SUM" => Ok(Self::Sum),
            "AVG" => Ok(Self::Avg),
            "MIN" => Ok(Self::Min),
            "MAX" => Ok(Self::Max),
            "TRIGGER" => Ok(Self::Trigger),
            "BEFORE" => Ok(Self::Before),
            "AFTER" => Ok(Self::After),
            "FOR" => Ok(Self::For),
            "EACH" => Ok(Self::Each),
            "ROW" => Ok(Self::Row),
            "RAISE" => Ok(Self::Raise),
            "ERROR" => Ok(Self::Error),
            "ADD" => Ok(Self::Add),
            "COLUMN" => Ok(Self::Column),
            "RENAME" => Ok(Self::Rename),
            "EXISTS" => Ok(Self::Exists),
            "CASE" => Ok(Self::Case),
            "WHEN" => Ok(Self::When),
            "THEN" => Ok(Self::Then),
            "ELSE" => Ok(Self::Else),
            "END" => Ok(Self::End),
            "UNION" => Ok(Self::Union),
            "ALL" => Ok(Self::All),
            _ => Err(ParseKeywordError),
        }
    }
}

/// A token in the SQL language
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    // Keywords
    Keyword(Keyword),

    // Identifiers and literals
    Identifier(String),
    String(String),
    Integer(i64),
    /// Float stored as bit representation for Hash/Eq support
    Float(FloatBits),

    // Punctuation
    LParen,    // (
    RParen,    // )
    Comma,     // ,
    Semicolon, // ;
    Dot,       // .
    Star,      // *

    // Operators
    Plus,    // +
    Minus,   // -
    Slash,   // /
    Percent, // %

    // Comparison
    Eq,    // =
    NotEq, // <> or !=
    Lt,    // <
    Gt,    // >
    LtEq,  // <=
    GtEq,  // >=
}
