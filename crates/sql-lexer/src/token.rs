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

    // Constraints
    Primary,
    Key,
    Foreign,
    References,
    Unique,
    Not,
    Default,

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

    // Aliases
    As,

    // Boolean literals
    True,
    False,
}

impl Keyword {
    /// Try to parse a keyword from a string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SELECT" => Some(Self::Select),
            "FROM" => Some(Self::From),
            "WHERE" => Some(Self::Where),
            "INSERT" => Some(Self::Insert),
            "INTO" => Some(Self::Into),
            "VALUES" => Some(Self::Values),
            "UPDATE" => Some(Self::Update),
            "SET" => Some(Self::Set),
            "DELETE" => Some(Self::Delete),
            "CREATE" => Some(Self::Create),
            "DROP" => Some(Self::Drop),
            "ALTER" => Some(Self::Alter),
            "TABLE" => Some(Self::Table),
            "INDEX" => Some(Self::Index),
            "INT" => Some(Self::Int),
            "INTEGER" => Some(Self::Integer),
            "TEXT" => Some(Self::Text),
            "VARCHAR" => Some(Self::Varchar),
            "BOOL" => Some(Self::Bool),
            "BOOLEAN" => Some(Self::Boolean),
            "FLOAT" => Some(Self::Float),
            "DOUBLE" => Some(Self::Double),
            "NULL" => Some(Self::Null),
            "PRIMARY" => Some(Self::Primary),
            "KEY" => Some(Self::Key),
            "FOREIGN" => Some(Self::Foreign),
            "REFERENCES" => Some(Self::References),
            "UNIQUE" => Some(Self::Unique),
            "NOT" => Some(Self::Not),
            "DEFAULT" => Some(Self::Default),
            "AND" => Some(Self::And),
            "OR" => Some(Self::Or),
            "IS" => Some(Self::Is),
            "IN" => Some(Self::In),
            "LIKE" => Some(Self::Like),
            "BETWEEN" => Some(Self::Between),
            "JOIN" => Some(Self::Join),
            "INNER" => Some(Self::Inner),
            "LEFT" => Some(Self::Left),
            "RIGHT" => Some(Self::Right),
            "OUTER" => Some(Self::Outer),
            "FULL" => Some(Self::Full),
            "CROSS" => Some(Self::Cross),
            "ON" => Some(Self::On),
            "ORDER" => Some(Self::Order),
            "BY" => Some(Self::By),
            "ASC" => Some(Self::Asc),
            "DESC" => Some(Self::Desc),
            "LIMIT" => Some(Self::Limit),
            "OFFSET" => Some(Self::Offset),
            "GROUP" => Some(Self::Group),
            "HAVING" => Some(Self::Having),
            "AS" => Some(Self::As),
            "TRUE" => Some(Self::True),
            "FALSE" => Some(Self::False),
            _ => None,
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
