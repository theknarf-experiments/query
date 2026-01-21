//! Intermediate Representation (IR) types for the Datalog planner
//!
//! These types represent the planner's output - a planned program ready for
//! evaluation. They are intentionally separate from the parser's AST types
//! to decouple parsing from planning/evaluation.
//!
//! The IR closely mirrors the AST structure for now, but is designed to
//! support future optimizations like:
//! - Body literal reordering
//! - Variable slot assignment
//! - Join strategy selection
//! - Builtin recognition

use internment::Intern;

/// Interned string for efficient storage and comparison
pub type Symbol = Intern<String>;

/// A planned Datalog program ready for evaluation
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedProgram {
    /// Rules organized by stratum (for stratified evaluation)
    pub strata: Vec<PlannedStratum>,
    /// Integrity constraints to check after evaluation
    pub constraints: Vec<PlannedConstraint>,
    /// Queries to execute
    pub queries: Vec<PlannedQuery>,
}

/// A stratum contains rules at the same stratification level
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedStratum {
    /// Rules in this stratum
    pub rules: Vec<PlannedRule>,
    /// Whether this stratum contains recursive rules
    pub is_recursive: bool,
    /// Predicates defined in this stratum
    pub predicates: Vec<Symbol>,
}

/// A planned rule ready for evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRule {
    /// The head atom (what this rule derives)
    pub head: Atom,
    /// Body literals (conditions to satisfy)
    pub body: Vec<Literal>,
}

/// A planned integrity constraint
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedConstraint {
    /// Body literals - constraint is violated if body is satisfiable
    pub body: Vec<Literal>,
}

/// A planned query
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedQuery {
    /// Body literals to satisfy
    pub body: Vec<Literal>,
}

/// A literal is either a positive atom, negative atom, comparison, or builtin
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    /// Positive atom: `parent(X, Y)`
    Positive(Atom),
    /// Negated atom: `not parent(X, Y)`
    Negative(Atom),
    /// Comparison: `X > 5`, `X = Y`
    Comparison(Comparison),
    /// Built-in predicate (classified at planning time)
    BuiltIn(BuiltIn),
}

/// Built-in predicates that can be evaluated directly without database lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    /// Arithmetic comparison: `=(X, Y)`, `<(X, 5)`, etc. (predicate syntax)
    Comparison(ComparisonOp, Term, Term),
    /// True (always succeeds)
    True,
    /// Fail (always fails)
    Fail,
}

/// A comparison between two terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    pub left: Term,
    pub op: ComparisonOp,
    pub right: Term,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

/// An atom is a predicate applied to terms: `parent(john, mary)`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub predicate: Symbol,
    pub terms: Vec<Term>,
}

/// A term can be a variable, constant, or compound term
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// Variable: uppercase or starts with underscore (X, Y, _tmp)
    Variable(Symbol),
    /// Constant value
    Constant(Value),
    /// Compound term: functor with arguments (f(a, b))
    Compound(Symbol, Vec<Term>),
}

/// Constant values
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(Symbol),
    /// Atom (lowercase identifier used as a constant)
    Atom(Symbol),
}

// Implement PartialEq, Eq, Hash for Value (needed for Term)
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Atom(a), Value::Atom(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Integer(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            Value::Float(f) => {
                1u8.hash(state);
                f.to_bits().hash(state);
            }
            Value::Boolean(b) => {
                2u8.hash(state);
                b.hash(state);
            }
            Value::String(s) => {
                3u8.hash(state);
                s.hash(state);
            }
            Value::Atom(a) => {
                4u8.hash(state);
                a.hash(state);
            }
        }
    }
}

// Helper methods for Literal
impl Literal {
    /// Get the underlying atom from a literal (None for comparisons/builtins)
    pub fn atom(&self) -> Option<&Atom> {
        match self {
            Literal::Positive(atom) | Literal::Negative(atom) => Some(atom),
            Literal::Comparison(_) | Literal::BuiltIn(_) => None,
        }
    }

    /// Check if the literal is positive
    pub fn is_positive(&self) -> bool {
        matches!(self, Literal::Positive(_))
    }

    /// Check if the literal is negative
    pub fn is_negative(&self) -> bool {
        matches!(self, Literal::Negative(_))
    }

    /// Check if the literal is a comparison
    pub fn is_comparison(&self) -> bool {
        matches!(self, Literal::Comparison(_))
    }

    /// Check if the literal is a builtin
    pub fn is_builtin(&self) -> bool {
        matches!(self, Literal::BuiltIn(_))
    }
}

// Helper methods for Atom
impl Atom {
    /// Create a new atom with the given predicate and terms
    pub fn new(predicate: impl Into<String>, terms: Vec<Term>) -> Self {
        Atom {
            predicate: Intern::new(predicate.into()),
            terms,
        }
    }
}

// Helper methods for Term
impl Term {
    /// Check if this term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    /// Check if this term is ground (contains no variables)
    pub fn is_ground(&self) -> bool {
        match self {
            Term::Variable(_) => false,
            Term::Constant(_) => true,
            Term::Compound(_, args) => args.iter().all(|t| t.is_ground()),
        }
    }
}

// Display implementations for debugging
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Atom(a) => write!(f, "{}", a),
        }
    }
}

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Variable(v) => write!(f, "{}", v),
            Term::Constant(c) => write!(f, "{}", c),
            Term::Compound(functor, args) => {
                write!(f, "{}(", functor)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.predicate)?;
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", term)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonOp::Equal => write!(f, "="),
            ComparisonOp::NotEqual => write!(f, "!="),
            ComparisonOp::LessThan => write!(f, "<"),
            ComparisonOp::LessOrEqual => write!(f, "<="),
            ComparisonOp::GreaterThan => write!(f, ">"),
            ComparisonOp::GreaterOrEqual => write!(f, ">="),
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Positive(atom) => write!(f, "{}", atom),
            Literal::Negative(atom) => write!(f, "not {}", atom),
            Literal::Comparison(comp) => write!(f, "{} {} {}", comp.left, comp.op, comp.right),
            Literal::BuiltIn(builtin) => write!(f, "{}", builtin),
        }
    }
}

impl std::fmt::Display for BuiltIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltIn::Comparison(op, left, right) => write!(f, "{} {} {}", left, op, right),
            BuiltIn::True => write!(f, "true"),
            BuiltIn::Fail => write!(f, "fail"),
        }
    }
}

impl std::fmt::Display for PlannedRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} :- ", self.head)?;
        for (i, lit) in self.body.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", lit)?;
        }
        Ok(())
    }
}
