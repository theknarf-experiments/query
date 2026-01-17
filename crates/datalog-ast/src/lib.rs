//! Abstract Syntax Tree (AST) definitions for Datalog
//!
//! This module defines the core data structures representing a Datalog program.
//!
//! # Key Components
//!
//! - **Program**: A collection of statements (facts, rules, queries)
//! - **Statement**: Top-level constructs (facts, rules, constraints)
//! - **Atom**: Predicate applied to terms (e.g., `parent(john, mary)`)
//! - **Term**: Variables, constants, or compound terms
//! - **Value**: Constant values (integers, floats, booleans, strings, atoms)
//! - **Literal**: Positive or negative atoms
//!
//! # Syntax Examples
//!
//! - **Facts**: `parent(john, mary).`
//! - **Rules**: `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`
//! - **Queries**: `?- ancestor(X, mary).`
//! - **Constraints**: `:- unsafe(X).`
//! - **Negation**: `not reachable(X, Y)`

use internment::Intern;

/// Interned string for efficient storage and comparison
pub type Symbol = Intern<String>;

/// A Datalog program consists of facts, rules, and queries
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

/// Top-level statements in a Datalog program
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A ground fact: `parent(john, mary).`
    Fact(Fact),
    /// A rule with head and body: `ancestor(X, Y) :- parent(X, Y).`
    Rule(Rule),
    /// An integrity constraint: `:- unsafe(X).`
    Constraint(Constraint),
    /// A query: `?- ancestor(X, mary).`
    Query(Query),
}

/// A fact is simply an atom: `parent(john, mary).`
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    pub atom: Atom,
}

/// A rule has a head and a body: `ancestor(X, Y) :- parent(X, Y).`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Literal>,
}

/// A constraint has no head, only a body: `:- unsafe(X).`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub body: Vec<Literal>,
}

/// A query: `?- parent(X, mary).`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub body: Vec<Literal>,
}

/// A literal is either a positive or negative atom
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    /// Positive atom: `parent(X, Y)`
    Positive(Atom),
    /// Negated atom: `not parent(X, Y)`
    Negative(Atom),
    /// Comparison: `X > 5`, `X = Y`
    Comparison(ComparisonLiteral),
}

/// A comparison literal for built-in predicates
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonLiteral {
    pub left: Term,
    pub op: ComparisonOp,
    pub right: Term,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal,          // =, ==
    NotEqual,       // !=, <>
    LessThan,       // <
    LessOrEqual,    // <=
    GreaterThan,    // >
    GreaterOrEqual, // >=
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

impl Program {
    /// Create a new empty program
    pub fn new() -> Self {
        Program {
            statements: Vec::new(),
        }
    }

    /// Add a statement to the program
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }

    /// Get all facts from the program
    pub fn facts(&self) -> impl Iterator<Item = &Fact> {
        self.statements.iter().filter_map(|s| match s {
            Statement::Fact(f) => Some(f),
            _ => None,
        })
    }

    /// Get all rules from the program
    pub fn rules(&self) -> impl Iterator<Item = &Rule> {
        self.statements.iter().filter_map(|s| match s {
            Statement::Rule(r) => Some(r),
            _ => None,
        })
    }

    /// Get all constraints from the program
    pub fn constraints(&self) -> impl Iterator<Item = &Constraint> {
        self.statements.iter().filter_map(|s| match s {
            Statement::Constraint(c) => Some(c),
            _ => None,
        })
    }

    /// Get all queries from the program
    pub fn queries(&self) -> impl Iterator<Item = &Query> {
        self.statements.iter().filter_map(|s| match s {
            Statement::Query(q) => Some(q),
            _ => None,
        })
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Literal {
    /// Get the underlying atom from a literal (None for comparisons)
    pub fn atom(&self) -> Option<&Atom> {
        match self {
            Literal::Positive(atom) | Literal::Negative(atom) => Some(atom),
            Literal::Comparison(_) => None,
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
}

impl Atom {
    /// Create a new atom with the given predicate and terms
    pub fn new(predicate: impl Into<String>, terms: Vec<Term>) -> Self {
        Atom {
            predicate: Intern::new(predicate.into()),
            terms,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_fact() {
        let fact = Fact {
            atom: Atom::new(
                "parent",
                vec![
                    Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                    Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
                ],
            ),
        };
        assert_eq!(fact.atom.predicate.as_ref(), "parent");
        assert_eq!(fact.atom.terms.len(), 2);
    }

    #[test]
    fn test_create_rule() {
        let rule = Rule {
            head: Atom::new(
                "ancestor",
                vec![
                    Term::Variable(Intern::new("X".to_string())),
                    Term::Variable(Intern::new("Y".to_string())),
                ],
            ),
            body: vec![Literal::Positive(Atom::new(
                "parent",
                vec![
                    Term::Variable(Intern::new("X".to_string())),
                    Term::Variable(Intern::new("Y".to_string())),
                ],
            ))],
        };
        assert_eq!(rule.head.predicate.as_ref(), "ancestor");
        assert_eq!(rule.body.len(), 1);
    }

    #[test]
    fn test_term_is_ground() {
        let ground = Term::Constant(Value::Integer(42));
        let variable = Term::Variable(Intern::new("X".to_string()));

        assert!(ground.is_ground());
        assert!(!variable.is_ground());
    }
}
