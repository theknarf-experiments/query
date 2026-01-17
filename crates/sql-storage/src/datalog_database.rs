//! Fact database with efficient indexing and querying
//!
//! This module provides a database for storing and querying ground facts.
//! Facts are indexed by predicate name for efficient lookup.
//!
//! # Features
//!
//! - **Indexing**: Facts are indexed by predicate for O(1) lookup
//! - **Ground facts only**: Only fully ground atoms (no variables) can be stored
//! - **Unification-based querying**: Query with patterns containing variables
//! - **Set semantics**: Duplicate facts are automatically deduplicated
//! - **Schema support**: Predicates can have associated schemas for validation
//!
//! # Example
//!
//! ```ignore
//! let mut db = FactDatabase::new();
//! db.insert(ground_fact).unwrap();
//! let results = db.query(&pattern_with_variables);
//! ```

use crate::datalog_unification::{unify_atoms, Substitution};
use crate::engine::{ColumnSchema, DataType, TableSchema};
use datalog_parser::{Atom, Symbol, Term};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Schema for a Datalog predicate (maps to SQL table schema)
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateSchema {
    /// The predicate name
    pub name: Symbol,
    /// Column schemas in positional order
    pub columns: Vec<ColumnSchema>,
}

impl PredicateSchema {
    /// Create a new predicate schema
    pub fn new(name: Symbol, columns: Vec<ColumnSchema>) -> Self {
        Self { name, columns }
    }

    /// Create from a SQL TableSchema
    pub fn from_table_schema(schema: &TableSchema) -> Self {
        Self {
            name: Symbol::new(schema.name.clone()),
            columns: schema.columns.clone(),
        }
    }

    /// Get the arity (number of columns/arguments)
    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    /// Get column name at position
    pub fn column_name(&self, index: usize) -> Option<&str> {
        self.columns.get(index).map(|c| c.name.as_str())
    }

    /// Get column type at position
    pub fn column_type(&self, index: usize) -> Option<&DataType> {
        self.columns.get(index).map(|c| &c.data_type)
    }
}

#[cfg(any(test, feature = "test-utils"))]
use std::cell::Cell;

/// A database of ground facts with efficient indexing
#[derive(Debug, Clone)]
pub struct FactDatabase {
    /// Index: predicate -> set of ground atoms
    facts_by_predicate: HashMap<Symbol, HashSet<Atom>>,
    /// Optional schemas for predicates (for SQL table interop)
    schemas: HashMap<Symbol, PredicateSchema>,
}

impl Default for FactDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for failed fact insertions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// Attempted to insert an atom containing variables
    NonGroundAtom(Atom),
    /// Atom has wrong number of arguments for its registered schema
    ArityMismatch {
        predicate: String,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for InsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InsertError::NonGroundAtom(atom) => {
                write!(f, "cannot insert non-ground atom: {:?}", atom)
            }
            InsertError::ArityMismatch {
                predicate,
                expected,
                found,
            } => {
                write!(
                    f,
                    "arity mismatch for {}: expected {} arguments, found {}",
                    predicate, expected, found
                )
            }
        }
    }
}

impl Error for InsertError {}

#[cfg(any(test, feature = "test-utils"))]
thread_local! {
    static TRACK_GROUND_QUERIES: Cell<bool> = const { Cell::new(false) };
    static GROUND_QUERY_COUNT: Cell<usize> = const { Cell::new(0) };
}

impl FactDatabase {
    pub fn new() -> Self {
        FactDatabase {
            facts_by_predicate: HashMap::new(),
            schemas: HashMap::new(),
        }
    }

    /// Register a schema for a predicate
    pub fn register_schema(&mut self, schema: PredicateSchema) {
        self.schemas.insert(schema.name, schema);
    }

    /// Get schema for a predicate (if registered)
    pub fn get_schema(&self, predicate: &Symbol) -> Option<&PredicateSchema> {
        self.schemas.get(predicate)
    }

    /// Check if predicate has a registered schema
    pub fn has_schema(&self, predicate: &Symbol) -> bool {
        self.schemas.contains_key(predicate)
    }

    /// Insert a ground fact into the database
    pub fn insert(&mut self, atom: Atom) -> Result<bool, InsertError> {
        // Check that atom is ground (no variables)
        if !is_ground(&atom) {
            return Err(InsertError::NonGroundAtom(atom));
        }

        // Validate arity if schema is registered (permissive: skip if no schema)
        if let Some(schema) = self.schemas.get(&atom.predicate) {
            if atom.terms.len() != schema.arity() {
                return Err(InsertError::ArityMismatch {
                    predicate: atom.predicate.to_string(),
                    expected: schema.arity(),
                    found: atom.terms.len(),
                });
            }
        }

        Ok(self
            .facts_by_predicate
            .entry(atom.predicate)
            .or_default()
            .insert(atom))
    }

    /// Merge another fact database into this one, consuming the other database.
    pub fn absorb(&mut self, mut other: FactDatabase) {
        for (predicate, mut facts) in other.facts_by_predicate.drain() {
            self.facts_by_predicate
                .entry(predicate)
                .or_default()
                .extend(facts.drain());
        }
        // Merge schemas (other's schemas take precedence for conflicts)
        self.schemas.extend(other.schemas);
    }

    /// Check if a fact exists in the database
    pub fn contains(&self, atom: &Atom) -> bool {
        if let Some(facts) = self.facts_by_predicate.get(&atom.predicate) {
            facts.contains(atom)
        } else {
            false
        }
    }

    /// Query for facts matching a pattern (may contain variables)
    /// Returns all substitutions that make the pattern match facts in the database
    pub fn query(&self, pattern: &Atom) -> Vec<Substitution> {
        #[cfg(any(test, feature = "test-utils"))]
        TRACK_GROUND_QUERIES.with(|flag| {
            if flag.get() && is_ground(pattern) {
                GROUND_QUERY_COUNT.with(|count| count.set(count.get() + 1));
            }
        });

        let mut results = Vec::new();

        // Get all facts with the same predicate
        if let Some(facts) = self.facts_by_predicate.get(&pattern.predicate) {
            for fact in facts {
                let mut subst = Substitution::new();
                if unify_atoms(pattern, fact, &mut subst) {
                    results.push(subst);
                }
            }
        }

        results
    }

    /// Get all facts with a specific predicate
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn get_by_predicate(&self, predicate: &Symbol) -> Vec<&Atom> {
        if let Some(facts) = self.facts_by_predicate.get(predicate) {
            facts.iter().collect()
        } else {
            vec![]
        }
    }

    /// Get all facts in the database
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn all_facts(&self) -> Vec<&Atom> {
        self.facts_by_predicate
            .values()
            .flat_map(|facts| facts.iter())
            .collect()
    }

    /// Count total number of facts
    pub fn len(&self) -> usize {
        self.facts_by_predicate
            .values()
            .map(|facts| facts.len())
            .sum()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn track_ground_queries() -> GroundQueryTracker {
        GroundQueryTracker::new()
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub struct GroundQueryTracker {
    _private: (),
}

#[cfg(any(test, feature = "test-utils"))]
impl GroundQueryTracker {
    fn new() -> Self {
        TRACK_GROUND_QUERIES.with(|flag| flag.set(true));
        GROUND_QUERY_COUNT.with(|count| count.set(0));
        GroundQueryTracker { _private: () }
    }

    pub fn count(&self) -> usize {
        GROUND_QUERY_COUNT.with(|count| count.get())
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl Drop for GroundQueryTracker {
    fn drop(&mut self) {
        TRACK_GROUND_QUERIES.with(|flag| flag.set(false));
    }
}

/// Check if an atom is ground (contains no variables)
fn is_ground(atom: &Atom) -> bool {
    atom.terms.iter().all(is_ground_term)
}

/// Check if a term is ground (contains no variables)
fn is_ground_term(term: &Term) -> bool {
    match term {
        Term::Variable(_) => false,
        Term::Constant(_) => true,
        Term::Compound(_, args) => args.iter().all(is_ground_term),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::Value;
    use internment::Intern;

    #[test]
    fn test_insert_ground_fact() {
        let mut db = FactDatabase::new();
        let atom = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert!(db.insert(atom).is_ok());
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_reject_non_ground_fact() {
        let mut db = FactDatabase::new();
        let atom = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert!(db.insert(atom).is_err());
    }

    #[test]
    fn test_query_with_variable() {
        let mut db = FactDatabase::new();
        db.insert(Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        })
        .unwrap();

        let pattern = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };

        let results = db.query(&pattern);
        assert_eq!(results.len(), 1);
    }

    // ===== Schema Support Tests =====

    #[test]
    fn test_register_and_get_schema() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db = FactDatabase::new();
        let predicate = Intern::new("person".to_string());

        let schema = PredicateSchema::new(
            predicate,
            vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
        );

        db.register_schema(schema);

        assert!(db.has_schema(&predicate));
        let retrieved = db.get_schema(&predicate).unwrap();
        assert_eq!(retrieved.arity(), 2);
        assert_eq!(retrieved.column_name(0), Some("id"));
        assert_eq!(retrieved.column_name(1), Some("name"));
    }

    #[test]
    fn test_insert_validates_arity_when_schema_exists() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db = FactDatabase::new();
        let predicate = Intern::new("person".to_string());

        // Register a 2-column schema
        let schema = PredicateSchema::new(
            predicate,
            vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
        );
        db.register_schema(schema);

        // Try to insert atom with wrong arity (1 instead of 2)
        let atom = Atom {
            predicate,
            terms: vec![Term::Constant(Value::Integer(1))],
        };

        let result = db.insert(atom);
        assert!(matches!(
            result,
            Err(InsertError::ArityMismatch {
                expected: 2,
                found: 1,
                ..
            })
        ));
    }

    #[test]
    fn test_insert_allows_unregistered_predicate() {
        let mut db = FactDatabase::new();

        // No schema registered for "unknown"
        let atom = Atom {
            predicate: Intern::new("unknown".to_string()),
            terms: vec![
                Term::Constant(Value::Integer(1)),
                Term::Constant(Value::Integer(2)),
                Term::Constant(Value::Integer(3)),
            ],
        };

        // Should succeed - permissive mode allows unregistered predicates
        assert!(db.insert(atom).is_ok());
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_insert_valid_arity() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db = FactDatabase::new();
        let predicate = Intern::new("person".to_string());

        // Register a 2-column schema
        let schema = PredicateSchema::new(
            predicate,
            vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
        );
        db.register_schema(schema);

        // Insert atom with correct arity
        let atom = Atom {
            predicate,
            terms: vec![
                Term::Constant(Value::Integer(1)),
                Term::Constant(Value::Atom(Intern::new("alice".to_string()))),
            ],
        };

        assert!(db.insert(atom).is_ok());
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_absorb_merges_schemas() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db1 = FactDatabase::new();
        let mut db2 = FactDatabase::new();

        let pred1 = Intern::new("table1".to_string());
        let pred2 = Intern::new("table2".to_string());

        // Register schema in db1
        db1.register_schema(PredicateSchema::new(
            pred1,
            vec![ColumnSchema {
                name: "col".to_string(),
                data_type: DataType::Int,
                nullable: false,
                primary_key: false,
                unique: false,
                default: None,
                references: None,
            }],
        ));

        // Register different schema in db2
        db2.register_schema(PredicateSchema::new(
            pred2,
            vec![ColumnSchema {
                name: "val".to_string(),
                data_type: DataType::Text,
                nullable: false,
                primary_key: false,
                unique: false,
                default: None,
                references: None,
            }],
        ));

        // Absorb db2 into db1
        db1.absorb(db2);

        // Both schemas should be present
        assert!(db1.has_schema(&pred1));
        assert!(db1.has_schema(&pred2));
    }

    #[test]
    fn test_predicate_schema_from_table_schema() {
        use crate::engine::{ColumnSchema, DataType, TableSchema};

        let table_schema = TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "email".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: true,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        };

        let pred_schema = PredicateSchema::from_table_schema(&table_schema);

        assert_eq!(pred_schema.name.as_ref(), "users");
        assert_eq!(pred_schema.arity(), 2);
        assert_eq!(pred_schema.column_name(0), Some("id"));
        assert_eq!(pred_schema.column_type(0), Some(&DataType::Int));
        assert_eq!(pred_schema.column_name(1), Some("email"));
        assert_eq!(pred_schema.column_type(1), Some(&DataType::Text));
    }
}
