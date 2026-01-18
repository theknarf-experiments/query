//! Datalog evaluation context
//!
//! This module provides `DatalogContext`, a bridge between SQL storage and Datalog
//! evaluation. It tracks metadata about predicates and provides unification-based
//! querying over SQL-stored facts.
//!
//! # Purpose
//!
//! `DatalogContext` serves as the evaluation context for Datalog programs:
//! - **Metadata tracking**: Schemas, EDB (base facts) vs IDB (derived facts)
//! - **Unification queries**: Pattern matching with variables over SQL rows
//! - **Auto table creation**: Creates derived tables with UNIQUE constraints
//!
//! All actual facts are stored in a `StorageEngine` (SQL storage). This struct
//! only holds metadata needed for correct evaluation.
//!
//! # Example
//!
//! ```ignore
//! let mut ctx = DatalogContext::new();
//! ctx.insert(ground_fact, &mut storage)?;
//! let results = ctx.query(&pattern_with_variables, &storage);
//! ```

use crate::datalog_unification::{unify_atoms, Substitution};
use crate::engine::{ColumnSchema, DataType, Row, StorageEngine, TableSchema};
use crate::Value as SqlValue;
use datalog_parser::{Atom, Symbol, Term, Value as DatalogValue};
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

/// Datalog evaluation context - bridges SQL storage with logic programming
///
/// Tracks metadata about predicates (schemas, EDB/IDB classification) and
/// provides unification-based querying over SQL-stored facts. All actual
/// facts are stored in the `StorageEngine`; this struct only holds metadata.
///
/// # EDB vs IDB
///
/// - **EDB (Extensional Database)**: Base facts from SQL tables
/// - **IDB (Intensional Database)**: Derived facts computed by Datalog rules
#[derive(Debug, Clone)]
pub struct DatalogContext {
    /// Schemas for predicates (maps predicate → column names/types)
    schemas: HashMap<Symbol, PredicateSchema>,
    /// EDB predicates backed by SQL tables
    storage_backed_predicates: HashSet<Symbol>,
    /// IDB predicates derived by rules
    derived_predicates: HashSet<Symbol>,
}

impl Default for DatalogContext {
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
    /// Storage operation failed
    StorageError(String),
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
            InsertError::StorageError(msg) => {
                write!(f, "storage error: {}", msg)
            }
        }
    }
}

impl Error for InsertError {}

/// Outcome of a successful insert operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    /// The fact was newly inserted (did not exist before)
    Inserted,
    /// The fact already existed (duplicate, no change)
    Duplicate,
}

impl InsertOutcome {
    /// Returns true if the fact was newly inserted
    pub fn is_new(&self) -> bool {
        matches!(self, InsertOutcome::Inserted)
    }

    /// Returns true if the fact was a duplicate
    pub fn is_duplicate(&self) -> bool {
        matches!(self, InsertOutcome::Duplicate)
    }
}

#[cfg(any(test, feature = "test-utils"))]
thread_local! {
    static TRACK_GROUND_QUERIES: Cell<bool> = const { Cell::new(false) };
    static GROUND_QUERY_COUNT: Cell<usize> = const { Cell::new(0) };
}

impl DatalogContext {
    /// Create a new empty context
    pub fn new() -> Self {
        DatalogContext {
            schemas: HashMap::new(),
            storage_backed_predicates: HashSet::new(),
            derived_predicates: HashSet::new(),
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

    /// Mark a predicate as backed by storage (SQL table - EDB base facts)
    pub fn mark_storage_backed(&mut self, predicate: Symbol) {
        self.storage_backed_predicates.insert(predicate);
    }

    /// Check if a predicate is backed by storage (EDB)
    pub fn is_storage_backed(&self, predicate: &Symbol) -> bool {
        self.storage_backed_predicates.contains(predicate)
    }

    /// Mark a predicate as derived (IDB - computed by rules)
    pub fn mark_derived(&mut self, predicate: Symbol) {
        self.derived_predicates.insert(predicate);
    }

    /// Check if a predicate is derived (IDB)
    pub fn is_derived(&self, predicate: &Symbol) -> bool {
        self.derived_predicates.contains(predicate)
    }

    /// Check if a predicate has storage (either EDB or IDB)
    pub fn has_storage(&self, predicate: &Symbol) -> bool {
        self.storage_backed_predicates.contains(predicate)
            || self.derived_predicates.contains(predicate)
    }

    /// Query for facts matching a pattern
    ///
    /// All predicates (EDB and IDB) are queried from storage.
    /// Uses indexes for O(1) lookups when constants are present.
    pub fn query<S: StorageEngine>(&self, pattern: &Atom, storage: &S) -> Vec<Substitution> {
        #[cfg(any(test, feature = "test-utils"))]
        TRACK_GROUND_QUERIES.with(|flag| {
            if flag.get() && is_ground(pattern) {
                GROUND_QUERY_COUNT.with(|count| count.set(count.get() + 1));
            }
        });

        // Check if this predicate has storage (EDB or IDB)
        if self.has_storage(&pattern.predicate) {
            // Try indexed lookup if pattern has constants
            if let Some(indexed_results) = self.query_via_index(pattern, storage) {
                return indexed_results;
            }
            // Fall back to table scan
            return self.query_via_scan(pattern, storage);
        }

        // Predicate not in storage - return empty
        vec![]
    }

    /// Try to query using an index if a constant column has one
    fn query_via_index<S: StorageEngine>(
        &self,
        pattern: &Atom,
        storage: &S,
    ) -> Option<Vec<Substitution>> {
        let table = pattern.predicate.as_ref();
        let schema = self.schemas.get(&pattern.predicate)?;

        // Find first constant in pattern that has an index
        for (i, term) in pattern.terms.iter().enumerate() {
            if let Term::Constant(value) = term {
                let column = schema.column_name(i)?;
                let sql_value = datalog_to_sql_value(value);

                if storage.has_index(table, column) {
                    if let Some(indices) = storage.index_lookup(table, column, &sql_value) {
                        let rows = storage.get_rows_by_indices(table, &indices).ok()?;
                        return Some(self.unify_rows_with_pattern(pattern, &rows));
                    }
                }
            }
        }
        None // No usable index found
    }

    /// Query by scanning all rows in the table
    fn query_via_scan<S: StorageEngine>(&self, pattern: &Atom, storage: &S) -> Vec<Substitution> {
        let table = pattern.predicate.as_ref();

        if let Ok(rows) = storage.scan(table) {
            return self.unify_rows_with_pattern(pattern, &rows);
        }
        vec![]
    }

    /// Convert rows to substitutions by unifying with the pattern
    fn unify_rows_with_pattern(&self, pattern: &Atom, rows: &[Row]) -> Vec<Substitution> {
        let mut results = Vec::new();
        for row in rows {
            // Convert row to atom
            let terms: Vec<Term> = row
                .iter()
                .map(|v| Term::Constant(sql_to_datalog_value(v)))
                .collect();
            let fact = Atom {
                predicate: pattern.predicate,
                terms,
            };

            // Try to unify
            let mut subst = Substitution::new();
            if unify_atoms(pattern, &fact, &mut subst) {
                results.push(subst);
            }
        }
        results
    }

    /// Insert a ground fact into storage
    ///
    /// Returns `Ok(InsertOutcome::Inserted)` if the fact was new,
    /// `Ok(InsertOutcome::Duplicate)` if it already existed.
    /// Uses UNIQUE constraints for O(1) deduplication.
    pub fn insert<S: StorageEngine>(
        &mut self,
        atom: Atom,
        storage: &mut S,
    ) -> Result<InsertOutcome, InsertError> {
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

        let predicate_name = atom.predicate.as_ref();
        let arity = atom.terms.len();

        // Ensure the derived table exists (creates with UNIQUE constraint)
        if !self.is_storage_backed(&atom.predicate) {
            ensure_derived_table(storage, predicate_name, arity)
                .map_err(|e| InsertError::StorageError(format!("{:?}", e)))?;
            self.mark_derived(atom.predicate);
        }

        // Convert atom to row and insert into storage
        let row = atom_to_row(&atom);
        match storage.insert(predicate_name, row) {
            Ok(()) => Ok(InsertOutcome::Inserted),
            Err(crate::engine::StorageError::ConstraintViolation(_)) => {
                Ok(InsertOutcome::Duplicate)
            }
            Err(e) => Err(InsertError::StorageError(format!("{:?}", e))),
        }
    }

    /// Check if a fact exists in storage
    ///
    /// Uses composite index lookup for O(1) when available, falls back to scan.
    pub fn contains<S: StorageEngine>(&self, atom: &Atom, storage: &S) -> bool {
        if !is_ground(atom) {
            // Non-ground atoms: check if any matching facts exist
            return !self.query(atom, storage).is_empty();
        }

        // Ground atom: check storage
        let predicate_name = atom.predicate.as_ref();

        // Check if predicate has storage
        if !self.has_storage(&atom.predicate) {
            return false;
        }

        // Get column names - use schema for storage-backed, col0/col1/... for derived
        let columns: Vec<String> = if let Some(schema) = self.schemas.get(&atom.predicate) {
            (0..schema.arity())
                .filter_map(|i| schema.column_name(i).map(|s| s.to_string()))
                .collect()
        } else {
            (0..atom.terms.len()).map(|i| format!("col{}", i)).collect()
        };
        let row = atom_to_row(atom);

        // Try O(1) index lookup first
        if let Some(indices) = storage.composite_index_lookup(predicate_name, &columns, &row) {
            return !indices.is_empty();
        }

        // Fall back to scan-based check
        if let Ok(rows) = storage.scan(predicate_name) {
            return rows.contains(&row);
        }

        false
    }

    /// Count total number of predicates with storage
    pub fn predicate_count(&self) -> usize {
        self.storage_backed_predicates.len() + self.derived_predicates.len()
    }

    /// Check if no predicates are registered
    pub fn is_empty(&self) -> bool {
        self.predicate_count() == 0
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

/// Convert SQL Value to Datalog Value
///
/// Note: SQL Text values are converted to Datalog Atoms so they can unify
/// with lowercase identifiers in Datalog queries. For example, SQL 'john'
/// becomes Datalog atom `john`, which matches `?- parent(john, X).`
fn sql_to_datalog_value(value: &SqlValue) -> DatalogValue {
    match value {
        SqlValue::Null => DatalogValue::Atom(Symbol::new("null".to_string())),
        SqlValue::Bool(b) => DatalogValue::Boolean(*b),
        SqlValue::Int(i) => DatalogValue::Integer(*i),
        SqlValue::Float(f) => DatalogValue::Float(*f),
        // Use Atom instead of String so it unifies with lowercase identifiers
        SqlValue::Text(s) => DatalogValue::Atom(Symbol::new(s.clone())),
        SqlValue::Date(d) => DatalogValue::Atom(Symbol::new(format!("{}", d))),
        SqlValue::Time(t) => DatalogValue::Atom(Symbol::new(format!("{}", t))),
        SqlValue::Timestamp(ts) => DatalogValue::Atom(Symbol::new(format!("{}", ts))),
    }
}

/// Convert Datalog Value to SQL Value
fn datalog_to_sql_value(value: &DatalogValue) -> SqlValue {
    match value {
        DatalogValue::Integer(i) => SqlValue::Int(*i),
        DatalogValue::Float(f) => SqlValue::Float(*f),
        DatalogValue::Boolean(b) => SqlValue::Bool(*b),
        DatalogValue::String(s) => SqlValue::Text(s.as_ref().clone()),
        DatalogValue::Atom(s) => {
            let name = s.as_ref();
            if name == "null" {
                SqlValue::Null
            } else {
                SqlValue::Text(name.clone())
            }
        }
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

// ============================================================================
// Derived Predicate Storage Support
// ============================================================================
//
// These functions enable storing derived Datalog facts (IDB) in the storage
// engine alongside SQL tables (EDB), creating a unified storage layer.

use crate::engine::{StorageError, TableConstraint};

/// Create a schema for a derived predicate based on arity
///
/// All columns use Text type to accommodate Datalog atoms.
/// A UNIQUE constraint on all columns ensures deduplication (set semantics).
pub fn create_derived_schema(predicate: &str, arity: usize) -> TableSchema {
    TableSchema {
        name: predicate.to_string(),
        columns: (0..arity)
            .map(|i| ColumnSchema {
                name: format!("col{}", i),
                data_type: DataType::Text,
                nullable: true,
                primary_key: false,
                unique: false,
                default: None,
                references: None,
            })
            .collect(),
        // UNIQUE constraint on all columns for deduplication
        constraints: vec![TableConstraint::Unique {
            columns: (0..arity).map(|i| format!("col{}", i)).collect(),
        }],
    }
}

/// Ensure a table exists for a derived predicate
///
/// Creates the table if it doesn't exist. Uses the arity to determine schema.
/// Returns Ok(true) if table was created, Ok(false) if it already existed.
pub fn ensure_derived_table<S: StorageEngine>(
    storage: &mut S,
    predicate: &str,
    arity: usize,
) -> Result<bool, StorageError> {
    // Check if table exists by trying to get schema
    if storage.get_schema(predicate).is_ok() {
        return Ok(false); // Already exists
    }

    let schema = create_derived_schema(predicate, arity);
    storage.create_table(schema)?;
    Ok(true) // Created
}

/// Convert a Datalog fact (Atom) to a SQL row for storage
///
/// Panics if the atom contains variables (must be ground).
pub fn atom_to_row(atom: &Atom) -> Row {
    atom.terms
        .iter()
        .map(|term| match term {
            Term::Constant(value) => datalog_to_sql_value(value),
            Term::Variable(v) => panic!("Cannot convert variable {} to SQL value", v.as_ref()),
            Term::Compound(_, _) => {
                // Serialize compound terms as text
                SqlValue::Text(format!("{:?}", term))
            }
        })
        .collect()
}

/// Convert a SQL row to a Datalog fact for a given predicate
pub fn row_to_atom(predicate: &str, row: &Row) -> Atom {
    Atom {
        predicate: Symbol::new(predicate.to_string()),
        terms: row
            .iter()
            .map(|v| Term::Constant(sql_to_datalog_value(v)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryEngine;
    use datalog_parser::Value;
    use internment::Intern;

    #[test]
    fn test_insert_ground_fact() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let atom = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert!(db.insert(atom, &mut storage).is_ok());
        assert!(db.is_derived(&Intern::new("parent".to_string())));
    }

    #[test]
    fn test_reject_non_ground_fact() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let atom = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert!(db.insert(atom, &mut storage).is_err());
    }

    #[test]
    fn test_query_with_variable() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            Atom {
                predicate: Intern::new("parent".to_string()),
                terms: vec![
                    Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                    Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
                ],
            },
            &mut storage,
        )
        .unwrap();

        let pattern = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };

        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 1);
    }

    // ===== Schema Support Tests =====

    #[test]
    fn test_register_and_get_schema() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db = DatalogContext::new();
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

        assert!(db.get_schema(&predicate).is_some());
        let retrieved = db.get_schema(&predicate).unwrap();
        assert_eq!(retrieved.arity(), 2);
        assert_eq!(retrieved.column_name(0), Some("id"));
        assert_eq!(retrieved.column_name(1), Some("name"));
    }

    #[test]
    fn test_insert_validates_arity_when_schema_exists() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db = DatalogContext::new();
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
        let mut storage = MemoryEngine::new();
        let atom = Atom {
            predicate,
            terms: vec![Term::Constant(Value::Integer(1))],
        };

        let result = db.insert(atom, &mut storage);
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
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

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
        assert!(db.insert(atom, &mut storage).is_ok());
        assert!(db.is_derived(&Intern::new("unknown".to_string())));
    }

    #[test]
    fn test_insert_valid_arity() {
        use crate::engine::{ColumnSchema, DataType};

        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
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

        assert!(db.insert(atom, &mut storage).is_ok());
        assert!(db.is_derived(&predicate));
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

    // ===== Storage-Backed Predicate Tests =====

    #[test]
    fn test_mark_storage_backed() {
        let mut db = DatalogContext::new();
        let pred = Intern::new("parent".to_string());

        assert!(!db.is_storage_backed(&pred));

        db.mark_storage_backed(pred);

        assert!(db.is_storage_backed(&pred));
    }

    #[test]
    fn test_query_derived_facts() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Insert a derived fact
        let pred = Intern::new("derived_fact".to_string());
        db.insert(
            Atom {
                predicate: pred,
                terms: vec![
                    Term::Constant(Value::Atom(Intern::new("a".to_string()))),
                    Term::Constant(Value::Atom(Intern::new("b".to_string()))),
                ],
            },
            &mut storage,
        )
        .unwrap();

        // Query with a variable
        let pattern = Atom {
            predicate: pred,
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Constant(Value::Atom(Intern::new("b".to_string()))),
            ],
        };

        // Should find the derived fact from storage
        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_storage_backed_predicate() {
        use crate::engine::{ColumnSchema, DataType, TableSchema};
        use crate::MemoryEngine;

        let mut storage = MemoryEngine::new();

        // Create a table in storage
        let schema = TableSchema {
            name: "parent".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "parent_name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "child_name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        };
        storage.create_table(schema).unwrap();

        // Insert data into storage
        storage
            .insert(
                "parent",
                vec![
                    SqlValue::Text("john".to_string()),
                    SqlValue::Text("mary".to_string()),
                ],
            )
            .unwrap();
        storage
            .insert(
                "parent",
                vec![
                    SqlValue::Text("john".to_string()),
                    SqlValue::Text("bob".to_string()),
                ],
            )
            .unwrap();

        // Set up DatalogContext with storage-backed predicate
        let mut db = DatalogContext::new();
        let pred = Intern::new("parent".to_string());
        db.register_schema(PredicateSchema::from_table_schema(
            storage.get_schema("parent").unwrap(),
        ));
        db.mark_storage_backed(pred);

        // Query for john's children
        let pattern = Atom {
            predicate: pred,
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Variable(Intern::new("X".to_string())),
            ],
        };

        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_uses_index_when_available() {
        use crate::engine::{ColumnSchema, DataType, TableSchema};
        use crate::MemoryEngine;

        let mut storage = MemoryEngine::new();

        // Create a table
        let schema = TableSchema {
            name: "person".to_string(),
            columns: vec![
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
            constraints: vec![],
        };
        storage.create_table(schema).unwrap();

        // Insert data
        for i in 0..100 {
            storage
                .insert(
                    "person",
                    vec![SqlValue::Int(i), SqlValue::Text(format!("person_{}", i))],
                )
                .unwrap();
        }

        // Create an index on the id column
        storage
            .create_index("person", "id", "idx_person_id")
            .unwrap();

        // Set up DatalogContext
        let mut db = DatalogContext::new();
        let pred = Intern::new("person".to_string());
        db.register_schema(PredicateSchema::from_table_schema(
            storage.get_schema("person").unwrap(),
        ));
        db.mark_storage_backed(pred);

        // Query for a specific id (should use the index)
        let pattern = Atom {
            predicate: pred,
            terms: vec![
                Term::Constant(Value::Integer(42)),
                Term::Variable(Intern::new("Name".to_string())),
            ],
        };

        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 1);

        // Verify the result
        let name = results[0].get(&Intern::new("Name".to_string())).unwrap();
        assert_eq!(
            name,
            &Term::Constant(Value::Atom(Intern::new("person_42".to_string())))
        );
    }

    #[test]
    fn test_query_falls_back_to_scan_without_index() {
        use crate::engine::{ColumnSchema, DataType, TableSchema};
        use crate::MemoryEngine;

        let mut storage = MemoryEngine::new();

        // Create a table without any indexes
        let schema = TableSchema {
            name: "data".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "key".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "value".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        };
        storage.create_table(schema).unwrap();

        // Insert data
        storage
            .insert(
                "data",
                vec![SqlValue::Text("a".to_string()), SqlValue::Int(1)],
            )
            .unwrap();
        storage
            .insert(
                "data",
                vec![SqlValue::Text("b".to_string()), SqlValue::Int(2)],
            )
            .unwrap();

        // Set up DatalogContext
        let mut db = DatalogContext::new();
        let pred = Intern::new("data".to_string());
        db.register_schema(PredicateSchema::from_table_schema(
            storage.get_schema("data").unwrap(),
        ));
        db.mark_storage_backed(pred);

        // Query with constant (no index available, falls back to scan)
        let pattern = Atom {
            predicate: pred,
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("a".to_string()))),
                Term::Variable(Intern::new("V".to_string())),
            ],
        };

        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_value_conversion_roundtrip() {
        // Test that values convert correctly between SQL and Datalog
        let test_cases = vec![
            (SqlValue::Int(42), Value::Integer(42)),
            (SqlValue::Float(3.14), Value::Float(3.14)),
            (SqlValue::Bool(true), Value::Boolean(true)),
            (
                SqlValue::Text("hello".to_string()),
                Value::Atom(Intern::new("hello".to_string())),
            ),
        ];

        for (sql_val, expected_datalog) in test_cases {
            let converted = sql_to_datalog_value(&sql_val);
            assert_eq!(converted, expected_datalog);
        }
    }
}
