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

use crate::operations::{insert as insert_row, OperationError};
use crate::runtime::Runtime;
use datalog_planner::{Atom, Symbol, Term, Value as DatalogValue};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use storage::{
    ColumnSchema, DataType, JsonValue, Row, StorageEngine, TableSchema, Value as SqlValue,
};

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
    /// Trigger aborted the operation
    TriggerAbort(String),
    /// Trigger depth exceeded (infinite recursion prevention)
    TriggerDepthExceeded { depth: u32, max_depth: u32 },
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
            InsertError::TriggerAbort(msg) => {
                write!(f, "trigger aborted: {}", msg)
            }
            InsertError::TriggerDepthExceeded { depth, max_depth } => {
                write!(
                    f,
                    "trigger depth exceeded: depth {} > max {}",
                    depth, max_depth
                )
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
    /// The insert was skipped by a BEFORE trigger
    Skipped,
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

    /// Query for rows matching a predicate
    ///
    /// All predicates (EDB and IDB) are queried from storage.
    /// Uses indexes for O(1) lookups when constants are present in the pattern.
    /// Returns raw rows - caller is responsible for unification.
    pub fn query<S: StorageEngine>(&self, pattern: &Atom, storage: &S) -> Vec<Row> {
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
    fn query_via_index<S: StorageEngine>(&self, pattern: &Atom, storage: &S) -> Option<Vec<Row>> {
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
                        // Still need to filter by other constants not covered by the index
                        return Some(self.filter_rows_by_constants(pattern, rows));
                    }
                }
            }
        }
        None // No usable index found
    }

    /// Query by scanning all rows in the table, filtering by constants in pattern
    fn query_via_scan<S: StorageEngine>(&self, pattern: &Atom, storage: &S) -> Vec<Row> {
        let table = pattern.predicate.as_ref();

        if let Ok(rows) = storage.scan(table) {
            return self.filter_rows_by_constants(pattern, rows);
        }
        vec![]
    }

    /// Filter rows to only those matching constants in the pattern
    fn filter_rows_by_constants(&self, pattern: &Atom, rows: Vec<Row>) -> Vec<Row> {
        rows.into_iter()
            .filter(|row| {
                for (i, term) in pattern.terms.iter().enumerate() {
                    if let Term::Constant(value) = term {
                        if i >= row.len() {
                            return false;
                        }
                        let expected = datalog_to_sql_value(value);
                        if row[i] != expected {
                            return false;
                        }
                    }
                }
                true
            })
            .collect()
    }

    /// Insert a ground fact into storage with trigger support
    ///
    /// Returns `Ok(InsertOutcome::Inserted)` if the fact was new,
    /// `Ok(InsertOutcome::Duplicate)` if it already existed,
    /// `Ok(InsertOutcome::Skipped)` if a BEFORE trigger skipped the insert.
    /// Uses UNIQUE constraints for O(1) deduplication.
    pub fn insert<S: StorageEngine, R: Runtime<S>>(
        &mut self,
        atom: Atom,
        storage: &mut S,
        runtime: &R,
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

        // Convert atom to row and insert into storage using trigger-aware insert
        let row = atom_to_row(&atom);
        match insert_row(storage, runtime, predicate_name, row) {
            Ok(true) => Ok(InsertOutcome::Inserted),
            Ok(false) => Ok(InsertOutcome::Skipped),
            Err(OperationError::Storage(storage::StorageError::ConstraintViolation(_))) => {
                Ok(InsertOutcome::Duplicate)
            }
            Err(OperationError::Storage(e)) => Err(InsertError::StorageError(format!("{:?}", e))),
            Err(OperationError::Runtime(e)) => Err(InsertError::StorageError(format!("{:?}", e))),
            Err(OperationError::TriggerAbort(msg)) => Err(InsertError::TriggerAbort(msg)),
            Err(OperationError::TriggerDepthExceeded { depth, max_depth }) => {
                Err(InsertError::TriggerDepthExceeded { depth, max_depth })
            }
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

    /// Count facts for a given predicate in storage
    pub fn count_facts<S: StorageEngine>(&self, predicate: &str, storage: &S) -> usize {
        if !self
            .storage_backed_predicates
            .contains(&Symbol::new(predicate.to_string()))
            && !self
                .derived_predicates
                .contains(&Symbol::new(predicate.to_string()))
        {
            return 0;
        }

        storage.scan(predicate).map(|rows| rows.len()).unwrap_or(0)
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
        // JSON values will be converted to compound terms in json_to_term
        // For now, fall back to text representation
        SqlValue::Json(j) => DatalogValue::Atom(Symbol::new(j.to_string())),
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

use storage::{StorageError, TableConstraint};

/// Create a schema for a derived predicate based on arity
///
/// All columns use Text type to accommodate Datalog atoms.
/// A UNIQUE constraint on all columns ensures deduplication (set semantics).
fn create_derived_schema(predicate: &str, arity: usize) -> TableSchema {
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
fn ensure_derived_table<S: StorageEngine>(
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
/// Compound terms are encoded as JSON for proper round-tripping.
fn atom_to_row(atom: &Atom) -> Row {
    atom.terms.iter().map(term_to_sql_value).collect()
}

/// Convert a SQL Value to a Datalog Term
///
/// JSON values are decoded back to compound terms.
/// Other values become constants.
pub fn sql_value_to_term(value: &SqlValue) -> Term {
    match value {
        SqlValue::Json(json) => json_to_term(json),
        other => Term::Constant(sql_to_datalog_value(other)),
    }
}

/// Convert a Datalog Term to a SQL Value
///
/// Compound terms are encoded as JSON.
/// Constants are converted to their SQL equivalents.
pub fn term_to_sql_value(term: &Term) -> SqlValue {
    match term {
        Term::Constant(value) => datalog_to_sql_value(value),
        Term::Variable(v) => panic!("Cannot convert variable {} to SQL value", v.as_ref()),
        Term::Compound(_, _) => SqlValue::Json(term_to_json(term)),
    }
}

/// Convert a Datalog Term to JsonValue
///
/// Encoding:
/// - Compound terms: `{"f": "functor", "a": [arg1, arg2, ...]}`
/// - Atoms: `{"t": "a", "v": "name"}`
/// - Integers: `{"t": "i", "v": 42}`
/// - Floats: `{"t": "f", "v": 3.14}`
/// - Booleans: `{"t": "b", "v": true}`
/// - Strings: `{"t": "s", "v": "text"}`
pub fn term_to_json(term: &Term) -> JsonValue {
    match term {
        Term::Compound(functor, args) => {
            let json_args: Vec<JsonValue> = args.iter().map(term_to_json).collect();
            JsonValue::Object(vec![
                ("f".to_string(), JsonValue::String(functor.as_ref().clone())),
                ("a".to_string(), JsonValue::Array(json_args)),
            ])
        }
        Term::Constant(value) => match value {
            DatalogValue::Atom(s) => JsonValue::Object(vec![
                ("t".to_string(), JsonValue::String("a".to_string())),
                ("v".to_string(), JsonValue::String(s.as_ref().clone())),
            ]),
            DatalogValue::Integer(n) => JsonValue::Object(vec![
                ("t".to_string(), JsonValue::String("i".to_string())),
                ("v".to_string(), JsonValue::Number(*n as f64)),
            ]),
            DatalogValue::Float(f) => JsonValue::Object(vec![
                ("t".to_string(), JsonValue::String("f".to_string())),
                ("v".to_string(), JsonValue::Number(*f)),
            ]),
            DatalogValue::Boolean(b) => JsonValue::Object(vec![
                ("t".to_string(), JsonValue::String("b".to_string())),
                ("v".to_string(), JsonValue::Bool(*b)),
            ]),
            DatalogValue::String(s) => JsonValue::Object(vec![
                ("t".to_string(), JsonValue::String("s".to_string())),
                ("v".to_string(), JsonValue::String(s.as_ref().clone())),
            ]),
        },
        Term::Variable(v) => panic!("Cannot convert variable {} to JSON", v.as_ref()),
    }
}

/// Convert JsonValue back to a Datalog Term
///
/// Decodes JSON according to the encoding in `term_to_json`.
pub fn json_to_term(json: &JsonValue) -> Term {
    match json {
        JsonValue::Object(pairs) => {
            // Check if it's a compound term (has "f" and "a" keys)
            let has_functor = pairs.iter().any(|(k, _)| k == "f");
            let has_args = pairs.iter().any(|(k, _)| k == "a");

            if has_functor && has_args {
                // Compound term
                let functor = pairs
                    .iter()
                    .find(|(k, _)| k == "f")
                    .and_then(|(_, v)| match v {
                        JsonValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .expect("compound term must have string functor");

                let args = pairs
                    .iter()
                    .find(|(k, _)| k == "a")
                    .and_then(|(_, v)| match v {
                        JsonValue::Array(arr) => Some(arr.iter().map(json_to_term).collect()),
                        _ => None,
                    })
                    .expect("compound term must have array arguments");

                Term::Compound(Symbol::new(functor), args)
            } else {
                // Constant value (has "t" and "v" keys)
                let type_tag = pairs
                    .iter()
                    .find(|(k, _)| k == "t")
                    .and_then(|(_, v)| match v {
                        JsonValue::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .expect("constant must have type tag");

                let value = pairs
                    .iter()
                    .find(|(k, _)| k == "v")
                    .map(|(_, v)| v)
                    .expect("constant must have value");

                match type_tag {
                    "a" => {
                        // Atom
                        if let JsonValue::String(s) = value {
                            Term::Constant(DatalogValue::Atom(Symbol::new(s.clone())))
                        } else {
                            panic!("atom value must be string")
                        }
                    }
                    "i" => {
                        // Integer
                        if let JsonValue::Number(n) = value {
                            Term::Constant(DatalogValue::Integer(*n as i64))
                        } else {
                            panic!("integer value must be number")
                        }
                    }
                    "f" => {
                        // Float
                        if let JsonValue::Number(n) = value {
                            Term::Constant(DatalogValue::Float(*n))
                        } else {
                            panic!("float value must be number")
                        }
                    }
                    "b" => {
                        // Boolean
                        if let JsonValue::Bool(b) = value {
                            Term::Constant(DatalogValue::Boolean(*b))
                        } else {
                            panic!("boolean value must be bool")
                        }
                    }
                    "s" => {
                        // String
                        if let JsonValue::String(s) = value {
                            Term::Constant(DatalogValue::String(Symbol::new(s.clone())))
                        } else {
                            panic!("string value must be string")
                        }
                    }
                    other => panic!("unknown type tag: {}", other),
                }
            }
        }
        _ => panic!("expected JSON object for term, got {:?}", json),
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::NoOpRuntime;
    use datalog_planner::Value;
    use internment::Intern;
    use storage::MemoryEngine;

    #[test]
    fn test_insert_ground_fact() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        let atom = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert!(db.insert(atom, &mut storage, &runtime).is_ok());
        assert!(db.is_derived(&Intern::new("parent".to_string())));
    }

    #[test]
    fn test_reject_non_ground_fact() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        let atom = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert!(db.insert(atom, &mut storage, &runtime).is_err());
    }

    #[test]
    fn test_query_with_variable() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            Atom {
                predicate: Intern::new("parent".to_string()),
                terms: vec![
                    Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                    Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
                ],
            },
            &mut storage,
            &runtime,
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
        use storage::{ColumnSchema, DataType};

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
        use storage::{ColumnSchema, DataType};

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
        let runtime = NoOpRuntime;
        let atom = Atom {
            predicate,
            terms: vec![Term::Constant(Value::Integer(1))],
        };

        let result = db.insert(atom, &mut storage, &runtime);
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

        let runtime = NoOpRuntime;

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
        assert!(db.insert(atom, &mut storage, &runtime).is_ok());
        assert!(db.is_derived(&Intern::new("unknown".to_string())));
    }

    #[test]
    fn test_insert_valid_arity() {
        use storage::{ColumnSchema, DataType};

        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
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

        assert!(db.insert(atom, &mut storage, &runtime).is_ok());
        assert!(db.is_derived(&predicate));
    }

    #[test]
    fn test_predicate_schema_from_table_schema() {
        use storage::{ColumnSchema, DataType, TableSchema};

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

        let runtime = NoOpRuntime;

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
            &runtime,
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
        use storage::MemoryEngine;
        use storage::{ColumnSchema, DataType, TableSchema};

        let mut storage = MemoryEngine::new();

        let _runtime = NoOpRuntime;

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
        use storage::MemoryEngine;
        use storage::{ColumnSchema, DataType, TableSchema};

        let mut storage = MemoryEngine::new();

        let _runtime = NoOpRuntime;

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

        // Verify the result - row is [id, name]
        // Name is the second column (index 1)
        assert_eq!(results[0][1], SqlValue::Text("person_42".to_string()));
    }

    #[test]
    fn test_query_falls_back_to_scan_without_index() {
        use storage::MemoryEngine;
        use storage::{ColumnSchema, DataType, TableSchema};

        let mut storage = MemoryEngine::new();

        let _runtime = NoOpRuntime;

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

    // ===== JSON Compound Term Serialization Tests =====

    #[test]
    fn test_term_to_json_atom() {
        let term = Term::Constant(Value::Atom(Intern::new("foo".to_string())));
        let json = term_to_json(&term);

        // Should be {"t": "a", "v": "foo"}
        assert_eq!(
            json.extract("$.t"),
            Some(&JsonValue::String("a".to_string()))
        );
        assert_eq!(
            json.extract("$.v"),
            Some(&JsonValue::String("foo".to_string()))
        );
    }

    #[test]
    fn test_term_to_json_integer() {
        let term = Term::Constant(Value::Integer(42));
        let json = term_to_json(&term);

        assert_eq!(
            json.extract("$.t"),
            Some(&JsonValue::String("i".to_string()))
        );
        assert_eq!(json.extract("$.v"), Some(&JsonValue::Number(42.0)));
    }

    #[test]
    fn test_term_to_json_compound() {
        // nest(value)
        let term = Term::Compound(
            Intern::new("nest".to_string()),
            vec![Term::Constant(Value::Atom(Intern::new(
                "value".to_string(),
            )))],
        );
        let json = term_to_json(&term);

        // Should be {"f": "nest", "a": [{"t": "a", "v": "value"}]}
        assert_eq!(
            json.extract("$.f"),
            Some(&JsonValue::String("nest".to_string()))
        );
        assert_eq!(
            json.extract("$.a[0].t"),
            Some(&JsonValue::String("a".to_string()))
        );
        assert_eq!(
            json.extract("$.a[0].v"),
            Some(&JsonValue::String("value".to_string()))
        );
    }

    #[test]
    fn test_term_to_json_nested_compound() {
        // nest(nest(value))
        let inner = Term::Compound(
            Intern::new("nest".to_string()),
            vec![Term::Constant(Value::Atom(Intern::new(
                "value".to_string(),
            )))],
        );
        let outer = Term::Compound(Intern::new("nest".to_string()), vec![inner]);
        let json = term_to_json(&outer);

        // Should be {"f": "nest", "a": [{"f": "nest", "a": [{"t": "a", "v": "value"}]}]}
        assert_eq!(
            json.extract("$.f"),
            Some(&JsonValue::String("nest".to_string()))
        );
        assert_eq!(
            json.extract("$.a[0].f"),
            Some(&JsonValue::String("nest".to_string()))
        );
        assert_eq!(
            json.extract("$.a[0].a[0].v"),
            Some(&JsonValue::String("value".to_string()))
        );
    }

    #[test]
    fn test_json_to_term_atom() {
        let json = JsonValue::Object(vec![
            ("t".to_string(), JsonValue::String("a".to_string())),
            ("v".to_string(), JsonValue::String("foo".to_string())),
        ]);
        let term = json_to_term(&json);

        assert_eq!(
            term,
            Term::Constant(Value::Atom(Intern::new("foo".to_string())))
        );
    }

    #[test]
    fn test_json_to_term_compound() {
        // {"f": "nest", "a": [{"t": "a", "v": "value"}]}
        let json = JsonValue::Object(vec![
            ("f".to_string(), JsonValue::String("nest".to_string())),
            (
                "a".to_string(),
                JsonValue::Array(vec![JsonValue::Object(vec![
                    ("t".to_string(), JsonValue::String("a".to_string())),
                    ("v".to_string(), JsonValue::String("value".to_string())),
                ])]),
            ),
        ]);
        let term = json_to_term(&json);

        let expected = Term::Compound(
            Intern::new("nest".to_string()),
            vec![Term::Constant(Value::Atom(Intern::new(
                "value".to_string(),
            )))],
        );
        assert_eq!(term, expected);
    }

    #[test]
    fn test_term_json_roundtrip() {
        // Test that term -> json -> term is identity
        let terms = vec![
            Term::Constant(Value::Atom(Intern::new("foo".to_string()))),
            Term::Constant(Value::Integer(42)),
            Term::Constant(Value::Float(3.14)),
            Term::Constant(Value::Boolean(true)),
            Term::Compound(
                Intern::new("nest".to_string()),
                vec![Term::Constant(Value::Atom(Intern::new(
                    "value".to_string(),
                )))],
            ),
            // Deeply nested: nest(nest(nest(value)))
            Term::Compound(
                Intern::new("nest".to_string()),
                vec![Term::Compound(
                    Intern::new("nest".to_string()),
                    vec![Term::Compound(
                        Intern::new("nest".to_string()),
                        vec![Term::Constant(Value::Atom(Intern::new(
                            "value".to_string(),
                        )))],
                    )],
                )],
            ),
            // Multiple args: pair(1, foo)
            Term::Compound(
                Intern::new("pair".to_string()),
                vec![
                    Term::Constant(Value::Integer(1)),
                    Term::Constant(Value::Atom(Intern::new("foo".to_string()))),
                ],
            ),
        ];

        for term in terms {
            let json = term_to_json(&term);
            let roundtripped = json_to_term(&json);
            assert_eq!(roundtripped, term, "Roundtrip failed for {:?}", term);
        }
    }

    #[test]
    fn test_compound_term_storage_roundtrip() {
        // Test that compound terms can be stored and retrieved via SQL
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        let runtime = NoOpRuntime;

        // Create a compound term: nest(value)
        let compound_term = Term::Compound(
            Intern::new("nest".to_string()),
            vec![Term::Constant(Value::Atom(Intern::new(
                "value".to_string(),
            )))],
        );

        let atom = Atom {
            predicate: Intern::new("test".to_string()),
            terms: vec![compound_term.clone()],
        };

        // Insert the fact
        db.insert(atom.clone(), &mut storage, &runtime).unwrap();

        // Query for it
        let pattern = Atom {
            predicate: Intern::new("test".to_string()),
            terms: vec![Term::Variable(Intern::new("X".to_string()))],
        };

        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 1);

        // The row should contain the compound term (as JSON)
        // Convert the row value back to a term to verify
        let bound = sql_value_to_term(&results[0][0]);
        assert_eq!(bound, compound_term);
    }

    #[test]
    fn test_deeply_nested_compound_storage() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        let runtime = NoOpRuntime;

        // Create deeply nested: nest(nest(nest(nest(value))))
        let mut term = Term::Constant(Value::Atom(Intern::new("value".to_string())));
        for _ in 0..4 {
            term = Term::Compound(Intern::new("nest".to_string()), vec![term]);
        }

        let atom = Atom {
            predicate: Intern::new("deep".to_string()),
            terms: vec![term.clone()],
        };

        db.insert(atom, &mut storage, &runtime).unwrap();

        let pattern = Atom {
            predicate: Intern::new("deep".to_string()),
            terms: vec![Term::Variable(Intern::new("X".to_string()))],
        };

        let results = db.query(&pattern, &storage);
        assert_eq!(results.len(), 1);

        // Convert the row value back to a term to verify
        let bound = sql_value_to_term(&results[0][0]);
        assert_eq!(bound, term);
    }
}
