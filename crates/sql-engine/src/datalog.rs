//! Datalog query support for the SQL engine
//!
//! This module allows executing Datalog queries against SQL tables.
//! SQL tables are converted to Datalog facts, queries are evaluated,
//! and results are converted back to SQL format.
//!
//! # Example
//!
//! ```ignore
//! let engine = Engine::new();
//! engine.execute("CREATE TABLE parent (parent TEXT, child TEXT)");
//! engine.execute("INSERT INTO parent VALUES ('john', 'mary'), ('mary', 'jane')");
//!
//! // Execute Datalog query to find all ancestors
//! let result = engine.execute_datalog(r#"
//!     ancestor(X, Y) :- parent(X, Y).
//!     ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
//!     ?- ancestor(X, Y).
//! "#)?;
//! ```

use datalog_eval::{evaluate, EvaluationError};
use datalog_parser::{
    Atom, Constraint, Literal, Query, Rule, SrcId, Symbol, Term, Value as DValue,
};
use sql_storage::FactDatabase;
use sql_storage::{StorageEngine, Value as SValue};

use crate::{ExecError, QueryResult};

/// Errors specific to Datalog execution
#[derive(Debug, Clone)]
pub enum DatalogError {
    /// Parse error in Datalog program
    ParseError(String),
    /// Evaluation error
    EvaluationError(String),
    /// No query found in program
    NoQuery,
    /// Table not found
    TableNotFound(String),
}

impl std::fmt::Display for DatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatalogError::ParseError(msg) => write!(f, "Datalog parse error: {}", msg),
            DatalogError::EvaluationError(msg) => write!(f, "Datalog evaluation error: {}", msg),
            DatalogError::NoQuery => write!(f, "No query found in Datalog program"),
            DatalogError::TableNotFound(name) => write!(f, "Table not found: {}", name),
        }
    }
}

impl From<DatalogError> for ExecError {
    fn from(err: DatalogError) -> Self {
        ExecError::InvalidExpression(err.to_string())
    }
}

impl From<EvaluationError> for DatalogError {
    fn from(err: EvaluationError) -> Self {
        DatalogError::EvaluationError(err.to_string())
    }
}

/// Convert SQL Value to Datalog Value
///
/// Note: SQL Text values are converted to Datalog Atoms so they can unify
/// with lowercase identifiers in Datalog queries. For example, SQL 'john'
/// becomes Datalog atom `john`, which matches `?- parent(john, X).`
fn sql_to_datalog_value(value: &SValue) -> DValue {
    match value {
        SValue::Null => DValue::Atom(Symbol::new("null".to_string())),
        SValue::Bool(b) => DValue::Boolean(*b),
        SValue::Int(i) => DValue::Integer(*i),
        SValue::Float(f) => DValue::Float(*f),
        // Use Atom instead of String so it unifies with lowercase identifiers
        SValue::Text(s) => DValue::Atom(Symbol::new(s.clone())),
        SValue::Date(d) => DValue::Atom(Symbol::new(format!("{}", d))),
        SValue::Time(t) => DValue::Atom(Symbol::new(format!("{}", t))),
        SValue::Timestamp(ts) => DValue::Atom(Symbol::new(format!("{}", ts))),
    }
}

/// Convert Datalog Value to SQL Value
fn datalog_to_sql_value(value: &DValue) -> SValue {
    match value {
        DValue::Integer(i) => SValue::Int(*i),
        DValue::Float(f) => SValue::Float(*f),
        DValue::Boolean(b) => SValue::Bool(*b),
        DValue::String(s) => SValue::Text(s.as_ref().clone()),
        DValue::Atom(s) => {
            let name = s.as_ref();
            if name == "null" {
                SValue::Null
            } else {
                SValue::Text(name.clone())
            }
        }
    }
}

/// Convert Datalog Term to SQL Value (for ground terms only)
fn term_to_sql_value(term: &Term) -> Option<SValue> {
    match term {
        Term::Constant(v) => Some(datalog_to_sql_value(v)),
        Term::Variable(_) => None,
        Term::Compound(_, _) => None,
    }
}

/// Load a SQL table into a Datalog FactDatabase
pub fn load_table_as_facts<S: StorageEngine>(
    storage: &S,
    table: &str,
    db: &mut FactDatabase,
) -> Result<(), DatalogError> {
    let rows = storage
        .scan(table)
        .map_err(|_| DatalogError::TableNotFound(table.to_string()))?;

    let predicate = Symbol::new(table.to_string());

    for row in rows {
        let terms: Vec<Term> = row
            .iter()
            .map(|v| Term::Constant(sql_to_datalog_value(v)))
            .collect();

        let atom = Atom { predicate, terms };

        // Ignore insert errors (e.g., duplicates)
        let _ = db.insert(atom);
    }

    Ok(())
}

/// Execute a Datalog program against SQL tables
///
/// The program can contain:
/// - Facts (will be added to the fact database)
/// - Rules (will be evaluated)
/// - Constraints (will be checked after evaluation)
/// - Queries (the last query's results will be returned)
///
/// SQL tables are automatically loaded as facts based on predicates used in the program.
pub fn execute_datalog_program<S: StorageEngine>(
    storage: &S,
    program_text: &str,
) -> Result<QueryResult, DatalogError> {
    // Parse the Datalog program
    let src = SrcId::repl();
    let program = datalog_parser::parse_program(program_text, src)
        .map_err(|errors| DatalogError::ParseError(format!("{:?}", errors)))?;

    // Separate statements into facts, rules, constraints, and queries
    let mut db = FactDatabase::new();
    let mut rules = Vec::new();
    let mut constraints = Vec::new();
    let mut queries = Vec::new();

    for stmt in program.statements {
        match stmt {
            datalog_parser::Statement::Fact(fact) => {
                let _ = db.insert(fact.atom);
            }
            datalog_parser::Statement::Rule(rule) => {
                rules.push(rule);
            }
            datalog_parser::Statement::Constraint(constraint) => {
                constraints.push(constraint);
            }
            datalog_parser::Statement::Query(query) => {
                queries.push(query);
            }
        }
    }

    // Find all predicates used in the program
    let predicates = collect_predicates(&rules, &constraints, &queries);

    // Load SQL tables as facts for any predicate that matches a table name
    for pred in &predicates {
        let table_name = pred.as_ref();
        // Try to load the table (ignore errors - it might be a derived predicate)
        let _ = load_table_as_facts(storage, table_name, &mut db);
    }

    // Evaluate the program
    let result_db = evaluate(&rules, &constraints, db)?;

    // If there's a query, evaluate it and return results
    if let Some(query) = queries.last() {
        return execute_query(&result_db, query);
    }

    // No query - return all derived facts as a generic result
    Err(DatalogError::NoQuery)
}

/// Collect all predicate names used in a program
fn collect_predicates(
    rules: &[Rule],
    constraints: &[Constraint],
    queries: &[Query],
) -> Vec<Symbol> {
    let mut predicates = Vec::new();

    for rule in rules {
        predicates.push(rule.head.predicate);
        for lit in &rule.body {
            if let Some(atom) = lit.atom() {
                predicates.push(atom.predicate);
            }
        }
    }

    for constraint in constraints {
        for lit in &constraint.body {
            if let Some(atom) = lit.atom() {
                predicates.push(atom.predicate);
            }
        }
    }

    for query in queries {
        for lit in &query.body {
            if let Some(atom) = lit.atom() {
                predicates.push(atom.predicate);
            }
        }
    }

    // Deduplicate
    predicates.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
    predicates.dedup();
    predicates
}

/// Execute a query against the fact database
fn execute_query(db: &FactDatabase, query: &Query) -> Result<QueryResult, DatalogError> {
    use datalog_grounding::satisfy_body;

    // Find all variables in the query
    let variables = collect_query_variables(query);

    // Get all substitutions that satisfy the query
    let substitutions = satisfy_body(&query.body, db);

    if substitutions.is_empty() {
        // No results
        return Ok(QueryResult::Select {
            columns: variables.iter().map(|v| v.as_ref().clone()).collect(),
            rows: vec![],
        });
    }

    // Convert substitutions to rows
    let columns: Vec<String> = variables.iter().map(|v| v.as_ref().clone()).collect();
    let mut rows = Vec::new();

    for subst in substitutions {
        let row: Vec<SValue> = variables
            .iter()
            .map(|var| {
                subst
                    .get(var)
                    .and_then(term_to_sql_value)
                    .unwrap_or(SValue::Null)
            })
            .collect();
        rows.push(row);
    }

    // Deduplicate rows
    rows.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
    rows.dedup();

    Ok(QueryResult::Select { columns, rows })
}

/// Collect all variables from a query
fn collect_query_variables(query: &Query) -> Vec<Symbol> {
    let mut variables = Vec::new();

    for lit in &query.body {
        collect_literal_variables(lit, &mut variables);
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    variables.retain(|v| {
        // Skip anonymous variables
        if v.as_ref().starts_with('_') {
            return false;
        }
        seen.insert(*v)
    });

    variables
}

/// Collect variables from a literal
fn collect_literal_variables(lit: &Literal, vars: &mut Vec<Symbol>) {
    match lit {
        Literal::Positive(atom) | Literal::Negative(atom) => {
            for term in &atom.terms {
                collect_term_variables(term, vars);
            }
        }
        Literal::Comparison(comp) => {
            collect_term_variables(&comp.left, vars);
            collect_term_variables(&comp.right, vars);
        }
    }
}

/// Collect variables from a term
fn collect_term_variables(term: &Term, vars: &mut Vec<Symbol>) {
    match term {
        Term::Variable(v) => vars.push(*v),
        Term::Constant(_) => {}
        Term::Compound(_, args) => {
            for arg in args {
                collect_term_variables(arg, vars);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sql_storage::MemoryEngine;

    fn setup_test_db() -> MemoryEngine {
        use sql_storage::{ColumnSchema, DataType, TableSchema};

        let mut storage = MemoryEngine::new();

        // Create parent table
        storage
            .create_table(TableSchema {
                name: "parent".to_string(),
                columns: vec![
                    ColumnSchema {
                        name: "parent".to_string(),
                        data_type: DataType::Text,
                        nullable: false,
                        default: None,
                        primary_key: false,
                        unique: false,
                        references: None,
                    },
                    ColumnSchema {
                        name: "child".to_string(),
                        data_type: DataType::Text,
                        nullable: false,
                        default: None,
                        primary_key: false,
                        unique: false,
                        references: None,
                    },
                ],
                constraints: vec![],
            })
            .unwrap();

        // Insert test data
        storage
            .insert(
                "parent",
                vec![SValue::Text("john".into()), SValue::Text("mary".into())],
            )
            .unwrap();
        storage
            .insert(
                "parent",
                vec![SValue::Text("mary".into()), SValue::Text("jane".into())],
            )
            .unwrap();
        storage
            .insert(
                "parent",
                vec![SValue::Text("jane".into()), SValue::Text("bob".into())],
            )
            .unwrap();

        storage
    }

    #[test]
    fn test_simple_query() {
        let storage = setup_test_db();

        let result = execute_datalog_program(
            &storage,
            r#"
            ?- parent(X, Y).
            "#,
        )
        .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Y"]);
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_transitive_closure() {
        let storage = setup_test_db();

        let result = execute_datalog_program(
            &storage,
            r#"
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
            ?- ancestor(X, Y).
            "#,
        )
        .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Y"]);
            // Should have: john->mary, mary->jane, jane->bob (direct)
            // Plus: john->jane, mary->bob, john->bob (transitive)
            assert_eq!(rows.len(), 6);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_query_with_constant() {
        let storage = setup_test_db();

        let result = execute_datalog_program(
            &storage,
            r#"
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
            ?- ancestor(john, Y).
            "#,
        )
        .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Y"]);
            // john's descendants: mary, jane, bob
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_negation() {
        let storage = setup_test_db();

        let result = execute_datalog_program(
            &storage,
            r#"
            has_child(X) :- parent(X, _Y).
            ?- has_child(X).
            "#,
        )
        .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // john, mary, jane all have children
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }
}
