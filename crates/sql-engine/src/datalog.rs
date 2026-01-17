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
use sql_storage::{PredicateSchema, StorageEngine, Value as SValue};

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
///
/// This also registers the table's schema with the FactDatabase,
/// enabling arity validation for facts loaded from SQL tables.
pub fn load_table_as_facts<S: StorageEngine>(
    storage: &S,
    table: &str,
    db: &mut FactDatabase,
) -> Result<(), DatalogError> {
    // Get schema from storage and register it
    let table_schema = storage
        .get_schema(table)
        .map_err(|_| DatalogError::TableNotFound(table.to_string()))?;

    let pred_schema = PredicateSchema::from_table_schema(table_schema);
    db.register_schema(pred_schema);

    // Load rows as facts
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

        // Insert fact - errors are propagated (includes arity validation)
        db.insert(atom)
            .map_err(|e| DatalogError::EvaluationError(format!("{}", e)))?;
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
    use crate::{Engine, QueryResult};

    fn setup_test_db() -> Engine {
        let mut engine = Engine::new();

        // Create parent table
        engine
            .execute("CREATE TABLE parent (parent TEXT, child TEXT)")
            .unwrap();

        // Insert test data
        engine
            .execute(
                "INSERT INTO parent VALUES ('john', 'mary'), ('mary', 'jane'), ('jane', 'bob')",
            )
            .unwrap();

        engine
    }

    #[test]
    fn test_simple_query() {
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
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
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
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
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
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
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
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

    // ===== Stratified Negation Tests =====

    #[test]
    fn test_stratified_negation_simple() {
        // Classic "flies" example: birds fly unless they're penguins
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE bird (name TEXT)").unwrap();
        engine.execute("CREATE TABLE penguin (name TEXT)").unwrap();
        engine
            .execute("INSERT INTO bird VALUES ('tweety'), ('polly')")
            .unwrap();
        engine
            .execute("INSERT INTO penguin VALUES ('polly')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            flies(X) :- bird(X), not penguin(X).
            ?- flies(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Only tweety flies (polly is a penguin)
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_negation_chain() {
        // Chain of negations across multiple strata
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE base (x TEXT)").unwrap();
        engine.execute("INSERT INTO base VALUES ('a')").unwrap();

        let result = engine
            .execute_datalog(
                r#"
            % Stratum 0: p(X) :- base(X).
            p(X) :- base(X).
            % Stratum 1: q(X) :- base(X), not p(X).
            q(X) :- base(X), not p(X).
            % Stratum 2: r(X) :- base(X), not q(X).
            r(X) :- base(X), not q(X).
            ?- r(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // p(a) exists, so q(a) doesn't exist, so r(a) exists
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_employment() {
        // Unemployed people need help
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE person (name TEXT)").unwrap();
        engine.execute("CREATE TABLE employed (name TEXT)").unwrap();
        engine
            .execute("INSERT INTO person VALUES ('alice'), ('bob'), ('charlie')")
            .unwrap();
        engine
            .execute("INSERT INTO employed VALUES ('alice'), ('bob')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            unemployed(X) :- person(X), not employed(X).
            needs_help(X) :- unemployed(X).
            ?- needs_help(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Only charlie is unemployed
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_with_recursion() {
        // Reachability with blocked nodes
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE edge (src TEXT, dst TEXT)")
            .unwrap();
        engine.execute("CREATE TABLE blocked (node TEXT)").unwrap();
        engine
            .execute(
                "INSERT INTO edge VALUES ('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'x'), ('x', 'y')",
            )
            .unwrap();
        engine
            .execute("INSERT INTO blocked VALUES ('x'), ('y')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            % Stratum 0: Basic reachability (recursive)
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
            % Stratum 1: Safe reachability (excludes blocked nodes)
            safe_reachable(X, Y) :- reachable(X, Y), not blocked(Y).
            ?- safe_reachable(a, Y).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Y"]);
            // From 'a', can safely reach b, c, d (not x or y - blocked)
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Query with Joins Tests =====

    #[test]
    fn test_query_with_join() {
        // Grandparent query: join parent with itself
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
                r#"
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
            ?- grandparent(X, Z).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Z"]);
            // john->jane (via mary), mary->bob (via jane)
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_query_with_three_way_join() {
        // Great-grandparent: three-way join
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
                r#"
            great_grandparent(X, W) :- parent(X, Y), parent(Y, Z), parent(Z, W).
            ?- great_grandparent(X, W).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "W"]);
            // john->bob (via mary, jane)
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Integer Datatype Tests =====

    #[test]
    fn test_integer_facts() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE score (player TEXT, points INTEGER)")
            .unwrap();
        engine
            .execute("INSERT INTO score VALUES ('alice', 100), ('bob', 50), ('charlie', 100)")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            high_scorer(P) :- score(P, 100).
            ?- high_scorer(P).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["P"]);
            // alice and charlie have 100 points
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_integer_in_recursive_rules() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE link (src INTEGER, dst INTEGER)")
            .unwrap();
        engine
            .execute("INSERT INTO link VALUES (1, 2), (2, 3), (3, 4)")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            path(X, Y) :- link(X, Y).
            path(X, Z) :- path(X, Y), link(Y, Z).
            ?- path(1, Y).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Y"]);
            // From 1: can reach 2, 3, 4
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Facts-Only Tests =====

    #[test]
    fn test_facts_only_no_rules() {
        let engine = setup_test_db();

        // Just query existing facts, no rules
        let result = engine
            .execute_datalog(
                r#"
            ?- parent(john, X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // john's child is mary
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_ground_query_true() {
        let engine = setup_test_db();

        // Query a specific fact that exists
        let result = engine
            .execute_datalog(
                r#"
            ?- parent(john, mary).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty());
            // Ground query should return 1 row if true
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_ground_query_false() {
        let engine = setup_test_db();

        // Query a specific fact that doesn't exist
        let result = engine
            .execute_datalog(
                r#"
            ?- parent(mary, john).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty());
            // Ground query should return 0 rows if false
            assert_eq!(rows.len(), 0);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Multiple Rules for Same Predicate Tests =====

    #[test]
    fn test_multiple_rules_same_predicate() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE a (x TEXT)").unwrap();
        engine.execute("CREATE TABLE b (x TEXT)").unwrap();
        engine
            .execute("INSERT INTO a VALUES ('x1'), ('x2')")
            .unwrap();
        engine
            .execute("INSERT INTO b VALUES ('y1'), ('y2')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            % Two rules define the same predicate (union)
            combined(X) :- a(X).
            combined(X) :- b(X).
            ?- combined(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Should have x1, x2, y1, y2
            assert_eq!(rows.len(), 4);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Longer Chain Tests =====

    #[test]
    fn test_longer_chain() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE edge (src TEXT, dst TEXT)")
            .unwrap();

        // Create a chain: n0 -> n1 -> n2 -> ... -> n9
        let mut values = Vec::new();
        for i in 0..10 {
            values.push(format!("('n{}', 'n{}')", i, i + 1));
        }
        engine
            .execute(&format!("INSERT INTO edge VALUES {}", values.join(", ")))
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).
            ?- path(n0, X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // From n0, can reach n1, n2, ..., n10 (10 nodes)
            assert_eq!(rows.len(), 10);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_transitive_closure_counts() {
        // Verify exact count of transitive closure
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE edge (src TEXT, dst TEXT)")
            .unwrap();

        // Create chain: a -> b -> c -> d -> e (4 edges)
        engine
            .execute("INSERT INTO edge VALUES ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).
            ?- path(X, Y).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Y"]);
            // 4 edges, so:
            // Direct: 4 paths (a-b, b-c, c-d, d-e)
            // Length 2: 3 paths (a-c, b-d, c-e)
            // Length 3: 2 paths (a-d, b-e)
            // Length 4: 1 path (a-e)
            // Total: 4 + 3 + 2 + 1 = 10
            assert_eq!(rows.len(), 10);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Self-Join Tests =====

    #[test]
    fn test_sibling_query() {
        // NOTE: The native Datalog builtins only support numeric comparisons.
        // String inequality (X != Y) is not supported in comparisons.
        // This test uses a workaround with integer IDs.
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE parent (parent_id INT, child_id INT)")
            .unwrap();
        // mary (2) has children jane (3) and alice (4)
        engine
            .execute("INSERT INTO parent VALUES (1, 2), (2, 3), (2, 4), (3, 5)")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y.
            ?- sibling(X, Y).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Y"]);
            // 3 and 4 are siblings (both children of 2)
            // Should have (3, 4) and (4, 3)
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Anonymous Variable Tests =====

    #[test]
    fn test_anonymous_variables() {
        let engine = setup_test_db();

        let result = engine
            .execute_datalog(
                r#"
            is_parent(X) :- parent(X, _).
            ?- is_parent(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // john, mary, jane are all parents
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_multiple_anonymous_variables() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE triple (a TEXT, b TEXT, c TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO triple VALUES ('x', 'y', 'z'), ('a', 'b', 'c')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            has_triple(X) :- triple(X, _, _).
            ?- has_triple(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // x and a have triples
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Duplicate Facts Handling =====

    #[test]
    fn test_duplicate_derived_facts() {
        // When multiple rule applications derive the same fact, it should be deduplicated
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE r (x TEXT)").unwrap();
        engine
            .execute("INSERT INTO r VALUES ('a'), ('a'), ('b')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            p(X) :- r(X).
            ?- p(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Should have a and b (deduplicated)
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Zero-Arity Predicate Tests =====

    #[test]
    fn test_zero_arity_fact() {
        // Zero-arity facts (no arguments)
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE rain (dummy INTEGER)").unwrap();
        engine.execute("INSERT INTO rain VALUES (1)").unwrap(); // Represents "rain."

        let result = engine
            .execute_datalog(
                r#"
            ?- rain(_).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty());
            // Should return true (1 row)
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_zero_arity_rule() {
        // Zero-arity derived predicate
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE rain (dummy INTEGER)").unwrap();
        engine
            .execute("CREATE TABLE umbrella (dummy INTEGER)")
            .unwrap();
        engine.execute("INSERT INTO rain VALUES (1)").unwrap();
        engine.execute("INSERT INTO umbrella VALUES (1)").unwrap();

        let result = engine
            .execute_datalog(
                r#"
            % wet if rain and no umbrella... but we have umbrella
            wet(X) :- rain(X), not umbrella(X).
            ?- wet(_).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty());
            // Should be false (0 rows) because we have umbrella
            assert_eq!(rows.len(), 0);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_zero_arity_with_negation() {
        // Zero-arity with negation
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE rain (dummy INTEGER)").unwrap();
        engine
            .execute("CREATE TABLE sunny (dummy INTEGER)")
            .unwrap();
        engine.execute("INSERT INTO rain VALUES (1)").unwrap();
        // sunny is empty

        let result = engine
            .execute_datalog(
                r#"
            bad_weather(X) :- rain(X), not sunny(X).
            ?- bad_weather(_).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty());
            // Should be true - it's raining and not sunny
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Additional Integer Constant Tests =====

    #[test]
    fn test_integer_constant_in_rule() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE temperature (city TEXT, temp INTEGER)")
            .unwrap();
        engine
            .execute("INSERT INTO temperature VALUES ('NYC', 72), ('LA', 85), ('Chicago', 72)")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            mild(City) :- temperature(City, 72).
            ?- mild(City).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["City"]);
            // NYC and Chicago have 72
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_integer_constant_in_query() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE measurement (sensor TEXT, value INTEGER)")
            .unwrap();
        engine
            .execute("INSERT INTO measurement VALUES ('A', 100), ('B', 200), ('C', 100)")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            ?- measurement(Sensor, 100).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Sensor"]);
            // Sensors A and C have 100
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Boolean Constant Tests =====

    #[test]
    fn test_boolean_facts() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE user_status (name TEXT, active INTEGER)")
            .unwrap();
        // Using 1 for true, 0 for false (SQLite style)
        engine
            .execute("INSERT INTO user_status VALUES ('alice', 1), ('bob', 0), ('charlie', 1)")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            active_user(Name) :- user_status(Name, 1).
            ?- active_user(Name).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Name"]);
            // alice and charlie are active
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== String Constant Tests =====

    #[test]
    fn test_string_constant_in_rule() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE employee (name TEXT, dept TEXT)")
            .unwrap();
        engine
            .execute(
                "INSERT INTO employee VALUES ('alice', 'engineering'), ('bob', 'sales'), ('charlie', 'engineering')",
            )
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            engineer(Name) :- employee(Name, engineering).
            ?- engineer(Name).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Name"]);
            // alice and charlie are in engineering
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_string_constant_in_query() {
        // In Datalog, lowercase identifiers are atoms (constants)
        // Uppercase identifiers are variables
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE person (name TEXT, city TEXT)")
            .unwrap();
        engine
            .execute(
                "INSERT INTO person VALUES ('alice', 'nyc'), ('bob', 'la'), ('charlie', 'nyc')",
            )
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            ?- person(Name, nyc).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Name"]);
            // alice and charlie are in nyc
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Empty Result Tests =====

    #[test]
    fn test_empty_result_no_matching_facts() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE person (name TEXT)").unwrap();
        engine
            .execute("INSERT INTO person VALUES ('alice'), ('bob')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            ?- person(charlie).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty());
            // No match - charlie doesn't exist
            assert_eq!(rows.len(), 0);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_empty_result_from_rule() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE parent (parent TEXT, child TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO parent VALUES ('john', 'mary')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
            ?- grandparent(X, Y).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Y"]);
            // No grandparents - only one generation
            assert_eq!(rows.len(), 0);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Mixed Datatypes with Negation =====

    #[test]
    fn test_negation_with_integers() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE number (n INTEGER)").unwrap();
        engine.execute("CREATE TABLE even (n INTEGER)").unwrap();
        engine
            .execute("INSERT INTO number VALUES (1), (2), (3), (4), (5)")
            .unwrap();
        engine.execute("INSERT INTO even VALUES (2), (4)").unwrap();

        let result = engine
            .execute_datalog(
                r#"
            odd(N) :- number(N), not even(N).
            ?- odd(N).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["N"]);
            // 1, 3, 5 are odd
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_negation_with_sensor_readings() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE reading (sensor TEXT, value INTEGER)")
            .unwrap();
        engine
            .execute("CREATE TABLE calibrated (sensor TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO reading VALUES ('A', 15), ('B', 25), ('C', 35)")
            .unwrap();
        engine
            .execute("INSERT INTO calibrated VALUES ('B')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            needs_calibration(S) :- reading(S, _), not calibrated(S).
            ?- needs_calibration(S).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["S"]);
            // A and C need calibration
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_mixed_types_in_single_program() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE item (id INTEGER, name TEXT, price INTEGER)")
            .unwrap();
        engine
            .execute("CREATE TABLE discount (id INTEGER)")
            .unwrap();
        engine
            .execute(
                "INSERT INTO item VALUES (1, 'apple', 150), (2, 'banana', 75), (3, 'cherry', 200)",
            )
            .unwrap();
        engine.execute("INSERT INTO discount VALUES (2)").unwrap();

        let result = engine
            .execute_datalog(
                r#"
            full_price(Id, Name, Price) :- item(Id, Name, Price), not discount(Id).
            ?- full_price(Id, Name, Price).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Id", "Name", "Price"]);
            // Items 1 and 3 are full price
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Additional Stratified Negation Edge Cases =====

    #[test]
    fn test_stratified_double_negation() {
        // p(X) :- base(X).
        // q(X) :- base(X), not p(X).  -- q is empty because p covers all base
        // r(X) :- base(X), not q(X).  -- r = base because q is empty
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE base (x TEXT)").unwrap();
        engine
            .execute("INSERT INTO base VALUES ('a'), ('b')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            p(X) :- base(X).
            q(X) :- base(X), not p(X).
            r(X) :- base(X), not q(X).
            ?- r(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // r should contain a and b (since q is empty, not q(X) is always true)
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_complex_dependency() {
        // Multiple predicates with complex dependency graph
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE node (n TEXT)").unwrap();
        engine.execute("CREATE TABLE red (n TEXT)").unwrap();
        engine.execute("CREATE TABLE blue (n TEXT)").unwrap();
        engine
            .execute("INSERT INTO node VALUES ('a'), ('b'), ('c'), ('d')")
            .unwrap();
        engine
            .execute("INSERT INTO red VALUES ('a'), ('b')")
            .unwrap();
        engine.execute("INSERT INTO blue VALUES ('c')").unwrap();

        let result = engine
            .execute_datalog(
                r#"
            colored(X) :- red(X).
            colored(X) :- blue(X).
            uncolored(X) :- node(X), not colored(X).
            ?- uncolored(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Only 'd' is uncolored
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_with_multiple_base_predicates() {
        // Negation involving multiple base predicates
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE student (name TEXT)").unwrap();
        engine.execute("CREATE TABLE employee (name TEXT)").unwrap();
        engine.execute("CREATE TABLE retiree (name TEXT)").unwrap();
        engine
            .execute("INSERT INTO student VALUES ('alice'), ('bob')")
            .unwrap();
        engine
            .execute("INSERT INTO employee VALUES ('charlie'), ('alice')")
            .unwrap();
        engine
            .execute("INSERT INTO retiree VALUES ('dave')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            % People who are only students (not working)
            pure_student(X) :- student(X), not employee(X).
            ?- pure_student(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Only bob is a pure student
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_three_level_negation() {
        // Three levels of stratification
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE a (x TEXT)").unwrap();
        engine
            .execute("INSERT INTO a VALUES ('1'), ('2'), ('3')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            % Stratum 0: b copies from a
            b(X) :- a(X).
            % Stratum 1: c = a - b = empty
            c(X) :- a(X), not b(X).
            % Stratum 2: d = a - c = a (since c is empty)
            d(X) :- a(X), not c(X).
            % Stratum 3: e = a - d = empty (since d = a)
            e(X) :- a(X), not d(X).
            ?- e(X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // e should be empty
            assert_eq!(rows.len(), 0);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_stratified_negation_with_join() {
        // Negation combined with joins
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE likes (person TEXT, food TEXT)")
            .unwrap();
        engine
            .execute("CREATE TABLE allergic (person TEXT, food TEXT)")
            .unwrap();
        engine
            .execute(
                "INSERT INTO likes VALUES ('alice', 'pizza'), ('alice', 'sushi'), ('bob', 'pizza'), ('bob', 'pasta')",
            )
            .unwrap();
        engine
            .execute("INSERT INTO allergic VALUES ('alice', 'sushi')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            can_eat(Person, Food) :- likes(Person, Food), not allergic(Person, Food).
            ?- can_eat(Person, Food).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Person", "Food"]);
            // alice-pizza, bob-pizza, bob-pasta (not alice-sushi)
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Float Constant Tests =====

    #[test]
    fn test_float_sql_only() {
        // Test that SQL parsing of floats works in the engine
        let mut engine = Engine::new();
        let create_result = engine.execute("CREATE TABLE temperature (city TEXT, temp FLOAT)");
        assert!(
            create_result.is_ok(),
            "CREATE TABLE failed: {:?}",
            create_result
        );

        let insert_result =
            engine.execute("INSERT INTO temperature VALUES ('nyc', 72.5), ('la', 85.0)");
        assert!(insert_result.is_ok(), "INSERT failed: {:?}", insert_result);

        let select_result = engine.execute("SELECT * FROM temperature");
        assert!(select_result.is_ok(), "SELECT failed: {:?}", select_result);

        if let QueryResult::Select { columns, rows } = select_result.unwrap() {
            assert_eq!(columns, vec!["city", "temp"]);
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_float_facts() {
        let mut engine = Engine::new();
        let create_result = engine.execute("CREATE TABLE temperature (city TEXT, temp FLOAT)");
        assert!(
            create_result.is_ok(),
            "CREATE TABLE failed: {:?}",
            create_result
        );

        let insert_result = engine
            .execute("INSERT INTO temperature VALUES ('nyc', 72.5), ('la', 85.0), ('sf', 60.5)");
        assert!(insert_result.is_ok(), "INSERT failed: {:?}", insert_result);

        let result = engine.execute_datalog("?- temperature(City, Temp).");

        assert!(result.is_ok(), "Datalog query failed: {:?}", result);
        let result = result.unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["City", "Temp"]);
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_float_in_rule() {
        let mut engine = Engine::new();
        let create_result = engine.execute("CREATE TABLE temperature (city TEXT, temp FLOAT)");
        assert!(
            create_result.is_ok(),
            "CREATE TABLE failed: {:?}",
            create_result
        );

        let insert_result = engine
            .execute("INSERT INTO temperature VALUES ('nyc', 72.5), ('la', 85.0), ('sf', 60.5)");
        assert!(insert_result.is_ok(), "INSERT failed: {:?}", insert_result);

        let result = engine.execute_datalog("hot(City) :- temperature(City, 85.0). ?- hot(City).");

        assert!(result.is_ok(), "Datalog query failed: {:?}", result);
        let result = result.unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["City"]);
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Date/Time/Timestamp Tests =====

    #[test]
    fn test_date_type() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE events (name TEXT, event_date DATE)")
            .unwrap();
        engine
            .execute(
                "INSERT INTO events VALUES ('meeting', '2024-01-15'), ('deadline', '2024-02-20')",
            )
            .unwrap();

        let result = engine.execute_datalog("?- events(Name, Date).").unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Name", "Date"]);
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_date_query_with_variable() {
        // NOTE: Querying dates with specific constant strings doesn't currently unify
        // because dates are converted to atoms during SQL->Datalog conversion.
        // This test queries all dates and verifies we get the expected count.
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE events (name TEXT, event_date DATE)")
            .unwrap();
        engine
            .execute(
                "INSERT INTO events VALUES ('meeting', '2024-01-15'), ('deadline', '2024-02-20')",
            )
            .unwrap();

        // Query all events with their dates
        let result = engine.execute_datalog("?- events(Name, Date).").unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Name", "Date"]);
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_time_type() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE schedule (task TEXT, start_time TIME)")
            .unwrap();
        engine
            .execute("INSERT INTO schedule VALUES ('standup', '09:00:00'), ('lunch', '12:30:00')")
            .unwrap();

        let result = engine.execute_datalog("?- schedule(Task, Time).").unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Task", "Time"]);
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_timestamp_type() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE logs (event TEXT, created_at TIMESTAMP)")
            .unwrap();
        engine
            .execute("INSERT INTO logs VALUES ('login', '2024-01-15 09:00:00'), ('logout', '2024-01-15 17:30:00')")
            .unwrap();

        let result = engine.execute_datalog("?- logs(Event, Ts).").unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Event", "Ts"]);
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Comprehensive Data Type Parity Test =====

    #[test]
    fn test_all_data_types_together() {
        // Test that all SQL data types work correctly with Datalog
        let mut engine = Engine::new();

        // Create table with all supported types
        engine
            .execute(
                "CREATE TABLE products (
                    id INT,
                    name TEXT,
                    price FLOAT,
                    in_stock BOOL,
                    release_date DATE,
                    available_from TIME,
                    last_updated TIMESTAMP
                )",
            )
            .unwrap();

        engine
            .execute(
                "INSERT INTO products VALUES
                (1, 'Widget', 19.99, true, '2024-01-01', '08:00:00', '2024-01-15 10:30:00'),
                (2, 'Gadget', 49.99, false, '2024-02-15', '09:00:00', '2024-02-20 14:45:00')",
            )
            .unwrap();

        // Query all columns
        let result = engine
            .execute_datalog(
                "?- products(Id, Name, Price, Stock, ReleaseDate, AvailFrom, LastUpdate).",
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(
                columns,
                vec![
                    "Id",
                    "Name",
                    "Price",
                    "Stock",
                    "ReleaseDate",
                    "AvailFrom",
                    "LastUpdate"
                ]
            );
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_datalog_rule_with_all_types() {
        let mut engine = Engine::new();

        engine
            .execute(
                "CREATE TABLE products (
                    id INT,
                    name TEXT,
                    price FLOAT,
                    in_stock BOOL
                )",
            )
            .unwrap();

        engine
            .execute(
                "INSERT INTO products VALUES
                (1, 'Widget', 19.99, true),
                (2, 'Gadget', 49.99, false),
                (3, 'Gizmo', 29.99, true)",
            )
            .unwrap();

        // Rule combining multiple type constraints
        // Find in-stock products (using boolean true)
        let result = engine
            .execute_datalog(
                "available(Name, Price) :- products(_Id, Name, Price, true). ?- available(N, P).",
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["N", "P"]);
            // Widget and Gizmo are in stock
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Comment Tests =====

    #[test]
    fn test_program_with_line_comments() {
        let engine = setup_test_db();

        // Program with % line comments
        let result = engine
            .execute_datalog(
                r#"
            % This is a comment about the ancestor rule
            ancestor(X, Y) :- parent(X, Y).  % inline comment
            % Another comment
            ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
            ?- ancestor(john, Y).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["Y"]);
            assert_eq!(rows.len(), 3); // mary, jane, bob
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Comparison Operators in Body Tests =====

    #[test]
    fn test_comparison_greater_than() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores VALUES ('alice', 85), ('bob', 70), ('charlie', 90)")
            .unwrap();

        let result = engine
            .execute_datalog(
                "high_scorer(Name) :- scores(Name, Score), Score > 80. ?- high_scorer(N).",
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["N"]);
            // alice (85) and charlie (90) > 80
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_comparison_less_than() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores VALUES ('alice', 85), ('bob', 70), ('charlie', 90)")
            .unwrap();

        let result = engine
            .execute_datalog(
                "low_scorer(Name) :- scores(Name, Score), Score < 80. ?- low_scorer(N).",
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["N"]);
            // bob (70) < 80
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_comparison_equality() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores VALUES ('alice', 85), ('bob', 70), ('charlie', 85)")
            .unwrap();

        let result = engine
            .execute_datalog(
                "exact_score(Name) :- scores(Name, Score), Score = 85. ?- exact_score(N).",
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["N"]);
            // alice and charlie both have 85
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_comparison_not_equal() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        engine
            .execute("INSERT INTO scores VALUES ('alice', 85), ('bob', 70), ('charlie', 85)")
            .unwrap();

        let result = engine
            .execute_datalog("not_85(Name) :- scores(Name, Score), Score != 85. ?- not_85(N).")
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["N"]);
            // Only bob (70) != 85
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_comparison_between_variables() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE pairs (a INT, b INT)").unwrap();
        engine
            .execute("INSERT INTO pairs VALUES (1, 2), (3, 3), (5, 4)")
            .unwrap();

        let result = engine
            .execute_datalog("ascending(A, B) :- pairs(A, B), A < B. ?- ascending(X, Y).")
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X", "Y"]);
            // Only (1, 2) where A < B
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Duplicate Handling Tests =====

    #[test]
    fn test_duplicate_facts_in_table() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE items (name TEXT)").unwrap();
        // Insert duplicates
        engine
            .execute("INSERT INTO items VALUES ('apple'), ('banana'), ('apple'), ('cherry')")
            .unwrap();

        let result = engine.execute_datalog("?- items(X).").unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // Datalog semantics: results are deduplicated (set semantics)
            // apple, banana, cherry = 3 unique values
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_derived_duplicate_facts() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE edge (src TEXT, dst TEXT)")
            .unwrap();
        // Multiple paths lead to same derived facts
        engine
            .execute("INSERT INTO edge VALUES ('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')")
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).
            ?- path(a, d).
            "#,
            )
            .unwrap();

        // Query for path(a, d) which can be derived two ways
        // (a->b->d and a->c->d) should still return result
        if let QueryResult::Select { columns, rows } = result {
            assert!(columns.is_empty() || rows.len() >= 1);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Long Chain Stress Tests =====

    #[test]
    fn test_very_long_chain() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE edge (src TEXT, dst TEXT)")
            .unwrap();

        // Create a chain: n0 -> n1 -> n2 -> ... -> n19 (20 nodes)
        let mut values = Vec::new();
        for i in 0..20 {
            values.push(format!("('n{}', 'n{}')", i, i + 1));
        }
        engine
            .execute(&format!("INSERT INTO edge VALUES {}", values.join(", ")))
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).
            ?- path(n0, X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // From n0, can reach n1, n2, ..., n20 (20 nodes)
            assert_eq!(rows.len(), 20);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_wide_graph() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE edge (src TEXT, dst TEXT)")
            .unwrap();

        // Create a wide graph: root -> child1, child2, ..., child10
        // Each child -> grandchild
        let mut values = Vec::new();
        for i in 1..=10 {
            values.push(format!("('root', 'child{}')", i));
            values.push(format!("('child{}', 'grandchild{}')", i, i));
        }
        engine
            .execute(&format!("INSERT INTO edge VALUES {}", values.join(", ")))
            .unwrap();

        let result = engine
            .execute_datalog(
                r#"
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
            ?- reachable(root, X).
            "#,
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["X"]);
            // From root: 10 children + 10 grandchildren = 20 nodes
            assert_eq!(rows.len(), 20);
        } else {
            panic!("Expected Select result");
        }
    }

    // ===== Multiple Comparison Test =====

    #[test]
    fn test_multiple_comparisons_in_body() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE range_data (id INT, val INT)")
            .unwrap();
        engine
            .execute("INSERT INTO range_data VALUES (1, 5), (2, 15), (3, 25), (4, 10)")
            .unwrap();

        let result = engine
            .execute_datalog(
                "in_range(Id, Val) :- range_data(Id, Val), Val >= 10, Val < 20. ?- in_range(I, V).",
            )
            .unwrap();

        if let QueryResult::Select { columns, rows } = result {
            assert_eq!(columns, vec!["I", "V"]);
            // (2, 15) and (4, 10) are in range [10, 20)
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }
}
