//! SQL and Datalog integration tests
//!
//! Tests demonstrating the interoperability between SQL and Datalog:
//! - Create tables and insert data with SQL
//! - Query with Datalog (including recursive queries)
//! - Query Datalog-derived results back with SQL

use query::{Engine, QueryResult};

/// Helper to extract rows from a QueryResult
fn get_rows(result: &QueryResult) -> Vec<Vec<String>> {
    match result {
        QueryResult::Select { rows, .. } => rows
            .iter()
            .map(|row| row.iter().map(|v| format!("{:?}", v)).collect())
            .collect(),
        _ => vec![],
    }
}

/// Helper to get row count
fn row_count(result: &QueryResult) -> usize {
    match result {
        QueryResult::Select { rows, .. } => rows.len(),
        QueryResult::RowsAffected(n) => *n,
        _ => 0,
    }
}

#[test]
fn test_sql_datalog_sql_roundtrip() {
    let mut engine = Engine::new();

    // Step 1: Create table with SQL
    let result = engine.execute("CREATE TABLE edge (src INT, dst INT)");
    assert!(result.is_ok());

    // Step 2: Insert data with SQL
    let result = engine.execute("INSERT INTO edge VALUES (1, 2), (2, 3), (3, 4), (4, 5)");
    assert!(result.is_ok());
    assert_eq!(row_count(&result.unwrap()), 4);

    // Verify data with SQL
    let result = engine.execute("SELECT * FROM edge ORDER BY src").unwrap();
    assert_eq!(row_count(&result), 4);

    // Step 3: Run Datalog query to compute transitive closure
    // This creates a derived predicate 'path' with all reachable pairs
    let datalog_result = engine
        .execute_datalog(
            r#"
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).
            ?- path(1, X).
        "#,
        )
        .unwrap();

    // Should find all nodes reachable from 1: 2, 3, 4, 5
    assert_eq!(row_count(&datalog_result), 4);

    // Step 4: Query the Datalog-derived 'path' predicate with SQL
    // The path predicate is stored as a table with auto-generated column names (col0, col1)
    let sql_on_datalog = engine
        .execute("SELECT * FROM path ORDER BY col0, col1")
        .unwrap();

    // path should contain all transitive pairs:
    // (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)
    assert_eq!(
        row_count(&sql_on_datalog),
        10,
        "Expected 10 path entries for transitive closure"
    );

    // Verify specific paths exist using SQL WHERE clause
    // Note: Datalog-derived tables use col0, col1, ... as column names
    let result = engine
        .execute("SELECT * FROM path WHERE col0 = 1 ORDER BY col1")
        .unwrap();
    assert_eq!(row_count(&result), 4); // 1 can reach 2, 3, 4, 5

    let result = engine
        .execute("SELECT * FROM path WHERE col0 = 3 ORDER BY col1")
        .unwrap();
    assert_eq!(row_count(&result), 2); // 3 can reach 4, 5

    // Step 5: Query with SQL aggregation on Datalog-derived data
    let result = engine
        .execute(
            "SELECT DISTINCT col0 FROM path
             WHERE col1 = 5
             ORDER BY col0",
        )
        .unwrap();
    // All nodes that can reach 5: 1, 2, 3, 4
    assert_eq!(row_count(&result), 4);
}

#[test]
fn test_datalog_negation_with_sql_data() {
    let mut engine = Engine::new();

    // Create tables with SQL
    engine.execute("CREATE TABLE person (name TEXT)").unwrap();
    engine.execute("CREATE TABLE has_car (name TEXT)").unwrap();

    // Insert data
    engine
        .execute("INSERT INTO person VALUES ('alice'), ('bob'), ('charlie'), ('diana')")
        .unwrap();
    engine
        .execute("INSERT INTO has_car VALUES ('bob'), ('diana')")
        .unwrap();

    // Use Datalog to find people without cars (negation)
    let result = engine
        .execute_datalog(
            r#"
            needs_ride(X) :- person(X), not has_car(X).
            ?- needs_ride(X).
        "#,
        )
        .unwrap();

    // alice and charlie don't have cars
    assert_eq!(row_count(&result), 2);

    // Query the derived predicate with SQL
    let sql_result = engine
        .execute("SELECT * FROM needs_ride ORDER BY name")
        .unwrap();

    let rows = get_rows(&sql_result);
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_recursive_ancestor_query() {
    let mut engine = Engine::new();

    // Create family tree with SQL
    engine
        .execute("CREATE TABLE parent (child TEXT, parent TEXT)")
        .unwrap();

    engine
        .execute(
            "INSERT INTO parent VALUES
             ('bob', 'alice'),
             ('charlie', 'bob'),
             ('diana', 'charlie'),
             ('eve', 'diana')",
        )
        .unwrap();

    // Compute ancestors with Datalog
    let result = engine
        .execute_datalog(
            r#"
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
            ?- ancestor(eve, X).
        "#,
        )
        .unwrap();

    // eve's ancestors: diana, charlie, bob, alice
    assert_eq!(row_count(&result), 4);

    // Query with SQL: who are alice's descendants?
    // Note: Datalog-derived tables use col0, col1, ... as column names
    // ancestor(X, Y) means col0 = descendant (X), col1 = ancestor (Y)
    let sql_result = engine
        .execute("SELECT col0 FROM ancestor WHERE col1 = 'alice' ORDER BY col0")
        .unwrap();

    // bob, charlie, diana, eve are all descendants of alice
    assert_eq!(row_count(&sql_result), 4);
}

#[test]
fn test_mixed_sql_datalog_workflow() {
    let mut engine = Engine::new();

    // SQL: Create and populate tables
    engine
        .execute("CREATE TABLE employee (id INT, name TEXT, dept TEXT)")
        .unwrap();
    engine
        .execute("CREATE TABLE reports_to (employee_id INT, manager_id INT)")
        .unwrap();

    engine
        .execute(
            "INSERT INTO employee VALUES
         (1, 'CEO', 'exec'),
         (2, 'VP1', 'sales'),
         (3, 'VP2', 'eng'),
         (4, 'Manager1', 'sales'),
         (5, 'Manager2', 'eng'),
         (6, 'Dev1', 'eng'),
         (7, 'Dev2', 'eng')",
        )
        .unwrap();

    engine
        .execute(
            "INSERT INTO reports_to VALUES
         (2, 1), (3, 1),
         (4, 2), (5, 3),
         (6, 5), (7, 5)",
        )
        .unwrap();

    // Datalog: Compute transitive management chain
    // manages(M, E) means: M manages E (directly or transitively)
    // This creates a derived table with col0 = manager_id, col1 = employee_id
    engine
        .execute_datalog(
            r#"
        manages(M, E) :- reports_to(E, M).
        manages(M, E) :- reports_to(E, X), manages(M, X).
        ?- manages(X, Y).
    "#,
        )
        .unwrap();

    // SQL: Verify the derived table exists and query it directly
    let manages_result = engine.execute("SELECT * FROM manages").unwrap();
    // Total management relationships (direct + transitive)
    assert_eq!(row_count(&manages_result), 12);

    // SQL: Query who the CEO manages (manager_id = 1)
    let ceo_manages = engine
        .execute("SELECT * FROM manages WHERE col0 = 1 ORDER BY col1")
        .unwrap();
    // CEO manages: 2, 3, 4, 5, 6, 7 (6 people)
    assert_eq!(row_count(&ceo_manages), 6);

    // SQL: Query who Manager2 (id=5) manages
    let manager2_manages = engine
        .execute("SELECT * FROM manages WHERE col0 = 5 ORDER BY col1")
        .unwrap();
    // Manager2 manages: 6, 7
    assert_eq!(row_count(&manager2_manages), 2);

    // SQL: Count distinct managers (people who manage at least one person)
    let distinct_managers = engine
        .execute("SELECT DISTINCT col0 FROM manages ORDER BY col0")
        .unwrap();
    // Managers: 1 (CEO), 2 (VP1), 3 (VP2), 5 (Manager2)
    // Note: VP1(2) manages Manager1(4), VP2(3) manages Manager2(5) and transitively 6,7
    assert_eq!(row_count(&distinct_managers), 4);
}
