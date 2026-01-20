//! Property-based tests for SQL queries using proptest
//!
//! These tests generate random but valid SQL queries and verify
//! that the database handles them correctly.

use db::Engine;
use logical::Value;
use proptest::prelude::*;

/// SQL reserved words that cannot be used as identifiers
const SQL_RESERVED_WORDS: &[&str] = &[
    // DML
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    // DDL
    "CREATE",
    "DROP",
    "ALTER",
    "TABLE",
    "INDEX",
    "VIEW",
    "TRIGGER",
    "FUNCTION",
    "PROCEDURE",
    // Data types
    "INT",
    "INTEGER",
    "TEXT",
    "VARCHAR",
    "BOOL",
    "BOOLEAN",
    "FLOAT",
    "DOUBLE",
    "DATE",
    "TIMESTAMP",
    "TIME",
    "NULL",
    // Constraints
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "UNIQUE",
    "NOT",
    "DEFAULT",
    "CONSTRAINT",
    "CHECK",
    "CASCADE",
    "RESTRICT",
    "ACTION",
    "NO",
    // Logical operators
    "AND",
    "OR",
    "IS",
    "IN",
    "LIKE",
    "BETWEEN",
    "EXISTS",
    // Joins
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "OUTER",
    "FULL",
    "CROSS",
    "ON",
    // Ordering and limits
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "OFFSET",
    "GROUP",
    "HAVING",
    // Aggregates
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    // Boolean literals
    "TRUE",
    "FALSE",
    // Case expressions
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    // Set operations
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "ALL",
    // Transaction control
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "TRANSACTION",
    "SAVEPOINT",
    "RELEASE",
    "TO",
    // Misc
    "AS",
    "WITH",
    "RECURSIVE",
    "FOR",
    "EACH",
    "ROW",
    "BEFORE",
    "AFTER",
    "RAISE",
    "ERROR",
    "ADD",
    "COLUMN",
    "RENAME",
    "OVER",
    "PARTITION",
    "RANK",
    "CALL",
    "DECLARE",
    "EXEC",
    "EXECUTE",
    "RETURNS",
    "RETURN",
    "LANGUAGE",
    "DISTINCT",
];

/// Generate a valid identifier (table or column name)
fn identifier_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z][a-z0-9_]{0,7}")
        .unwrap()
        .prop_filter("must not be reserved word", |s| {
            let upper = s.to_uppercase();
            !SQL_RESERVED_WORDS.contains(&upper.as_str())
        })
}

/// Generate a random integer value
fn int_value_strategy() -> impl Strategy<Value = i64> {
    prop::num::i64::ANY.prop_map(|v| v % 10000)
}

/// Generate a random text value (simple alphanumeric)
fn text_value_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 ]{1,20}").unwrap()
}

/// Column definition for table creation
#[derive(Debug, Clone)]
struct ColumnDef {
    name: String,
    col_type: String,
}

/// Generate column definitions
fn column_def_strategy() -> impl Strategy<Value = ColumnDef> {
    (
        identifier_strategy(),
        prop::sample::select(vec!["INT", "TEXT", "BOOL"]),
    )
        .prop_map(|(name, col_type)| ColumnDef {
            name,
            col_type: col_type.to_string(),
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_create_table_valid(
        table_name in identifier_strategy(),
        columns in prop::collection::vec(column_def_strategy(), 1..5)
    ) {
        // Deduplicate column names
        let mut seen = std::collections::HashSet::new();
        let columns: Vec<_> = columns.into_iter()
            .filter(|c| seen.insert(c.name.clone()))
            .collect();

        if columns.is_empty() {
            return Ok(());
        }

        let cols_sql = columns.iter()
            .map(|c| format!("{} {}", c.name, c.col_type))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!("CREATE TABLE {} ({})", table_name, cols_sql);

        let mut engine = Engine::new();
        let result = engine.execute(&sql);
        prop_assert!(result.is_ok(), "CREATE TABLE should succeed: {:?}", result);
    }

    #[test]
    fn test_insert_and_select(
        values in prop::collection::vec(int_value_strategy(), 1..5)
    ) {
        let mut engine = Engine::new();

        // Create table with matching columns
        let cols: Vec<String> = (0..values.len()).map(|i| format!("c{} INT", i)).collect();
        let create_sql = format!("CREATE TABLE t ({})", cols.join(", "));
        engine.execute(&create_sql).unwrap();

        // Insert values
        let vals_str = values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let insert_sql = format!("INSERT INTO t VALUES ({})", vals_str);
        let result = engine.execute(&insert_sql);
        prop_assert!(result.is_ok(), "INSERT should succeed: {:?}", result);

        // Select and verify
        let select_result = engine.execute("SELECT * FROM t");
        prop_assert!(select_result.is_ok());

        if let Ok(db::QueryResult::Select { rows, .. }) = select_result {
            prop_assert_eq!(rows.len(), 1);
            for (i, val) in values.iter().enumerate() {
                prop_assert_eq!(rows[0][i].clone(), Value::Int(*val));
            }
        }
    }

    #[test]
    fn test_where_clause_comparison(
        target in 0i64..100,
        values in prop::collection::vec(0i64..100, 1..20)
    ) {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (id INT, val INT)").unwrap();

        // Insert all values
        for (i, val) in values.iter().enumerate() {
            let sql = format!("INSERT INTO t VALUES ({}, {})", i, val);
            engine.execute(&sql).unwrap();
        }

        // Count expected matches
        let expected_count = values.iter().filter(|&&v| v == target).count();

        // Query with WHERE
        let sql = format!("SELECT * FROM t WHERE val = {}", target);
        let result = engine.execute(&sql);
        prop_assert!(result.is_ok());

        if let Ok(db::QueryResult::Select { rows, .. }) = result {
            prop_assert_eq!(rows.len(), expected_count,
                "Expected {} rows where val = {}, got {}", expected_count, target, rows.len());
        }
    }

    #[test]
    fn test_update_modifies_correct_rows(
        target_id in 0usize..10,
        new_value in int_value_strategy(),
        num_rows in 5usize..15
    ) {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (id INT, val INT)").unwrap();

        // Insert rows
        for i in 0..num_rows {
            let sql = format!("INSERT INTO t VALUES ({}, {})", i, i * 10);
            engine.execute(&sql).unwrap();
        }

        // Update specific row
        let target = target_id % num_rows;
        let sql = format!("UPDATE t SET val = {} WHERE id = {}", new_value, target);
        let result = engine.execute(&sql);
        prop_assert!(result.is_ok());

        // Verify update
        let sql = format!("SELECT val FROM t WHERE id = {}", target);
        let result = engine.execute(&sql);
        prop_assert!(result.is_ok());

        if let Ok(db::QueryResult::Select { rows, .. }) = result {
            if !rows.is_empty() {
                prop_assert_eq!(rows[0][0].clone(), Value::Int(new_value));
            }
        }
    }

    #[test]
    fn test_delete_removes_correct_rows(
        delete_threshold in 0i64..100,
        values in prop::collection::vec(0i64..100, 5..20)
    ) {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (id INT, val INT)").unwrap();

        // Insert all values
        for (i, val) in values.iter().enumerate() {
            let sql = format!("INSERT INTO t VALUES ({}, {})", i, val);
            engine.execute(&sql).unwrap();
        }

        let initial_count = values.len();
        let to_delete = values.iter().filter(|&&v| v < delete_threshold).count();
        let expected_remaining = initial_count - to_delete;

        // Delete rows below threshold
        let sql = format!("DELETE FROM t WHERE val < {}", delete_threshold);
        let result = engine.execute(&sql);
        prop_assert!(result.is_ok());

        // Count remaining
        let result = engine.execute("SELECT * FROM t");
        prop_assert!(result.is_ok());

        if let Ok(db::QueryResult::Select { rows, .. }) = result {
            prop_assert_eq!(rows.len(), expected_remaining,
                "After DELETE WHERE val < {}, expected {} rows, got {}",
                delete_threshold, expected_remaining, rows.len());
        }
    }

    #[test]
    fn test_aggregate_count(
        num_rows in 0usize..50
    ) {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (id INT)").unwrap();

        for i in 0..num_rows {
            let sql = format!("INSERT INTO t VALUES ({})", i);
            engine.execute(&sql).unwrap();
        }

        let result = engine.execute("SELECT COUNT(*) FROM t");
        prop_assert!(result.is_ok());

        if let Ok(db::QueryResult::Select { rows, .. }) = result {
            prop_assert_eq!(rows.len(), 1);
            prop_assert_eq!(rows[0][0].clone(), Value::Int(num_rows as i64));
        }
    }

    #[test]
    fn test_aggregate_sum(
        values in prop::collection::vec(-1000i64..1000, 1..20)
    ) {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (val INT)").unwrap();

        for val in &values {
            let sql = format!("INSERT INTO t VALUES ({})", val);
            engine.execute(&sql).unwrap();
        }

        let expected_sum: i64 = values.iter().sum();

        let result = engine.execute("SELECT SUM(val) FROM t");
        prop_assert!(result.is_ok());

        if let Ok(db::QueryResult::Select { rows, .. }) = result {
            prop_assert_eq!(rows.len(), 1);
            prop_assert_eq!(rows[0][0].clone(), Value::Int(expected_sum));
        }
    }

    #[test]
    fn test_multiple_inserts_selects(
        operations in prop::collection::vec(
            prop::sample::select(vec!["insert", "select"]),
            1..30
        )
    ) {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (id INT, val INT)").unwrap();

        let mut insert_count = 0;

        for op in operations {
            match op {
                "insert" => {
                    let sql = format!("INSERT INTO t VALUES ({}, {})", insert_count, insert_count * 2);
                    let result = engine.execute(&sql);
                    prop_assert!(result.is_ok());
                    insert_count += 1;
                }
                "select" => {
                    let result = engine.execute("SELECT * FROM t");
                    prop_assert!(result.is_ok());
                    if let Ok(db::QueryResult::Select { rows, .. }) = result {
                        prop_assert_eq!(rows.len(), insert_count);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn test_empty_table_aggregates() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE empty (id INT, val INT)")
        .unwrap();

    // COUNT on empty table should return 0
    let result = engine.execute("SELECT COUNT(*) FROM empty");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(0));
    }
}

#[test]
fn test_boolean_operations() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE flags (id INT, active BOOL)")
        .unwrap();

    engine
        .execute("INSERT INTO flags VALUES (1, TRUE)")
        .unwrap();
    engine
        .execute("INSERT INTO flags VALUES (2, FALSE)")
        .unwrap();
    engine
        .execute("INSERT INTO flags VALUES (3, TRUE)")
        .unwrap();

    let result = engine.execute("SELECT * FROM flags WHERE active = TRUE");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 2);
    }
}

#[test]
fn test_null_handling() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE nullable (id INT, val INT)")
        .unwrap();

    engine
        .execute("INSERT INTO nullable VALUES (1, NULL)")
        .unwrap();
    engine
        .execute("INSERT INTO nullable VALUES (2, 100)")
        .unwrap();

    let result = engine.execute("SELECT * FROM nullable");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][1], Value::Null);
        assert_eq!(rows[1][1], Value::Int(100));
    }
}

#[test]
fn test_text_values() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE texts (id INT, name TEXT)")
        .unwrap();

    engine
        .execute("INSERT INTO texts VALUES (1, 'hello')")
        .unwrap();
    engine
        .execute("INSERT INTO texts VALUES (2, 'world')")
        .unwrap();

    let result = engine.execute("SELECT * FROM texts WHERE name = 'hello'");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(1));
    }
}

#[test]
fn test_arithmetic_expressions() {
    let mut engine = Engine::new();
    engine.execute("CREATE TABLE nums (a INT, b INT)").unwrap();

    engine.execute("INSERT INTO nums VALUES (10, 3)").unwrap();

    // Test various arithmetic operations in WHERE clause
    let result = engine.execute("SELECT * FROM nums WHERE a + b = 13");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1);
    }

    let result = engine.execute("SELECT * FROM nums WHERE a - b = 7");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1);
    }

    let result = engine.execute("SELECT * FROM nums WHERE a * b = 30");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1);
    }
}

#[test]
fn test_comparison_operators() {
    let mut engine = Engine::new();
    engine.execute("CREATE TABLE cmp (val INT)").unwrap();

    for i in 1..=10 {
        engine
            .execute(&format!("INSERT INTO cmp VALUES ({})", i))
            .unwrap();
    }

    // Less than
    let result = engine.execute("SELECT * FROM cmp WHERE val < 5");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 4);
    }

    // Greater than
    let result = engine.execute("SELECT * FROM cmp WHERE val > 5");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 5);
    }

    // Less than or equal
    let result = engine.execute("SELECT * FROM cmp WHERE val <= 5");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 5);
    }

    // Greater than or equal
    let result = engine.execute("SELECT * FROM cmp WHERE val >= 5");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 6);
    }

    // Not equal
    let result = engine.execute("SELECT * FROM cmp WHERE val != 5");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 9);
    }
}

#[test]
fn test_logical_operators() {
    let mut engine = Engine::new();
    engine.execute("CREATE TABLE logic (a INT, b INT)").unwrap();

    engine.execute("INSERT INTO logic VALUES (1, 10)").unwrap();
    engine.execute("INSERT INTO logic VALUES (2, 20)").unwrap();
    engine.execute("INSERT INTO logic VALUES (3, 30)").unwrap();
    engine.execute("INSERT INTO logic VALUES (4, 40)").unwrap();

    // AND
    let result = engine.execute("SELECT * FROM logic WHERE a > 1 AND b < 40");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 2); // rows 2 and 3
    }

    // OR
    let result = engine.execute("SELECT * FROM logic WHERE a = 1 OR a = 4");
    assert!(result.is_ok());
    if let Ok(db::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 2);
    }
}
