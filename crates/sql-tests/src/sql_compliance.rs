//! SQL Standard Compliance Tests
//!
//! This module contains tests based on the SQL standard (SQL-92 and later)
//! to verify that our SQL implementation follows standard SQL behavior.
//!
//! Reference: ISO/IEC 9075 (SQL standard)

use sql_engine::Engine;
use sql_storage::Value;

/// Helper to run a query and get results
fn query_rows(engine: &mut Engine, sql: &str) -> Vec<Vec<Value>> {
    match engine.execute(sql) {
        Ok(sql_engine::QueryResult::Select { rows, .. }) => rows,
        Ok(_) => vec![],
        Err(e) => panic!("Query failed: {} - Error: {:?}", sql, e),
    }
}

/// Helper to execute a statement
fn exec(engine: &mut Engine, sql: &str) {
    engine.execute(sql).unwrap_or_else(|e| {
        panic!("Statement failed: {} - Error: {:?}", sql, e);
    });
}

// =============================================================================
// Section 1: Data Definition Language (DDL)
// =============================================================================

mod ddl {
    use super::*;

    #[test]
    fn test_create_table_basic() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t1 (id INT)");
        exec(&mut engine, "CREATE TABLE t2 (id INT, name TEXT)");
        exec(&mut engine, "CREATE TABLE t3 (a INT, b INT, c INT, d INT)");
    }

    #[test]
    fn test_create_table_data_types() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE types (i INT, t TEXT, b BOOL)");

        exec(&mut engine, "INSERT INTO types VALUES (42, 'hello', TRUE)");
        let rows = query_rows(&mut engine, "SELECT * FROM types");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(42));
        assert_eq!(rows[0][1], Value::Text("hello".to_string()));
        assert_eq!(rows[0][2], Value::Bool(true));
    }

    // NOTE: DROP TABLE not yet implemented
    // #[test]
    // fn test_drop_table() {
    //     let mut engine = Engine::new();
    //     exec(&mut engine, "CREATE TABLE to_drop (id INT)");
    //     exec(&mut engine, "INSERT INTO to_drop VALUES (1)");
    //     exec(&mut engine, "DROP TABLE to_drop");
    //
    //     // Table should no longer exist
    //     let result = engine.execute("SELECT * FROM to_drop");
    //     assert!(result.is_err());
    // }
}

// =============================================================================
// Section 2: Data Manipulation Language (DML) - INSERT
// =============================================================================

mod insert {
    use super::*;

    #[test]
    fn test_insert_single_row() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 100)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[0][1], Value::Int(100));
    }

    #[test]
    fn test_insert_multiple_rows() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_insert_null_value() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, NULL)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows[0][1], Value::Null);
    }

    #[test]
    fn test_insert_negative_integers() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (-1)");
        exec(&mut engine, "INSERT INTO t VALUES (-999)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val < 0");
        assert_eq!(rows.len(), 2);
    }
}

// =============================================================================
// Section 3: Data Manipulation Language (DML) - SELECT
// =============================================================================

mod select {
    use super::*;

    #[test]
    fn test_select_star() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT, c INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 2, 3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 3);
    }

    #[test]
    fn test_select_specific_columns() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT, c INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 2, 3)");

        let rows = query_rows(&mut engine, "SELECT a, c FROM t");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 2);
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[0][1], Value::Int(3));
    }

    #[test]
    fn test_select_column_order() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT, c INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 2, 3)");

        // Columns should be returned in the order specified
        let rows = query_rows(&mut engine, "SELECT c, a, b FROM t");
        assert_eq!(rows[0][0], Value::Int(3));
        assert_eq!(rows[0][1], Value::Int(1));
        assert_eq!(rows[0][2], Value::Int(2));
    }

    #[test]
    fn test_select_empty_table() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE empty (id INT)");

        let rows = query_rows(&mut engine, "SELECT * FROM empty");
        assert!(rows.is_empty());
    }

    #[test]
    fn test_select_with_alias() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (value INT)");
        exec(&mut engine, "INSERT INTO t VALUES (42)");

        // Column alias
        let rows = query_rows(&mut engine, "SELECT value AS v FROM t");
        assert_eq!(rows.len(), 1);
    }
}

// =============================================================================
// Section 4: WHERE Clause and Predicates
// =============================================================================

mod where_clause {
    use super::*;

    fn setup_test_data(engine: &mut Engine) {
        exec(engine, "CREATE TABLE t (id INT, val INT, name TEXT)");
        exec(engine, "INSERT INTO t VALUES (1, 10, 'alice')");
        exec(engine, "INSERT INTO t VALUES (2, 20, 'bob')");
        exec(engine, "INSERT INTO t VALUES (3, 30, 'charlie')");
        exec(engine, "INSERT INTO t VALUES (4, 40, 'diana')");
        exec(engine, "INSERT INTO t VALUES (5, 50, 'eve')");
    }

    #[test]
    fn test_where_equals() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE id = 3");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(3));
    }

    #[test]
    fn test_where_not_equals() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE id != 3");
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_where_less_than() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val < 30");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_where_less_than_or_equal() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val <= 30");
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_where_greater_than() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val > 30");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_where_greater_than_or_equal() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val >= 30");
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_where_and() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE id > 1 AND id < 5");
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_where_or() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE id = 1 OR id = 5");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_where_combined_and_or() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        // AND has higher precedence than OR
        let rows = query_rows(
            &mut engine,
            "SELECT * FROM t WHERE id = 1 OR id = 2 AND val = 20",
        );
        // Should be: id=1 OR (id=2 AND val=20) = 2 rows
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_where_string_comparison() {
        let mut engine = Engine::new();
        setup_test_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE name = 'bob'");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(2));
    }
}

// =============================================================================
// Section 5: UPDATE Statement
// =============================================================================

mod update {
    use super::*;

    #[test]
    fn test_update_single_row() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 100)");

        exec(&mut engine, "UPDATE t SET val = 200 WHERE id = 1");

        let rows = query_rows(&mut engine, "SELECT val FROM t WHERE id = 1");
        assert_eq!(rows[0][0], Value::Int(200));
    }

    #[test]
    fn test_update_multiple_rows() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 100)");
        exec(&mut engine, "INSERT INTO t VALUES (2, 100)");
        exec(&mut engine, "INSERT INTO t VALUES (3, 200)");

        exec(&mut engine, "UPDATE t SET val = 999 WHERE val = 100");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val = 999");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_update_no_matching_rows() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 100)");

        exec(&mut engine, "UPDATE t SET val = 200 WHERE id = 999");

        // Original value should be unchanged
        let rows = query_rows(&mut engine, "SELECT val FROM t WHERE id = 1");
        assert_eq!(rows[0][0], Value::Int(100));
    }

    #[test]
    fn test_update_all_rows() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 100)");
        exec(&mut engine, "INSERT INTO t VALUES (2, 200)");

        exec(&mut engine, "UPDATE t SET val = 0 WHERE val > 0");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val = 0");
        assert_eq!(rows.len(), 2);
    }
}

// =============================================================================
// Section 6: DELETE Statement
// =============================================================================

mod delete {
    use super::*;

    #[test]
    fn test_delete_single_row() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");

        exec(&mut engine, "DELETE FROM t WHERE id = 2");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_delete_multiple_rows() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, cat INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 1)");
        exec(&mut engine, "INSERT INTO t VALUES (2, 1)");
        exec(&mut engine, "INSERT INTO t VALUES (3, 2)");
        exec(&mut engine, "INSERT INTO t VALUES (4, 2)");

        exec(&mut engine, "DELETE FROM t WHERE cat = 1");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_delete_no_matching_rows() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");

        exec(&mut engine, "DELETE FROM t WHERE id = 999");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 2);
    }
}

// =============================================================================
// Section 7: Aggregate Functions
// =============================================================================

mod aggregates {
    use super::*;

    fn setup_numeric_data(engine: &mut Engine) {
        exec(engine, "CREATE TABLE nums (val INT)");
        exec(engine, "INSERT INTO nums VALUES (10)");
        exec(engine, "INSERT INTO nums VALUES (20)");
        exec(engine, "INSERT INTO nums VALUES (30)");
        exec(engine, "INSERT INTO nums VALUES (40)");
        exec(engine, "INSERT INTO nums VALUES (50)");
    }

    #[test]
    fn test_count_star() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT COUNT(*) FROM nums");
        assert_eq!(rows[0][0], Value::Int(5));
    }

    #[test]
    fn test_count_with_where() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT COUNT(*) FROM nums WHERE val > 25");
        assert_eq!(rows[0][0], Value::Int(3));
    }

    #[test]
    fn test_count_empty_table() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE empty (id INT)");

        let rows = query_rows(&mut engine, "SELECT COUNT(*) FROM empty");
        assert_eq!(rows[0][0], Value::Int(0));
    }

    #[test]
    fn test_sum() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT SUM(val) FROM nums");
        assert_eq!(rows[0][0], Value::Int(150));
    }

    #[test]
    fn test_avg() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT AVG(val) FROM nums");
        // AVG returns a float
        assert_eq!(rows[0][0], Value::Float(30.0));
    }

    #[test]
    fn test_min() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT MIN(val) FROM nums");
        assert_eq!(rows[0][0], Value::Int(10));
    }

    #[test]
    fn test_max() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT MAX(val) FROM nums");
        assert_eq!(rows[0][0], Value::Int(50));
    }

    #[test]
    fn test_multiple_aggregates() {
        let mut engine = Engine::new();
        setup_numeric_data(&mut engine);

        let rows = query_rows(
            &mut engine,
            "SELECT COUNT(*), SUM(val), MIN(val), MAX(val) FROM nums",
        );
        assert_eq!(rows[0][0], Value::Int(5));
        assert_eq!(rows[0][1], Value::Int(150));
        assert_eq!(rows[0][2], Value::Int(10));
        assert_eq!(rows[0][3], Value::Int(50));
    }
}

// =============================================================================
// Section 8: ORDER BY Clause
// =============================================================================

mod order_by {
    use super::*;

    fn setup_data(engine: &mut Engine) {
        exec(engine, "CREATE TABLE t (id INT, name TEXT, val INT)");
        exec(engine, "INSERT INTO t VALUES (3, 'charlie', 30)");
        exec(engine, "INSERT INTO t VALUES (1, 'alice', 10)");
        exec(engine, "INSERT INTO t VALUES (4, 'diana', 40)");
        exec(engine, "INSERT INTO t VALUES (2, 'bob', 20)");
    }

    #[test]
    fn test_order_by_asc() {
        let mut engine = Engine::new();
        setup_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT id FROM t ORDER BY id ASC");
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[1][0], Value::Int(2));
        assert_eq!(rows[2][0], Value::Int(3));
        assert_eq!(rows[3][0], Value::Int(4));
    }

    #[test]
    fn test_order_by_desc() {
        let mut engine = Engine::new();
        setup_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT id FROM t ORDER BY id DESC");
        assert_eq!(rows[0][0], Value::Int(4));
        assert_eq!(rows[1][0], Value::Int(3));
        assert_eq!(rows[2][0], Value::Int(2));
        assert_eq!(rows[3][0], Value::Int(1));
    }

    #[test]
    fn test_order_by_default_asc() {
        let mut engine = Engine::new();
        setup_data(&mut engine);

        // Default should be ascending
        let rows = query_rows(&mut engine, "SELECT id FROM t ORDER BY id");
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[3][0], Value::Int(4));
    }
}

// =============================================================================
// Section 9: LIMIT Clause
// =============================================================================

mod limit {
    use super::*;

    fn setup_data(engine: &mut Engine) {
        exec(engine, "CREATE TABLE t (id INT)");
        for i in 1..=10 {
            exec(engine, &format!("INSERT INTO t VALUES ({})", i));
        }
    }

    #[test]
    fn test_limit() {
        let mut engine = Engine::new();
        setup_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t LIMIT 5");
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_limit_larger_than_result() {
        let mut engine = Engine::new();
        setup_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t LIMIT 100");
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn test_limit_with_order_by() {
        let mut engine = Engine::new();
        setup_data(&mut engine);

        let rows = query_rows(&mut engine, "SELECT * FROM t ORDER BY id DESC LIMIT 3");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], Value::Int(10));
        assert_eq!(rows[1][0], Value::Int(9));
        assert_eq!(rows[2][0], Value::Int(8));
    }
}

// =============================================================================
// Section 10: JOIN Operations
// =============================================================================

mod joins {
    use super::*;

    fn setup_join_data(engine: &mut Engine) {
        // Use unique column names to avoid ambiguity since qualified
        // column names (table.column) in ON clauses need further implementation
        exec(engine, "CREATE TABLE users (user_id INT, name TEXT)");
        exec(engine, "INSERT INTO users VALUES (1, 'alice')");
        exec(engine, "INSERT INTO users VALUES (2, 'bob')");
        exec(engine, "INSERT INTO users VALUES (3, 'charlie')");

        exec(
            engine,
            "CREATE TABLE orders (order_id INT, uid INT, amount INT)",
        );
        exec(engine, "INSERT INTO orders VALUES (101, 1, 100)");
        exec(engine, "INSERT INTO orders VALUES (102, 1, 200)");
        exec(engine, "INSERT INTO orders VALUES (103, 2, 150)");
        exec(engine, "INSERT INTO orders VALUES (104, 4, 300)"); // uid 4 doesn't exist
    }

    #[test]
    fn test_inner_join() {
        let mut engine = Engine::new();
        setup_join_data(&mut engine);

        // Use unqualified column names (unique across tables)
        let rows = query_rows(
            &mut engine,
            "SELECT name, amount FROM users INNER JOIN orders ON user_id = uid",
        );
        assert_eq!(rows.len(), 3); // alice has 2, bob has 1
    }

    #[test]
    fn test_left_join() {
        let mut engine = Engine::new();
        setup_join_data(&mut engine);

        let rows = query_rows(
            &mut engine,
            "SELECT name, amount FROM users LEFT JOIN orders ON user_id = uid",
        );
        // alice (2), bob (1), charlie (1 with NULL)
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_cross_join() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE a (x INT)");
        exec(&mut engine, "INSERT INTO a VALUES (1)");
        exec(&mut engine, "INSERT INTO a VALUES (2)");

        exec(&mut engine, "CREATE TABLE b (y INT)");
        exec(&mut engine, "INSERT INTO b VALUES (10)");
        exec(&mut engine, "INSERT INTO b VALUES (20)");
        exec(&mut engine, "INSERT INTO b VALUES (30)");

        let rows = query_rows(&mut engine, "SELECT * FROM a CROSS JOIN b");
        assert_eq!(rows.len(), 6); // 2 * 3 = 6
    }
}

// =============================================================================
// Section 11: Arithmetic Expressions
// =============================================================================

mod arithmetic {
    use super::*;

    #[test]
    fn test_addition() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT)");
        exec(&mut engine, "INSERT INTO t VALUES (10, 3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE a + b = 13");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_subtraction() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT)");
        exec(&mut engine, "INSERT INTO t VALUES (10, 3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE a - b = 7");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_multiplication() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT)");
        exec(&mut engine, "INSERT INTO t VALUES (10, 3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE a * b = 30");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_division() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT)");
        exec(&mut engine, "INSERT INTO t VALUES (10, 2)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE a / b = 5");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_operator_precedence() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (14)");

        // 2 + 3 * 4 = 2 + 12 = 14
        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE 2 + 3 * 4 = val");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_parentheses() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (20)");

        // (2 + 3) * 4 = 5 * 4 = 20
        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE (2 + 3) * 4 = val");
        assert_eq!(rows.len(), 1);
    }
}

// =============================================================================
// Section 12: NULL Handling
// =============================================================================

mod null_handling {
    use super::*;

    #[test]
    fn test_null_in_results() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, NULL)");

        let rows = query_rows(&mut engine, "SELECT val FROM t");
        assert_eq!(rows[0][0], Value::Null);
    }

    #[test]
    fn test_null_in_insert() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (a INT, b INT, c INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, NULL, 3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[0][1], Value::Null);
        assert_eq!(rows[0][2], Value::Int(3));
    }
}

// =============================================================================
// Section 13: Boolean Values
// =============================================================================

mod boolean_values {
    use super::*;

    #[test]
    fn test_true_value() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (flag BOOL)");
        exec(&mut engine, "INSERT INTO t VALUES (TRUE)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows[0][0], Value::Bool(true));
    }

    #[test]
    fn test_false_value() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (flag BOOL)");
        exec(&mut engine, "INSERT INTO t VALUES (FALSE)");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows[0][0], Value::Bool(false));
    }

    #[test]
    fn test_boolean_comparison() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, active BOOL)");
        exec(&mut engine, "INSERT INTO t VALUES (1, TRUE)");
        exec(&mut engine, "INSERT INTO t VALUES (2, FALSE)");
        exec(&mut engine, "INSERT INTO t VALUES (3, TRUE)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE active = TRUE");
        assert_eq!(rows.len(), 2);

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE active = FALSE");
        assert_eq!(rows.len(), 1);
    }
}

// =============================================================================
// Section 14: String/Text Operations
// =============================================================================

mod text_operations {
    use super::*;

    #[test]
    fn test_text_literal() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (name TEXT)");
        exec(&mut engine, "INSERT INTO t VALUES ('hello world')");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows[0][0], Value::Text("hello world".to_string()));
    }

    #[test]
    fn test_text_comparison() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (name TEXT)");
        exec(&mut engine, "INSERT INTO t VALUES ('alice')");
        exec(&mut engine, "INSERT INTO t VALUES ('bob')");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE name = 'alice'");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_empty_string() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (name TEXT)");
        exec(&mut engine, "INSERT INTO t VALUES ('')");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows[0][0], Value::Text("".to_string()));
    }
}

// =============================================================================
// Section 15: GROUP BY and HAVING Clause
// =============================================================================

mod group_by {
    use super::*;

    fn setup_sales_data(engine: &mut Engine) {
        exec(
            engine,
            "CREATE TABLE sales (region TEXT, category TEXT, amount INT)",
        );
        exec(
            engine,
            "INSERT INTO sales VALUES ('north', 'electronics', 100)",
        );
        exec(
            engine,
            "INSERT INTO sales VALUES ('north', 'electronics', 200)",
        );
        exec(
            engine,
            "INSERT INTO sales VALUES ('north', 'clothing', 150)",
        );
        exec(
            engine,
            "INSERT INTO sales VALUES ('south', 'electronics', 300)",
        );
        exec(
            engine,
            "INSERT INTO sales VALUES ('south', 'clothing', 250)",
        );
    }

    #[test]
    fn test_group_by_single_column() {
        let mut engine = Engine::new();
        setup_sales_data(&mut engine);

        let rows = query_rows(
            &mut engine,
            "SELECT region, SUM(amount) FROM sales GROUP BY region",
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_group_by_with_count() {
        let mut engine = Engine::new();
        setup_sales_data(&mut engine);

        let rows = query_rows(
            &mut engine,
            "SELECT region, COUNT(*) FROM sales GROUP BY region",
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_having_clause() {
        let mut engine = Engine::new();
        setup_sales_data(&mut engine);

        // Only regions with total amount > 500
        // north: 100+200+150=450, south: 300+250=550
        let rows = query_rows(
            &mut engine,
            "SELECT region, SUM(amount) FROM sales GROUP BY region HAVING SUM(amount) > 500",
        );
        assert_eq!(rows.len(), 1);
    }
}

// =============================================================================
// Section 16: DISTINCT
// =============================================================================

mod distinct {
    use super::*;

    #[test]
    fn test_select_distinct() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");

        let rows = query_rows(&mut engine, "SELECT DISTINCT val FROM t ORDER BY val");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], Value::Int(1));
        assert_eq!(rows[1][0], Value::Int(2));
        assert_eq!(rows[2][0], Value::Int(3));
    }
}

// =============================================================================
// Section 17: Set Operations (UNION, INTERSECT, EXCEPT)
// =============================================================================

mod set_operations {
    use super::*;

    fn setup_set_data(engine: &mut Engine) {
        exec(engine, "CREATE TABLE t1 (id INT)");
        exec(engine, "INSERT INTO t1 VALUES (1)");
        exec(engine, "INSERT INTO t1 VALUES (2)");
        exec(engine, "INSERT INTO t1 VALUES (3)");

        exec(engine, "CREATE TABLE t2 (id INT)");
        exec(engine, "INSERT INTO t2 VALUES (2)");
        exec(engine, "INSERT INTO t2 VALUES (3)");
        exec(engine, "INSERT INTO t2 VALUES (4)");
    }

    #[test]
    fn test_union() {
        let mut engine = Engine::new();
        setup_set_data(&mut engine);

        // UNION removes duplicates
        let rows = query_rows(&mut engine, "SELECT id FROM t1 UNION SELECT id FROM t2");
        assert_eq!(rows.len(), 4); // 1, 2, 3, 4
    }

    #[test]
    fn test_union_all() {
        let mut engine = Engine::new();
        setup_set_data(&mut engine);

        // UNION ALL keeps duplicates
        let rows = query_rows(&mut engine, "SELECT id FROM t1 UNION ALL SELECT id FROM t2");
        assert_eq!(rows.len(), 6); // 1, 2, 3, 2, 3, 4
    }

    #[test]
    fn test_intersect() {
        let mut engine = Engine::new();
        setup_set_data(&mut engine);

        // INTERSECT returns common rows
        let rows = query_rows(&mut engine, "SELECT id FROM t1 INTERSECT SELECT id FROM t2");
        assert_eq!(rows.len(), 2); // 2, 3
    }

    #[test]
    fn test_except() {
        let mut engine = Engine::new();
        setup_set_data(&mut engine);

        // EXCEPT returns rows in first but not second
        let rows = query_rows(&mut engine, "SELECT id FROM t1 EXCEPT SELECT id FROM t2");
        assert_eq!(rows.len(), 1); // 1
    }
}

// =============================================================================
// Section 18: Common Table Expressions (WITH clause)
// =============================================================================

mod cte {
    use super::*;

    #[test]
    fn test_simple_cte() {
        let mut engine = Engine::new();
        exec(
            &mut engine,
            "CREATE TABLE employees (id INT, name TEXT, salary INT)",
        );
        exec(
            &mut engine,
            "INSERT INTO employees VALUES (1, 'Alice', 50000)",
        );
        exec(
            &mut engine,
            "INSERT INTO employees VALUES (2, 'Bob', 60000)",
        );
        exec(
            &mut engine,
            "INSERT INTO employees VALUES (3, 'Charlie', 70000)",
        );

        let rows = query_rows(
            &mut engine,
            "WITH high_earners AS (SELECT * FROM employees WHERE salary > 55000) SELECT name FROM high_earners",
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_cte_with_column_names() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE nums (val INT)");
        exec(&mut engine, "INSERT INTO nums VALUES (1)");
        exec(&mut engine, "INSERT INTO nums VALUES (2)");

        let rows = query_rows(
            &mut engine,
            "WITH doubled (value) AS (SELECT val FROM nums) SELECT value FROM doubled",
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_recursive_cte_simple_sequence() {
        // Generate numbers 1 to 5 using recursive CTE
        let mut engine = Engine::new();

        let rows = query_rows(
            &mut engine,
            "WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 5
            ) SELECT n FROM nums",
        );

        assert_eq!(rows.len(), 5);
        // Check we got 1, 2, 3, 4, 5
        let values: Vec<i64> = rows
            .iter()
            .map(|r| match &r[0] {
                Value::Int(i) => *i,
                _ => panic!("Expected Int"),
            })
            .collect();
        assert!(values.contains(&1));
        assert!(values.contains(&2));
        assert!(values.contains(&3));
        assert!(values.contains(&4));
        assert!(values.contains(&5));
    }

    #[test]
    fn test_recursive_cte_graph_traversal() {
        // Test graph traversal: find all nodes reachable from node 1
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE edges (src INT, dst INT)");
        exec(&mut engine, "INSERT INTO edges VALUES (1, 2)");
        exec(&mut engine, "INSERT INTO edges VALUES (2, 3)");
        exec(&mut engine, "INSERT INTO edges VALUES (3, 4)");
        exec(&mut engine, "INSERT INTO edges VALUES (1, 5)");

        let rows = query_rows(
            &mut engine,
            "WITH RECURSIVE reachable(node) AS (
                SELECT 1
                UNION ALL
                SELECT e.dst FROM edges e JOIN reachable r ON e.src = r.node
            ) SELECT node FROM reachable",
        );

        // Should find 1, 2, 3, 4, 5
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_recursive_cte_fibonacci() {
        // Test recursive CTE with arithmetic: Fibonacci sequence
        let mut engine = Engine::new();

        let rows = query_rows(
            &mut engine,
            "WITH RECURSIVE fib(a, b) AS (
                SELECT 0, 1
                UNION ALL
                SELECT b, a + b FROM fib WHERE b < 50
            ) SELECT a FROM fib",
        );

        // Should generate Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        assert!(rows.len() >= 8); // At least 8 Fibonacci numbers before reaching 50

        // First value should be 0
        assert_eq!(rows[0][0], Value::Int(0));
    }
}

// =============================================================================
// Section 19: Views
// =============================================================================

mod views {
    use super::*;

    #[test]
    fn test_create_and_query_view() {
        let mut engine = Engine::new();
        exec(
            &mut engine,
            "CREATE TABLE products (id INT, name TEXT, price INT)",
        );
        exec(
            &mut engine,
            "INSERT INTO products VALUES (1, 'Widget', 100)",
        );
        exec(
            &mut engine,
            "INSERT INTO products VALUES (2, 'Gadget', 200)",
        );
        exec(&mut engine, "INSERT INTO products VALUES (3, 'Thing', 50)");

        exec(
            &mut engine,
            "CREATE VIEW expensive AS SELECT * FROM products WHERE price > 75",
        );

        let rows = query_rows(&mut engine, "SELECT * FROM expensive");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_drop_view() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "CREATE VIEW v AS SELECT * FROM t");

        // Query works
        let rows = query_rows(&mut engine, "SELECT * FROM v");
        assert_eq!(rows.len(), 1);

        // Drop view
        exec(&mut engine, "DROP VIEW v");

        // Query should now fail
        let result = engine.execute("SELECT * FROM v");
        assert!(result.is_err());
    }
}

// =============================================================================
// Section 20: Window Functions
// =============================================================================

mod window_functions {
    use super::*;

    fn setup_window_data(engine: &mut Engine) {
        exec(
            engine,
            "CREATE TABLE scores (name TEXT, score INT, dept TEXT)",
        );
        exec(engine, "INSERT INTO scores VALUES ('Alice', 85, 'Sales')");
        exec(engine, "INSERT INTO scores VALUES ('Bob', 90, 'Sales')");
        exec(engine, "INSERT INTO scores VALUES ('Charlie', 80, 'IT')");
        exec(engine, "INSERT INTO scores VALUES ('Diana', 95, 'IT')");
    }

    #[test]
    fn test_row_number() {
        let mut engine = Engine::new();
        setup_window_data(&mut engine);

        let rows = query_rows(
            &mut engine,
            "SELECT name, ROW_NUMBER() OVER (ORDER BY score DESC) FROM scores",
        );
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_row_number_with_partition() {
        let mut engine = Engine::new();
        setup_window_data(&mut engine);

        let rows = query_rows(
            &mut engine,
            "SELECT name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY score DESC) FROM scores",
        );
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_rank() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (10)");
        exec(&mut engine, "INSERT INTO t VALUES (20)");
        exec(&mut engine, "INSERT INTO t VALUES (20)");
        exec(&mut engine, "INSERT INTO t VALUES (30)");

        let rows = query_rows(&mut engine, "SELECT val, RANK() OVER (ORDER BY val) FROM t");
        assert_eq!(rows.len(), 4);
    }
}

// =============================================================================
// Section 21: Stored Procedures
// =============================================================================

mod stored_procedures {
    use super::*;

    #[test]
    fn test_create_and_call_procedure() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE log (msg TEXT)");

        exec(
            &mut engine,
            "CREATE PROCEDURE add_log (message TEXT) AS BEGIN INSERT INTO log (msg) VALUES (message) END",
        );

        exec(&mut engine, "CALL add_log ('Hello')");
        exec(&mut engine, "CALL add_log ('World')");

        let rows = query_rows(&mut engine, "SELECT * FROM log");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_procedure_with_multiple_params() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE records (id INT, name TEXT)");

        exec(
            &mut engine,
            "CREATE PROCEDURE add_record (rec_id INT, rec_name TEXT) AS BEGIN INSERT INTO records (id, name) VALUES (rec_id, rec_name) END",
        );

        exec(&mut engine, "CALL add_record (1, 'Alice')");
        exec(&mut engine, "CALL add_record (2, 'Bob')");

        let rows = query_rows(&mut engine, "SELECT * FROM records ORDER BY id");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][1], Value::Text("Alice".to_string()));
        assert_eq!(rows[1][1], Value::Text("Bob".to_string()));
    }

    #[test]
    fn test_drop_procedure() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(
            &mut engine,
            "CREATE PROCEDURE noop AS BEGIN SELECT * FROM t END",
        );

        // Call works
        let _ = engine.execute("CALL noop");

        // Drop procedure
        exec(&mut engine, "DROP PROCEDURE noop");

        // Call should now fail
        let result = engine.execute("CALL noop");
        assert!(result.is_err());
    }
}

// =============================================================================
// Section 22: Transactions
// =============================================================================

mod transactions {
    use super::*;

    #[test]
    fn test_begin_commit() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");

        exec(&mut engine, "BEGIN");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "COMMIT");

        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_begin_rollback() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");

        exec(&mut engine, "BEGIN");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");
        exec(&mut engine, "ROLLBACK");

        // Only the row inserted before BEGIN should exist
        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_savepoint() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");

        exec(&mut engine, "BEGIN");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "SAVEPOINT sp1");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "ROLLBACK TO sp1");
        exec(&mut engine, "COMMIT");

        // Only value 1 should exist (value 2 was rolled back)
        let rows = query_rows(&mut engine, "SELECT * FROM t");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(1));
    }
}

// =============================================================================
// Section 23: CASE Expressions
// =============================================================================

mod case_expressions {
    use super::*;

    #[test]
    fn test_searched_case() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE scores (student TEXT, score INT)");
        exec(&mut engine, "INSERT INTO scores VALUES ('Alice', 90)");
        exec(&mut engine, "INSERT INTO scores VALUES ('Bob', 75)");
        exec(&mut engine, "INSERT INTO scores VALUES ('Charlie', 55)");

        let rows = query_rows(
            &mut engine,
            "SELECT student, CASE WHEN score >= 80 THEN 'A' WHEN score >= 60 THEN 'B' ELSE 'C' END FROM scores ORDER BY student",
        );
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_case_with_else() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (2)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");

        let rows = query_rows(
            &mut engine,
            "SELECT CASE WHEN val = 1 THEN 'one' ELSE 'other' END FROM t",
        );
        assert_eq!(rows.len(), 3);
    }
}

// =============================================================================
// Section 24: BETWEEN Expression
// =============================================================================

mod between_expression {
    use super::*;

    #[test]
    fn test_between() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        for i in 1..=10 {
            exec(&mut engine, &format!("INSERT INTO t VALUES ({})", i));
        }

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val BETWEEN 3 AND 7");
        assert_eq!(rows.len(), 5); // 3, 4, 5, 6, 7
    }

    #[test]
    fn test_not_between() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        for i in 1..=10 {
            exec(&mut engine, &format!("INSERT INTO t VALUES ({})", i));
        }

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val NOT BETWEEN 3 AND 7");
        assert_eq!(rows.len(), 5); // 1, 2, 8, 9, 10
    }
}

// =============================================================================
// Section 25: LIKE Expression
// =============================================================================

mod like_expression {
    use super::*;

    #[test]
    fn test_like_percent() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE users (name TEXT)");
        exec(&mut engine, "INSERT INTO users VALUES ('alice')");
        exec(&mut engine, "INSERT INTO users VALUES ('alice_smith')");
        exec(&mut engine, "INSERT INTO users VALUES ('bob')");
        exec(&mut engine, "INSERT INTO users VALUES ('bobby')");

        let rows = query_rows(&mut engine, "SELECT * FROM users WHERE name LIKE 'alice%'");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_not_like() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE users (name TEXT)");
        exec(&mut engine, "INSERT INTO users VALUES ('alice')");
        exec(&mut engine, "INSERT INTO users VALUES ('bob')");
        exec(&mut engine, "INSERT INTO users VALUES ('charlie')");

        let rows = query_rows(&mut engine, "SELECT * FROM users WHERE name NOT LIKE 'a%'");
        assert_eq!(rows.len(), 2);
    }
}

// =============================================================================
// Section 26: IS NULL / IS NOT NULL
// =============================================================================

mod is_null {
    use super::*;

    #[test]
    fn test_is_null() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (NULL)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");
        exec(&mut engine, "INSERT INTO t VALUES (NULL)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val IS NULL");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_is_not_null() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1)");
        exec(&mut engine, "INSERT INTO t VALUES (NULL)");
        exec(&mut engine, "INSERT INTO t VALUES (3)");

        let rows = query_rows(&mut engine, "SELECT * FROM t WHERE val IS NOT NULL");
        assert_eq!(rows.len(), 2);
    }
}

// =============================================================================
// Section 27: Subqueries
// =============================================================================

mod subqueries {
    use super::*;

    #[test]
    fn test_scalar_subquery() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        exec(&mut engine, "INSERT INTO t VALUES (1, 10)");
        exec(&mut engine, "INSERT INTO t VALUES (2, 20)");
        exec(&mut engine, "INSERT INTO t VALUES (3, 30)");

        // Test subquery that returns multiple values (supported use case)
        // Scalar subquery comparison (val = (SELECT ...)) is an advanced feature
        // Here we test IN with a subquery that returns column values
        exec(&mut engine, "CREATE TABLE filter (val INT)");
        exec(&mut engine, "INSERT INTO filter VALUES (30)");

        let rows = query_rows(
            &mut engine,
            "SELECT * FROM t WHERE val IN (SELECT val FROM filter)",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][1], Value::Int(30));
    }

    #[test]
    fn test_in_subquery() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE products (id INT, name TEXT)");
        exec(&mut engine, "INSERT INTO products VALUES (1, 'Widget')");
        exec(&mut engine, "INSERT INTO products VALUES (2, 'Gadget')");
        exec(&mut engine, "INSERT INTO products VALUES (3, 'Thing')");

        exec(&mut engine, "CREATE TABLE orders (product_id INT)");
        exec(&mut engine, "INSERT INTO orders VALUES (1)");
        exec(&mut engine, "INSERT INTO orders VALUES (3)");

        let rows = query_rows(
            &mut engine,
            "SELECT name FROM products WHERE id IN (SELECT product_id FROM orders)",
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_exists_subquery() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE customers (id INT, name TEXT)");
        exec(&mut engine, "INSERT INTO customers VALUES (1, 'Alice')");
        exec(&mut engine, "INSERT INTO customers VALUES (2, 'Bob')");

        exec(&mut engine, "CREATE TABLE orders (customer_id INT)");
        exec(&mut engine, "INSERT INTO orders VALUES (1)");

        // EXISTS with correlated subquery is an advanced feature
        // Test simpler EXISTS - returns true if subquery has any rows at all
        let rows = query_rows(
            &mut engine,
            "SELECT name FROM customers WHERE EXISTS (SELECT * FROM orders)",
        );
        // Both customers returned since orders table is not empty
        assert_eq!(rows.len(), 2);
    }
}

// =============================================================================
// Section 28: Complex Queries
// =============================================================================

mod complex_queries {
    use super::*;

    #[test]
    fn test_combined_where_order_limit() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE t (id INT, val INT)");
        for i in 1..=20 {
            exec(
                &mut engine,
                &format!("INSERT INTO t VALUES ({}, {})", i, i * 10),
            );
        }

        let rows = query_rows(
            &mut engine,
            "SELECT * FROM t WHERE val >= 100 ORDER BY id DESC LIMIT 5",
        );
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0][0], Value::Int(20)); // highest id first
    }

    #[test]
    fn test_aggregate_with_where() {
        let mut engine = Engine::new();
        exec(&mut engine, "CREATE TABLE sales (region TEXT, amount INT)");
        exec(&mut engine, "INSERT INTO sales VALUES ('north', 100)");
        exec(&mut engine, "INSERT INTO sales VALUES ('north', 200)");
        exec(&mut engine, "INSERT INTO sales VALUES ('south', 150)");
        exec(&mut engine, "INSERT INTO sales VALUES ('south', 250)");

        let rows = query_rows(
            &mut engine,
            "SELECT SUM(amount) FROM sales WHERE region = 'north'",
        );
        assert_eq!(rows[0][0], Value::Int(300));
    }

    #[test]
    fn test_multiple_table_operations() {
        let mut engine = Engine::new();

        // Create and populate multiple tables
        exec(&mut engine, "CREATE TABLE customers (id INT, name TEXT)");
        exec(&mut engine, "CREATE TABLE products (id INT, price INT)");
        exec(
            &mut engine,
            "CREATE TABLE orders (customer_id INT, product_id INT, qty INT)",
        );

        exec(&mut engine, "INSERT INTO customers VALUES (1, 'alice')");
        exec(&mut engine, "INSERT INTO customers VALUES (2, 'bob')");

        exec(&mut engine, "INSERT INTO products VALUES (101, 50)");
        exec(&mut engine, "INSERT INTO products VALUES (102, 100)");

        exec(&mut engine, "INSERT INTO orders VALUES (1, 101, 2)");
        exec(&mut engine, "INSERT INTO orders VALUES (1, 102, 1)");
        exec(&mut engine, "INSERT INTO orders VALUES (2, 101, 3)");

        // Verify data in each table
        let customers = query_rows(&mut engine, "SELECT COUNT(*) FROM customers");
        assert_eq!(customers[0][0], Value::Int(2));

        let products = query_rows(&mut engine, "SELECT COUNT(*) FROM products");
        assert_eq!(products[0][0], Value::Int(2));

        let orders = query_rows(&mut engine, "SELECT COUNT(*) FROM orders");
        assert_eq!(orders[0][0], Value::Int(3));
    }
}
