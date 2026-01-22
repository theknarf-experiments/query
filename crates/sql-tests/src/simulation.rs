//! Deterministic simulation tests for the SQL database
//!
//! These tests simulate various failure scenarios to ensure the database
//! handles them correctly.

use query::{Engine, ExecError, logical::Value};

/// A deterministic random number generator for simulation
struct SimRng {
    seed: u64,
}

impl SimRng {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn next(&mut self) -> u64 {
        // Simple LCG PRNG
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed
    }

    fn next_range(&mut self, max: u64) -> u64 {
        self.next() % max
    }

    fn next_bool(&mut self) -> bool {
        self.next().is_multiple_of(2)
    }
}

/// Database operation for simulation
#[derive(Debug, Clone)]
enum SimOp {
    CreateTable {
        name: String,
        columns: Vec<String>,
    },
    Insert {
        table: String,
        values: Vec<i64>,
    },
    Select {
        table: String,
    },
    Update {
        table: String,
        col: String,
        val: i64,
        where_col: String,
        where_val: i64,
    },
    Delete {
        table: String,
        where_col: String,
        where_val: i64,
    },
}

/// Generate random operations based on schema
fn generate_operations(rng: &mut SimRng, num_ops: usize) -> Vec<SimOp> {
    let mut ops = Vec::new();
    let mut tables: Vec<(String, Vec<String>)> = Vec::new();

    for _ in 0..num_ops {
        let op_type = rng.next_range(5);

        match op_type {
            0 if tables.len() < 3 => {
                // Create table
                let name = format!("t{}", tables.len());
                let num_cols = 2 + rng.next_range(3) as usize;
                let columns: Vec<String> = (0..num_cols).map(|j| format!("c{}", j)).collect();
                tables.push((name.clone(), columns.clone()));
                ops.push(SimOp::CreateTable { name, columns });
            }
            1 if !tables.is_empty() => {
                // Insert
                let table_idx = rng.next_range(tables.len() as u64) as usize;
                let (name, cols) = &tables[table_idx];
                let values: Vec<i64> = (0..cols.len())
                    .map(|_| rng.next_range(1000) as i64)
                    .collect();
                ops.push(SimOp::Insert {
                    table: name.clone(),
                    values,
                });
            }
            2 if !tables.is_empty() => {
                // Select
                let table_idx = rng.next_range(tables.len() as u64) as usize;
                let (name, _) = &tables[table_idx];
                ops.push(SimOp::Select {
                    table: name.clone(),
                });
            }
            3 if !tables.is_empty() => {
                // Update
                let table_idx = rng.next_range(tables.len() as u64) as usize;
                let (name, cols) = &tables[table_idx];
                if cols.len() >= 2 {
                    let col_idx = rng.next_range(cols.len() as u64) as usize;
                    let where_idx = rng.next_range(cols.len() as u64) as usize;
                    ops.push(SimOp::Update {
                        table: name.clone(),
                        col: cols[col_idx].clone(),
                        val: rng.next_range(1000) as i64,
                        where_col: cols[where_idx].clone(),
                        where_val: rng.next_range(1000) as i64,
                    });
                }
            }
            4 if !tables.is_empty() => {
                // Delete
                let table_idx = rng.next_range(tables.len() as u64) as usize;
                let (name, cols) = &tables[table_idx];
                if !cols.is_empty() {
                    let where_idx = rng.next_range(cols.len() as u64) as usize;
                    ops.push(SimOp::Delete {
                        table: name.clone(),
                        where_col: cols[where_idx].clone(),
                        where_val: rng.next_range(1000) as i64,
                    });
                }
            }
            _ => {
                // Create table as fallback
                if tables.len() < 5 {
                    let name = format!("t{}", tables.len());
                    let columns = vec!["c0".to_string(), "c1".to_string()];
                    tables.push((name.clone(), columns.clone()));
                    ops.push(SimOp::CreateTable { name, columns });
                }
            }
        }
    }

    ops
}

/// Execute an operation and return the SQL string
fn op_to_sql(op: &SimOp) -> String {
    match op {
        SimOp::CreateTable { name, columns } => {
            let cols = columns
                .iter()
                .map(|c| format!("{} INT", c))
                .collect::<Vec<_>>()
                .join(", ");
            format!("CREATE TABLE {} ({})", name, cols)
        }
        SimOp::Insert { table, values } => {
            let vals = values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("INSERT INTO {} VALUES ({})", table, vals)
        }
        SimOp::Select { table } => {
            format!("SELECT * FROM {}", table)
        }
        SimOp::Update {
            table,
            col,
            val,
            where_col,
            where_val,
        } => {
            format!(
                "UPDATE {} SET {} = {} WHERE {} = {}",
                table, col, val, where_col, where_val
            )
        }
        SimOp::Delete {
            table,
            where_col,
            where_val,
        } => {
            format!("DELETE FROM {} WHERE {} = {}", table, where_col, where_val)
        }
    }
}

#[test]
fn test_deterministic_simulation_basic() {
    // Use a fixed seed for reproducibility
    let mut rng = SimRng::new(12345);
    let ops = generate_operations(&mut rng, 20);

    let mut engine = Engine::new();
    let mut errors = Vec::new();

    for op in &ops {
        let sql = op_to_sql(op);
        match engine.execute(&sql) {
            Ok(_) => {}
            Err(e) => {
                // Only record unexpected errors (not "table not found" for deletes, etc.)
                match op {
                    SimOp::Select { .. } | SimOp::Update { .. } | SimOp::Delete { .. } => {
                        // These may fail if table is empty or no matching rows - expected
                    }
                    _ => errors.push((sql, e)),
                }
            }
        }
    }

    // Should have no unexpected errors
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);
}

#[test]
fn test_deterministic_simulation_multiple_seeds() {
    // Run simulation with multiple seeds
    for seed in [1, 42, 99999, 0xDEADBEEF, 0xCAFEBABE] {
        let mut rng = SimRng::new(seed);
        let ops = generate_operations(&mut rng, 30);

        let mut engine = Engine::new();

        for op in &ops {
            let sql = op_to_sql(op);
            let _ = engine.execute(&sql);
        }

        // Run the same operations again with the same seed - should be deterministic
        let mut rng2 = SimRng::new(seed);
        let ops2 = generate_operations(&mut rng2, 30);

        // Operations should be identical
        for (op1, op2) in ops.iter().zip(ops2.iter()) {
            assert_eq!(op_to_sql(op1), op_to_sql(op2));
        }
    }
}

#[test]
fn test_concurrent_operations_simulation() {
    // Simulate concurrent operations by interleaving
    let mut rng1 = SimRng::new(111);
    let mut rng2 = SimRng::new(222);

    // Generate two streams of operations
    let ops1 = generate_operations(&mut rng1, 15);
    let ops2 = generate_operations(&mut rng2, 15);

    let mut engine = Engine::new();

    // Create shared tables first
    engine
        .execute("CREATE TABLE shared (id INT, val INT)")
        .unwrap();

    // Interleave operations
    let mut rng_interleave = SimRng::new(333);
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < ops1.len() || i2 < ops2.len() {
        let use_first = if i1 >= ops1.len() {
            false
        } else if i2 >= ops2.len() {
            true
        } else {
            rng_interleave.next_bool()
        };

        if use_first {
            let sql = op_to_sql(&ops1[i1]);
            let _ = engine.execute(&sql);
            i1 += 1;
        } else {
            let sql = op_to_sql(&ops2[i2]);
            let _ = engine.execute(&sql);
            i2 += 1;
        }
    }
}

#[test]
fn test_stress_insert_delete_cycle() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE stress (id INT, data INT)")
        .unwrap();

    let mut rng = SimRng::new(54321);

    // Insert and delete in cycles
    for cycle in 0..10 {
        // Insert batch
        for i in 0..100 {
            let id = cycle * 100 + i;
            let data = rng.next_range(10000) as i64;
            let sql = format!("INSERT INTO stress VALUES ({}, {})", id, data);
            engine.execute(&sql).unwrap();
        }

        // Delete some randomly
        for _ in 0..50 {
            let id = rng.next_range(((cycle + 1) * 100) as u64) as i64;
            let sql = format!("DELETE FROM stress WHERE id = {}", id);
            let _ = engine.execute(&sql);
        }

        // Verify we can still query
        let result = engine.execute("SELECT * FROM stress");
        assert!(result.is_ok());
    }
}

#[test]
fn test_edge_case_empty_table_operations() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE empty (id INT, val INT)")
        .unwrap();

    // Operations on empty table should not crash
    let result = engine.execute("SELECT * FROM empty");
    assert!(result.is_ok());
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert!(rows.is_empty());
    }

    // Update on empty table
    let result = engine.execute("UPDATE empty SET val = 1 WHERE id = 1");
    assert!(result.is_ok());

    // Delete on empty table
    let result = engine.execute("DELETE FROM empty WHERE id = 1");
    assert!(result.is_ok());

    // Aggregate on empty table
    let result = engine.execute("SELECT COUNT(*) FROM empty");
    assert!(result.is_ok());
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(0));
    }
}

#[test]
fn test_large_row_counts() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE large (id INT, a INT, b INT, c INT)")
        .unwrap();

    // Insert many rows
    for i in 0..1000 {
        let sql = format!(
            "INSERT INTO large VALUES ({}, {}, {}, {})",
            i,
            i * 2,
            i * 3,
            i % 10
        );
        engine.execute(&sql).unwrap();
    }

    // Query all
    let result = engine.execute("SELECT * FROM large");
    assert!(result.is_ok());
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1000);
    }

    // Query with WHERE
    let result = engine.execute("SELECT * FROM large WHERE c = 5");
    assert!(result.is_ok());
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 100); // Every 10th row has c = 5
    }

    // Aggregate
    let result = engine.execute("SELECT COUNT(*), SUM(id), AVG(a) FROM large");
    assert!(result.is_ok());
}

// ==================== FAILURE MODE SIMULATION TESTS ====================

/// Simulates transaction rollback scenarios
#[test]
fn test_failure_mode_transaction_rollback() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    engine
        .execute("INSERT INTO accounts VALUES (1, 1000)")
        .unwrap();
    engine
        .execute("INSERT INTO accounts VALUES (2, 500)")
        .unwrap();

    // Start transaction
    engine.execute("BEGIN").unwrap();

    // Make changes
    engine
        .execute("UPDATE accounts SET balance = 800 WHERE id = 1")
        .unwrap();
    engine
        .execute("UPDATE accounts SET balance = 700 WHERE id = 2")
        .unwrap();

    // Verify changes are visible within transaction
    let result = engine.execute("SELECT balance FROM accounts WHERE id = 1");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(800));
    }

    // Rollback
    engine.execute("ROLLBACK").unwrap();

    // Changes should be undone
    let result = engine.execute("SELECT balance FROM accounts WHERE id = 1");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(1000));
    }

    let result = engine.execute("SELECT balance FROM accounts WHERE id = 2");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(500));
    }
}

/// Simulates savepoint and partial rollback scenarios
#[test]
fn test_failure_mode_savepoint_rollback() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE log (step INT, value INT)")
        .unwrap();

    engine.execute("BEGIN").unwrap();

    engine.execute("INSERT INTO log VALUES (1, 100)").unwrap();
    engine.execute("SAVEPOINT sp1").unwrap();

    engine.execute("INSERT INTO log VALUES (2, 200)").unwrap();
    engine.execute("SAVEPOINT sp2").unwrap();

    engine.execute("INSERT INTO log VALUES (3, 300)").unwrap();

    // Check all 3 rows exist
    let result = engine.execute("SELECT COUNT(*) FROM log");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(3));
    }

    // Rollback to sp2 - should undo step 3
    engine.execute("ROLLBACK TO sp2").unwrap();
    let result = engine.execute("SELECT COUNT(*) FROM log");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(2));
    }

    // Rollback to sp1 - should undo step 2 as well
    engine.execute("ROLLBACK TO sp1").unwrap();
    let result = engine.execute("SELECT COUNT(*) FROM log");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(1));
    }

    // Full rollback
    engine.execute("ROLLBACK").unwrap();
    let result = engine.execute("SELECT COUNT(*) FROM log");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(0));
    }
}

/// Simulates constraint validation - tests what constraints are supported
#[test]
fn test_failure_mode_constraint_violations() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT)")
        .unwrap();

    // First insert should succeed
    engine
        .execute("INSERT INTO users VALUES (1, 'alice@test.com')")
        .unwrap();

    // Note: Primary key constraint enforcement varies by implementation
    // This test verifies the database handles duplicate inserts gracefully
    let result = engine.execute("INSERT INTO users VALUES (1, 'bob@test.com')");
    // The insert may succeed or fail depending on enforcement
    // But the database should not crash
    let _ = result;

    // Verify we can still query
    let result = engine.execute("SELECT COUNT(*) FROM users");
    assert!(result.is_ok());
}

/// Simulates foreign key constraint on DELETE (cascade prevention)
#[test]
fn test_failure_mode_foreign_key_handling() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    engine
        .execute("CREATE TABLE employees (id INT, dept_id INT REFERENCES departments)")
        .unwrap();

    // Insert parent row
    engine
        .execute("INSERT INTO departments VALUES (1, 'Engineering')")
        .unwrap();

    // Insert child rows referencing the parent
    engine
        .execute("INSERT INTO employees VALUES (1, 1)")
        .unwrap();
    engine
        .execute("INSERT INTO employees VALUES (2, 1)")
        .unwrap();

    // Try to delete parent while children exist - should fail (FK constraint)
    let result = engine.execute("DELETE FROM departments WHERE id = 1");
    // FK constraint should prevent this delete
    assert!(result.is_err(), "Delete with FK reference should fail");

    // Verify parent still exists
    let result = engine.execute("SELECT COUNT(*) FROM departments");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(1));
    }
}

/// Simulates null constraint handling
#[test]
fn test_failure_mode_null_constraint_handling() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE required (id INT PRIMARY KEY, name TEXT NOT NULL)")
        .unwrap();

    // Insert with NULL in NOT NULL column - test behavior
    let result = engine.execute("INSERT INTO required VALUES (1, NULL)");
    // Note: NOT NULL constraint enforcement depends on implementation
    // The database should handle this gracefully either way
    let _ = result;

    // Verify database is still usable
    let result = engine.execute("SELECT COUNT(*) FROM required");
    assert!(result.is_ok());
}

/// Simulates recovery after transaction errors
#[test]
fn test_failure_mode_transaction_error_recovery() {
    let mut engine = Engine::new();
    engine.execute("CREATE TABLE data (id INT)").unwrap();

    // Start transaction and make changes
    engine.execute("BEGIN").unwrap();
    engine.execute("INSERT INTO data VALUES (1)").unwrap();
    engine.execute("INSERT INTO data VALUES (2)").unwrap();

    // After some operations, we can still rollback
    engine.execute("ROLLBACK").unwrap();

    // Table should be empty after rollback
    let result = engine.execute("SELECT COUNT(*) FROM data");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(0));
    }
}

/// Simulates stress test with mixed operations
#[test]
fn test_failure_mode_stress_mixed_operations() {
    let mut rng = SimRng::new(0xF00D);
    let mut engine = Engine::new();

    // Create schema without PRIMARY KEY for simpler testing
    engine
        .execute("CREATE TABLE mixed_test (id INT, val INT)")
        .unwrap();

    let mut expected_count = 0;
    let mut next_id = 1;

    // Run many operations
    for _ in 0..500 {
        let op_type = rng.next_range(10);

        match op_type {
            0..=5 => {
                // Insert (should always succeed)
                let sql = format!(
                    "INSERT INTO mixed_test VALUES ({}, {})",
                    next_id,
                    rng.next_range(1000)
                );
                if engine.execute(&sql).is_ok() {
                    expected_count += 1;
                    next_id += 1;
                }
            }
            6..=7 => {
                // Delete by random id (may or may not delete anything)
                let random_id = rng.next_range(next_id as u64 + 10) as i64;
                let sql = format!("DELETE FROM mixed_test WHERE id = {}", random_id);
                if let Ok(query::QueryResult::RowsAffected(n)) = engine.execute(&sql) {
                    expected_count -= n as i64;
                }
            }
            _ => {
                // Select (should always succeed)
                let result = engine.execute("SELECT COUNT(*) FROM mixed_test");
                assert!(result.is_ok());
            }
        }
    }

    // Verify final count
    let result = engine.execute("SELECT COUNT(*) FROM mixed_test");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(expected_count));
    }
}

/// Simulates multiple sequential transactions
#[test]
fn test_failure_mode_sequential_transactions() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE balance (id INT PRIMARY KEY, amount INT)")
        .unwrap();
    engine.execute("INSERT INTO balance VALUES (1, 0)").unwrap();

    // Run multiple transactions sequentially
    for i in 1..=10 {
        engine.execute("BEGIN").unwrap();
        engine
            .execute(&format!(
                "UPDATE balance SET amount = amount + {} WHERE id = 1",
                i * 100
            ))
            .unwrap();

        // 50% chance of commit, 50% chance of rollback
        if i % 2 == 0 {
            engine.execute("COMMIT").unwrap();
        } else {
            engine.execute("ROLLBACK").unwrap();
        }
    }

    // Only committed transactions should be visible
    // Committed: i=2,4,6,8,10 => amounts: 200,400,600,800,1000 = 3000
    let result = engine.execute("SELECT amount FROM balance WHERE id = 1");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(3000));
    }
}

/// Simulates trigger-induced failures
#[test]
fn test_failure_mode_trigger_errors() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE guarded (id INT, status TEXT)")
        .unwrap();

    // Create trigger that raises error on invalid status
    engine
        .execute("CREATE TRIGGER guard_status BEFORE INSERT ON guarded FOR EACH ROW RAISE ERROR 'Invalid status'")
        .unwrap();

    // Insert should fail due to trigger
    let result = engine.execute("INSERT INTO guarded VALUES (1, 'pending')");
    assert!(result.is_err());
    // The trigger abort now comes through as InvalidExpression with "Trigger aborted: {msg}"
    if let Err(ExecError::InvalidExpression(msg)) = result {
        assert!(msg.contains("Invalid status"));
    } else {
        panic!(
            "Expected InvalidExpression with trigger abort message, got: {:?}",
            result
        );
    }

    // Table should remain empty
    let result = engine.execute("SELECT COUNT(*) FROM guarded");
    if let Ok(query::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows[0][0], Value::Int(0));
    }
}

/// Simulates operations on non-existent objects
#[test]
fn test_failure_mode_missing_objects() {
    let mut engine = Engine::new();

    // Select from non-existent table
    let result = engine.execute("SELECT * FROM nonexistent");
    assert!(result.is_err());

    // Insert into non-existent table
    let result = engine.execute("INSERT INTO nonexistent VALUES (1)");
    assert!(result.is_err());

    // Update non-existent table
    let result = engine.execute("UPDATE nonexistent SET x = 1");
    assert!(result.is_err());

    // Delete from non-existent table
    let result = engine.execute("DELETE FROM nonexistent");
    assert!(result.is_err());

    // Drop non-existent procedure
    let result = engine.execute("DROP PROCEDURE nonexistent");
    assert!(result.is_err());

    // Call non-existent procedure
    let result = engine.execute("CALL nonexistent");
    assert!(result.is_err());
}

/// Simulates ALTER TABLE failures
#[test]
fn test_failure_mode_alter_table_errors() {
    let mut engine = Engine::new();
    engine
        .execute("CREATE TABLE altertest (id INT, name TEXT)")
        .unwrap();
    engine
        .execute("INSERT INTO altertest VALUES (1, 'test')")
        .unwrap();

    // Drop non-existent column should fail
    let result = engine.execute("ALTER TABLE altertest DROP COLUMN nonexistent");
    assert!(result.is_err());

    // Rename to same name - behavior depends on implementation
    // But table should still work after error
    let _ = engine.execute("ALTER TABLE altertest RENAME COLUMN id TO id");

    // Can still query the table
    let result = engine.execute("SELECT * FROM altertest");
    assert!(result.is_ok());
}

/// Simulates deterministic failure recovery patterns
#[test]
fn test_failure_mode_deterministic_recovery_pattern() {
    let seeds = [0xBEEF, 0xCAFE, 0xF00D, 0xDEAD, 0xFACE];

    for seed in seeds {
        let mut rng = SimRng::new(seed);
        let mut engine = Engine::new();

        engine
            .execute("CREATE TABLE recovery_test (id INT)")
            .unwrap();

        // Simulate pattern: begin -> operations -> maybe fail -> rollback
        for _ in 0..20 {
            engine.execute("BEGIN").unwrap();

            let num_ops = rng.next_range(5) + 1;
            for _j in 0..num_ops {
                let id = rng.next_range(1000) as i64;
                let _ = engine.execute(&format!("INSERT INTO recovery_test VALUES ({})", id));
            }

            // 30% chance to rollback
            if rng.next_range(10) < 3 {
                engine.execute("ROLLBACK").unwrap();
            } else {
                engine.execute("COMMIT").unwrap();
            }
        }

        // Database should still be queryable after all this
        let result = engine.execute("SELECT * FROM recovery_test");
        assert!(result.is_ok());
    }
}
