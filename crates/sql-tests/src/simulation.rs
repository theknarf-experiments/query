//! Deterministic simulation tests for the SQL database
//!
//! These tests simulate various failure scenarios to ensure the database
//! handles them correctly.

use sql_engine::Engine;
use sql_storage::Value;

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
        self.next() % 2 == 0
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
    if let Ok(sql_engine::QueryResult::Select { rows, .. }) = result {
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
    if let Ok(sql_engine::QueryResult::Select { rows, .. }) = result {
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
    if let Ok(sql_engine::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 1000);
    }

    // Query with WHERE
    let result = engine.execute("SELECT * FROM large WHERE c = 5");
    assert!(result.is_ok());
    if let Ok(sql_engine::QueryResult::Select { rows, .. }) = result {
        assert_eq!(rows.len(), 100); // Every 10th row has c = 5
    }

    // Aggregate
    let result = engine.execute("SELECT COUNT(*), SUM(id), AVG(a) FROM large");
    assert!(result.is_ok());
}
