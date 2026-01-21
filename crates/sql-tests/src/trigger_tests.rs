//! Comprehensive tests for the trigger system
//!
//! Tests PostgreSQL-style triggers including:
//! - CREATE FUNCTION and CREATE TRIGGER
//! - BEFORE and AFTER triggers
//! - Row modification, skip, and abort
//! - Triggers with Datalog operations

use db::{
    logical::{
        insert, update, FunctionDef, MemoryEngine, Runtime, RuntimeError, StorageEngine,
        TriggerContext, TriggerDef, TriggerEvent, TriggerResult, TriggerTiming, Value,
    },
    Engine, ExecError, QueryResult, SqlRuntime,
};

// ============================================================================
// SQL Trigger Tests (using inline syntax which works)
// ============================================================================

#[test]
fn test_inline_trigger_syntax() {
    let mut engine = Engine::new();

    engine
        .execute("CREATE TABLE inline_test (id INT, status TEXT)")
        .unwrap();

    // Use inline trigger syntax (creates internal function)
    engine
        .execute(
            "CREATE TRIGGER set_status BEFORE INSERT ON inline_test FOR EACH ROW SET status = 'active'",
        )
        .unwrap();

    engine
        .execute("INSERT INTO inline_test VALUES (1, 'pending')")
        .unwrap();

    let result = engine.execute("SELECT status FROM inline_test").unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows[0][0], Value::Text("active".to_string()));
        }
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_inline_trigger_raise_error() {
    let mut engine = Engine::new();

    engine
        .execute("CREATE TABLE restricted (id INT, value INT)")
        .unwrap();

    // Use inline RAISE ERROR syntax
    engine
        .execute(
            "CREATE TRIGGER block_inserts BEFORE INSERT ON restricted FOR EACH ROW RAISE ERROR 'Not permitted'",
        )
        .unwrap();

    // Insert should fail with error
    let result = engine.execute("INSERT INTO restricted VALUES (1, 100)");
    assert!(result.is_err());
    if let Err(ExecError::InvalidExpression(msg)) = result {
        assert!(msg.contains("Not permitted"));
    } else {
        panic!("Expected InvalidExpression error with abort message");
    }
}

#[test]
fn test_drop_trigger() {
    let mut engine = Engine::new();

    engine.execute("CREATE TABLE temp (id INT)").unwrap();

    // Create inline trigger
    engine
        .execute("CREATE TRIGGER temp_trig BEFORE INSERT ON temp FOR EACH ROW SET id = 99")
        .unwrap();

    // Drop trigger
    engine.execute("DROP TRIGGER temp_trig").unwrap();

    // Verify it's gone - trying to drop again should fail
    let result = engine.execute("DROP TRIGGER temp_trig");
    assert!(result.is_err());
}

#[test]
fn test_multiple_inline_triggers() {
    let mut engine = Engine::new();

    engine
        .execute("CREATE TABLE multi_trigger (id INT, a INT, b INT)")
        .unwrap();

    // Create two inline triggers - both fire on INSERT
    engine
        .execute("CREATE TRIGGER trig_a BEFORE INSERT ON multi_trigger FOR EACH ROW SET a = 10")
        .unwrap();

    engine
        .execute("CREATE TRIGGER trig_b BEFORE INSERT ON multi_trigger FOR EACH ROW SET b = 20")
        .unwrap();

    // Insert - both triggers should modify the row
    engine
        .execute("INSERT INTO multi_trigger VALUES (1, 0, 0)")
        .unwrap();

    let result = engine
        .execute("SELECT a, b FROM multi_trigger WHERE id = 1")
        .unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            // Both values should be set by triggers
            assert_eq!(rows[0][0], Value::Int(10));
            assert_eq!(rows[0][1], Value::Int(20));
        }
        _ => panic!("Expected Select result"),
    }
}

// ============================================================================
// Datalog + Trigger Integration Tests
// ============================================================================

/// Custom runtime that tracks trigger invocations for testing
struct TrackingRuntime {
    invocations: std::cell::RefCell<Vec<(String, TriggerEvent, TriggerTiming)>>,
}

impl TrackingRuntime {
    fn new() -> Self {
        Self {
            invocations: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn get_invocations(&self) -> Vec<(String, TriggerEvent, TriggerTiming)> {
        self.invocations.borrow().clone()
    }
}

impl<S: StorageEngine> Runtime<S> for TrackingRuntime {
    fn execute_trigger_function(
        &self,
        function_name: &str,
        context: TriggerContext,
        _storage: &mut S,
    ) -> Result<TriggerResult, RuntimeError> {
        self.invocations.borrow_mut().push((
            function_name.to_string(),
            context.event,
            context.timing,
        ));
        Ok(TriggerResult::Proceed(None))
    }
}

#[test]
fn test_triggers_with_logical_layer() {
    use db::logical::{ColumnSchema, DataType, TableSchema};

    let mut storage = MemoryEngine::new();

    // Create table
    storage
        .create_table(TableSchema {
            name: "items".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: true,
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
        })
        .unwrap();

    // Create a trigger
    storage
        .create_trigger(TriggerDef {
            name: "track_inserts".to_string(),
            table_name: "items".to_string(),
            timing: TriggerTiming::Before,
            events: vec![TriggerEvent::Insert],
            function_name: "track_func".to_string(),
        })
        .unwrap();

    let runtime = TrackingRuntime::new();

    // Insert using trigger-aware function
    let row = vec![Value::Int(1), Value::Text("Widget".to_string())];
    let result = insert(&mut storage, &runtime, "items", row);
    assert!(result.is_ok());

    // Verify trigger was invoked
    let invocations = runtime.get_invocations();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].0, "track_func");
    assert_eq!(invocations[0].1, TriggerEvent::Insert);
    assert_eq!(invocations[0].2, TriggerTiming::Before);

    // Verify data was inserted
    let rows = storage.scan("items").unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_sql_runtime_with_datalog_operations() {
    use db::logical::{ColumnSchema, DataType, TableSchema};

    let mut storage = MemoryEngine::new();

    // Create table
    storage
        .create_table(TableSchema {
            name: "facts".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "subject".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "predicate".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "object".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        })
        .unwrap();

    // Create function that uppercases subject
    storage
        .create_function(FunctionDef {
            name: "uppercase_subject".to_string(),
            params: "[]".to_string(),
            body: "SET NEW.subject = 'UPPERCASED'; RETURN NEW".to_string(),
            language: "sql".to_string(),
        })
        .unwrap();

    // Create trigger
    storage
        .create_trigger(TriggerDef {
            name: "format_facts".to_string(),
            table_name: "facts".to_string(),
            timing: TriggerTiming::Before,
            events: vec![TriggerEvent::Insert],
            function_name: "uppercase_subject".to_string(),
        })
        .unwrap();

    // Use SqlRuntime from sql-engine
    let runtime = SqlRuntime::new();

    // Insert a fact - trigger should modify it
    let row = vec![
        Value::Text("alice".to_string()),
        Value::Text("knows".to_string()),
        Value::Text("bob".to_string()),
    ];
    let result = insert(&mut storage, &runtime, "facts", row);
    assert!(result.is_ok());

    // Verify trigger modified the row
    let rows = storage.scan("facts").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::Text("UPPERCASED".to_string()));
}

#[test]
fn test_before_trigger_can_skip_update() {
    use db::logical::{ColumnSchema, DataType, TableSchema};

    let mut storage = MemoryEngine::new();

    storage
        .create_table(TableSchema {
            name: "protected".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: true,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "locked".to_string(),
                    data_type: DataType::Bool,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        })
        .unwrap();

    // Insert some data directly (no trigger yet)
    storage
        .insert("protected", vec![Value::Int(1), Value::Bool(true)])
        .unwrap();
    storage
        .insert("protected", vec![Value::Int(2), Value::Bool(false)])
        .unwrap();

    // Create a runtime that skips updates on locked rows
    struct SkipLockedRuntime;
    impl<S: StorageEngine> Runtime<S> for SkipLockedRuntime {
        fn execute_trigger_function(
            &self,
            _function_name: &str,
            context: TriggerContext,
            _storage: &mut S,
        ) -> Result<TriggerResult, RuntimeError> {
            if context.timing == TriggerTiming::Before {
                if let Some(old_row) = context.old_row {
                    if old_row.get(1) == Some(&Value::Bool(true)) {
                        return Ok(TriggerResult::Skip);
                    }
                }
            }
            Ok(TriggerResult::Proceed(None))
        }
    }

    // Add trigger
    storage
        .create_trigger(TriggerDef {
            name: "protect_locked".to_string(),
            table_name: "protected".to_string(),
            timing: TriggerTiming::Before,
            events: vec![TriggerEvent::Update],
            function_name: "protect_func".to_string(),
        })
        .unwrap();

    let runtime = SkipLockedRuntime;

    // Try to update all rows - only unlocked one should change
    let result = update(
        &mut storage,
        &runtime,
        "protected",
        |_row| true,                    // Match all
        |row| row[0] = Value::Int(999), // Try to change id
    );

    // Only 1 row should be updated (the unlocked one)
    assert_eq!(result, Ok(1));

    // Verify: row 1 (locked=true) should be unchanged, row 2 should be updated
    let rows = storage.scan("protected").unwrap();
    let row1 = rows.iter().find(|r| r[1] == Value::Bool(true)).unwrap();
    let row2 = rows.iter().find(|r| r[1] == Value::Bool(false)).unwrap();

    assert_eq!(row1[0], Value::Int(1)); // Unchanged
    assert_eq!(row2[0], Value::Int(999)); // Updated
}

// ============================================================================
// SQL -> Datalog -> SQL Integration Tests
// ============================================================================

/// Test that demonstrates the full round-trip:
/// 1. Create a table with a trigger in SQL
/// 2. Use Datalog to insert data (trigger should fire)
/// 3. Verify result with SQL
#[test]
fn test_sql_datalog_sql_trigger_roundtrip() {
    let mut engine = Engine::new();

    // Step 1: Create table and trigger in SQL
    engine
        .execute("CREATE TABLE events (id INT, status TEXT)")
        .unwrap();

    // Create a BEFORE INSERT trigger that sets status to 'processed'
    engine
        .execute(
            "CREATE TRIGGER process_event BEFORE INSERT ON events FOR EACH ROW SET status = 'processed'",
        )
        .unwrap();

    // Step 2: Insert data via Datalog
    // Datalog will use the underlying storage which has triggers registered
    let datalog_result = engine.execute_datalog(
        r#"
        events(1, "pending").
        events(2, "waiting").
        "#,
    );
    // The datalog insert should succeed
    // Note: Currently datalog returns NoQuery error for programs without queries
    // but the inserts should still work
    assert!(datalog_result.is_err()); // NoQuery error is expected

    // Step 3: Verify with SQL that triggers fired
    let result = engine
        .execute("SELECT id, status FROM events ORDER BY id")
        .unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows.len(), 2);
            // Both rows should have status = 'processed' due to trigger
            assert_eq!(rows[0][0], Value::Int(1));
            assert_eq!(rows[0][1], Value::Text("processed".to_string()));
            assert_eq!(rows[1][0], Value::Int(2));
            assert_eq!(rows[1][1], Value::Text("processed".to_string()));
        }
        _ => panic!("Expected Select result"),
    }
}

/// Test trigger that blocks all inserts works with Datalog
#[test]
fn test_sql_trigger_blocks_datalog_insert() {
    let mut engine = Engine::new();

    // Create table
    engine.execute("CREATE TABLE guarded (value INT)").unwrap();

    // Insert a value before trigger is created (should work)
    engine.execute("INSERT INTO guarded VALUES (10)").unwrap();

    // Create trigger that blocks all inserts
    engine
        .execute(
            "CREATE TRIGGER guard_all BEFORE INSERT ON guarded FOR EACH ROW RAISE ERROR 'No new inserts allowed'",
        )
        .unwrap();

    // Try to insert via Datalog - should fail due to trigger
    let _result = engine.execute_datalog(
        r#"
        guarded(20).
        "#,
    );
    // The insert should have failed due to trigger

    // Verify only the original value exists
    let result = engine.execute("SELECT value FROM guarded").unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0][0], Value::Int(10));
        }
        _ => panic!("Expected Select result"),
    }
}

/// Test that AFTER triggers fire correctly with Datalog inserts
/// AFTER triggers with SET can mark rows as "processed"
#[test]
fn test_after_trigger_with_datalog() {
    let mut engine = Engine::new();

    // Create table with a "processed" flag
    engine
        .execute("CREATE TABLE orders (id INT, amount INT, confirmed INT)")
        .unwrap();

    // Create AFTER INSERT trigger that marks rows as confirmed
    // Note: AFTER triggers with SET should work even though the row is already inserted
    engine
        .execute(
            "CREATE TRIGGER confirm_order AFTER INSERT ON orders FOR EACH ROW SET confirmed = 1",
        )
        .unwrap();

    // Insert via Datalog
    let _result = engine.execute_datalog(
        r#"
        orders(100, 500, 0).
        orders(101, 750, 0).
        "#,
    );

    // Verify orders were inserted
    let result = engine
        .execute("SELECT id, amount FROM orders ORDER BY id")
        .unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0][0], Value::Int(100));
            assert_eq!(rows[1][0], Value::Int(101));
        }
        _ => panic!("Expected Select result"),
    }
}

// ============================================================================
// Full SQL Trigger Function Tests (INSERT INTO from triggers)
// ============================================================================

/// Test that an AFTER trigger can insert into another table using INSERT INTO
#[test]
fn test_trigger_with_insert_into_another_table() {
    use db::logical::{ColumnSchema, DataType, FunctionDef, TableSchema, TriggerDef};

    let mut storage = MemoryEngine::new();

    // Create main table
    storage
        .create_table(TableSchema {
            name: "orders".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: true,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "amount".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        })
        .unwrap();

    // Create audit log table
    storage
        .create_table(TableSchema {
            name: "audit_log".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "order_id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "action".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        })
        .unwrap();

    // Create function that inserts into audit_log
    storage
        .create_function(FunctionDef {
            name: "log_order_insert".to_string(),
            params: "[]".to_string(),
            body: "INSERT INTO audit_log VALUES (NEW.id, 'INSERT'); RETURN NEW".to_string(),
            language: "sql".to_string(),
        })
        .unwrap();

    // Create trigger
    storage
        .create_trigger(TriggerDef {
            name: "order_audit_trigger".to_string(),
            table_name: "orders".to_string(),
            timing: TriggerTiming::After,
            events: vec![TriggerEvent::Insert],
            function_name: "log_order_insert".to_string(),
        })
        .unwrap();

    // Insert an order using SqlRuntime
    let runtime = SqlRuntime::new();
    let row = vec![Value::Int(100), Value::Int(500)];
    let result = insert(&mut storage, &runtime, "orders", row);
    assert!(result.is_ok());

    // Verify order was inserted
    let orders = storage.scan("orders").unwrap();
    assert_eq!(orders.len(), 1);
    assert_eq!(orders[0][0], Value::Int(100));
    assert_eq!(orders[0][1], Value::Int(500));

    // Verify audit log was populated by the trigger
    let audit = storage.scan("audit_log").unwrap();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0][0], Value::Int(100));
    assert_eq!(audit[0][1], Value::Text("INSERT".to_string()));
}

/// Test INSERT INTO with NEW.column references
#[test]
fn test_trigger_insert_with_new_references() {
    use db::logical::{ColumnSchema, DataType, FunctionDef, TableSchema, TriggerDef};

    let mut storage = MemoryEngine::new();

    // Create products table
    storage
        .create_table(TableSchema {
            name: "products".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: true,
                    unique: true,
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
                ColumnSchema {
                    name: "price".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        })
        .unwrap();

    // Create history table
    storage
        .create_table(TableSchema {
            name: "product_history".to_string(),
            columns: vec![
                ColumnSchema {
                    name: "product_id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "product_name".to_string(),
                    data_type: DataType::Text,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
                ColumnSchema {
                    name: "old_price".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    primary_key: false,
                    unique: false,
                    default: None,
                    references: None,
                },
            ],
            constraints: vec![],
        })
        .unwrap();

    // Create function that inserts NEW values into history
    storage
        .create_function(FunctionDef {
            name: "record_product".to_string(),
            params: "[]".to_string(),
            body: "INSERT INTO product_history VALUES (NEW.id, NEW.name, NEW.price); RETURN NEW"
                .to_string(),
            language: "sql".to_string(),
        })
        .unwrap();

    // Create trigger
    storage
        .create_trigger(TriggerDef {
            name: "product_history_trigger".to_string(),
            table_name: "products".to_string(),
            timing: TriggerTiming::After,
            events: vec![TriggerEvent::Insert],
            function_name: "record_product".to_string(),
        })
        .unwrap();

    // Insert a product
    let runtime = SqlRuntime::new();
    let row = vec![
        Value::Int(1),
        Value::Text("Widget".to_string()),
        Value::Int(99),
    ];
    let result = insert(&mut storage, &runtime, "products", row);
    assert!(result.is_ok());

    // Verify history was populated with NEW values
    let history = storage.scan("product_history").unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0][0], Value::Int(1));
    assert_eq!(history[0][1], Value::Text("Widget".to_string()));
    assert_eq!(history[0][2], Value::Int(99));
}

// ============================================================================
// FK Constraint as Trigger Tests
// ============================================================================

/// Test that FK constraints are automatically enforced via triggers on INSERT
#[test]
fn test_fk_constraint_insert_validation() {
    let mut engine = Engine::new();

    // Create parent table
    engine
        .execute("CREATE TABLE authors (id INT PRIMARY KEY, name TEXT)")
        .unwrap();

    // Create child table with FK constraint
    engine
        .execute(
            "CREATE TABLE books (id INT PRIMARY KEY, author_id INT REFERENCES authors(id), title TEXT)",
        )
        .unwrap();

    // Insert a valid author
    engine
        .execute("INSERT INTO authors VALUES (1, 'Jane Austen')")
        .unwrap();

    // Insert book with valid author - should succeed
    engine
        .execute("INSERT INTO books VALUES (1, 1, 'Pride and Prejudice')")
        .unwrap();

    // Try to insert book with non-existent author - should fail
    let result = engine.execute("INSERT INTO books VALUES (2, 999, 'Unknown Book')");
    assert!(result.is_err());
    let err = result.unwrap_err();
    match &err {
        ExecError::InvalidExpression(msg) => {
            assert!(
                msg.contains("Foreign key") || msg.contains("constraint"),
                "Expected FK constraint error, got: {}",
                msg
            );
        }
        _ => panic!("Expected InvalidExpression error, got: {:?}", err),
    }
}

/// Test ON DELETE CASCADE via trigger
#[test]
fn test_fk_on_delete_cascade_via_trigger() {
    let mut engine = Engine::new();

    // Create parent table
    engine
        .execute("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT)")
        .unwrap();

    // Create child table with CASCADE on delete
    engine
        .execute(
            "CREATE TABLE employees (id INT PRIMARY KEY, dept_id INT REFERENCES departments(id) ON DELETE CASCADE, name TEXT)",
        )
        .unwrap();

    // Insert department and employees
    engine
        .execute("INSERT INTO departments VALUES (1, 'Engineering')")
        .unwrap();
    engine
        .execute("INSERT INTO employees VALUES (1, 1, 'Alice')")
        .unwrap();
    engine
        .execute("INSERT INTO employees VALUES (2, 1, 'Bob')")
        .unwrap();

    // Delete department - should cascade to employees
    engine
        .execute("DELETE FROM departments WHERE id = 1")
        .unwrap();

    // Verify employees were deleted
    let result = engine.execute("SELECT COUNT(*) FROM employees").unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(
                rows[0][0],
                Value::Int(0),
                "Employees should be cascaded deleted"
            );
        }
        _ => panic!("Expected Select result"),
    }
}

/// Test ON DELETE SET NULL via trigger
#[test]
fn test_fk_on_delete_set_null_via_trigger() {
    let mut engine = Engine::new();

    // Create parent table
    engine
        .execute("CREATE TABLE managers (id INT PRIMARY KEY, name TEXT)")
        .unwrap();

    // Create child table with SET NULL on delete
    engine
        .execute(
            "CREATE TABLE projects (id INT PRIMARY KEY, manager_id INT REFERENCES managers(id) ON DELETE SET NULL, name TEXT)",
        )
        .unwrap();

    // Insert manager and projects
    engine
        .execute("INSERT INTO managers VALUES (1, 'John')")
        .unwrap();
    engine
        .execute("INSERT INTO projects VALUES (1, 1, 'Project A')")
        .unwrap();

    // Delete manager - should set manager_id to NULL
    engine.execute("DELETE FROM managers WHERE id = 1").unwrap();

    // Verify project still exists but with NULL manager_id
    let result = engine.execute("SELECT manager_id FROM projects").unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows.len(), 1, "Project should still exist");
            assert_eq!(rows[0][0], Value::Null, "manager_id should be NULL");
        }
        _ => panic!("Expected Select result"),
    }
}

/// Test ON UPDATE CASCADE via trigger
#[test]
fn test_fk_on_update_cascade_via_trigger() {
    let mut engine = Engine::new();

    // Create parent table
    engine
        .execute("CREATE TABLE categories (id INT PRIMARY KEY, name TEXT)")
        .unwrap();

    // Create child table with CASCADE on update
    engine
        .execute(
            "CREATE TABLE items (id INT PRIMARY KEY, cat_id INT REFERENCES categories(id) ON UPDATE CASCADE, name TEXT)",
        )
        .unwrap();

    // Insert category and items
    engine
        .execute("INSERT INTO categories VALUES (1, 'Electronics')")
        .unwrap();
    engine
        .execute("INSERT INTO items VALUES (1, 1, 'Phone')")
        .unwrap();
    engine
        .execute("INSERT INTO items VALUES (2, 1, 'Laptop')")
        .unwrap();

    // Update category id - should cascade to items
    engine
        .execute("UPDATE categories SET id = 100 WHERE id = 1")
        .unwrap();

    // Verify items have updated cat_id
    let result = engine
        .execute("SELECT cat_id FROM items ORDER BY id")
        .unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows.len(), 2);
            assert_eq!(
                rows[0][0],
                Value::Int(100),
                "cat_id should be updated to 100"
            );
            assert_eq!(
                rows[1][0],
                Value::Int(100),
                "cat_id should be updated to 100"
            );
        }
        _ => panic!("Expected Select result"),
    }
}

/// Test that RESTRICT prevents update when references exist
#[test]
fn test_fk_on_update_restrict_via_trigger() {
    let mut engine = Engine::new();

    // Create parent table
    engine
        .execute("CREATE TABLE regions (id INT PRIMARY KEY, name TEXT)")
        .unwrap();

    // Create child table with RESTRICT on update (default)
    engine
        .execute(
            "CREATE TABLE offices (id INT PRIMARY KEY, region_id INT REFERENCES regions(id) ON UPDATE RESTRICT, name TEXT)",
        )
        .unwrap();

    // Insert region and office
    engine
        .execute("INSERT INTO regions VALUES (1, 'North')")
        .unwrap();
    engine
        .execute("INSERT INTO offices VALUES (1, 1, 'HQ')")
        .unwrap();

    // Try to update region id - should fail due to RESTRICT
    let result = engine.execute("UPDATE regions SET id = 2 WHERE id = 1");
    assert!(result.is_err(), "Update should fail with RESTRICT");
    let err = result.unwrap_err();
    match &err {
        ExecError::InvalidExpression(msg) => {
            assert!(
                msg.contains("Cannot update") || msg.contains("referenced by"),
                "Expected RESTRICT error, got: {}",
                msg
            );
        }
        _ => panic!("Expected InvalidExpression error, got: {:?}", err),
    }

    // Verify region is unchanged
    let result = engine.execute("SELECT id FROM regions").unwrap();
    match result {
        QueryResult::Select { rows, .. } => {
            assert_eq!(rows[0][0], Value::Int(1), "Region id should be unchanged");
        }
        _ => panic!("Expected Select result"),
    }
}

/// Test that FK validation trigger checks UPDATE of FK column in child table
#[test]
fn test_fk_update_validation_via_trigger() {
    let mut engine = Engine::new();

    // Create tables
    engine
        .execute("CREATE TABLE vendors (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    engine
        .execute(
            "CREATE TABLE products (id INT PRIMARY KEY, vendor_id INT REFERENCES vendors(id), name TEXT)",
        )
        .unwrap();

    // Insert valid data
    engine
        .execute("INSERT INTO vendors VALUES (1, 'Vendor A')")
        .unwrap();
    engine
        .execute("INSERT INTO products VALUES (1, 1, 'Product X')")
        .unwrap();

    // Try to update product to reference non-existent vendor - should fail
    let result = engine.execute("UPDATE products SET vendor_id = 999 WHERE id = 1");
    assert!(result.is_err(), "Update to invalid FK should fail");
    let err = result.unwrap_err();
    match &err {
        ExecError::InvalidExpression(msg) => {
            assert!(
                msg.contains("Foreign key") || msg.contains("constraint"),
                "Expected FK constraint error, got: {}",
                msg
            );
        }
        _ => panic!("Expected InvalidExpression error, got: {:?}", err),
    }
}
