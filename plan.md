# Trigger System Implementation Plan

## Overview

Implement PostgreSQL-style triggers where triggers reference functions stored in metadata tables. The logical layer enforces triggers automatically on insert/update/delete operations, making them work for both SQL and Datalog.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         db crate                                │
│  (top-level, creates Runtime, passes to sql-engine)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      sql-engine crate                           │
│  - Implements SqlRuntime (executes SQL expressions/statements)  │
│  - Passes runtime to logical layer operations                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      logical crate                              │
│  - Defines Runtime trait                                        │
│  - TriggerDefinition (table, timing, events, function_name)     │
│  - On insert/update/delete: lookup triggers, call runtime       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      storage crate                              │
│  - _functions metadata table                                    │
│  - _triggers metadata table                                     │
│  - StorageEngine methods for CRUD on these tables               │
└─────────────────────────────────────────────────────────────────┘
```

## Metadata Tables

### `_functions` table
| Column   | Type | Description                          |
|----------|------|--------------------------------------|
| name     | TEXT | Primary key, function name           |
| params   | TEXT | JSON array of parameter definitions  |
| body     | TEXT | Function body (SQL statements)       |
| language | TEXT | Language identifier (e.g., "sql")    |

### `_triggers` table
| Column        | Type | Description                              |
|---------------|------|------------------------------------------|
| name          | TEXT | Primary key, trigger name                |
| table_name    | TEXT | Table the trigger is attached to         |
| timing        | TEXT | "BEFORE" or "AFTER"                      |
| events        | TEXT | JSON array: ["INSERT"], ["UPDATE"], etc. |
| function_name | TEXT | References _functions.name               |

## Runtime Trait

Defined in the `logical` crate:

```rust
/// Event that fired the trigger
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
}

/// Timing of trigger execution
pub enum TriggerTiming {
    Before,
    After,
}

/// Result of executing a trigger function
pub enum TriggerResult {
    /// Continue with the (possibly modified) row
    Proceed(Option<Row>),
    /// Skip this row (BEFORE trigger returned NULL equivalent)
    Skip,
    /// Abort the operation with an error message
    Abort(String),
}

/// Context passed to trigger functions
pub struct TriggerContext<'a> {
    pub event: TriggerEvent,
    pub timing: TriggerTiming,
    pub table: &'a str,
    pub old_row: Option<&'a Row>,
    pub new_row: Option<&'a Row>,
    pub column_names: &'a [String],
}

/// Runtime trait for executing trigger functions
pub trait Runtime {
    fn execute_trigger_function(
        &self,
        function_name: &str,
        context: TriggerContext,
        storage: &dyn StorageEngine,
    ) -> Result<TriggerResult, RuntimeError>;
}
```

## Implementation Steps

### Phase 1: Storage Layer
- [ ] Add `_functions` and `_triggers` table creation on engine init
- [ ] Add methods to StorageEngine trait:
  - `create_function(name, params, body, language)`
  - `drop_function(name)`
  - `get_function(name) -> Option<FunctionDef>`
  - `create_trigger(name, table, timing, events, function_name)`
  - `drop_trigger(name)`
  - `get_triggers_for_table(table) -> Vec<TriggerDef>`
- [ ] Implement in MemoryEngine

### Phase 2: Logical Layer
- [ ] Define `TriggerEvent`, `TriggerTiming`, `TriggerResult`, `TriggerContext`
- [ ] Define `Runtime` trait
- [ ] Define `RuntimeError` type
- [ ] Create wrapper functions that check triggers:
  - `insert_with_triggers(storage, runtime, table, row)`
  - `update_with_triggers(storage, runtime, table, predicate, updater)`
  - `delete_with_triggers(storage, runtime, table, predicate)`
- [ ] Update `DatalogContext::insert` to optionally use triggers

### Phase 3: SQL Engine
- [ ] Implement `SqlRuntime` that can execute function bodies
- [ ] Update executor to use `insert_with_triggers` etc.
- [ ] Handle CREATE FUNCTION / DROP FUNCTION statements
- [ ] Handle CREATE TRIGGER / DROP TRIGGER statements
- [ ] Remove old in-memory trigger storage from Engine

### Phase 4: Integration
- [ ] Update `db` crate to wire everything together
- [ ] Add tests for SQL triggers (existing behavior)
- [ ] Add tests for Datalog triggering SQL triggers
- [ ] Add tests for trigger ordering (alphabetical)
- [ ] Add tests for BEFORE trigger row modification
- [ ] Add tests for BEFORE trigger skip (return NULL)

## Trigger Execution Flow

### INSERT with BEFORE trigger
1. Caller: `insert_with_triggers(storage, runtime, "users", row)`
2. Logical: Query `_triggers` for table="users", event="INSERT", timing="BEFORE"
3. Logical: Sort triggers alphabetically by name
4. Logical: For each trigger, call `runtime.execute_trigger_function(...)`
5. Logical: If result is `Skip`, return without inserting
6. Logical: If result is `Proceed(Some(modified_row))`, use modified row
7. Logical: If result is `Abort(msg)`, return error
8. Logical: Call `storage.insert(table, final_row)`
9. Logical: Query `_triggers` for timing="AFTER", execute those
10. Return success

### UPDATE/DELETE follow similar pattern

## Open Questions

1. **Row-level vs Statement-level**: Start with row-level only (simpler)
2. **FOR EACH ROW**: Implicit for now, can add statement-level later
3. **Recursive triggers**: Should triggers trigger other triggers? (Start with no)
4. **Transaction semantics**: Triggers run in same transaction (implicit with current design)
