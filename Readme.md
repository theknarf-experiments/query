# Query

An embeddable query engine in Rust supporting both SQL and Datalog.

## Overview

Query is designed for applications that need to provide query capabilities over their own data. Whether you're building a database, an application with user-queryable data, or a tool that needs to reason over structured information, Query provides the query layer while you control the storage.

**Key design principles:**
- **Embeddable**: Integrate directly into your Rust application
- **Pluggable storage**: Implement the `StorageEngine` trait to use your own storage backend
- **Dual query languages**: SQL for familiar relational queries, Datalog for recursive and logic-based reasoning

## Features

### SQL Support
- `SELECT` with columns, expressions, aliases, `*`
- `WHERE`, `ORDER BY`, `LIMIT`, `OFFSET`
- `GROUP BY` with `HAVING`
- Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
- `JOIN` (inner, left, right, cross)
- Subqueries and CTEs (`WITH`)
- `INSERT`, `UPDATE`, `DELETE`
- `CREATE TABLE`, `CREATE INDEX`, `CREATE VIEW`
- Transactions (`BEGIN`, `COMMIT`, `ROLLBACK`)
- Triggers and stored procedures

### Datalog Support
- Recursive queries with automatic stratification
- Negation with stratified semantics
- Semi-naive evaluation for efficient fixpoint computation
- Seamless integration with SQL tables

### Data Types
- `INTEGER`, `FLOAT`, `TEXT`, `BOOLEAN`
- `DATE`, `TIME`, `TIMESTAMP`
- `JSON`

## Usage

### Basic Example

```rust
use query::Engine;

fn main() -> Result<(), query::ExecError> {
    let mut engine = Engine::new();

    // Create a table
    engine.execute("CREATE TABLE users (id INT, name TEXT, age INT)")?;

    // Insert data
    engine.execute("INSERT INTO users VALUES (1, 'Alice', 30)")?;
    engine.execute("INSERT INTO users VALUES (2, 'Bob', 25)")?;

    // Query with SQL
    let result = engine.execute("SELECT * FROM users WHERE age > 26")?;
    println!("{:?}", result);

    Ok(())
}
```

### Datalog Queries

```rust
use query::Engine;

fn main() -> Result<(), query::ExecError> {
    let mut engine = Engine::new();

    // Create base relation
    engine.execute("CREATE TABLE edge (src INT, dst INT)")?;
    engine.execute("INSERT INTO edge VALUES (1, 2), (2, 3), (3, 4)")?;

    // Compute transitive closure with Datalog
    let result = engine.execute_datalog(r#"
        path(X, Y) :- edge(X, Y).
        path(X, Z) :- path(X, Y), edge(Y, Z).
        ?- path(1, X).
    "#)?;

    println!("{:?}", result);  // All nodes reachable from 1
    Ok(())
}
```

### Custom Storage Backend

Implement the `StorageEngine` trait to use your own storage:

```rust
use query::logical::{StorageEngine, TableSchema, Row, StorageResult};

struct MyStorage {
    // Your storage implementation
}

impl StorageEngine for MyStorage {
    fn create_table(&mut self, name: &str, schema: TableSchema) -> StorageResult<()> {
        // ...
    }

    fn insert(&mut self, table: &str, row: Row) -> StorageResult<()> {
        // ...
    }

    // ... other methods
}
```

## Architecture

```
query/                 # Top-level crate, re-exports everything
  sql-parser/          # SQL lexing and parsing (Chumsky + Ariadne)
  sql-planner/         # SQL AST to logical plan
  sql-engine/          # Query execution
  datalog-parser/      # Datalog parsing
  datalog-planner/     # Datalog to IR, stratification
  datalog-eval/        # Datalog evaluation (semi-naive)
  storage/             # StorageEngine trait + MemoryEngine
  logical/             # Shared logical types
  json-value/          # JSON value type
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## License

MIT
