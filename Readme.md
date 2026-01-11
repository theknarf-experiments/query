# SQL (in Rust)

A SQL database implementation in pure Rust with pluggable storage engines.

## Architecture

The project is organized into multiple crates:

- **sql-lexer** - Lexical analysis using Chumsky, tokenizes SQL input
- **sql-parser** - Parses tokens into an Abstract Syntax Tree (AST)
- **sql-planner** - Converts AST into logical query plans
- **sql-storage** - Pluggable storage engine interface with in-memory implementation
- **sql-engine** - Ties everything together to execute SQL queries

## Features

- SQL parsing with Chumsky + Ariadne for error reporting
- Pluggable storage engines via trait
- In-memory storage engine included
- Expression evaluation with proper operator precedence
- Support for common SQL operations

### Supported Statements

- `SELECT` with `*`, columns, expressions, aliases
- `WHERE` filtering with comparison operators
- `ORDER BY` with `ASC`/`DESC`
- `LIMIT` and `OFFSET`
- `INSERT INTO ... VALUES`
- `CREATE TABLE` with column types and constraints

### Data Types

- `INT` / `INTEGER`
- `FLOAT` / `DOUBLE`
- `TEXT` / `VARCHAR`
- `BOOL` / `BOOLEAN`

## Usage

```rust
use sql_engine::Engine;

fn main() {
    let mut engine = Engine::new();

    // Create a table
    engine.execute("CREATE TABLE users (id INT, name TEXT)").unwrap();

    // Insert data
    engine.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();
    engine.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')").unwrap();

    // Query data
    let result = engine.execute("SELECT * FROM users WHERE id = 1").unwrap();
    println!("{:?}", result);
}
```

## Running Tests

```bash
cargo test
```

## Building

```bash
cargo build --release
```
