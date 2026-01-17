# Crate Consolidation Plan

## Goal

Unify SQL and Datalog frontends to:
1. Support both SQL and Datalog as query languages
2. Reuse semi-naive stratified evaluation for SQL recursive CTEs
3. Share optimizations between both query languages

## Current Architecture (11 crates, down from 13)

```
SQL Frontend:
sql-parser ──► sql-planner ──► sql-engine ◄── sql-storage
                    │               │
                    │          sql-tests ◄─── sql-wal
                    │
                    └── (uses datalog-safety for Datalog compilation)

Datalog Frontend (two execution paths):

Path 1 - Unified (new):
datalog-parser ──► sql-planner::datalog ──► sql-engine (Recursive nodes)

Path 2 - Direct (legacy):
datalog-parser ──► datalog-safety ──► datalog-grounding ──► datalog-eval
                        │                   │
                        └── datalog-builtins┘
```

Note: datalog-ast merged into datalog-parser, datalog-core merged into sql-storage.

## Target Architecture (8 crates)

```
sql-parser ────┐
               ├──► planner ──► engine ◄── storage
datalog-parser ┘                  │
                                  ▼
                          sql-tests ◄─── sql-wal
```

### Crate Mapping

| New Crate | Combines | Purpose |
|-----------|----------|---------|
| sql-parser | (unchanged) | SQL text → SQL AST |
| datalog-parser | datalog-ast + datalog-parser | Datalog text → Datalog AST |
| storage | sql-storage + datalog-core | Unified relation/table model |
| planner | sql-planner + datalog-safety + datalog-grounding | AST → LogicalPlan with recursion |
| engine | sql-engine + datalog-eval + datalog-builtins | Plan execution with semi-naive recursion |
| sql-wal | (unchanged) | Write-ahead log |
| sql-tests | (unchanged) | Integration tests |

## Key Design Decisions

### 1. Unified Storage Model

Datalog relations and SQL tables are both sets of tuples.

```rust
// storage/src/lib.rs
pub struct Relation {
    pub schema: Schema,
    pub rows: HashSet<Row>,
    pub indexes: HashMap<String, Index>,
}

pub struct Row {
    pub values: Vec<Value>,
}

pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    // Datalog uses interned symbols - consider Intern<String> variant
}
```

### 2. Extended LogicalPlan with Recursion

```rust
// planner/src/plan.rs
pub enum LogicalPlan {
    // Existing variants...
    Scan { table: String },
    Filter { input: Box<LogicalPlan>, predicate: Expr },
    Projection { input: Box<LogicalPlan>, exprs: Vec<(Expr, Option<String>)> },
    Join { left: Box<LogicalPlan>, right: Box<LogicalPlan>, ... },
    Aggregate { ... },
    SetOperation { ... },

    // NEW: Recursive query (for both Datalog rules and SQL recursive CTEs)
    Recursive {
        /// Name of the recursive relation
        name: String,
        /// Column names/schema
        columns: Vec<String>,
        /// Base case (non-recursive)
        base: Box<LogicalPlan>,
        /// Recursive step (references `name`)
        step: Box<LogicalPlan>,
    },

    // NEW: Stratified execution (for negation)
    Stratify {
        /// Plans to execute in order (each stratum)
        strata: Vec<LogicalPlan>,
    },
}
```

### 3. Datalog Compilation to LogicalPlan

Datalog rules compile to relational algebra:

```
% Datalog rule:
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compiles to LogicalPlan:
Recursive {
    name: "ancestor",
    columns: ["X", "Z"],
    base: Scan { table: "parent" },  // or base facts
    step: Join {
        left: Scan { table: "parent" },
        right: Scan { table: "ancestor" },  // recursive reference
        on: parent.Y = ancestor.X,
    } -> Project [parent.X, ancestor.Z]
}
```

### 4. Semi-Naive Evaluation in Engine

The engine executes `Recursive` nodes using semi-naive evaluation:

```rust
// engine/src/recursive.rs
fn evaluate_recursive(
    name: &str,
    base: &LogicalPlan,
    step: &LogicalPlan,
    ctx: &mut Context,
) -> Result<Relation> {
    // 1. Evaluate base case
    let mut result = evaluate(base, ctx)?;
    let mut delta = result.clone();

    // 2. Iterate until fixpoint
    while !delta.is_empty() {
        // Bind delta to the recursive relation for this iteration
        ctx.bind_delta(name, &delta);

        // Evaluate step using delta (semi-naive: only new facts)
        let new_facts = evaluate(step, ctx)?;

        // Delta = new facts not already in result
        delta = new_facts.difference(&result);

        // Add to result
        result = result.union(&delta);
    }

    Ok(result)
}
```

### 5. Stratification for Negation

Negation requires stratification - compute relations in dependency order:

```rust
// planner/src/stratify.rs
fn stratify(rules: &[Rule]) -> Result<Vec<Vec<Rule>>> {
    // Build dependency graph
    // Detect negative cycles (error: not stratifiable)
    // Topological sort into strata
    // Rules with negation on R must be in higher stratum than R's definition
}
```

## Migration Steps

### Phase 1: Merge datalog-ast into datalog-parser ✓
- [x] Currently datalog-ast is separate
- [x] Fold AST types into datalog-parser (they're tightly coupled)
- [x] Update imports in other datalog crates

### Phase 2: Create unified storage crate ✓
- [x] Merge datalog-core into sql-storage (keeping sql-storage name)
- [x] Port datalog-core's FactDatabase, Substitution, ConstantEnv
- [x] Update all dependent crates to use sql-storage
- [x] Remove datalog-core from workspace

### Phase 3: Extend LogicalPlan with recursion ✓
- [x] Add `Recursive` variant to LogicalPlan (for fixpoint evaluation)
- [x] Add `Stratify` variant for negation handling
- [x] Add `RecursiveRef` variant for referencing recursive relations
- [x] Keep existing sql-planner functionality

### Phase 4: Add Datalog → LogicalPlan compilation ✓
- [x] Create datalog-to-plan module in planner (sql-planner/src/datalog.rs)
- [x] Compile Datalog atoms to Scan nodes
- [x] Compile rule bodies to Join chains with Projection
- [x] Compile recursive rules to Recursive nodes
- [x] Use safety checking from datalog-safety
- [x] Use stratification for negation handling (Stratify nodes)

### Phase 5: Add semi-naive execution to engine ✓
- [x] Add semi-naive evaluation for Recursive nodes
- [x] Execute Recursive nodes with fixpoint iteration (delta-based)
- [x] Execute Stratify nodes in stratum order
- [x] Add RecursiveRef handling via CTE context

### Phase 6: Clean up old crates ✓
- [x] datalog-ast: Removed (merged into datalog-parser in Phase 1)
- [x] datalog-core: Removed (merged into sql-storage in Phase 2)
- [x] datalog-safety: Kept (used by sql-planner for compilation)
- [x] datalog-grounding: Kept (provides alternative eval path)
- [x] datalog-builtins: Kept (provides alternative eval path)
- [x] datalog-eval: Kept (provides alternative eval path via Engine::execute_datalog)

Note: The remaining Datalog crates provide two execution paths:
1. New path: compile_datalog() -> LogicalPlan -> SQL engine (unified)
2. Legacy path: datalog-eval::evaluate() (direct Datalog evaluation)

### Phase 7: Add SQL recursive CTE support ✓
- [x] Parse WITH RECURSIVE in sql-parser (extended CTE query to support UNION)
- [x] Compile recursive CTEs to Recursive LogicalPlan (WithRecursiveCte node)
- [x] Verify shared semi-naive execution works (tests pass for sequences, graphs, Fibonacci)

## Testing Strategy

1. **Keep existing tests passing** throughout migration
2. **Add integration tests** that run same queries in SQL and Datalog
3. **Property tests** for semi-naive correctness (compare to naive iteration)
4. **Stratification tests** for correct negation handling

## File Structure After Migration

```
crates/
├── sql-parser/          # SQL parsing (unchanged)
│   └── src/
│       ├── ast.rs
│       ├── parser.rs
│       └── lib.rs
│
├── datalog-parser/      # Datalog parsing (absorbs datalog-ast)
│   └── src/
│       ├── ast.rs       # From datalog-ast
│       ├── parser.rs
│       └── lib.rs
│
├── storage/             # Unified storage (sql-storage + datalog-core)
│   └── src/
│       ├── relation.rs  # Relation, Row, Value
│       ├── index.rs     # Indexing
│       ├── schema.rs    # Schema types
│       └── lib.rs
│
├── planner/             # Unified planning (sql-planner + datalog-grounding + datalog-safety)
│   └── src/
│       ├── plan.rs      # LogicalPlan with Recursive
│       ├── sql.rs       # SQL AST → LogicalPlan
│       ├── datalog.rs   # Datalog AST → LogicalPlan
│       ├── stratify.rs  # Stratification analysis
│       ├── safety.rs    # Datalog safety checking
│       ├── optimize.rs  # Plan optimizations
│       └── lib.rs
│
├── engine/              # Unified execution (sql-engine + datalog-eval + datalog-builtins)
│   └── src/
│       ├── execute.rs   # Plan execution
│       ├── recursive.rs # Semi-naive evaluation
│       ├── expr.rs      # Expression evaluation (includes builtins)
│       ├── context.rs   # Execution context
│       └── lib.rs
│
├── sql-wal/             # Write-ahead log (unchanged)
└── sql-tests/           # Integration tests (unchanged)
```

## Dependencies After Migration

```
sql-parser ─────────────────────────────┐
                                        ▼
datalog-parser ──────────────────────► planner ──► engine
                                        ▲            ▲
                                        │            │
                                     storage ────────┘
```

```toml
# planner/Cargo.toml
[dependencies]
sql-parser = { path = "../sql-parser" }
datalog-parser = { path = "../datalog-parser" }
storage = { path = "../storage" }

# engine/Cargo.toml
[dependencies]
planner = { path = "../planner" }
storage = { path = "../storage" }
sql-parser = { path = "../sql-parser" }  # For Expr types
```
