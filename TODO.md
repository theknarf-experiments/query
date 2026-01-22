# TODO

## Bugs

### JOIN returns NULL for Datalog-derived tables

When performing a JOIN between a SQL table and a Datalog-derived table, all columns from both tables return NULL values.

**Reproduction:**

```sql
-- Create SQL tables
CREATE TABLE employee (id INT, name TEXT, dept TEXT);
INSERT INTO employee VALUES (1, 'CEO', 'exec'), (2, 'VP1', 'sales');

CREATE TABLE reports_to (employee_id INT, manager_id INT);
INSERT INTO reports_to VALUES (2, 1);
```

```datalog
-- Derive management relationships
manages(M, E) :- reports_to(E, M).
?- manages(X, Y).
```

```sql
-- This works (direct query on derived table):
SELECT * FROM manages WHERE col0 = 1;
-- Returns: [[Int(1), Int(2)]]

-- This fails (cross join with SQL table):
SELECT manages.col1, employee.id, employee.name
FROM manages, employee
WHERE manages.col0 = 1;
-- Returns: [[Null, Null, Null], [Null, Null, Null]]

-- JOINs also fail:
SELECT employee.name
FROM manages
JOIN employee ON manages.col1 = employee.id
WHERE manages.col0 = 1;
-- Returns: empty result set
```

**Expected:** JOIN should match rows and return actual values.

**Workaround:** Query Datalog-derived tables directly without JOINs.

**Location:** Likely in `sql-engine` crate, in how multi-table queries resolve columns for dynamically created tables.
