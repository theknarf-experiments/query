//! Query executor - executes logical plans against storage

use sql_parser::{BinaryOp, ColumnDef, DataType, Expr, UnaryOp};
use sql_planner::LogicalPlan;
use sql_storage::{
    ColumnSchema, DataType as StorageDataType, MemoryEngine, Row, StorageEngine, StorageError,
    TableSchema, Value,
};

/// Result type for execution operations
pub type ExecResult = Result<QueryResult, ExecError>;

/// Execution error
#[derive(Debug, Clone, PartialEq)]
pub enum ExecError {
    /// Storage error
    Storage(StorageError),
    /// Table not found
    TableNotFound(String),
    /// Column not found
    ColumnNotFound(String),
    /// Type error
    TypeError(String),
    /// Invalid expression
    InvalidExpression(String),
}

impl From<StorageError> for ExecError {
    fn from(err: StorageError) -> Self {
        ExecError::Storage(err)
    }
}

/// Result of a query execution
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    /// SELECT result with column names and rows
    Select {
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
    },
    /// Number of rows affected (INSERT, UPDATE, DELETE)
    RowsAffected(usize),
    /// DDL success (CREATE TABLE, DROP TABLE)
    Success,
}

/// Database engine that executes queries
pub struct Engine {
    storage: MemoryEngine,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Create a new database engine
    pub fn new() -> Self {
        Self {
            storage: MemoryEngine::new(),
        }
    }

    /// Execute a SQL string
    pub fn execute(&mut self, sql: &str) -> ExecResult {
        let stmt = sql_parser::parse(sql)
            .map_err(|_| ExecError::InvalidExpression("Parse error".to_string()))?;
        let plan = sql_planner::plan(stmt)
            .map_err(|_| ExecError::InvalidExpression("Planning error".to_string()))?;
        self.execute_plan(plan)
    }

    /// Execute a logical plan
    fn execute_plan(&mut self, plan: LogicalPlan) -> ExecResult {
        match plan {
            LogicalPlan::CreateTable { name, columns } => {
                self.execute_create_table(&name, &columns)
            }
            LogicalPlan::Insert {
                table,
                columns,
                values,
            } => self.execute_insert(&table, columns.as_deref(), &values),
            _ => self.execute_query(plan),
        }
    }

    /// Execute a CREATE TABLE
    fn execute_create_table(&mut self, name: &str, columns: &[ColumnDef]) -> ExecResult {
        let schema = TableSchema {
            name: name.to_string(),
            columns: columns
                .iter()
                .map(|c| ColumnSchema {
                    name: c.name.clone(),
                    data_type: convert_data_type(&c.data_type),
                    nullable: c.nullable,
                    primary_key: c.primary_key,
                })
                .collect(),
        };
        self.storage.create_table(schema)?;
        Ok(QueryResult::Success)
    }

    /// Execute an INSERT
    fn execute_insert(
        &mut self,
        table: &str,
        _columns: Option<&[String]>,
        values: &[Vec<Expr>],
    ) -> ExecResult {
        let mut count = 0;
        for value_row in values {
            let row: Row = value_row.iter().map(eval_literal).collect();
            self.storage.insert(table, row)?;
            count += 1;
        }
        Ok(QueryResult::RowsAffected(count))
    }

    /// Execute a SELECT query
    fn execute_query(&self, plan: LogicalPlan) -> ExecResult {
        match plan {
            LogicalPlan::Scan { table } => {
                let schema = self.storage.get_schema(&table)?;
                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();
                let rows = self.storage.scan(&table)?;
                Ok(QueryResult::Select {
                    columns: column_names,
                    rows,
                })
            }
            LogicalPlan::Filter { input, predicate } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        let filtered: Vec<Vec<Value>> = rows
                            .into_iter()
                            .filter(|row| eval_predicate(&predicate, row, &columns))
                            .collect();
                        Ok(QueryResult::Select {
                            columns,
                            rows: filtered,
                        })
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::Projection { input, exprs } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        // Handle SELECT *
                        if exprs.len() == 1 {
                            if let (Expr::Column(c), _) = &exprs[0] {
                                if c == "*" {
                                    return Ok(QueryResult::Select { columns, rows });
                                }
                            }
                        }

                        let new_columns: Vec<String> = exprs
                            .iter()
                            .enumerate()
                            .map(|(i, (expr, alias))| {
                                alias.clone().unwrap_or_else(|| match expr {
                                    Expr::Column(c) => c.clone(),
                                    _ => format!("col{}", i),
                                })
                            })
                            .collect();

                        let new_rows: Vec<Vec<Value>> = rows
                            .iter()
                            .map(|row| {
                                exprs
                                    .iter()
                                    .map(|(expr, _)| eval_expr(expr, row, &columns))
                                    .collect()
                            })
                            .collect();

                        Ok(QueryResult::Select {
                            columns: new_columns,
                            rows: new_rows,
                        })
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::Sort { input, order_by } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, mut rows } => {
                        // Sort by first order_by expression
                        if let Some(ob) = order_by.first() {
                            rows.sort_by(|a, b| {
                                let val_a = eval_expr(&ob.expr, a, &columns);
                                let val_b = eval_expr(&ob.expr, b, &columns);
                                let cmp = compare_values(&val_a, &val_b);
                                if ob.desc {
                                    cmp.reverse()
                                } else {
                                    cmp
                                }
                            });
                        }
                        Ok(QueryResult::Select { columns, rows })
                    }
                    other => Ok(other),
                }
            }
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let result = self.execute_query(*input)?;
                match result {
                    QueryResult::Select { columns, rows } => {
                        let limited: Vec<Vec<Value>> =
                            rows.into_iter().skip(offset).take(limit).collect();
                        Ok(QueryResult::Select {
                            columns,
                            rows: limited,
                        })
                    }
                    other => Ok(other),
                }
            }
            _ => Err(ExecError::InvalidExpression("Unsupported plan".to_string())),
        }
    }
}

/// Convert parser DataType to storage DataType
fn convert_data_type(dt: &DataType) -> StorageDataType {
    match dt {
        DataType::Int => StorageDataType::Int,
        DataType::Float => StorageDataType::Float,
        DataType::Text => StorageDataType::Text,
        DataType::Bool => StorageDataType::Bool,
    }
}

/// Evaluate a literal expression to a Value
fn eval_literal(expr: &Expr) -> Value {
    match expr {
        Expr::Integer(n) => Value::Int(*n),
        Expr::Float(f) => Value::Float(*f),
        Expr::String(s) => Value::Text(s.clone()),
        Expr::Boolean(b) => Value::Bool(*b),
        Expr::Null => Value::Null,
        Expr::UnaryOp {
            op: UnaryOp::Neg,
            expr,
        } => match eval_literal(expr) {
            Value::Int(n) => Value::Int(-n),
            Value::Float(f) => Value::Float(-f),
            other => other,
        },
        _ => Value::Null,
    }
}

/// Evaluate an expression against a row
fn eval_expr(expr: &Expr, row: &[Value], columns: &[String]) -> Value {
    match expr {
        Expr::Column(name) => {
            if let Some(idx) = columns.iter().position(|c| c == name) {
                row.get(idx).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        Expr::Integer(n) => Value::Int(*n),
        Expr::Float(f) => Value::Float(*f),
        Expr::String(s) => Value::Text(s.clone()),
        Expr::Boolean(b) => Value::Bool(*b),
        Expr::Null => Value::Null,
        Expr::UnaryOp { op, expr } => {
            let val = eval_expr(expr, row, columns);
            match op {
                UnaryOp::Neg => match val {
                    Value::Int(n) => Value::Int(-n),
                    Value::Float(f) => Value::Float(-f),
                    _ => Value::Null,
                },
                UnaryOp::Not => match val {
                    Value::Bool(b) => Value::Bool(!b),
                    _ => Value::Null,
                },
            }
        }
        Expr::BinaryOp { left, op, right } => {
            let l = eval_expr(left, row, columns);
            let r = eval_expr(right, row, columns);
            eval_binary_op(&l, op, &r)
        }
    }
}

/// Evaluate a binary operation
fn eval_binary_op(left: &Value, op: &BinaryOp, right: &Value) -> Value {
    match op {
        BinaryOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Mod => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a % b),
            _ => Value::Null,
        },
        BinaryOp::Eq => Value::Bool(values_equal(left, right)),
        BinaryOp::NotEq => Value::Bool(!values_equal(left, right)),
        BinaryOp::Lt => Value::Bool(matches!(
            compare_values(left, right),
            std::cmp::Ordering::Less
        )),
        BinaryOp::Gt => Value::Bool(matches!(
            compare_values(left, right),
            std::cmp::Ordering::Greater
        )),
        BinaryOp::LtEq => Value::Bool(!matches!(
            compare_values(left, right),
            std::cmp::Ordering::Greater
        )),
        BinaryOp::GtEq => Value::Bool(!matches!(
            compare_values(left, right),
            std::cmp::Ordering::Less
        )),
        BinaryOp::And => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a && *b),
            _ => Value::Null,
        },
        BinaryOp::Or => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a || *b),
            _ => Value::Null,
        },
    }
}

/// Check if two values are equal
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

/// Compare two values
fn compare_values(left: &Value, right: &Value) -> std::cmp::Ordering {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::Text(a), Value::Text(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    }
}

/// Evaluate a predicate expression
fn eval_predicate(expr: &Expr, row: &[Value], columns: &[String]) -> bool {
    match eval_expr(expr, row, columns) {
        Value::Bool(b) => b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_and_insert() {
        let mut engine = Engine::new();

        let result = engine.execute("CREATE TABLE users (id INT, name TEXT)");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), QueryResult::Success);

        let result = engine.execute("INSERT INTO users (id, name) VALUES (1, 'alice')");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), QueryResult::RowsAffected(1));
    }

    #[test]
    fn test_select_all() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_with_where() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users WHERE id = 1");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Int(1));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_with_order_by() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (3, 'charlie')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users ORDER BY id");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows[0][0], Value::Int(1));
                assert_eq!(rows[1][0], Value::Int(2));
                assert_eq!(rows[2][0], Value::Int(3));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_with_limit() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (3, 'charlie')")
            .unwrap();

        let result = engine.execute("SELECT * FROM users LIMIT 2");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_select_projection() {
        let mut engine = Engine::new();
        engine
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        engine
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();

        let result = engine.execute("SELECT name FROM users");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["name"]);
                assert_eq!(rows[0][0], Value::Text("alice".to_string()));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_arithmetic_expression() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE nums (a INT, b INT)").unwrap();
        engine
            .execute("INSERT INTO nums (a, b) VALUES (10, 3)")
            .unwrap();

        let result = engine.execute("SELECT a + b AS sum FROM nums");
        assert!(result.is_ok());
        match result.unwrap() {
            QueryResult::Select { columns, rows } => {
                assert_eq!(columns, vec!["sum"]);
                assert_eq!(rows[0][0], Value::Int(13));
            }
            _ => panic!("Expected Select result"),
        }
    }

    #[test]
    fn test_comparison_operators() {
        let mut engine = Engine::new();
        engine.execute("CREATE TABLE t (x INT)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (5)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (10)").unwrap();
        engine.execute("INSERT INTO t (x) VALUES (15)").unwrap();

        let result = engine.execute("SELECT * FROM t WHERE x > 5");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }

        let result = engine.execute("SELECT * FROM t WHERE x >= 10");
        match result.unwrap() {
            QueryResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected Select result"),
        }
    }
}
