//! Query planner - converts AST statements to logical plans

use sql_parser::{
    CreateTableStatement, Expr, InsertStatement, SelectColumn, SelectStatement, Statement,
};

use crate::plan::LogicalPlan;

/// Result type for planning operations
pub type PlanResult = Result<LogicalPlan, PlanError>;

/// Planning error
#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    /// No FROM clause in SELECT that requires data
    NoTableReference,
    /// Invalid LIMIT value
    InvalidLimit(String),
    /// Invalid OFFSET value
    InvalidOffset(String),
}

/// Convert a Statement to a LogicalPlan
pub fn plan(statement: Statement) -> PlanResult {
    match statement {
        Statement::Select(select) => plan_select(select),
        Statement::Insert(insert) => plan_insert(insert),
        Statement::CreateTable(create) => plan_create_table(create),
    }
}

/// Plan a SELECT statement
fn plan_select(select: SelectStatement) -> PlanResult {
    // Start with a table scan (or empty if no FROM)
    let mut plan = match select.from {
        Some(table_ref) => LogicalPlan::Scan {
            table: table_ref.name,
        },
        None => {
            // SELECT without FROM - just project expressions
            // We need to check if all columns are literals
            let exprs = extract_select_columns(&select.columns);
            return Ok(LogicalPlan::Projection {
                input: Box::new(LogicalPlan::Scan {
                    table: "dual".to_string(),
                }),
                exprs,
            });
        }
    };

    // Apply WHERE filter
    if let Some(predicate) = select.where_clause {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // Apply projection
    let exprs = extract_select_columns(&select.columns);
    plan = LogicalPlan::Projection {
        input: Box::new(plan),
        exprs,
    };

    // Apply ORDER BY
    if !select.order_by.is_empty() {
        plan = LogicalPlan::Sort {
            input: Box::new(plan),
            order_by: select.order_by,
        };
    }

    // Apply LIMIT/OFFSET
    if select.limit.is_some() || select.offset.is_some() {
        let limit = match select.limit {
            Some(Expr::Integer(n)) if n >= 0 => n as usize,
            Some(other) => {
                return Err(PlanError::InvalidLimit(format!("{:?}", other)));
            }
            None => usize::MAX,
        };
        let offset = match select.offset {
            Some(Expr::Integer(n)) if n >= 0 => n as usize,
            Some(other) => {
                return Err(PlanError::InvalidOffset(format!("{:?}", other)));
            }
            None => 0,
        };
        plan = LogicalPlan::Limit {
            input: Box::new(plan),
            limit,
            offset,
        };
    }

    Ok(plan)
}

/// Extract expressions from SELECT columns
fn extract_select_columns(columns: &[SelectColumn]) -> Vec<(Expr, Option<String>)> {
    columns
        .iter()
        .map(|col| match col {
            SelectColumn::Star => (Expr::Column("*".to_string()), None),
            SelectColumn::Expr { expr, alias } => (expr.clone(), alias.clone()),
        })
        .collect()
}

/// Plan an INSERT statement
fn plan_insert(insert: InsertStatement) -> PlanResult {
    Ok(LogicalPlan::Insert {
        table: insert.table,
        columns: insert.columns,
        values: insert.values,
    })
}

/// Plan a CREATE TABLE statement
fn plan_create_table(create: CreateTableStatement) -> PlanResult {
    Ok(LogicalPlan::CreateTable {
        name: create.name,
        columns: create.columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sql_parser::parse;

    fn plan_sql(sql: &str) -> PlanResult {
        let stmt = parse(sql).expect("Failed to parse SQL");
        plan(stmt)
    }

    #[test]
    fn test_plan_select_star() {
        let result = plan_sql("SELECT * FROM users");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Projection -> Scan
        match plan {
            LogicalPlan::Projection { input, exprs } => {
                assert_eq!(exprs.len(), 1);
                assert!(matches!(&exprs[0].0, Expr::Column(c) if c == "*"));
                assert!(matches!(*input, LogicalPlan::Scan { table } if table == "users"));
            }
            _ => panic!("Expected Projection"),
        }
    }

    #[test]
    fn test_plan_select_with_where() {
        let result = plan_sql("SELECT id FROM users WHERE id = 1");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Projection -> Filter -> Scan
        match plan {
            LogicalPlan::Projection { input, .. } => match *input {
                LogicalPlan::Filter { input, .. } => {
                    assert!(matches!(*input, LogicalPlan::Scan { table } if table == "users"));
                }
                _ => panic!("Expected Filter"),
            },
            _ => panic!("Expected Projection"),
        }
    }

    #[test]
    fn test_plan_select_with_order_by() {
        let result = plan_sql("SELECT id FROM users ORDER BY id DESC");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Sort -> Projection -> Scan
        match plan {
            LogicalPlan::Sort { input, order_by } => {
                assert_eq!(order_by.len(), 1);
                assert!(order_by[0].desc);
                assert!(matches!(*input, LogicalPlan::Projection { .. }));
            }
            _ => panic!("Expected Sort"),
        }
    }

    #[test]
    fn test_plan_select_with_limit() {
        let result = plan_sql("SELECT id FROM users LIMIT 10 OFFSET 5");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Limit -> Projection -> Scan
        match plan {
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                assert_eq!(limit, 10);
                assert_eq!(offset, 5);
                assert!(matches!(*input, LogicalPlan::Projection { .. }));
            }
            _ => panic!("Expected Limit"),
        }
    }

    #[test]
    fn test_plan_insert() {
        let result = plan_sql("INSERT INTO users (id, name) VALUES (1, 'alice')");
        assert!(result.is_ok());
        let plan = result.unwrap();
        match plan {
            LogicalPlan::Insert {
                table,
                columns,
                values,
            } => {
                assert_eq!(table, "users");
                assert_eq!(columns, Some(vec!["id".to_string(), "name".to_string()]));
                assert_eq!(values.len(), 1);
            }
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_plan_create_table() {
        let result = plan_sql("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)");
        assert!(result.is_ok());
        let plan = result.unwrap();
        match plan {
            LogicalPlan::CreateTable { name, columns } => {
                assert_eq!(name, "users");
                assert_eq!(columns.len(), 2);
            }
            _ => panic!("Expected CreateTable"),
        }
    }

    #[test]
    fn test_plan_select_literal() {
        let result = plan_sql("SELECT 42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_query_plan() {
        let result = plan_sql("SELECT id, name FROM users WHERE id > 10 ORDER BY name LIMIT 20");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Limit -> Sort -> Projection -> Filter -> Scan
        match plan {
            LogicalPlan::Limit { input, limit, .. } => {
                assert_eq!(limit, 20);
                match *input {
                    LogicalPlan::Sort { input, .. } => match *input {
                        LogicalPlan::Projection { input, exprs } => {
                            assert_eq!(exprs.len(), 2);
                            match *input {
                                LogicalPlan::Filter { input, .. } => {
                                    assert!(matches!(*input, LogicalPlan::Scan { .. }));
                                }
                                _ => panic!("Expected Filter"),
                            }
                        }
                        _ => panic!("Expected Projection"),
                    },
                    _ => panic!("Expected Sort"),
                }
            }
            _ => panic!("Expected Limit"),
        }
    }
}
