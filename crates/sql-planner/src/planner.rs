//! Query planner - converts AST statements to logical plans

use sql_parser::{
    CreateTableStatement, DeleteStatement, Expr, InsertStatement, SelectColumn, SelectStatement,
    Statement, UpdateStatement,
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
        Statement::Update(update) => plan_update(update),
        Statement::Delete(delete) => plan_delete(delete),
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

    // Apply JOINs
    for join in select.joins {
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(LogicalPlan::Scan {
                table: join.table.name,
            }),
            join_type: join.join_type,
            on: join.on,
        };
    }

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

/// Plan an UPDATE statement
fn plan_update(update: UpdateStatement) -> PlanResult {
    Ok(LogicalPlan::Update {
        table: update.table,
        assignments: update.assignments,
        where_clause: update.where_clause,
    })
}

/// Plan a DELETE statement
fn plan_delete(delete: DeleteStatement) -> PlanResult {
    Ok(LogicalPlan::Delete {
        table: delete.table,
        where_clause: delete.where_clause,
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

    #[test]
    fn test_plan_update() {
        let result = plan_sql("UPDATE users SET name = 'bob' WHERE id = 1");
        assert!(result.is_ok());
        let plan = result.unwrap();
        match plan {
            LogicalPlan::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].column, "name");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Update"),
        }
    }

    #[test]
    fn test_plan_delete() {
        let result = plan_sql("DELETE FROM users WHERE id = 1");
        assert!(result.is_ok());
        let plan = result.unwrap();
        match plan {
            LogicalPlan::Delete {
                table,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_plan_inner_join() {
        let result = plan_sql("SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Projection -> Join -> (Scan, Scan)
        match plan {
            LogicalPlan::Projection { input, .. } => match *input {
                LogicalPlan::Join {
                    left,
                    right,
                    join_type,
                    on,
                } => {
                    assert_eq!(join_type, sql_parser::JoinType::Inner);
                    assert!(on.is_some());
                    assert!(matches!(*left, LogicalPlan::Scan { table } if table == "users"));
                    assert!(matches!(*right, LogicalPlan::Scan { table } if table == "orders"));
                }
                _ => panic!("Expected Join"),
            },
            _ => panic!("Expected Projection"),
        }
    }

    #[test]
    fn test_plan_multiple_joins() {
        let result = plan_sql(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id JOIN items ON orders.id = items.order_id"
        );
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should be Projection -> Join -> (Join -> (Scan, Scan), Scan)
        match plan {
            LogicalPlan::Projection { input, .. } => match *input {
                LogicalPlan::Join { left, right, .. } => {
                    // right should be items scan
                    assert!(matches!(*right, LogicalPlan::Scan { table } if table == "items"));
                    // left should be another join
                    match *left {
                        LogicalPlan::Join { left, right, .. } => {
                            assert!(
                                matches!(*left, LogicalPlan::Scan { table } if table == "users")
                            );
                            assert!(
                                matches!(*right, LogicalPlan::Scan { table } if table == "orders")
                            );
                        }
                        _ => panic!("Expected nested Join"),
                    }
                }
                _ => panic!("Expected Join"),
            },
            _ => panic!("Expected Projection"),
        }
    }
}
