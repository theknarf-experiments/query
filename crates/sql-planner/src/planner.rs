//! Query planner - converts AST statements to logical plans

use sql_parser::{
    CallProcedureStatement, CreateFunctionStatement, CreateIndexStatement,
    CreateProcedureStatement, CreateTableStatement, CreateTriggerStatement, CreateViewStatement,
    Cte, CteQuery, CteSetOperation, DeleteStatement, Expr, InsertStatement, SelectColumn,
    SelectOrSet, SelectStatement, SetOperationStatement, SetOperator, Statement, UpdateStatement,
    WithClause,
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
        Statement::Select(select) => plan_select(*select),
        Statement::SetOperation(set_op) => plan_set_operation(set_op),
        Statement::Insert(insert) => plan_insert(insert),
        Statement::Update(update) => plan_update(update),
        Statement::Delete(delete) => plan_delete(delete),
        Statement::CreateTable(create) => plan_create_table(create),
        Statement::CreateFunction(create) => plan_create_function(create),
        Statement::DropFunction(name) => Ok(LogicalPlan::DropFunction { name }),
        Statement::CreateTrigger(create) => plan_create_trigger(create),
        Statement::DropTrigger(name) => Ok(LogicalPlan::DropTrigger { name }),
        Statement::DropTable(name) => Ok(LogicalPlan::DropTable { name }),
        Statement::AlterTable(alter) => Ok(LogicalPlan::AlterTable {
            table: alter.table,
            action: alter.action,
        }),
        Statement::CreateIndex(create) => plan_create_index(create),
        Statement::DropIndex(name) => Ok(LogicalPlan::DropIndex { name }),
        Statement::CreateView(create) => plan_create_view(create),
        Statement::DropView(name) => Ok(LogicalPlan::DropView { name }),
        Statement::CreateProcedure(create) => plan_create_procedure(create),
        Statement::DropProcedure(name) => Ok(LogicalPlan::DropProcedure { name }),
        Statement::CallProcedure(call) => plan_call_procedure(call),
        Statement::Begin => Ok(LogicalPlan::Begin),
        Statement::Commit => Ok(LogicalPlan::Commit),
        Statement::Rollback => Ok(LogicalPlan::Rollback),
        Statement::Savepoint(name) => Ok(LogicalPlan::Savepoint { name }),
        Statement::ReleaseSavepoint(name) => Ok(LogicalPlan::ReleaseSavepoint { name }),
        Statement::RollbackTo(name) => Ok(LogicalPlan::RollbackTo { name }),
    }
}

/// Plan a set operation (UNION, INTERSECT, EXCEPT)
fn plan_set_operation(set_op: SetOperationStatement) -> PlanResult {
    let left = plan_select_or_set(*set_op.left)?;
    let right = plan_select_or_set(*set_op.right)?;
    Ok(LogicalPlan::SetOperation {
        left: Box::new(left),
        right: Box::new(right),
        op: set_op.op,
        all: set_op.all,
    })
}

/// Plan a SelectOrSet (recursive helper)
fn plan_select_or_set(node: SelectOrSet) -> PlanResult {
    match node {
        SelectOrSet::Select(select) => plan_select(*select),
        SelectOrSet::SetOp(set_op) => plan_set_operation(set_op),
    }
}

/// Plan a SELECT statement
fn plan_select(select: SelectStatement) -> PlanResult {
    // Save WITH clause for later wrapping
    let with_clause = select.with_clause.clone();

    // Plan the main query
    let plan = plan_select_core(select)?;

    // Wrap with CTE if WITH clause is present
    wrap_with_cte(plan, with_clause)
}

/// Wrap a plan with CTE if WITH clause is present
fn wrap_with_cte(plan: LogicalPlan, with_clause: Option<WithClause>) -> PlanResult {
    match with_clause {
        Some(with) if with.recursive => {
            // Handle recursive CTEs by compiling to Recursive nodes
            compile_recursive_ctes(plan, with.ctes)
        }
        Some(with) => Ok(LogicalPlan::WithCte {
            ctes: with.ctes,
            recursive: false,
            input: Box::new(plan),
        }),
        None => Ok(plan),
    }
}

/// Compile recursive CTEs into appropriate LogicalPlan nodes
fn compile_recursive_ctes(plan: LogicalPlan, ctes: Vec<Cte>) -> PlanResult {
    // Find the first recursive CTE (one that references itself)
    let recursive_idx = ctes.iter().position(cte_references_self);

    match recursive_idx {
        Some(idx) => {
            let recursive_cte = &ctes[idx];
            let pre_ctes: Vec<Cte> = ctes[..idx].to_vec();

            // Compile the recursive CTE into base and step plans
            let (base, step) = compile_recursive_cte_query(recursive_cte)?;

            // Get column names
            let columns = recursive_cte.columns.clone();

            Ok(LogicalPlan::WithRecursiveCte {
                name: recursive_cte.name.clone(),
                columns,
                base: Box::new(base),
                step: Box::new(step),
                pre_ctes,
                input: Box::new(plan),
            })
        }
        None => {
            // No recursive CTEs - just use regular CTE handling
            Ok(LogicalPlan::WithCte {
                ctes,
                recursive: true,
                input: Box::new(plan),
            })
        }
    }
}

/// Check if a CTE references itself (is recursive)
fn cte_references_self(cte: &Cte) -> bool {
    cte_query_references_table(&cte.query, &cte.name)
}

/// Check if a CteQuery references a given table name
fn cte_query_references_table(query: &CteQuery, table_name: &str) -> bool {
    match query {
        CteQuery::Select(select) => select_references_table(select, table_name),
        CteQuery::SetOp(set_op) => {
            cte_query_references_table(&set_op.left, table_name)
                || cte_query_references_table(&set_op.right, table_name)
        }
    }
}

/// Check if a SELECT statement references a given table name
fn select_references_table(query: &SelectStatement, table_name: &str) -> bool {
    // Check FROM clause
    if let Some(ref table_ref) = query.from {
        if table_ref.name == table_name {
            return true;
        }
    }

    // Check JOINs
    for join in &query.joins {
        if join.table.name == table_name {
            return true;
        }
    }

    // Check subqueries in WHERE clause
    if let Some(ref where_clause) = query.where_clause {
        if expr_references_table(where_clause, table_name) {
            return true;
        }
    }

    // Check nested WITH clause
    if let Some(ref with) = query.with_clause {
        for nested_cte in &with.ctes {
            if cte_query_references_table(&nested_cte.query, table_name) {
                return true;
            }
        }
    }

    false
}

/// Check if an expression references a table (via subqueries)
fn expr_references_table(expr: &Expr, table_name: &str) -> bool {
    match expr {
        Expr::Subquery(select) => select_references_table(select, table_name),
        Expr::InSubquery { subquery, .. } => select_references_table(subquery, table_name),
        Expr::Exists(select) => select_references_table(select, table_name),
        Expr::BinaryOp { left, right, .. } => {
            expr_references_table(left, table_name) || expr_references_table(right, table_name)
        }
        Expr::UnaryOp { expr, .. } => expr_references_table(expr, table_name),
        _ => false,
    }
}

/// Compile a recursive CTE query into base and step LogicalPlan nodes
///
/// For a proper recursive CTE like:
///   WITH RECURSIVE x(n) AS (
///       SELECT 1             -- base case
///       UNION ALL
///       SELECT n + 1 FROM x  -- recursive step
///   )
///
/// This function separates the base case (non-recursive) from the step (recursive).
fn compile_recursive_cte_query(cte: &Cte) -> Result<(LogicalPlan, LogicalPlan), PlanError> {
    let cte_name = &cte.name;

    match &cte.query {
        CteQuery::SetOp(set_op) => {
            // This is a UNION/UNION ALL - separate base and recursive parts
            let (base_queries, step_queries) = separate_base_and_step(set_op, cte_name);

            if base_queries.is_empty() {
                return Err(PlanError::InvalidLimit(
                    "Recursive CTE must have a non-recursive base case".to_string(),
                ));
            }

            // Plan base case (union of all non-recursive queries)
            let base = plan_cte_queries(&base_queries)?;

            // Plan step case (union of all recursive queries)
            let step = if step_queries.is_empty() {
                // No recursive step - this shouldn't happen for a recursive CTE
                LogicalPlan::Scan {
                    table: "__empty__".to_string(),
                }
            } else {
                plan_cte_queries(&step_queries)?
            };

            Ok((base, step))
        }
        CteQuery::Select(select) => {
            // Single SELECT that references itself - this is just the step
            // with an empty base (unusual but handle it)
            if select_references_table(select, cte_name) {
                let step = plan_select_core((**select).clone())?;
                Ok((
                    LogicalPlan::Scan {
                        table: "__empty__".to_string(),
                    },
                    step,
                ))
            } else {
                // Not actually recursive
                let base = plan_select_core((**select).clone())?;
                Ok((
                    base,
                    LogicalPlan::Scan {
                        table: "__empty__".to_string(),
                    },
                ))
            }
        }
    }
}

/// Separate a CTE set operation into base (non-recursive) and step (recursive) queries
fn separate_base_and_step(
    set_op: &CteSetOperation,
    cte_name: &str,
) -> (Vec<SelectStatement>, Vec<SelectStatement>) {
    let mut base_queries = Vec::new();
    let mut step_queries = Vec::new();

    collect_union_members(
        &CteQuery::SetOp(Box::new(set_op.clone())),
        cte_name,
        &mut base_queries,
        &mut step_queries,
    );

    (base_queries, step_queries)
}

/// Recursively collect all SELECT statements from a CteQuery, categorizing them
fn collect_union_members(
    query: &CteQuery,
    cte_name: &str,
    base: &mut Vec<SelectStatement>,
    step: &mut Vec<SelectStatement>,
) {
    match query {
        CteQuery::Select(select) => {
            if select_references_table(select, cte_name) {
                step.push((**select).clone());
            } else {
                base.push((**select).clone());
            }
        }
        CteQuery::SetOp(set_op) => {
            collect_union_members(&set_op.left, cte_name, base, step);
            collect_union_members(&set_op.right, cte_name, base, step);
        }
    }
}

/// Plan a list of SELECT statements as a UNION ALL
fn plan_cte_queries(queries: &[SelectStatement]) -> PlanResult {
    if queries.is_empty() {
        return Ok(LogicalPlan::Scan {
            table: "__empty__".to_string(),
        });
    }

    let mut plans: Vec<LogicalPlan> = queries
        .iter()
        .map(|q| plan_select_core(q.clone()))
        .collect::<Result<Vec<_>, _>>()?;

    if plans.len() == 1 {
        return Ok(plans.remove(0));
    }

    // Combine with UNION ALL
    let mut result = plans.remove(0);
    for plan in plans {
        result = LogicalPlan::SetOperation {
            left: Box::new(result),
            right: Box::new(plan),
            op: SetOperator::Union,
            all: true,
        };
    }

    Ok(result)
}

/// Plan the core of a SELECT statement (without CTE handling)
fn plan_select_core(select: SelectStatement) -> PlanResult {
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

    // Check if we have GROUP BY or aggregates
    let has_group_by = !select.group_by.is_empty();
    let exprs = extract_select_columns(&select.columns);
    let has_aggregates = exprs.iter().any(|(e, _)| contains_aggregate(e));

    if has_group_by || has_aggregates {
        // Apply aggregation
        plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_by: select.group_by,
            aggregates: exprs,
            having: select.having,
        };
    } else {
        // Apply projection
        plan = LogicalPlan::Projection {
            input: Box::new(plan),
            exprs,
        };
    }

    // Apply DISTINCT
    if select.distinct {
        plan = LogicalPlan::Distinct {
            input: Box::new(plan),
        };
    }

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

/// Check if an expression contains an aggregate function
fn contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Aggregate { .. } => true,
        Expr::BinaryOp { left, right, .. } => contains_aggregate(left) || contains_aggregate(right),
        Expr::UnaryOp { expr, .. } => contains_aggregate(expr),
        _ => false,
    }
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
        constraints: create.constraints,
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

/// Plan a CREATE FUNCTION statement
fn plan_create_function(create: CreateFunctionStatement) -> PlanResult {
    Ok(LogicalPlan::CreateFunction {
        name: create.name,
        body: create.body,
        language: create.language,
    })
}

/// Plan a CREATE TRIGGER statement
fn plan_create_trigger(create: CreateTriggerStatement) -> PlanResult {
    Ok(LogicalPlan::CreateTrigger {
        name: create.name,
        timing: create.timing,
        events: create.events,
        table: create.table,
        action: create.action,
    })
}

/// Plan a CREATE INDEX statement
fn plan_create_index(create: CreateIndexStatement) -> PlanResult {
    Ok(LogicalPlan::CreateIndex {
        name: create.name,
        table: create.table,
        columns: create.columns,
    })
}

/// Plan a CREATE VIEW statement
fn plan_create_view(create: CreateViewStatement) -> PlanResult {
    // Plan the view query
    let query_plan = plan_select_core(*create.query)?;
    Ok(LogicalPlan::CreateView {
        name: create.name,
        columns: create.columns,
        query: Box::new(query_plan),
    })
}

/// Plan a CREATE PROCEDURE statement
fn plan_create_procedure(create: CreateProcedureStatement) -> PlanResult {
    Ok(LogicalPlan::CreateProcedure {
        name: create.name,
        params: create.params,
        body: create.body,
    })
}

/// Plan a CALL procedure statement
fn plan_call_procedure(call: CallProcedureStatement) -> PlanResult {
    Ok(LogicalPlan::CallProcedure {
        name: call.name,
        args: call.args,
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
            LogicalPlan::CreateTable { name, columns, .. } => {
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
