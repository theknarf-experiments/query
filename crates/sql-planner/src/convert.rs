//! Conversion functions from parser AST types to planner IR types

use crate::ir;
use crate::plan::LogicalPlan;
use crate::planner::{plan_select_core, PlanError};
use sql_parser as ast;

/// Convert parser Expr to IR Expr
pub fn convert_expr(expr: ast::Expr) -> Result<ir::Expr, PlanError> {
    Ok(match expr {
        ast::Expr::Column(name) => ir::Expr::Column(name),
        ast::Expr::Integer(n) => ir::Expr::Integer(n),
        ast::Expr::Float(f) => ir::Expr::Float(f),
        ast::Expr::String(s) => ir::Expr::String(s),
        ast::Expr::Boolean(b) => ir::Expr::Boolean(b),
        ast::Expr::Null => ir::Expr::Null,
        ast::Expr::BinaryOp { left, op, right } => ir::Expr::BinaryOp {
            left: Box::new(convert_expr(*left)?),
            op: convert_binary_op(op),
            right: Box::new(convert_expr(*right)?),
        },
        ast::Expr::UnaryOp { op, expr } => ir::Expr::UnaryOp {
            op: convert_unary_op(op),
            expr: Box::new(convert_expr(*expr)?),
        },
        ast::Expr::Aggregate { func, arg } => ir::Expr::Aggregate {
            func: convert_aggregate_func(func),
            arg: Box::new(convert_expr(*arg)?),
        },
        ast::Expr::Subquery(select) => ir::Expr::Subquery(Box::new(plan_select_core(*select)?)),
        ast::Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => ir::Expr::InSubquery {
            expr: Box::new(convert_expr(*expr)?),
            subquery: Box::new(plan_select_core(*subquery)?),
            negated,
        },
        ast::Expr::Exists(select) => ir::Expr::Exists(Box::new(plan_select_core(*select)?)),
        ast::Expr::Like {
            expr,
            pattern,
            negated,
        } => ir::Expr::Like {
            expr: Box::new(convert_expr(*expr)?),
            pattern: Box::new(convert_expr(*pattern)?),
            negated,
        },
        ast::Expr::IsNull { expr, negated } => ir::Expr::IsNull {
            expr: Box::new(convert_expr(*expr)?),
            negated,
        },
        ast::Expr::Case {
            operand,
            when_clauses,
            else_result,
        } => ir::Expr::Case {
            operand: operand.map(|e| convert_expr(*e)).transpose()?.map(Box::new),
            when_clauses: when_clauses
                .into_iter()
                .map(|(cond, result)| Ok((convert_expr(cond)?, convert_expr(result)?)))
                .collect::<Result<Vec<_>, PlanError>>()?,
            else_result: else_result
                .map(|e| convert_expr(*e))
                .transpose()?
                .map(Box::new),
        },
        ast::Expr::Between {
            expr,
            low,
            high,
            negated,
        } => ir::Expr::Between {
            expr: Box::new(convert_expr(*expr)?),
            low: Box::new(convert_expr(*low)?),
            high: Box::new(convert_expr(*high)?),
            negated,
        },
        ast::Expr::WindowFunction {
            func,
            partition_by,
            order_by,
        } => ir::Expr::WindowFunction {
            func: convert_window_func(func),
            partition_by: partition_by
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?,
            order_by: order_by
                .into_iter()
                .map(convert_order_by)
                .collect::<Result<Vec<_>, _>>()?,
        },
        ast::Expr::Function { name, args } => ir::Expr::Function {
            name,
            args: args
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?,
        },
    })
}

/// Convert parser BinaryOp to IR BinaryOp
pub fn convert_binary_op(op: ast::BinaryOp) -> ir::BinaryOp {
    match op {
        ast::BinaryOp::Add => ir::BinaryOp::Add,
        ast::BinaryOp::Sub => ir::BinaryOp::Sub,
        ast::BinaryOp::Mul => ir::BinaryOp::Mul,
        ast::BinaryOp::Div => ir::BinaryOp::Div,
        ast::BinaryOp::Mod => ir::BinaryOp::Mod,
        ast::BinaryOp::Eq => ir::BinaryOp::Eq,
        ast::BinaryOp::NotEq => ir::BinaryOp::NotEq,
        ast::BinaryOp::Lt => ir::BinaryOp::Lt,
        ast::BinaryOp::Gt => ir::BinaryOp::Gt,
        ast::BinaryOp::LtEq => ir::BinaryOp::LtEq,
        ast::BinaryOp::GtEq => ir::BinaryOp::GtEq,
        ast::BinaryOp::And => ir::BinaryOp::And,
        ast::BinaryOp::Or => ir::BinaryOp::Or,
    }
}

/// Convert parser UnaryOp to IR UnaryOp
pub fn convert_unary_op(op: ast::UnaryOp) -> ir::UnaryOp {
    match op {
        ast::UnaryOp::Neg => ir::UnaryOp::Neg,
        ast::UnaryOp::Not => ir::UnaryOp::Not,
    }
}

/// Convert parser AggregateFunc to IR AggregateFunc
pub fn convert_aggregate_func(func: ast::AggregateFunc) -> ir::AggregateFunc {
    match func {
        ast::AggregateFunc::Count => ir::AggregateFunc::Count,
        ast::AggregateFunc::Sum => ir::AggregateFunc::Sum,
        ast::AggregateFunc::Avg => ir::AggregateFunc::Avg,
        ast::AggregateFunc::Min => ir::AggregateFunc::Min,
        ast::AggregateFunc::Max => ir::AggregateFunc::Max,
    }
}

/// Convert parser WindowFunc to IR WindowFunc
pub fn convert_window_func(func: ast::WindowFunc) -> ir::WindowFunc {
    match func {
        ast::WindowFunc::RowNumber => ir::WindowFunc::RowNumber,
        ast::WindowFunc::Rank => ir::WindowFunc::Rank,
        ast::WindowFunc::DenseRank => ir::WindowFunc::DenseRank,
    }
}

/// Convert parser OrderBy to IR OrderBy
pub fn convert_order_by(ob: ast::OrderBy) -> Result<ir::OrderBy, PlanError> {
    Ok(ir::OrderBy {
        expr: convert_expr(ob.expr)?,
        desc: ob.desc,
    })
}

/// Convert parser JoinType to IR JoinType
pub fn convert_join_type(jt: ast::JoinType) -> ir::JoinType {
    match jt {
        ast::JoinType::Inner => ir::JoinType::Inner,
        ast::JoinType::Left => ir::JoinType::Left,
        ast::JoinType::Right => ir::JoinType::Right,
        ast::JoinType::Full => ir::JoinType::Full,
        ast::JoinType::Cross => ir::JoinType::Cross,
    }
}

/// Convert parser SetOperator to IR SetOperator
pub fn convert_set_operator(op: ast::SetOperator) -> ir::SetOperator {
    match op {
        ast::SetOperator::Union => ir::SetOperator::Union,
        ast::SetOperator::Intersect => ir::SetOperator::Intersect,
        ast::SetOperator::Except => ir::SetOperator::Except,
    }
}

/// Convert parser Assignment to IR Assignment
pub fn convert_assignment(asgn: ast::Assignment) -> Result<ir::Assignment, PlanError> {
    Ok(ir::Assignment {
        column: asgn.column,
        value: convert_expr(asgn.value)?,
    })
}

/// Convert parser DataType to IR DataType
pub fn convert_data_type(dt: ast::DataType) -> ir::DataType {
    match dt {
        ast::DataType::Int => ir::DataType::Int,
        ast::DataType::Float => ir::DataType::Float,
        ast::DataType::Text => ir::DataType::Text,
        ast::DataType::Bool => ir::DataType::Bool,
        ast::DataType::Date => ir::DataType::Date,
        ast::DataType::Time => ir::DataType::Time,
        ast::DataType::Timestamp => ir::DataType::Timestamp,
    }
}

/// Convert parser ReferentialAction to IR ReferentialAction
pub fn convert_referential_action(ra: ast::ReferentialAction) -> ir::ReferentialAction {
    match ra {
        ast::ReferentialAction::NoAction => ir::ReferentialAction::NoAction,
        ast::ReferentialAction::Cascade => ir::ReferentialAction::Cascade,
        ast::ReferentialAction::SetNull => ir::ReferentialAction::SetNull,
        ast::ReferentialAction::SetDefault => ir::ReferentialAction::SetDefault,
        ast::ReferentialAction::Restrict => ir::ReferentialAction::Restrict,
    }
}

/// Convert parser ForeignKeyRef to IR ForeignKeyRef
pub fn convert_foreign_key_ref(fk: ast::ForeignKeyRef) -> ir::ForeignKeyRef {
    ir::ForeignKeyRef {
        table: fk.table,
        column: fk.column,
        on_delete: convert_referential_action(fk.on_delete),
        on_update: convert_referential_action(fk.on_update),
    }
}

/// Convert parser ColumnDef to IR ColumnDef
pub fn convert_column_def(col: ast::ColumnDef) -> Result<ir::ColumnDef, PlanError> {
    Ok(ir::ColumnDef {
        name: col.name,
        data_type: convert_data_type(col.data_type),
        nullable: col.nullable,
        primary_key: col.primary_key,
        unique: col.unique,
        default: col.default.map(convert_expr).transpose()?,
        references: col.references.map(convert_foreign_key_ref),
    })
}

/// Convert parser TableConstraint to IR TableConstraint
pub fn convert_table_constraint(
    tc: ast::TableConstraint,
) -> Result<ir::TableConstraint, PlanError> {
    Ok(match tc {
        ast::TableConstraint::PrimaryKey { name, columns } => {
            ir::TableConstraint::PrimaryKey { name, columns }
        }
        ast::TableConstraint::ForeignKey {
            name,
            columns,
            references_table,
            references_columns,
            on_delete,
            on_update,
        } => ir::TableConstraint::ForeignKey {
            name,
            columns,
            references_table,
            references_columns,
            on_delete: convert_referential_action(on_delete),
            on_update: convert_referential_action(on_update),
        },
        ast::TableConstraint::Unique { name, columns } => {
            ir::TableConstraint::Unique { name, columns }
        }
        ast::TableConstraint::Check { name, expr } => ir::TableConstraint::Check {
            name,
            expr: convert_expr(expr)?,
        },
    })
}

/// Convert parser TriggerTiming to IR TriggerTiming
pub fn convert_trigger_timing(tt: ast::TriggerTiming) -> ir::TriggerTiming {
    match tt {
        ast::TriggerTiming::Before => ir::TriggerTiming::Before,
        ast::TriggerTiming::After => ir::TriggerTiming::After,
    }
}

/// Convert parser TriggerEvent to IR TriggerEvent
pub fn convert_trigger_event(te: ast::TriggerEvent) -> ir::TriggerEvent {
    match te {
        ast::TriggerEvent::Insert => ir::TriggerEvent::Insert,
        ast::TriggerEvent::Update => ir::TriggerEvent::Update,
        ast::TriggerEvent::Delete => ir::TriggerEvent::Delete,
    }
}

/// Convert parser TriggerAction to IR TriggerAction
pub fn convert_trigger_action(ta: ast::TriggerAction) -> Result<ir::TriggerAction, PlanError> {
    Ok(match ta {
        ast::TriggerAction::SetColumn { column, value } => ir::TriggerAction::SetColumn {
            column,
            value: convert_expr(value)?,
        },
        ast::TriggerAction::RaiseError(msg) => ir::TriggerAction::RaiseError(msg),
    })
}

/// Convert parser TriggerActionType to IR TriggerActionType
pub fn convert_trigger_action_type(
    tat: ast::TriggerActionType,
) -> Result<ir::TriggerActionType, PlanError> {
    Ok(match tat {
        ast::TriggerActionType::ExecuteFunction(name) => {
            ir::TriggerActionType::ExecuteFunction(name)
        }
        ast::TriggerActionType::InlineActions(actions) => ir::TriggerActionType::InlineActions(
            actions
                .into_iter()
                .map(convert_trigger_action)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

/// Convert parser AlterAction to IR AlterAction
pub fn convert_alter_action(aa: ast::AlterAction) -> Result<ir::AlterAction, PlanError> {
    Ok(match aa {
        ast::AlterAction::AddColumn(col) => ir::AlterAction::AddColumn(convert_column_def(col)?),
        ast::AlterAction::DropColumn(name) => ir::AlterAction::DropColumn(name),
        ast::AlterAction::RenameColumn { old_name, new_name } => {
            ir::AlterAction::RenameColumn { old_name, new_name }
        }
        ast::AlterAction::RenameTable(name) => ir::AlterAction::RenameTable(name),
    })
}

/// Convert parser ProcedureParam to IR ProcedureParam
pub fn convert_procedure_param(pp: ast::ProcedureParam) -> ir::ProcedureParam {
    ir::ProcedureParam {
        name: pp.name,
        data_type: convert_data_type(pp.data_type),
    }
}

/// Convert parser ProcedureStatement to IR ProcedureStatement
pub fn convert_procedure_statement(
    ps: ast::ProcedureStatement,
) -> Result<ir::ProcedureStatement, PlanError> {
    Ok(match ps {
        ast::ProcedureStatement::Sql(stmt) => {
            ir::ProcedureStatement::Plan(Box::new(crate::planner::plan(*stmt)?))
        }
        ast::ProcedureStatement::Declare { name, data_type } => ir::ProcedureStatement::Declare {
            name,
            data_type: convert_data_type(data_type),
        },
        ast::ProcedureStatement::SetVar { name, value } => ir::ProcedureStatement::SetVar {
            name,
            value: convert_expr(value)?,
        },
        ast::ProcedureStatement::Return(expr) => {
            ir::ProcedureStatement::Return(expr.map(convert_expr).transpose()?)
        }
    })
}

/// Convert a parser Cte to an IR Cte by planning the query
pub fn convert_cte(cte: ast::Cte) -> Result<ir::Cte, PlanError> {
    let plan = convert_cte_query(cte.query)?;
    Ok(ir::Cte {
        name: cte.name,
        columns: cte.columns,
        plan: Box::new(plan),
    })
}

/// Convert a parser CteQuery to a LogicalPlan
pub fn convert_cte_query(query: ast::CteQuery) -> Result<LogicalPlan, PlanError> {
    match query {
        ast::CteQuery::Select(select) => plan_select_core(*select),
        ast::CteQuery::SetOp(set_op) => {
            let left = convert_cte_query(set_op.left)?;
            let right = convert_cte_query(set_op.right)?;
            Ok(LogicalPlan::SetOperation {
                left: Box::new(left),
                right: Box::new(right),
                op: convert_set_operator(set_op.op),
                all: set_op.all,
            })
        }
    }
}
