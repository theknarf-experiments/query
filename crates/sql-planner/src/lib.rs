//! SQL Query Planner
//!
//! This crate converts parsed AST into logical and physical query plans.
//! It also supports compiling Datalog programs to LogicalPlan.

mod convert;
pub mod datalog;
pub mod ir;
pub mod plan;
pub mod planner;

pub use plan::*;
pub use planner::{plan, PlanError, PlanResult};

// Re-export IR types for use by sql-engine
pub use ir::{
    AggregateFunc, AlterAction, Assignment, BinaryOp, ColumnDef, Cte, DataType, Expr,
    ForeignKeyRef, JoinType, OrderBy, ProcedureParam, ProcedureStatement, ReferentialAction,
    SetOperator, TableConstraint, TriggerAction, TriggerActionType, TriggerEvent, TriggerTiming,
    UnaryOp, WindowFunc,
};

// Re-export parse function so sql-engine only needs to depend on sql-planner
pub use sql_parser::parse;

// Datalog compilation
pub use datalog::{compile_datalog, DatalogPlanError};
