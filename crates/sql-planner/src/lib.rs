//! SQL Query Planner
//!
//! This crate converts parsed AST into logical and physical query plans.
//! It also supports compiling Datalog programs to LogicalPlan.

pub mod datalog;
pub mod plan;
pub mod planner;

pub use plan::*;
pub use planner::{plan, PlanError, PlanResult};

// Datalog compilation
pub use datalog::{compile_datalog, DatalogPlanError};
