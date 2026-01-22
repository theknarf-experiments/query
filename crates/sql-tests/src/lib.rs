//! SQL database simulation and property-based tests
//!
//! This crate provides deterministic simulation testing for the SQL database.
//! It uses property-based testing to explore edge cases and failure modes.
//! It also includes SQL standard compliance tests.

#[cfg(test)]
mod simulation;

#[cfg(test)]
mod proptest_queries;

#[cfg(test)]
mod sql_compliance;

#[cfg(test)]
mod sql_datalog_integration;

#[cfg(test)]
mod trigger_tests;
