//! SQL database simulation and property-based tests
//!
//! This crate provides deterministic simulation testing for the SQL database.
//! It uses property-based testing to explore edge cases and failure modes.

#[cfg(test)]
mod simulation;

#[cfg(test)]
mod proptest_queries;
