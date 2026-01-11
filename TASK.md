# SQL

Create a full sql database in pure Rust.
Well organized code split into multiple crates (one for parsing sql, one for creating a querying plan, etc).
Use Test driven development to ensure good code quality.
Use Chumsky + Ariadne for parsing (split into a lexer step and a parser step)
Plugable storage engines, so that one can swap out which engine one want to use as storage.
Find a standardized test for SQL queries to run against to ensure you follow the full SQL standard.
Use deterministic simulation testing to fully test the database in all kinds of failure modes.
A write ahead log to ensure reliability.
