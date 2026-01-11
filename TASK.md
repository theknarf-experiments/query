# SQL

Create a full sql database in pure Rust.
Well organized code split into multiple crates (one for parsing sql, one for creating a querying plan, etc).
Use Test driven development to ensure good code quality.
Use Chumsky + Ariadne for parsing (split into a lexer step and a parser step)
Plugable storage engines, so that one can swap out which engine one want to use as storage.
