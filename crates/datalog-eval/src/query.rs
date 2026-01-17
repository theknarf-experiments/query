//! Query evaluation
//!
//! This module implements query evaluation against a fact database.
//! Queries find all substitutions that satisfy the query body.
//!
//! # Query Types
//!
//! - **Ground queries**: No variables, returns true/false
//! - **Variable queries**: With variables, returns all matching substitutions
//!
//! # Example
//!
//! ```ignore
//! // Query: ?- parent(X, mary).
//! let results = evaluate_query(&query, &db);
//! // Returns: [{X -> john}, {X -> alice}]
//! ```

use datalog_ast::{Query, Symbol};
use datalog_core::{FactDatabase, Substitution};
use datalog_grounding::satisfy_body;
use std::collections::HashSet;

/// Result of query evaluation - list of substitutions that satisfy the query
pub type QueryResult = Vec<Substitution>;

/// Evaluate a query against a fact database
///
/// Returns all substitutions that satisfy the query body.
/// For ground queries (no variables), returns either:
/// - `vec![Substitution::new()]` if the query is true
/// - `vec![]` if the query is false
///
/// For queries with variables, returns all matching substitutions.
pub fn evaluate_query(query: &Query, db: &FactDatabase) -> QueryResult {
    // Use satisfy_body from grounding module to find all substitutions
    satisfy_body(&query.body, db)
}

/// Extract variable bindings from a substitution for a set of variables
///
/// Returns a mapping from variable names to their bound values (as strings)
#[cfg_attr(not(test), allow(dead_code))]
pub fn extract_bindings(
    subst: &Substitution,
    variables: &HashSet<Symbol>,
) -> Vec<(Symbol, String)> {
    variables
        .iter()
        .filter_map(|var| subst.get(var).map(|term| (*var, format!("{:?}", term))))
        .collect()
}

/// Extract all variables from a query
#[cfg_attr(not(test), allow(dead_code))]
pub fn query_variables(query: &Query) -> HashSet<Symbol> {
    use datalog_ast::Term;

    let mut vars = HashSet::new();

    fn collect_term_vars(term: &Term, vars: &mut HashSet<Symbol>) {
        match term {
            Term::Variable(v) => {
                vars.insert(*v);
            }
            Term::Compound(_, args) => {
                for arg in args {
                    collect_term_vars(arg, vars);
                }
            }
            _ => {}
        }
    }

    for literal in &query.body {
        if let Some(atom) = literal.atom() {
            for term in &atom.terms {
                collect_term_vars(term, &mut vars);
            }
        }
        // For aggregates, variables are tracked separately in the aggregate structure
    }

    vars
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_ast::{Atom, Literal, Term, Value};

    fn sym(s: &str) -> Symbol {
        Symbol::new(s.to_string())
    }

    fn var_term(name: &str) -> Term {
        Term::Variable(sym(name))
    }

    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    #[test]
    fn test_query_with_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![
                Term::Constant(Value::Atom(sym("john"))),
                Term::Constant(Value::Atom(sym("mary"))),
            ],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![
                Term::Constant(Value::Atom(sym("mary"))),
                Term::Constant(Value::Atom(sym("jane"))),
            ],
        ))
        .unwrap();

        // Query: ?- parent(X, mary).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), Term::Constant(Value::Atom(sym("mary")))],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_query() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![
                Term::Constant(Value::Atom(sym("john"))),
                Term::Constant(Value::Atom(sym("mary"))),
            ],
        ))
        .unwrap();

        // Query: ?- parent(john, mary). (ground query - should succeed)
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![
                    Term::Constant(Value::Atom(sym("john"))),
                    Term::Constant(Value::Atom(sym("mary"))),
                ],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1); // One empty substitution = true

        // Query: ?- parent(john, jane). (ground query - should fail)
        let query2 = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![
                    Term::Constant(Value::Atom(sym("john"))),
                    Term::Constant(Value::Atom(sym("jane"))),
                ],
            ))],
        };

        let results2 = evaluate_query(&query2, &db);
        assert_eq!(results2.len(), 0); // No substitution = false
    }
}
