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
//! let results = evaluate_query(&query, &db, &storage);
//! // Returns: [{X -> john}, {X -> alice}]
//! ```

use datalog_grounding::satisfy_body;
use datalog_parser::{Query, Symbol};
use sql_storage::{DatalogContext, StorageEngine, Substitution};
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
pub fn evaluate_query<S: StorageEngine>(
    query: &Query,
    db: &DatalogContext,
    storage: &S,
) -> QueryResult {
    // Use satisfy_body from grounding module to find all substitutions
    satisfy_body(&query.body, db, storage)
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
    use datalog_parser::Term;

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
    use datalog_parser::{Atom, Literal, Term, Value};
    use sql_storage::MemoryEngine;

    fn sym(s: &str) -> Symbol {
        Symbol::new(s.to_string())
    }

    fn var_term(name: &str) -> Term {
        Term::Variable(sym(name))
    }

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(sym(name)))
    }

    fn int_term(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    // ===== Ground Query Tests =====

    #[test]
    fn test_query_ground_true() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- parent(john, mary).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("john"), atom_term("mary")],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 1, "Ground query should return one result");
    }

    #[test]
    fn test_query_ground_false() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- parent(alice, bob). (not in database)
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("alice"), atom_term("bob")],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 0, "False query should return empty result");
    }

    // ===== Variable Query Tests =====

    #[test]
    fn test_query_with_one_variable() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("alice"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("bob"), atom_term("sue")]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- parent(X, mary).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), atom_term("mary")],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 2, "Should find 2 parents of mary");

        // Verify the bindings
        let vars = query_variables(&query);
        for subst in &results {
            let bindings = extract_bindings(subst, &vars);
            assert_eq!(bindings.len(), 1);
            // X should be bound to either john or alice
            let x_binding = &bindings[0].1;
            assert!(
                x_binding.contains("john") || x_binding.contains("alice"),
                "X should be john or alice, got: {}",
                x_binding
            );
        }
    }

    #[test]
    fn test_query_with_multiple_variables() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("alice"), atom_term("bob")]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- parent(X, Y).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 2, "Should find 2 parent relationships");
    }

    // ===== Join Query Tests =====

    #[test]
    fn test_query_with_join() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("sue")]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- parent(X, Y), parent(Y, Z).
        // Should find: X=john, Y=mary, Z=sue
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 1, "Should find 1 grandparent relationship");
    }

    #[test]
    fn test_query_join_no_match() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("bob"), atom_term("sue")]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- parent(X, Y), parent(Y, Z).
        // No chains exist
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 0, "Should find no chains");
    }

    // ===== Negation Query Tests =====

    #[test]
    fn test_query_with_negation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(make_atom("person", vec![atom_term("john")]), &mut storage)
            .unwrap();
        db.insert(make_atom("person", vec![atom_term("mary")]), &mut storage)
            .unwrap();
        db.insert(make_atom("dead", vec![atom_term("john")]), &mut storage)
            .unwrap();

        // Query: ?- person(X), not dead(X).
        // Should find only mary
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("person", vec![var_term("X")])),
                Literal::Negative(make_atom("dead", vec![var_term("X")])),
            ],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 1, "Should find 1 living person");
    }

    // ===== Query Variables Extraction Tests =====

    #[test]
    fn test_query_variables_extraction() {
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("age", vec![var_term("Y"), var_term("A")])),
            ],
        };

        let vars = query_variables(&query);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&sym("X")));
        assert!(vars.contains(&sym("Y")));
        assert!(vars.contains(&sym("A")));
    }

    #[test]
    fn test_query_variables_compound_terms() {
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "has",
                vec![
                    var_term("X"),
                    Term::Compound(sym("item"), vec![var_term("Y"), int_term(5)]),
                ],
            ))],
        };

        let vars = query_variables(&query);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&sym("X")));
        assert!(vars.contains(&sym("Y")));
    }

    #[test]
    fn test_query_variables_no_variables() {
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "fact",
                vec![atom_term("a"), atom_term("b")],
            ))],
        };

        let vars = query_variables(&query);
        assert!(vars.is_empty());
    }

    // ===== Extract Bindings Tests =====

    #[test]
    fn test_extract_bindings() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();

        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 1);

        let vars = query_variables(&query);
        let bindings = extract_bindings(&results[0], &vars);

        assert_eq!(bindings.len(), 2);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_query_empty_database() {
        let db = DatalogContext::new();
        let storage = MemoryEngine::new();

        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "anything",
                vec![var_term("X")],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_with_integers() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("age", vec![atom_term("john"), int_term(30)]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("age", vec![atom_term("mary"), int_term(25)]),
            &mut storage,
        )
        .unwrap();

        // Query: ?- age(X, 30).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "age",
                vec![var_term("X"), int_term(30)],
            ))],
        };

        let results = evaluate_query(&query, &db, &storage);
        assert_eq!(results.len(), 1);
    }
}
