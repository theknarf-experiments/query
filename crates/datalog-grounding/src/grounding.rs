//! Rule grounding - generating ground instances of rules
//!
//! This module implements the grounding phase of logic programming evaluation.
//! Grounding replaces variables in rules with concrete values from the database.
//!
//! # Key Functions
//!
//! - `ground_rule`: Standard grounding for a single rule
//! - `ground_rule_semi_naive`: Optimized grounding using delta (newly derived facts)
//! - `satisfy_body`: Find all substitutions that satisfy a rule body
//! - `ground_choice_rule`: Ground choice rules with their elements
//!
//! # Example
//!
//! ```ignore
//! // Given rule: ancestor(X, Z) :- parent(X, Y), parent(Y, Z)
//! // And facts: parent(a, b), parent(b, c)
//! // Produces: ancestor(a, c)
//! let groundings = ground_rule(&rule, &db, &const_env);
//! ```

use datalog_builtins as builtins;
use datalog_parser::{Atom, Literal, Rule, Term};
use sql_storage::{FactDatabase, Substitution};

#[cfg(test)]
mod allocation_tracker {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator;

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

    pub fn reset() {
        ALLOCATIONS.store(0, Ordering::SeqCst);
    }

    pub fn allocations() -> usize {
        ALLOCATIONS.load(Ordering::SeqCst)
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.alloc_zeroed(layout)
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.realloc(ptr, layout, new_size)
        }
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL: allocation_tracker::CountingAllocator = allocation_tracker::CountingAllocator;

/// Ground a rule: generate all ground instances by substituting variables
/// For a rule like `ancestor(X, Z) :- parent(X, Y), parent(Y, Z)`
/// This finds all ways to satisfy the body and applies those substitutions to the head
pub fn ground_rule(rule: &Rule, db: &FactDatabase) -> Vec<Atom> {
    let mut results = Vec::new();

    // Get all substitutions that satisfy the entire body
    let substitutions = satisfy_body(&rule.body, db);

    // Apply each substitution to the head to get ground facts
    for subst in substitutions {
        let ground_head = subst.apply_atom(&rule.head);
        results.push(ground_head);
    }

    results
}

/// Find all substitutions that satisfy a conjunction of literals
pub fn satisfy_body(body: &[Literal], db: &FactDatabase) -> Vec<Substitution> {
    satisfy_body_with_selector(body, db, None, &|_, _| DatabaseSelection::Full)
}

enum DatabaseSelection {
    Full,
    Delta,
}

fn satisfy_body_with_selector<F>(
    body: &[Literal],
    full_db: &FactDatabase,
    delta: Option<&FactDatabase>,
    selector: &F,
) -> Vec<Substitution>
where
    F: Fn(usize, &Literal) -> DatabaseSelection,
{
    satisfy_body_with_selector_recursive(body, full_db, delta, selector, 0, &Substitution::new())
}

fn satisfy_body_with_selector_recursive<F>(
    body: &[Literal],
    full_db: &FactDatabase,
    delta: Option<&FactDatabase>,
    selector: &F,
    index: usize,
    current_subst: &Substitution,
) -> Vec<Substitution>
where
    F: Fn(usize, &Literal) -> DatabaseSelection,
{
    if index == body.len() {
        return vec![current_subst.clone()];
    }

    let literal = &body[index];

    match literal {
        Literal::Positive(atom) => {
            if let Some(builtin) = builtins::parse_builtin(atom) {
                // Built-ins act as filters - evaluate them after satisfying the rest.
                let rest_substs = satisfy_body_with_selector_recursive(
                    body,
                    full_db,
                    delta,
                    selector,
                    index + 1,
                    current_subst,
                );
                let mut result = Vec::new();

                for subst in rest_substs {
                    let applied_builtin = apply_subst_to_builtin(&subst, &builtin);
                    if let Some(true) = builtins::eval_builtin(&applied_builtin, &subst) {
                        result.push(subst);
                    }
                }

                result
            } else {
                // Apply the current substitution to the atom before querying.
                let grounded_atom = current_subst.apply_atom(atom);
                let db = match selector(index, literal) {
                    DatabaseSelection::Full => full_db,
                    DatabaseSelection::Delta => delta.unwrap_or(full_db),
                };
                let mut result = Vec::new();

                for atom_subst in db.query(&grounded_atom) {
                    if let Some(combined) = combine_substs(current_subst, &atom_subst) {
                        let mut rest_results = satisfy_body_with_selector_recursive(
                            body,
                            full_db,
                            delta,
                            selector,
                            index + 1,
                            &combined,
                        );
                        result.append(&mut rest_results);
                    }
                }

                result
            }
        }
        Literal::Negative(atom) => {
            // Negation filters substitutions produced by the rest of the body.
            let rest_substs = satisfy_body_with_selector_recursive(
                body,
                full_db,
                delta,
                selector,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();

            for subst in rest_substs {
                let grounded_atom = subst.apply_atom(atom);
                if !database_has_match(full_db, &grounded_atom) {
                    result.push(subst);
                }
            }

            result
        }
        Literal::Comparison(comp) => {
            // Comparisons act as filters - evaluate after satisfying the rest
            let rest_substs = satisfy_body_with_selector_recursive(
                body,
                full_db,
                delta,
                selector,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();

            for subst in rest_substs {
                // Evaluate the comparison with the substitution
                let left = subst.apply(&comp.left);
                let right = subst.apply(&comp.right);
                let builtin =
                    builtins::BuiltIn::Comparison(comp_op_to_builtin(&comp.op), left, right);
                if let Some(true) = builtins::eval_builtin(&builtin, &subst) {
                    result.push(subst);
                }
            }

            result
        }
    }
}

/// Convert AST comparison op to builtins comparison op
fn comp_op_to_builtin(op: &datalog_parser::ComparisonOp) -> builtins::CompOp {
    use datalog_parser::ComparisonOp;
    match op {
        ComparisonOp::Equal => builtins::CompOp::Eq,
        ComparisonOp::NotEqual => builtins::CompOp::Neq,
        ComparisonOp::LessThan => builtins::CompOp::Lt,
        ComparisonOp::LessOrEqual => builtins::CompOp::Lte,
        ComparisonOp::GreaterThan => builtins::CompOp::Gt,
        ComparisonOp::GreaterOrEqual => builtins::CompOp::Gte,
    }
}

/// Combine two substitutions, returning `None` if they conflict.
fn combine_substs(s1: &Substitution, s2: &Substitution) -> Option<Substitution> {
    let mut combined = s1.clone();

    for (var, term) in s2.iter() {
        // Apply bindings from both substitutions before comparing.
        let term_applied_s2 = s2.apply(term);
        let candidate = combined.apply(&term_applied_s2);

        if let Some(existing) = combined.get(var) {
            let existing_applied_s2 = s2.apply(existing);
            let existing_resolved = combined.apply(&existing_applied_s2);

            if existing_resolved != candidate {
                return None;
            }
        }

        combined.bind(*var, candidate);
    }

    Some(combined)
}

fn database_has_match(db: &FactDatabase, atom: &Atom) -> bool {
    if atom_is_ground(atom) {
        db.contains(atom)
    } else {
        !db.query(atom).is_empty()
    }
}

fn atom_is_ground(atom: &Atom) -> bool {
    atom.terms.iter().all(term_is_ground)
}

fn term_is_ground(term: &Term) -> bool {
    match term {
        Term::Variable(_) => false,
        Term::Constant(_) => true,
        Term::Compound(_, args) => args.iter().all(term_is_ground),
    }
}

/// Apply substitution to a built-in predicate
fn apply_subst_to_builtin(subst: &Substitution, builtin: &builtins::BuiltIn) -> builtins::BuiltIn {
    match builtin {
        builtins::BuiltIn::Comparison(op, left, right) => {
            builtins::BuiltIn::Comparison(op.clone(), subst.apply(left), subst.apply(right))
        }
        builtins::BuiltIn::True => builtins::BuiltIn::True,
        builtins::BuiltIn::Fail => builtins::BuiltIn::Fail,
    }
}

/// Ground a rule using semi-naive evaluation
/// For multi-literal rules, this tries using delta at each position
/// and the full database for other positions
pub fn ground_rule_semi_naive(
    rule: &Rule,
    delta: &FactDatabase,
    full_db: &FactDatabase,
) -> Vec<Atom> {
    let mut results = Vec::new();

    if rule.body.is_empty() {
        // No body - just return the head
        return vec![rule.head.clone()];
    }

    // For each position i in the body, use delta for position i and full_db for others
    for delta_pos in 0..rule.body.len() {
        let substs = satisfy_body_mixed(&rule.body, delta, full_db, delta_pos);
        for subst in substs {
            let ground_head = subst.apply_atom(&rule.head);
            results.push(ground_head);
        }
    }

    results
}

/// Satisfy body literals using delta for one position and full DB for others
fn satisfy_body_mixed(
    body: &[Literal],
    delta: &FactDatabase,
    full_db: &FactDatabase,
    delta_pos: usize,
) -> Vec<Substitution> {
    satisfy_body_with_selector(body, full_db, Some(delta), &|index, literal| {
        if index == delta_pos {
            match literal {
                Literal::Positive(atom) if builtins::parse_builtin(atom).is_some() => {
                    DatabaseSelection::Full
                }
                Literal::Positive(_) => DatabaseSelection::Delta,
                Literal::Negative(_) => DatabaseSelection::Full,
                Literal::Comparison(_) => DatabaseSelection::Full,
            }
        } else {
            DatabaseSelection::Full
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::{Atom, Rule, Symbol, Value};

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
    fn test_ground_simple_rule() {
        // parent(john, mary). parent(mary, jane).
        // ancestor(X, Y) :- parent(X, Y).
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

        let rule = Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        };

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_transitive_rule() {
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "ancestor",
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

        let rule = Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
            body: vec![
                Literal::Positive(make_atom("ancestor", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        };

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate.as_ref(), "ancestor");
    }

    #[test]
    fn test_negation() {
        // not_parent(X) :- person(X), not parent(X, _).
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "person",
            vec![Term::Constant(Value::Atom(sym("john")))],
        ))
        .unwrap();
        db.insert(make_atom(
            "person",
            vec![Term::Constant(Value::Atom(sym("mary")))],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![
                Term::Constant(Value::Atom(sym("john"))),
                Term::Constant(Value::Atom(sym("jane"))),
            ],
        ))
        .unwrap();

        // Check that mary (who is not a parent) would be included, john (who is) would not
        let body = vec![
            Literal::Positive(make_atom("person", vec![var_term("X")])),
            Literal::Negative(make_atom("parent", vec![var_term("X"), var_term("_Y")])),
        ];

        let substs = satisfy_body(&body, &db);
        // Mary should be the only one without a parent entry
        assert_eq!(substs.len(), 1);
    }
}
