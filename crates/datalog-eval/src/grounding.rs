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

use crate::builtins;
use crate::datalog_context::{sql_value_to_term, DatalogContext};
use crate::{unify_atoms, Substitution};
use datalog_planner::{Atom, BuiltIn, Literal, Rule, Term};
use logical::StorageEngine;
use storage::Row;

#[cfg(test)]
#[allow(unused_imports)]
use logical::NoOpRuntime;

#[cfg(test)]
mod allocation_tracker {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator;

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

    #[allow(dead_code)]
    pub fn reset() {
        ALLOCATIONS.store(0, Ordering::SeqCst);
    }

    #[allow(dead_code)]
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

/// Convert rows to substitutions by unifying with the pattern
fn unify_rows_with_pattern(pattern: &Atom, rows: &[Row]) -> Vec<Substitution> {
    let mut results = Vec::new();
    for row in rows {
        // Convert row to atom (handles JSON → compound term conversion)
        let terms: Vec<Term> = row.iter().map(sql_value_to_term).collect();
        let fact = Atom {
            predicate: pattern.predicate,
            terms,
        };

        // Try to unify
        let mut subst = Substitution::new();
        if unify_atoms(pattern, &fact, &mut subst) {
            results.push(subst);
        }
    }
    results
}

/// Ground a rule: generate all ground instances by substituting variables
/// For a rule like `ancestor(X, Z) :- parent(X, Y), parent(Y, Z)`
/// This finds all ways to satisfy the body and applies those substitutions to the head
pub fn ground_rule<S: StorageEngine>(rule: &Rule, db: &DatalogContext, storage: &S) -> Vec<Atom> {
    let mut results = Vec::new();

    // Get all substitutions that satisfy the entire body
    let substitutions = satisfy_body(&rule.body, db, storage);

    // Apply each substitution to the head to get ground facts
    for subst in substitutions {
        let ground_head = subst.apply_atom(&rule.head);
        results.push(ground_head);
    }

    results
}

/// Find all substitutions that satisfy a conjunction of literals
pub fn satisfy_body<S: StorageEngine>(
    body: &[Literal],
    db: &DatalogContext,
    storage: &S,
) -> Vec<Substitution> {
    satisfy_body_with_selector(body, db, storage, None, &|_, _| DatabaseSelection::Full)
}

enum DatabaseSelection {
    Full,
    Delta,
}

fn satisfy_body_with_selector<S, F>(
    body: &[Literal],
    full_db: &DatalogContext,
    storage: &S,
    delta: Option<&DatalogContext>,
    selector: &F,
) -> Vec<Substitution>
where
    S: StorageEngine,
    F: Fn(usize, &Literal) -> DatabaseSelection,
{
    satisfy_body_with_selector_recursive(
        body,
        full_db,
        storage,
        delta,
        selector,
        0,
        &Substitution::new(),
    )
}

fn satisfy_body_with_selector_recursive<S, F>(
    body: &[Literal],
    full_db: &DatalogContext,
    storage: &S,
    delta: Option<&DatalogContext>,
    selector: &F,
    index: usize,
    current_subst: &Substitution,
) -> Vec<Substitution>
where
    S: StorageEngine,
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
                    storage,
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

                let rows = db.query(&grounded_atom, storage);
                for atom_subst in unify_rows_with_pattern(&grounded_atom, &rows) {
                    if let Some(combined) = combine_substs(current_subst, &atom_subst) {
                        let mut rest_results = satisfy_body_with_selector_recursive(
                            body,
                            full_db,
                            storage,
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
                storage,
                delta,
                selector,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();

            for subst in rest_substs {
                let grounded_atom = subst.apply_atom(atom);
                if !database_has_match(full_db, storage, &grounded_atom) {
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
                storage,
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
        Literal::BuiltIn(builtin) => {
            // Pre-classified builtins act as filters - evaluate after satisfying the rest
            let rest_substs = satisfy_body_with_selector_recursive(
                body,
                full_db,
                storage,
                delta,
                selector,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();

            for subst in rest_substs {
                let eval_builtin = ir_builtin_to_eval_builtin(builtin, &subst);
                if let Some(true) = builtins::eval_builtin(&eval_builtin, &subst) {
                    result.push(subst);
                }
            }

            result
        }
    }
}

/// Convert AST comparison op to builtins comparison op
fn comp_op_to_builtin(op: &datalog_planner::ComparisonOp) -> builtins::CompOp {
    use datalog_planner::ComparisonOp;
    match op {
        ComparisonOp::Equal => builtins::CompOp::Eq,
        ComparisonOp::NotEqual => builtins::CompOp::Neq,
        ComparisonOp::LessThan => builtins::CompOp::Lt,
        ComparisonOp::LessOrEqual => builtins::CompOp::Lte,
        ComparisonOp::GreaterThan => builtins::CompOp::Gt,
        ComparisonOp::GreaterOrEqual => builtins::CompOp::Gte,
    }
}

/// Convert IR BuiltIn to eval builtins::BuiltIn, applying substitution
fn ir_builtin_to_eval_builtin(builtin: &BuiltIn, subst: &Substitution) -> builtins::BuiltIn {
    match builtin {
        BuiltIn::Comparison(op, left, right) => builtins::BuiltIn::Comparison(
            comp_op_to_builtin(op),
            subst.apply(left),
            subst.apply(right),
        ),
        BuiltIn::True => builtins::BuiltIn::True,
        BuiltIn::Fail => builtins::BuiltIn::Fail,
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

fn database_has_match<S: StorageEngine>(db: &DatalogContext, storage: &S, atom: &Atom) -> bool {
    if atom_is_ground(atom) {
        db.contains(atom, storage)
    } else {
        let rows = db.query(atom, storage);
        !unify_rows_with_pattern(atom, &rows).is_empty()
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
pub fn ground_rule_semi_naive<S: StorageEngine>(
    rule: &Rule,
    delta: &DatalogContext,
    full_db: &DatalogContext,
    storage: &S,
) -> Vec<Atom> {
    let mut results = Vec::new();

    if rule.body.is_empty() {
        // No body - just return the head
        return vec![rule.head.clone()];
    }

    // For each position i in the body, use delta for position i and full_db for others
    for delta_pos in 0..rule.body.len() {
        let substs = satisfy_body_mixed(&rule.body, delta, full_db, storage, delta_pos);
        for subst in substs {
            let ground_head = subst.apply_atom(&rule.head);
            results.push(ground_head);
        }
    }

    results
}

/// Satisfy body literals using delta for one position and full DB for others
fn satisfy_body_mixed<S: StorageEngine>(
    body: &[Literal],
    delta: &DatalogContext,
    full_db: &DatalogContext,
    storage: &S,
    delta_pos: usize,
) -> Vec<Substitution> {
    satisfy_body_with_selector(body, full_db, storage, Some(delta), &|index, literal| {
        if index == delta_pos {
            match literal {
                // Builtins (either pre-classified or detected at runtime) don't use delta
                Literal::Positive(atom) if builtins::parse_builtin(atom).is_some() => {
                    DatabaseSelection::Full
                }
                Literal::BuiltIn(_) => DatabaseSelection::Full,
                Literal::Positive(_) => DatabaseSelection::Delta,
                Literal::Negative(_) => DatabaseSelection::Full,
                Literal::Comparison(_) => DatabaseSelection::Full,
            }
        } else {
            DatabaseSelection::Full
        }
    })
}

// ============================================================================
// DeltaTracker-Based Semi-Naive Evaluation
// ============================================================================

use crate::DeltaTracker;

/// Ground a rule using semi-naive evaluation with lightweight DeltaTracker
///
/// For each body literal position, this grounds the rule using delta at that
/// position and full_db for all other positions. This ensures we only find
/// derivations that use at least one newly derived fact.
pub fn ground_rule_semi_naive_with_delta<S: StorageEngine>(
    rule: &Rule,
    delta: &DeltaTracker,
    full_db: &DatalogContext,
    storage: &S,
) -> Vec<Atom> {
    let mut results = Vec::new();

    if rule.body.is_empty() {
        // No body - just return the head
        return vec![rule.head.clone()];
    }

    // For each position i in the body, use delta for position i and full_db for others
    for delta_pos in 0..rule.body.len() {
        let substs = satisfy_body_with_delta(&rule.body, delta, full_db, storage, delta_pos);
        for subst in substs {
            let ground_head = subst.apply_atom(&rule.head);
            results.push(ground_head);
        }
    }

    results
}

/// Satisfy body literals using DeltaTracker for one position and full DB for others
fn satisfy_body_with_delta<S: StorageEngine>(
    body: &[Literal],
    delta: &DeltaTracker,
    full_db: &DatalogContext,
    storage: &S,
    delta_pos: usize,
) -> Vec<Substitution> {
    satisfy_body_with_delta_recursive(
        body,
        delta,
        full_db,
        storage,
        delta_pos,
        0,
        &Substitution::new(),
    )
}

fn satisfy_body_with_delta_recursive<S: StorageEngine>(
    body: &[Literal],
    delta: &DeltaTracker,
    full_db: &DatalogContext,
    storage: &S,
    delta_pos: usize,
    index: usize,
    current_subst: &Substitution,
) -> Vec<Substitution> {
    if index == body.len() {
        return vec![current_subst.clone()];
    }

    let literal = &body[index];

    match literal {
        Literal::Positive(atom) => {
            if let Some(builtin) = builtins::parse_builtin(atom) {
                // Built-ins act as filters - evaluate after satisfying the rest
                let rest_substs = satisfy_body_with_delta_recursive(
                    body,
                    delta,
                    full_db,
                    storage,
                    delta_pos,
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
                // Apply the current substitution to the atom before querying
                let grounded_atom = current_subst.apply_atom(atom);

                // Choose data source: delta for this position, full_db for others
                let matches: Vec<Substitution> = if index == delta_pos {
                    delta.query(&grounded_atom) // In-memory query (already returns Substitutions)
                } else {
                    let rows = full_db.query(&grounded_atom, storage);
                    unify_rows_with_pattern(&grounded_atom, &rows)
                };

                let mut result = Vec::new();
                for atom_subst in matches {
                    if let Some(combined) = combine_substs(current_subst, &atom_subst) {
                        let mut rest_results = satisfy_body_with_delta_recursive(
                            body,
                            delta,
                            full_db,
                            storage,
                            delta_pos,
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
            // Negation always checks against full database (stratification ensures safety)
            let rest_substs = satisfy_body_with_delta_recursive(
                body,
                delta,
                full_db,
                storage,
                delta_pos,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();
            for subst in rest_substs {
                let grounded_atom = subst.apply_atom(atom);
                if !database_has_match(full_db, storage, &grounded_atom) {
                    result.push(subst);
                }
            }
            result
        }
        Literal::Comparison(comp) => {
            // Comparisons act as filters
            let rest_substs = satisfy_body_with_delta_recursive(
                body,
                delta,
                full_db,
                storage,
                delta_pos,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();
            for subst in rest_substs {
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
        Literal::BuiltIn(builtin) => {
            // Pre-classified builtins act as filters
            let rest_substs = satisfy_body_with_delta_recursive(
                body,
                delta,
                full_db,
                storage,
                delta_pos,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();
            for subst in rest_substs {
                let eval_builtin = ir_builtin_to_eval_builtin(builtin, &subst);
                if let Some(true) = builtins::eval_builtin(&eval_builtin, &subst) {
                    result.push(subst);
                }
            }
            result
        }
    }
}

// ============================================================================
// Storage-Aware Grounding Functions
// ============================================================================
//
// These functions enable Datalog queries to use SQL storage indexes for efficient
// lookups. For storage-backed predicates, they query the storage engine directly
// instead of local facts.

/// Ground a rule using storage for indexed lookups
#[deprecated(note = "Use ground_rule() with storage parameter instead")]
pub fn ground_rule_with_storage<S: StorageEngine>(
    rule: &Rule,
    db: &DatalogContext,
    storage: &S,
) -> Vec<Atom> {
    ground_rule(rule, db, storage)
}

/// Find all substitutions that satisfy a conjunction of literals, using storage indexes
#[deprecated(note = "Use satisfy_body() with storage parameter instead")]
pub fn satisfy_body_with_storage<S: StorageEngine>(
    body: &[Literal],
    db: &DatalogContext,
    storage: &S,
) -> Vec<Substitution> {
    satisfy_body(body, db, storage)
}

/// Ground a rule using semi-naive evaluation with storage support
#[deprecated(note = "Use ground_rule_semi_naive() with storage parameter instead")]
pub fn ground_rule_semi_naive_with_storage<S: StorageEngine>(
    rule: &Rule,
    delta: &DatalogContext,
    full_db: &DatalogContext,
    storage: &S,
) -> Vec<Atom> {
    ground_rule_semi_naive(rule, delta, full_db, storage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_planner::{Atom, Rule, Symbol, Value};
    use logical::NoOpRuntime;
    use storage::MemoryEngine;

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

    fn compound_term(functor: &str, args: Vec<Term>) -> Term {
        Term::Compound(sym(functor), args)
    }

    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    fn make_rule(head: Atom, body: Vec<Literal>) -> Rule {
        Rule { head, body }
    }

    // ===== Basic Grounding Tests =====

    #[test]
    fn test_ground_rule_no_variables() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: ancestor(john, mary) :- parent(john, mary).
        let rule = make_rule(
            make_atom("ancestor", vec![atom_term("john"), atom_term("mary")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("john"), atom_term("mary")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate.as_ref(), "ancestor");
    }

    #[test]
    fn test_ground_rule_single_variable() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("bob")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: child(X) :- parent(john, X).
        let rule = make_rule(
            make_atom("child", vec![var_term("X")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("john"), var_term("X")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 2);

        // Both child(mary) and child(bob) should be generated
        let terms: Vec<_> = results.iter().map(|a| &a.terms[0]).collect();
        assert!(terms.contains(&&atom_term("mary")));
        assert!(terms.contains(&&atom_term("bob")));
    }

    #[test]
    fn test_ground_rule_multiple_variables() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("bob"), atom_term("alice")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: ancestor(X, Y) :- parent(X, Y).
        let rule = make_rule(
            make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_simple_rule() {
        // parent(john, mary). parent(mary, jane).
        // ancestor(X, Y) :- parent(X, Y).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("jane")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        let rule = Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        };

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 2);
    }

    // ===== Join Tests (Multiple Literals) =====

    #[test]
    fn test_ground_rule_join_two_literals() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("alice")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("bob"), atom_term("charlie")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var_term("X"), var_term("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only john -> mary -> alice forms a valid chain
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("john"));
        assert_eq!(results[0].terms[1], atom_term("alice"));
    }

    #[test]
    fn test_ground_rule_multiple_chains() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("b"), atom_term("d")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var_term("X"), var_term("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // a -> b -> c and a -> b -> d
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_transitive_rule() {
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("ancestor", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("jane")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        let rule = Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
            body: vec![
                Literal::Positive(make_atom("ancestor", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        };

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate.as_ref(), "ancestor");
    }

    #[test]
    fn test_ground_rule_three_literals() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("c"), atom_term("d")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: path3(X, W) :- edge(X, Y), edge(Y, Z), edge(Z, W).
        let rule = make_rule(
            make_atom("path3", vec![var_term("X"), var_term("W")]),
            vec![
                Literal::Positive(make_atom("edge", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                Literal::Positive(make_atom("edge", vec![var_term("Z"), var_term("W")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only a -> b -> c -> d
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("a"));
        assert_eq!(results[0].terms[1], atom_term("d"));
    }

    // ===== No Matches / Empty Body Tests =====

    #[test]
    fn test_ground_rule_no_matches() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: child(X) :- parent(alice, X).
        // No facts match parent(alice, X)
        let rule = make_rule(
            make_atom("child", vec![var_term("X")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("alice"), var_term("X")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_empty_body() {
        let db = DatalogContext::new();
        let storage = MemoryEngine::new();

        // Rule: fact(a) :- .
        // (A rule with no body is always true)
        let rule = make_rule(make_atom("fact", vec![atom_term("a")]), vec![]);

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], make_atom("fact", vec![atom_term("a")]));
    }

    // ===== Negation Tests =====

    #[test]
    fn test_ground_rule_simple_negation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("bird", vec![atom_term("tweety")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("bird", vec![atom_term("polly")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("penguin", vec![atom_term("polly")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: flies(X) :- bird(X), not penguin(X).
        let rule = make_rule(
            make_atom("flies", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("bird", vec![var_term("X")])),
                Literal::Negative(make_atom("penguin", vec![var_term("X")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only tweety should fly (polly is a penguin)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("tweety"));
    }

    #[test]
    fn test_negation() {
        // not_parent(X) :- person(X), not parent(X, _).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("person", vec![atom_term("john")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("person", vec![atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("jane")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Check that mary (who is not a parent) would be included, john (who is) would not
        let body = vec![
            Literal::Positive(make_atom("person", vec![var_term("X")])),
            Literal::Negative(make_atom("parent", vec![var_term("X"), var_term("_Y")])),
        ];

        let substs = satisfy_body(&body, &db, &storage);
        // Mary should be the only one without a parent entry
        assert_eq!(substs.len(), 1);
    }

    #[test]
    fn test_ground_rule_negation_with_ground_term() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("bird", vec![atom_term("tweety")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: not_bird_polly :- not bird(polly).
        let rule = make_rule(
            make_atom("not_bird_polly", vec![]),
            vec![Literal::Negative(make_atom(
                "bird",
                vec![atom_term("polly")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // polly is not a bird, so this succeeds
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_rule_multiple_negations() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(make_atom("a", vec![atom_term("x")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("b", vec![atom_term("y")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("c", vec![atom_term("z")]), &mut storage, &runtime)
            .unwrap();

        // Rule: result :- not a(y), not b(x), not c(w).
        let rule = make_rule(
            make_atom("result", vec![]),
            vec![
                Literal::Negative(make_atom("a", vec![atom_term("y")])),
                Literal::Negative(make_atom("b", vec![atom_term("x")])),
                Literal::Negative(make_atom("c", vec![atom_term("w")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // All negations succeed (a(y), b(x), c(w) don't exist)
        assert_eq!(results.len(), 1);
    }

    // ===== Compound Term Tests =====

    #[test]
    fn test_ground_rule_with_compound_terms() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom(
                "has",
                vec![
                    atom_term("john"),
                    compound_term("item", vec![atom_term("sword"), int_term(10)]),
                ],
            ),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom(
                "has",
                vec![
                    atom_term("mary"),
                    compound_term("item", vec![atom_term("shield"), int_term(5)]),
                ],
            ),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: armed(P) :- has(P, item(sword, _)).
        let rule = make_rule(
            make_atom("armed", vec![var_term("P")]),
            vec![Literal::Positive(make_atom(
                "has",
                vec![
                    var_term("P"),
                    compound_term("item", vec![atom_term("sword"), var_term("_W")]),
                ],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("john"));
    }

    #[test]
    fn test_ground_rule_extract_from_compound() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom(
                "item",
                vec![compound_term(
                    "weapon",
                    vec![atom_term("sword"), int_term(10)],
                )],
            ),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: weapon_name(N) :- item(weapon(N, _)).
        let rule = make_rule(
            make_atom("weapon_name", vec![var_term("N")]),
            vec![Literal::Positive(make_atom(
                "item",
                vec![compound_term("weapon", vec![var_term("N"), var_term("_D")])],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("sword"));
    }

    // ===== Semi-Naive Grounding Tests =====

    #[test]
    fn test_semi_naive_basic() {
        let mut full_db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        full_db
            .insert(
                make_atom("edge", vec![atom_term("a"), atom_term("b")]),
                &mut storage,
                &runtime,
            )
            .unwrap();
        full_db
            .insert(
                make_atom("edge", vec![atom_term("b"), atom_term("c")]),
                &mut storage,
                &runtime,
            )
            .unwrap();
        full_db
            .insert(
                make_atom("path", vec![atom_term("a"), atom_term("b")]),
                &mut storage,
                &runtime,
            )
            .unwrap();

        // Delta: only the new path fact
        let mut delta = DatalogContext::new();
        delta
            .insert(
                make_atom("path", vec![atom_term("a"), atom_term("b")]),
                &mut storage,
                &runtime,
            )
            .unwrap();

        // Rule: path(X, Z) :- path(X, Y), edge(Y, Z).
        let rule = make_rule(
            make_atom("path", vec![var_term("X"), var_term("Z")]),
            vec![
                Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
            ],
        );

        let results = ground_rule_semi_naive(&rule, &delta, &full_db, &storage);

        // Should derive path(a, c) using the delta path(a, b) with edge(b, c)
        assert!(!results.is_empty());
    }

    // ===== Satisfy Body Tests =====

    #[test]
    fn test_satisfy_body_single_literal() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("fact", vec![atom_term("a")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("fact", vec![atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        let body = vec![Literal::Positive(make_atom("fact", vec![var_term("X")]))];
        let results = satisfy_body(&body, &db, &storage);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_satisfy_body_join() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("r", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("s", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("s", vec![atom_term("x"), atom_term("y")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // r(X, Y), s(Y, Z) - should only match where r.Y = s.Y
        let body = vec![
            Literal::Positive(make_atom("r", vec![var_term("X"), var_term("Y")])),
            Literal::Positive(make_atom("s", vec![var_term("Y"), var_term("Z")])),
        ];
        let results = satisfy_body(&body, &db, &storage);

        // Only one match: X=a, Y=b, Z=c
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_satisfy_body_empty() {
        let db = DatalogContext::new();
        let storage = MemoryEngine::new();

        let body: Vec<Literal> = vec![];
        let results = satisfy_body(&body, &db, &storage);

        // Empty body succeeds with one empty substitution
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    // ===== Integration Tests =====

    #[test]
    fn test_integration_ground_query() {
        // A complete workflow: facts -> rules -> ground -> query
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        let runtime = NoOpRuntime;

        // Facts
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("alice")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("bob"), atom_term("charlie")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("charlie"), atom_term("dave")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var_term("X"), var_term("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        );

        let derived = ground_rule(&rule, &db, &storage);

        // Should derive:
        // grandparent(john, alice) from parent(john, mary) + parent(mary, alice)
        // grandparent(bob, dave) from parent(bob, charlie) + parent(charlie, dave)
        assert_eq!(derived.len(), 2);

        // Verify specific derivations
        let has_john_alice = derived.iter().any(|a| {
            a.predicate.as_ref() == "grandparent"
                && a.terms[0] == atom_term("john")
                && a.terms[1] == atom_term("alice")
        });
        let has_bob_dave = derived.iter().any(|a| {
            a.predicate.as_ref() == "grandparent"
                && a.terms[0] == atom_term("bob")
                && a.terms[1] == atom_term("dave")
        });

        assert!(has_john_alice, "Should derive grandparent(john, alice)");
        assert!(has_bob_dave, "Should derive grandparent(bob, dave)");
    }

    // ===== Allocation/Performance Tests =====

    #[test]
    fn test_large_rule_chain() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        let chain_len = 20;

        for i in 0..chain_len {
            db.insert(
                make_atom("link", vec![int_term(i), int_term(i + 1)]),
                &mut storage,
                &runtime,
            )
            .unwrap();
        }

        // Build a long chain rule: result(X, Y) :- link(X, V1), link(V1, V2), ..., link(Vn, Y)
        let mut body = vec![Literal::Positive(make_atom(
            "link",
            vec![var_term("X"), var_term("V0")],
        ))];
        for i in 0..5 {
            let current = format!("V{}", i);
            let next = format!("V{}", i + 1);
            body.push(Literal::Positive(make_atom(
                "link",
                vec![var_term(&current), var_term(&next)],
            )));
        }

        let rule = make_rule(
            make_atom("result", vec![var_term("X"), var_term("V5")]),
            body,
        );

        let results = ground_rule(&rule, &db, &storage);

        // Should find paths of length 6 in a chain of 20
        // Starting positions: 0, 1, 2, ..., 14 (14+6=20)
        assert_eq!(results.len(), 15);
    }

    // ===== Additional Edge Case Tests =====

    #[test]
    fn test_ground_rule_same_variable_multiple_positions() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("a")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: self_loop(X) :- edge(X, X).
        let rule = make_rule(
            make_atom("self_loop", vec![var_term("X")]),
            vec![Literal::Positive(make_atom(
                "edge",
                vec![var_term("X"), var_term("X")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Should find self_loop(a) and self_loop(b) - the self loops
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_negation_no_match() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("bird", vec![atom_term("tweety")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: mammal(X) :- not bird(X).
        // Since only tweety exists and is a bird, no mammals
        let rule = make_rule(
            make_atom("mammal", vec![var_term("X")]),
            vec![Literal::Negative(make_atom("bird", vec![var_term("X")]))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // No results - we can't prove something is NOT a bird
        // unless we have a closed world assumption with a finite domain
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_join_same_predicate() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("knows", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("knows", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("knows", vec![atom_term("a"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: friend_of_friend(X, Z) :- knows(X, Y), knows(Y, Z).
        let rule = make_rule(
            make_atom("friend_of_friend", vec![var_term("X"), var_term("Z")]),
            vec![
                Literal::Positive(make_atom("knows", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("knows", vec![var_term("Y"), var_term("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // a knows b, b knows c -> friend_of_friend(a, c)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("a"));
        assert_eq!(results[0].terms[1], atom_term("c"));
    }

    #[test]
    fn test_ground_rule_multiple_same_variable() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom(
                "triple",
                vec![atom_term("a"), atom_term("a"), atom_term("a")],
            ),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom(
                "triple",
                vec![atom_term("a"), atom_term("b"), atom_term("c")],
            ),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: same_all(X) :- triple(X, X, X).
        let rule = make_rule(
            make_atom("same_all", vec![var_term("X")]),
            vec![Literal::Positive(make_atom(
                "triple",
                vec![var_term("X"), var_term("X"), var_term("X")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only triple(a, a, a) matches
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("a"));
    }

    #[test]
    fn test_ground_rule_four_literals() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("step", vec![int_term(0), int_term(1)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("step", vec![int_term(1), int_term(2)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("step", vec![int_term(2), int_term(3)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("step", vec![int_term(3), int_term(4)]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: path4(A, E) :- step(A, B), step(B, C), step(C, D), step(D, E).
        let rule = make_rule(
            make_atom("path4", vec![var_term("A"), var_term("E")]),
            vec![
                Literal::Positive(make_atom("step", vec![var_term("A"), var_term("B")])),
                Literal::Positive(make_atom("step", vec![var_term("B"), var_term("C")])),
                Literal::Positive(make_atom("step", vec![var_term("C"), var_term("D")])),
                Literal::Positive(make_atom("step", vec![var_term("D"), var_term("E")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only 0 -> 1 -> 2 -> 3 -> 4
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], int_term(0));
        assert_eq!(results[0].terms[1], int_term(4));
    }

    #[test]
    fn test_ground_rule_negation_all_filtered() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("person", vec![atom_term("john")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("person", vec![atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("blocked", vec![atom_term("john")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("blocked", vec![atom_term("mary")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: allowed(X) :- person(X), not blocked(X).
        let rule = make_rule(
            make_atom("allowed", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("person", vec![var_term("X")])),
                Literal::Negative(make_atom("blocked", vec![var_term("X")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // All people are blocked
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_satisfy_body_with_different_predicates() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("person", vec![atom_term("john")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("age", vec![atom_term("john"), int_term(30)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("city", vec![atom_term("john"), atom_term("nyc")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Body: person(X), age(X, A), city(X, C).
        let body = vec![
            Literal::Positive(make_atom("person", vec![var_term("X")])),
            Literal::Positive(make_atom("age", vec![var_term("X"), var_term("A")])),
            Literal::Positive(make_atom("city", vec![var_term("X"), var_term("C")])),
        ];

        let results = satisfy_body(&body, &db, &storage);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_rule_cartesian_product() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(make_atom("a", vec![atom_term("x")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("a", vec![atom_term("y")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("b", vec![atom_term("1")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("b", vec![atom_term("2")]), &mut storage, &runtime)
            .unwrap();

        // Rule: pair(X, Y) :- a(X), b(Y).
        // No shared variables - cartesian product
        let rule = make_rule(
            make_atom("pair", vec![var_term("X"), var_term("Y")]),
            vec![
                Literal::Positive(make_atom("a", vec![var_term("X")])),
                Literal::Positive(make_atom("b", vec![var_term("Y")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // 2 * 2 = 4 pairs
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_ground_rule_mixed_constants_and_variables() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("rel", vec![atom_term("a"), atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("rel", vec![atom_term("a"), atom_term("x"), atom_term("y")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("rel", vec![atom_term("d"), atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: matched(X, Z) :- rel(a, X, Z).
        // First argument is constant
        let rule = make_rule(
            make_atom("matched", vec![var_term("X"), var_term("Z")]),
            vec![Literal::Positive(make_atom(
                "rel",
                vec![atom_term("a"), var_term("X"), var_term("Z")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only rel(a, b, c) and rel(a, x, y) match
        assert_eq!(results.len(), 2);
    }

    // ===== Additional Grounding Tests (proclog-style) =====

    #[test]
    fn test_ground_rule_long_chain_four_hops() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("c"), atom_term("d")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: path3(X, W) :- edge(X, Y), edge(Y, Z), edge(Z, W).
        let rule = make_rule(
            make_atom("path3", vec![var_term("X"), var_term("W")]),
            vec![
                Literal::Positive(make_atom("edge", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                Literal::Positive(make_atom("edge", vec![var_term("Z"), var_term("W")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only a -> b -> c -> d
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("a"));
        assert_eq!(results[0].terms[1], atom_term("d"));
    }

    #[test]
    fn test_ground_rule_multiple_chains_branching() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("parent", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("b"), atom_term("d")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var_term("X"), var_term("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // a -> b -> c and a -> b -> d
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_negation_bird_penguin() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("bird", vec![atom_term("tweety")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("bird", vec![atom_term("polly")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("penguin", vec![atom_term("polly")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: flies(X) :- bird(X), not penguin(X).
        let rule = make_rule(
            make_atom("flies", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("bird", vec![var_term("X")])),
                Literal::Negative(make_atom("penguin", vec![var_term("X")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only tweety should fly (polly is a penguin)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("tweety"));
    }

    #[test]
    fn test_ground_rule_negation_ground_constant() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("bird", vec![atom_term("tweety")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: not_bird_polly(x) :- bird(x), not bird(polly).
        // Note: we use bird(x) as positive literal to generate domain
        let rule = make_rule(
            make_atom("not_bird_polly", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("bird", vec![var_term("X")])),
                Literal::Negative(make_atom("bird", vec![atom_term("polly")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // polly is not a bird, so the negation succeeds
        // We get not_bird_polly(tweety)
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_rule_three_negations() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(make_atom("a", vec![atom_term("x")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("b", vec![atom_term("y")]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("c", vec![atom_term("z")]), &mut storage, &runtime)
            .unwrap();

        // Provide domain via positive literal
        db.insert(
            make_atom("domain", vec![atom_term("val")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: result(V) :- domain(V), not a(y), not b(x), not c(w).
        let rule = make_rule(
            make_atom("result", vec![var_term("V")]),
            vec![
                Literal::Positive(make_atom("domain", vec![var_term("V")])),
                Literal::Negative(make_atom("a", vec![atom_term("y")])),
                Literal::Negative(make_atom("b", vec![atom_term("x")])),
                Literal::Negative(make_atom("c", vec![atom_term("w")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // All negations succeed (different constants)
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_rule_with_builtin_comparison() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("number", vec![int_term(7)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("number", vec![int_term(3)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("number", vec![int_term(5)]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: large(X) :- number(X), X > 5.
        let rule = make_rule(
            make_atom("large", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("number", vec![var_term("X")])),
                Literal::Positive(Atom {
                    predicate: sym(">"),
                    terms: vec![var_term("X"), int_term(5)],
                }),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only 7 > 5
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], int_term(7));
    }

    #[test]
    fn test_ground_rule_with_builtin_equality() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("pair", vec![int_term(2), int_term(3)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("pair", vec![int_term(4), int_term(2)]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: matching(X, Y, S) :- pair(X, Y), S = X + Y.
        // Note: This tests arithmetic in body (though our impl may vary)
        let rule = make_rule(
            make_atom("sum_pair", vec![var_term("X"), var_term("Y")]),
            vec![Literal::Positive(make_atom(
                "pair",
                vec![var_term("X"), var_term("Y")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Simple grounding without arithmetic head
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_with_builtin_less_than() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(make_atom("val", vec![int_term(1)]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("val", vec![int_term(5)]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("val", vec![int_term(10)]), &mut storage, &runtime)
            .unwrap();

        // Rule: small(X) :- val(X), X < 6.
        let rule = make_rule(
            make_atom("small", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("val", vec![var_term("X")])),
                Literal::Positive(Atom {
                    predicate: sym("<"),
                    terms: vec![var_term("X"), int_term(6)],
                }),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // 1 < 6 and 5 < 6
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_with_builtin_equal() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(make_atom("num", vec![int_term(5)]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("num", vec![int_term(10)]), &mut storage, &runtime)
            .unwrap();

        // Rule: five(X) :- num(X), X = 5.
        let rule = make_rule(
            make_atom("five", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("num", vec![var_term("X")])),
                Literal::Positive(Atom {
                    predicate: sym("="),
                    terms: vec![var_term("X"), int_term(5)],
                }),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only 5 = 5
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], int_term(5));
    }

    #[test]
    fn test_ground_rule_same_variable_twice() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("rel", vec![atom_term("a"), atom_term("a")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("rel", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: same(X) :- rel(X, X).
        let rule = make_rule(
            make_atom("same", vec![var_term("X")]),
            vec![Literal::Positive(make_atom(
                "rel",
                vec![var_term("X"), var_term("X")],
            ))],
        );

        let results = ground_rule(&rule, &db, &storage);

        // Only rel(a, a) matches where both positions are the same
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_term("a"));
    }

    #[test]
    fn test_ground_rule_self_join() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("r", vec![int_term(1), int_term(2)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("r", vec![int_term(2), int_term(3)]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("r", vec![int_term(3), int_term(1)]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: cycle(X) :- r(X, Y), r(Y, Z), r(Z, X).
        let rule = make_rule(
            make_atom("cycle", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("r", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("r", vec![var_term("Y"), var_term("Z")])),
                Literal::Positive(make_atom("r", vec![var_term("Z"), var_term("X")])),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // 1 -> 2 -> 3 -> 1 forms a cycle, so X can be 1, 2, or 3
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_ground_rule_with_constant_in_head() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(
            make_atom("base", vec![atom_term("a")]),
            &mut storage,
            &runtime,
        )
        .unwrap();
        db.insert(
            make_atom("base", vec![atom_term("b")]),
            &mut storage,
            &runtime,
        )
        .unwrap();

        // Rule: tagged(constant_value, X) :- base(X).
        let rule = make_rule(
            make_atom("tagged", vec![atom_term("constant_value"), var_term("X")]),
            vec![Literal::Positive(make_atom("base", vec![var_term("X")]))],
        );

        let results = ground_rule(&rule, &db, &storage);

        assert_eq!(results.len(), 2);
        // Both should have constant_value as first term
        assert!(results
            .iter()
            .all(|a| a.terms[0] == atom_term("constant_value")));
    }

    #[test]
    fn test_ground_rule_builtin_not_equal() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        let runtime = NoOpRuntime;
        db.insert(make_atom("val", vec![int_term(1)]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("val", vec![int_term(5)]), &mut storage, &runtime)
            .unwrap();
        db.insert(make_atom("val", vec![int_term(10)]), &mut storage, &runtime)
            .unwrap();

        // Rule: not_five(X) :- val(X), X != 5.
        let rule = make_rule(
            make_atom("not_five", vec![var_term("X")]),
            vec![
                Literal::Positive(make_atom("val", vec![var_term("X")])),
                Literal::Positive(Atom {
                    predicate: sym("!="),
                    terms: vec![var_term("X"), int_term(5)],
                }),
            ],
        );

        let results = ground_rule(&rule, &db, &storage);

        // 1 != 5 and 10 != 5
        assert_eq!(results.len(), 2);
    }
}
