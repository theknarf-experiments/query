//! Datalog evaluation
//!
//! This module provides the main entry point for evaluating Datalog programs.
//! It uses stratified semi-naive evaluation for efficient and correct handling
//! of recursion and negation.
//!
//! # Example
//!
//! ```ignore
//! use datalog_eval::evaluate;
//!
//! let result = evaluate(&rules, &constraints, initial_facts)?;
//! ```

use datalog_parser::{Constraint, Rule};
use datalog_planner::{
    check_program_safety, ground_rule, ground_rule_semi_naive_with_delta, satisfy_body, stratify,
    SafetyError, StratificationError,
};
use sql_storage::{DatalogContext, DeltaTracker, InsertError, StorageEngine, StorageError};

/// Errors that can occur during evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// Program violates safety rules
    Safety(SafetyError),
    /// Program is not stratifiable (cycle through negation)
    Stratification(StratificationError),
    /// Derived a non-ground fact during evaluation
    Derivation(InsertError),
    /// Constraint violation (integrity constraint failed)
    ConstraintViolation {
        constraint: String,
        violation_count: usize,
    },
    /// Storage error when writing derived facts
    Storage(StorageError),
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluationError::Safety(e) => write!(f, "Safety error: {}", e),
            EvaluationError::Stratification(e) => {
                write!(f, "Stratification error: {}", e)
            }
            EvaluationError::Derivation(e) => write!(f, "Derivation error: {}", e),
            EvaluationError::ConstraintViolation {
                constraint,
                violation_count,
            } => {
                write!(
                    f,
                    "Constraint '{}' violated {} time(s)",
                    constraint, violation_count
                )
            }
            EvaluationError::Storage(e) => write!(f, "Storage error: {:?}", e),
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<SafetyError> for EvaluationError {
    fn from(e: SafetyError) -> Self {
        EvaluationError::Safety(e)
    }
}

impl From<StratificationError> for EvaluationError {
    fn from(e: StratificationError) -> Self {
        EvaluationError::Stratification(e)
    }
}

impl From<InsertError> for EvaluationError {
    fn from(e: InsertError) -> Self {
        EvaluationError::Derivation(e)
    }
}

impl From<StorageError> for EvaluationError {
    fn from(e: StorageError) -> Self {
        EvaluationError::Storage(e)
    }
}

/// Statistics about evaluation performance
///
/// Used by `evaluate_instrumented` to provide insight into evaluation behavior.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvaluationStats {
    /// Number of fixed-point iterations performed
    pub iterations: usize,
    /// Total number of rule applications (rule evaluated once = 1 application)
    pub rule_applications: usize,
    /// Number of new facts derived (not counting duplicates)
    pub facts_derived: usize,
}

/// Evaluate a Datalog program to fixed point.
///
/// This is the main entry point for Datalog evaluation. It:
/// 1. Checks that all rules are safe (variables properly bound)
/// 2. Stratifies the program to handle negation correctly
/// 3. Evaluates each stratum using semi-naive evaluation
/// 4. Checks constraints after evaluation
///
/// # Arguments
///
/// * `rules` - The Datalog rules to evaluate
/// * `constraints` - Integrity constraints that must not be violated
/// * `initial_facts` - The initial fact database (EDB - extensional database)
/// * `storage` - The storage engine for fact storage
///
/// # Returns
///
/// The complete fact database including all derived facts (IDB - intensional database)
///
/// # Example
///
/// ```ignore
/// // Define rules for transitive closure
/// // ancestor(X, Y) :- parent(X, Y).
/// // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
///
/// let result = evaluate(&rules, &[], facts, &mut storage)?;
/// ```
pub fn evaluate<S: StorageEngine>(
    rules: &[Rule],
    constraints: &[Constraint],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<DatalogContext, EvaluationError> {
    // Check safety first (variables in negation must appear in positive literals, etc.)
    check_program_safety(rules)?;

    // Stratify the program to handle negation correctly
    let stratification = stratify(rules)?;

    let mut db = initial_facts;

    // Evaluate each stratum to fixed point before moving to the next
    for stratum_rules in &stratification.rules_by_stratum {
        db = semi_naive_evaluate(stratum_rules, db, storage)?;
    }

    // Check constraints after evaluation
    check_constraints(constraints, &db, storage)?;

    Ok(db)
}

/// Evaluate a Datalog program with instrumentation to collect statistics.
///
/// This is identical to `evaluate()` but also returns `EvaluationStats`
/// with information about iteration count, rule applications, and facts derived.
///
/// Useful for:
/// - Performance testing and optimization
/// - Verifying semi-naive efficiency vs naive evaluation
/// - Understanding evaluation behavior on complex programs
///
/// # Example
///
/// ```ignore
/// let (result, stats) = evaluate_instrumented(&rules, &[], facts, &mut storage)?;
/// println!("Iterations: {}, Facts derived: {}", stats.iterations, stats.facts_derived);
/// ```
pub fn evaluate_instrumented<S: StorageEngine>(
    rules: &[Rule],
    constraints: &[Constraint],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<(DatalogContext, EvaluationStats), EvaluationError> {
    // Check safety first
    check_program_safety(rules)?;

    // Stratify the program
    let stratification = stratify(rules)?;

    let mut db = initial_facts;
    let mut total_stats = EvaluationStats::default();

    // Evaluate each stratum to fixed point
    for stratum_rules in &stratification.rules_by_stratum {
        let (new_db, stratum_stats) = semi_naive_evaluate_instrumented(stratum_rules, db, storage)?;
        db = new_db;

        // Accumulate stats from each stratum
        total_stats.iterations += stratum_stats.iterations;
        total_stats.rule_applications += stratum_stats.rule_applications;
        total_stats.facts_derived += stratum_stats.facts_derived;
    }

    // Check constraints after evaluation
    check_constraints(constraints, &db, storage)?;

    Ok((db, total_stats))
}

/// Check constraints against the database.
///
/// A constraint is violated if its body can be satisfied (i.e., there exist
/// substitutions that make all literals in the body true).
pub fn check_constraints<S: StorageEngine>(
    constraints: &[Constraint],
    db: &DatalogContext,
    storage: &S,
) -> Result<(), EvaluationError> {
    for constraint in constraints {
        let violations = satisfy_body(&constraint.body, db, storage);

        if !violations.is_empty() {
            return Err(EvaluationError::ConstraintViolation {
                constraint: format!("{:?}", constraint.body),
                violation_count: violations.len(),
            });
        }
    }
    Ok(())
}

/// Fixed-point evaluation for a set of rules within a single stratum.
///
/// Uses semi-naive evaluation: after the first iteration, only considers
/// derivations that use at least one newly derived fact from the previous iteration.
/// Storage handles deduplication via UNIQUE constraints on all columns.
fn semi_naive_evaluate<S: StorageEngine>(
    rules: &[Rule],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<DatalogContext, EvaluationError> {
    let mut db = initial_facts;
    let mut delta = DeltaTracker::new();
    let mut first_iteration = true;

    loop {
        let mut new_delta = DeltaTracker::new();

        for rule in rules {
            // First iteration: use naive evaluation (consider all facts as "new")
            // Subsequent iterations: use semi-naive (only derivations using delta)
            let derived = if first_iteration {
                ground_rule(rule, &db, storage)
            } else {
                ground_rule_semi_naive_with_delta(rule, &delta, &db, storage)
            };

            for fact in derived {
                // Try to insert - storage UNIQUE constraint handles deduplication
                if db.insert(fact.clone(), storage)?.is_new() {
                    // Only truly new facts go into the next iteration's delta
                    new_delta.insert(fact);
                }
            }
        }

        first_iteration = false;

        // Fixed point reached when no new facts are derived
        if new_delta.is_empty() {
            break;
        }

        // Swap: new_delta becomes the delta for next iteration
        delta = new_delta;
    }

    Ok(db)
}

/// Instrumented version of semi-naive evaluation that tracks statistics.
fn semi_naive_evaluate_instrumented<S: StorageEngine>(
    rules: &[Rule],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<(DatalogContext, EvaluationStats), EvaluationError> {
    let mut db = initial_facts;
    let mut delta = DeltaTracker::new();
    let mut first_iteration = true;
    let mut stats = EvaluationStats::default();

    loop {
        let mut new_delta = DeltaTracker::new();
        stats.iterations += 1;

        for rule in rules {
            stats.rule_applications += 1;

            let derived = if first_iteration {
                ground_rule(rule, &db, storage)
            } else {
                ground_rule_semi_naive_with_delta(rule, &delta, &db, storage)
            };

            for fact in derived {
                if db.insert(fact.clone(), storage)?.is_new() {
                    new_delta.insert(fact);
                    stats.facts_derived += 1;
                }
            }
        }

        first_iteration = false;

        if new_delta.is_empty() {
            break;
        }

        delta = new_delta;
    }

    Ok((db, stats))
}

/// Naive evaluation: repeatedly apply all rules until fixed point.
///
/// This is a simple but inefficient evaluation strategy that re-evaluates
/// all facts every iteration. It serves as a baseline for comparing
/// against semi-naive evaluation.
///
/// # Differences from semi-naive
///
/// - Does NOT track deltas (new facts from previous iteration)
/// - Re-grounds all rules against ALL facts each iteration
/// - Relies solely on storage deduplication for termination
/// - Much slower for recursive rules but simpler to understand
///
/// # Use cases
///
/// - Performance comparison/benchmarking
/// - Debugging evaluation issues
/// - Educational purposes
///
/// Note: This does NOT do stratification. For programs with negation,
/// use `evaluate()` or `evaluate_instrumented()` instead.
pub fn naive_evaluate<S: StorageEngine>(
    rules: &[Rule],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<DatalogContext, EvaluationError> {
    let mut db = initial_facts;
    let mut changed = true;

    while changed {
        changed = false;

        // Apply ALL rules against ALL facts
        for rule in rules {
            let derived = ground_rule(rule, &db, storage);
            for fact in derived {
                if db.insert(fact, storage)?.is_new() {
                    changed = true;
                }
            }
        }
    }

    Ok(db)
}

/// Naive evaluation with instrumentation for statistics.
///
/// Same as `naive_evaluate()` but also collects `EvaluationStats`.
/// Useful for comparing efficiency against `evaluate_instrumented()`.
///
/// # Example
///
/// ```ignore
/// let (result_naive, naive_stats) = naive_evaluate_instrumented(&rules, facts, &mut storage)?;
/// let (result_semi, semi_stats) = evaluate_instrumented(&rules, &[], facts, &mut storage)?;
///
/// // Semi-naive should have fewer iterations for recursive rules
/// assert!(semi_stats.iterations <= naive_stats.iterations);
/// ```
pub fn naive_evaluate_instrumented<S: StorageEngine>(
    rules: &[Rule],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<(DatalogContext, EvaluationStats), EvaluationError> {
    let mut db = initial_facts;
    let mut changed = true;
    let mut stats = EvaluationStats::default();

    while changed {
        changed = false;
        stats.iterations += 1;

        for rule in rules {
            stats.rule_applications += 1;
            let derived = ground_rule(rule, &db, storage);
            for fact in derived {
                if db.insert(fact, storage)?.is_new() {
                    changed = true;
                    stats.facts_derived += 1;
                }
            }
        }
    }

    Ok((db, stats))
}

/// Evaluate a Datalog program using storage for indexed lookups.
///
/// Deprecated: Use `evaluate()` with storage parameter instead.
#[deprecated(note = "Use evaluate() with storage parameter instead")]
pub fn evaluate_with_storage<S: StorageEngine>(
    rules: &[Rule],
    constraints: &[Constraint],
    initial_facts: DatalogContext,
    storage: &mut S,
) -> Result<DatalogContext, EvaluationError> {
    evaluate(rules, constraints, initial_facts, storage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::{Atom, Literal, Symbol, Term, Value};
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

    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    #[test]
    fn test_simple_derivation() {
        // parent(john, mary). parent(mary, jane).
        // ancestor(X, Y) :- parent(X, Y).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("jane")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should have derived 2 ancestor facts
        // predicate_count() returns count of distinct predicates, not total facts
        assert_eq!(result.predicate_count(), 2); // parent + ancestor
    }

    #[test]
    fn test_transitive_closure() {
        // ancestor(X, Y) :- parent(X, Y).
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("c"), atom_term("d")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "parent",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("ancestor", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should have 2 predicates: parent + ancestor
        assert_eq!(result.predicate_count(), 2);
    }

    #[test]
    fn test_negation_with_stratification() {
        // person(alice). person(bob). parent(alice, charlie).
        // childless(X) :- person(X), not parent(X, _).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(make_atom("person", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(make_atom("person", vec![atom_term("bob")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("alice"), atom_term("charlie")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![Rule {
            head: make_atom("childless", vec![var_term("X")]),
            body: vec![
                Literal::Positive(make_atom("person", vec![var_term("X")])),
                Literal::Negative(make_atom("parent", vec![var_term("X"), var_term("_Y")])),
            ],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should have 3 predicates: person, parent, childless
        assert_eq!(result.predicate_count(), 3);
    }

    #[test]
    fn test_constraint_violation() {
        // Constraint: :- dangerous(X).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("dangerous", vec![atom_term("bomb")]),
            &mut storage,
        )
        .unwrap();

        let constraints = vec![Constraint {
            body: vec![Literal::Positive(make_atom(
                "dangerous",
                vec![var_term("X")],
            ))],
        }];

        let result = evaluate(&[], &constraints, db, &mut storage);

        assert!(matches!(
            result,
            Err(EvaluationError::ConstraintViolation { .. })
        ));
    }

    #[test]
    fn test_cycle_through_negation_error() {
        // p(X) :- base(X), not q(X).
        // q(X) :- base(X), not p(X).
        // These rules are safe (X bound by base) but form a cycle through negation
        let rules = vec![
            Rule {
                head: make_atom("p", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("q", vec![var_term("X")])),
                ],
            },
            Rule {
                head: make_atom("q", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("p", vec![var_term("X")])),
                ],
            },
        ];

        let mut storage = MemoryEngine::new();
        let result = evaluate(&rules, &[], DatalogContext::new(), &mut storage);

        assert!(matches!(result, Err(EvaluationError::Stratification(_))));
    }

    #[test]
    fn test_evaluate_instrumented_basic() {
        // Simple rule: ancestor(X, Y) :- parent(X, Y).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("jane")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        }];

        let (result, stats) = evaluate_instrumented(&rules, &[], db, &mut storage).unwrap();

        assert_eq!(result.predicate_count(), 2); // parent + ancestor
        assert_eq!(stats.facts_derived, 2); // 2 ancestor facts derived
        assert!(stats.iterations >= 1);
        assert!(stats.rule_applications >= 1);
    }

    #[test]
    fn test_evaluate_instrumented_transitive_closure() {
        // Transitive closure: multiple iterations needed
        // ancestor(X, Y) :- parent(X, Y).
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Chain: a -> b -> c -> d (3 edges, should derive 6 ancestor facts)
        db.insert(
            make_atom("parent", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("c"), atom_term("d")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "parent",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("ancestor", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let (result, stats) = evaluate_instrumented(&rules, &[], db, &mut storage).unwrap();

        assert_eq!(result.predicate_count(), 2);
        // ancestor facts: (a,b), (b,c), (c,d), (a,c), (b,d), (a,d) = 6
        assert_eq!(stats.facts_derived, 6);
        // Should take multiple iterations (chain length)
        assert!(
            stats.iterations >= 3,
            "Expected at least 3 iterations, got {}",
            stats.iterations
        );
    }

    #[test]
    fn test_evaluate_instrumented_no_rules() {
        // No rules - should have 0 facts derived, minimal iterations
        let db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        let (_, stats) = evaluate_instrumented(&[], &[], db, &mut storage).unwrap();

        assert_eq!(stats.facts_derived, 0);
        assert_eq!(stats.rule_applications, 0);
        // No strata means 0 iterations
        assert_eq!(stats.iterations, 0);
    }

    // ===== Stratification Edge Case Tests =====

    #[test]
    fn test_stratified_chain_of_negations() {
        // Stratum 0: p(X) :- base(X).
        // Stratum 1: q(X) :- base(X), not p(X).
        // Stratum 2: r(X) :- base(X), not q(X).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(make_atom("base", vec![atom_term("a")]), &mut storage)
            .unwrap();

        let rules = vec![
            Rule {
                head: make_atom("p", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("base", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("q", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("p", vec![var_term("X")])),
                ],
            },
            Rule {
                head: make_atom("r", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("q", vec![var_term("X")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // p(a) should be derived in stratum 0
        assert!(result.contains(&make_atom("p", vec![atom_term("a")]), &storage));

        // q(a) should NOT be derived (because p(a) exists)
        assert!(!result.contains(&make_atom("q", vec![atom_term("a")]), &storage));

        // r(a) should be derived (because q(a) doesn't exist)
        assert!(result.contains(&make_atom("r", vec![atom_term("a")]), &storage));
    }

    #[test]
    fn test_stratified_double_negation() {
        // Test multiple levels of negation
        // base(a). base(b). base(c).
        // excluded(a).
        // included(X) :- base(X), not excluded(X).
        // non_included(X) :- base(X), not included(X).
        // definitely(X) :- base(X), not non_included(X).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(make_atom("base", vec![atom_term("a")]), &mut storage)
            .unwrap();
        db.insert(make_atom("base", vec![atom_term("b")]), &mut storage)
            .unwrap();
        db.insert(make_atom("base", vec![atom_term("c")]), &mut storage)
            .unwrap();
        db.insert(make_atom("excluded", vec![atom_term("a")]), &mut storage)
            .unwrap();

        let rules = vec![
            Rule {
                head: make_atom("included", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("excluded", vec![var_term("X")])),
                ],
            },
            Rule {
                head: make_atom("non_included", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("included", vec![var_term("X")])),
                ],
            },
            Rule {
                head: make_atom("definitely", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("base", vec![var_term("X")])),
                    Literal::Negative(make_atom("non_included", vec![var_term("X")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // b and c are included (not excluded)
        assert!(result.contains(&make_atom("included", vec![atom_term("b")]), &storage));
        assert!(result.contains(&make_atom("included", vec![atom_term("c")]), &storage));

        // a is not included (excluded)
        assert!(!result.contains(&make_atom("included", vec![atom_term("a")]), &storage));

        // a is non_included
        assert!(result.contains(&make_atom("non_included", vec![atom_term("a")]), &storage));

        // b and c are definitely (not non_included)
        assert!(result.contains(&make_atom("definitely", vec![atom_term("b")]), &storage));
        assert!(result.contains(&make_atom("definitely", vec![atom_term("c")]), &storage));

        // a is NOT definitely (it is non_included)
        assert!(!result.contains(&make_atom("definitely", vec![atom_term("a")]), &storage));
    }

    #[test]
    fn test_stratified_with_transitive_closure_and_negation() {
        // Graph reachability with blocked nodes
        // edge(a, b). edge(b, c). edge(c, d). edge(a, x).
        // blocked(x).
        // reachable(X, Y) :- edge(X, Y).
        // reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        // safe_reachable(X, Y) :- reachable(X, Y), not blocked(Y).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("c"), atom_term("d")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("x")]),
            &mut storage,
        )
        .unwrap();
        db.insert(make_atom("blocked", vec![atom_term("x")]), &mut storage)
            .unwrap();

        let rules = vec![
            Rule {
                head: make_atom("reachable", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("reachable", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("reachable", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
            Rule {
                head: make_atom("safe_reachable", vec![var_term("X"), var_term("Y")]),
                body: vec![
                    Literal::Positive(make_atom("reachable", vec![var_term("X"), var_term("Y")])),
                    Literal::Negative(make_atom("blocked", vec![var_term("Y")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should reach d from a (via b->c->d)
        assert!(result.contains(
            &make_atom("safe_reachable", vec![atom_term("a"), atom_term("d")]),
            &storage
        ));

        // Should NOT reach x (blocked)
        assert!(!result.contains(
            &make_atom("safe_reachable", vec![atom_term("a"), atom_term("x")]),
            &storage
        ));

        // But x IS reachable (just not safe_reachable)
        assert!(result.contains(
            &make_atom("reachable", vec![atom_term("a"), atom_term("x")]),
            &storage
        ));
    }

    #[test]
    fn test_stratified_game_states() {
        // Game state exploration with forbidden states
        // initial(start).
        // transition(start, s1). transition(s1, s2). transition(s1, danger).
        // forbidden(danger).
        // reachable(S) :- initial(S).
        // reachable(S) :- reachable(S0), transition(S0, S).
        // safe(S) :- reachable(S), not forbidden(S).
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();
        db.insert(make_atom("initial", vec![atom_term("start")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom("transition", vec![atom_term("start"), atom_term("s1")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("transition", vec![atom_term("s1"), atom_term("s2")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("transition", vec![atom_term("s1"), atom_term("danger")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("forbidden", vec![atom_term("danger")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![
            Rule {
                head: make_atom("reachable", vec![var_term("S")]),
                body: vec![Literal::Positive(make_atom("initial", vec![var_term("S")]))],
            },
            Rule {
                head: make_atom("reachable", vec![var_term("S")]),
                body: vec![
                    Literal::Positive(make_atom("reachable", vec![var_term("S0")])),
                    Literal::Positive(make_atom("transition", vec![var_term("S0"), var_term("S")])),
                ],
            },
            Rule {
                head: make_atom("safe", vec![var_term("S")]),
                body: vec![
                    Literal::Positive(make_atom("reachable", vec![var_term("S")])),
                    Literal::Negative(make_atom("forbidden", vec![var_term("S")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // All states are reachable
        assert!(result.contains(&make_atom("reachable", vec![atom_term("start")]), &storage));
        assert!(result.contains(&make_atom("reachable", vec![atom_term("s1")]), &storage));
        assert!(result.contains(&make_atom("reachable", vec![atom_term("s2")]), &storage));
        assert!(result.contains(&make_atom("reachable", vec![atom_term("danger")]), &storage));

        // Danger is NOT safe
        assert!(!result.contains(&make_atom("safe", vec![atom_term("danger")]), &storage));

        // Other states ARE safe
        assert!(result.contains(&make_atom("safe", vec![atom_term("start")]), &storage));
        assert!(result.contains(&make_atom("safe", vec![atom_term("s1")]), &storage));
        assert!(result.contains(&make_atom("safe", vec![atom_term("s2")]), &storage));
    }

    // Note: Zero-arity predicate tests from proclog are not included because
    // SQL storage requires at least one column per table. Zero-arity predicates
    // (propositional logic) would require a different storage approach.

    // ===== Long Recursive Chain Stress Tests =====

    #[test]
    fn test_very_long_chain() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Create a long chain: n0->n1->n2->...->n100
        for i in 0..100 {
            let from = format!("n{}", i);
            let to = format!("n{}", i + 1);
            db.insert(
                make_atom("edge", vec![atom_term(&from), atom_term(&to)]),
                &mut storage,
            )
            .unwrap();
        }

        // path(X, Y) :- edge(X, Y).
        // path(X, Z) :- path(X, Y), edge(Y, Z).
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should be able to reach from n0 to n100
        assert!(result.contains(
            &make_atom("path", vec![atom_term("n0"), atom_term("n100")]),
            &storage
        ));

        // We have 100 edges, so we should have:
        // - 100 direct paths
        // - 99 paths of length 2
        // - 98 paths of length 3
        // - ...
        // - 1 path of length 100
        // Total = 100 + 99 + 98 + ... + 1 = 100*101/2 = 5050
        let path_count = result.count_facts("path", &storage);
        assert_eq!(path_count, 5050, "Expected 5050 path facts");
    }

    #[test]
    fn test_long_chain_with_stats() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Create a chain of 50 nodes
        for i in 0..50 {
            let from = format!("n{}", i);
            let to = format!("n{}", i + 1);
            db.insert(
                make_atom("edge", vec![atom_term(&from), atom_term(&to)]),
                &mut storage,
            )
            .unwrap();
        }

        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let (result, stats) = evaluate_instrumented(&rules, &[], db, &mut storage)
            .expect("evaluation should succeed");

        // Check that semi-naive is efficient
        // Should converge in approximately 50 iterations (length of chain)
        println!(
            "Long chain (50 nodes) stats: {} iterations, {} facts derived",
            stats.iterations, stats.facts_derived
        );
        assert!(
            stats.iterations <= 51,
            "Too many iterations: {}",
            stats.iterations
        );

        // Verify correctness
        assert!(result.contains(
            &make_atom("path", vec![atom_term("n0"), atom_term("n50")]),
            &storage
        ));
    }

    #[test]
    fn test_wide_graph_stress() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Create a wide graph: single source to many targets
        // This tests performance with many facts in a single relation
        for i in 0..100 {
            let target = format!("t{}", i);
            db.insert(
                make_atom("edge", vec![atom_term("source"), atom_term(&target)]),
                &mut storage,
            )
            .unwrap();
        }

        // reachable(X) :- edge(source, X).
        let rules = vec![Rule {
            head: make_atom("reachable", vec![var_term("X")]),
            body: vec![Literal::Positive(make_atom(
                "edge",
                vec![atom_term("source"), var_term("X")],
            ))],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // All targets should be reachable
        for i in 0..100 {
            let target = format!("t{}", i);
            assert!(
                result.contains(&make_atom("reachable", vec![atom_term(&target)]), &storage),
                "t{} should be reachable",
                i
            );
        }
    }

    #[test]
    fn test_multi_level_derivation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Create facts that will derive through multiple predicates
        for i in 0..20 {
            let name = format!("e{}", i);
            db.insert(make_atom("base", vec![atom_term(&name)]), &mut storage)
                .unwrap();
        }

        // level1(X) :- base(X).
        // level2(X) :- level1(X).
        // level3(X) :- level2(X).
        // level4(X) :- level3(X).
        // level5(X) :- level4(X).
        let rules = vec![
            Rule {
                head: make_atom("level1", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("base", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level2", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("level1", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level3", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("level2", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level4", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("level3", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level5", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("level4", vec![var_term("X")]))],
            },
        ];

        let (result, stats) = evaluate_instrumented(&rules, &[], db, &mut storage)
            .expect("evaluation should succeed");

        // Each level should have all 20 entities
        for level in 1..=5 {
            let level_name = format!("level{}", level);
            let count = result.count_facts(&level_name, &storage);
            assert_eq!(count, 20, "Level {} should have 20 facts", level);
        }

        println!(
            "Multi-level derivation stats: {} iterations, {} facts derived",
            stats.iterations, stats.facts_derived
        );

        // Should derive in about 5 iterations (one per level)
        assert!(
            stats.iterations <= 6,
            "Too many iterations: {}",
            stats.iterations
        );
    }

    // ===== Naive Evaluation Tests =====

    #[test]
    fn test_naive_evaluate_basic() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom("parent", vec![atom_term("alice"), atom_term("bob")]),
            &mut storage,
        )
        .unwrap();

        // ancestor(X, Y) :- parent(X, Y).
        let rules = vec![Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        }];

        let result = naive_evaluate(&rules, db, &mut storage).unwrap();

        assert!(result.contains(
            &make_atom("ancestor", vec![atom_term("alice"), atom_term("bob")]),
            &storage
        ));
    }

    #[test]
    fn test_naive_evaluate_transitive_closure() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // a -> b -> c
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();

        // path(X, Y) :- edge(X, Y).
        // path(X, Z) :- path(X, Y), edge(Y, Z).
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let result = naive_evaluate(&rules, db, &mut storage).unwrap();

        // Should derive path(a, c) through transitive closure
        assert!(result.contains(
            &make_atom("path", vec![atom_term("a"), atom_term("c")]),
            &storage
        ));
    }

    #[test]
    fn test_naive_vs_semi_naive_correctness() {
        // Both should produce the same result
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        // Create initial facts in separate databases for each evaluation
        let mut storage_naive = MemoryEngine::new();
        let mut db_naive = DatalogContext::new();
        db_naive
            .insert(
                make_atom("edge", vec![atom_term("a"), atom_term("b")]),
                &mut storage_naive,
            )
            .unwrap();
        db_naive
            .insert(
                make_atom("edge", vec![atom_term("b"), atom_term("c")]),
                &mut storage_naive,
            )
            .unwrap();
        db_naive
            .insert(
                make_atom("edge", vec![atom_term("c"), atom_term("d")]),
                &mut storage_naive,
            )
            .unwrap();

        let mut storage_semi = MemoryEngine::new();
        let mut db_semi = DatalogContext::new();
        db_semi
            .insert(
                make_atom("edge", vec![atom_term("a"), atom_term("b")]),
                &mut storage_semi,
            )
            .unwrap();
        db_semi
            .insert(
                make_atom("edge", vec![atom_term("b"), atom_term("c")]),
                &mut storage_semi,
            )
            .unwrap();
        db_semi
            .insert(
                make_atom("edge", vec![atom_term("c"), atom_term("d")]),
                &mut storage_semi,
            )
            .unwrap();

        let result_naive = naive_evaluate(&rules, db_naive, &mut storage_naive).unwrap();
        let result_semi = evaluate(&rules, &[], db_semi, &mut storage_semi).unwrap();

        // Both should derive the same paths
        assert_eq!(
            result_naive.count_facts("path", &storage_naive),
            result_semi.count_facts("path", &storage_semi)
        );

        // Check specific paths exist in both
        let expected_paths = vec![
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("a", "c"),
            ("b", "d"),
            ("a", "d"),
        ];

        for (from, to) in expected_paths {
            let fact = make_atom("path", vec![atom_term(from), atom_term(to)]);
            assert!(
                result_naive.contains(&fact, &storage_naive),
                "Naive should contain path({}, {})",
                from,
                to
            );
            assert!(
                result_semi.contains(&fact, &storage_semi),
                "Semi-naive should contain path({}, {})",
                from,
                to
            );
        }
    }

    #[test]
    fn test_naive_vs_semi_naive_efficiency() {
        // Semi-naive should be more efficient (fewer rule applications)
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        // Build a chain for testing
        let mut storage_naive = MemoryEngine::new();
        let mut db_naive = DatalogContext::new();
        let mut storage_semi = MemoryEngine::new();
        let mut db_semi = DatalogContext::new();

        for i in 0..20 {
            let from = format!("n{}", i);
            let to = format!("n{}", i + 1);
            db_naive
                .insert(
                    make_atom("edge", vec![atom_term(&from), atom_term(&to)]),
                    &mut storage_naive,
                )
                .unwrap();
            db_semi
                .insert(
                    make_atom("edge", vec![atom_term(&from), atom_term(&to)]),
                    &mut storage_semi,
                )
                .unwrap();
        }

        let (_, naive_stats) =
            naive_evaluate_instrumented(&rules, db_naive, &mut storage_naive).unwrap();
        let (_, semi_stats) =
            evaluate_instrumented(&rules, &[], db_semi, &mut storage_semi).unwrap();

        println!(
            "Naive: {} iterations, {} rule apps, {} facts",
            naive_stats.iterations, naive_stats.rule_applications, naive_stats.facts_derived
        );
        println!(
            "Semi-naive: {} iterations, {} rule apps, {} facts",
            semi_stats.iterations, semi_stats.rule_applications, semi_stats.facts_derived
        );

        // Both should derive the same number of facts
        assert_eq!(
            naive_stats.facts_derived, semi_stats.facts_derived,
            "Both should derive same number of facts"
        );

        // Semi-naive should have fewer or equal rule applications
        // (it processes less redundantly)
        assert!(
            semi_stats.rule_applications <= naive_stats.rule_applications,
            "Semi-naive should have <= rule applications than naive"
        );
    }

    // ===== Compound Term Tests =====

    fn compound_term(functor: &str, args: Vec<Term>) -> Term {
        Term::Compound(sym(functor), args)
    }

    #[test]
    fn test_compound_term_in_fact() {
        // Test storing and retrieving compound terms as facts
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Insert: data(pair(1, foo))
        let compound = compound_term(
            "pair",
            vec![Term::Constant(Value::Integer(1)), atom_term("foo")],
        );
        db.insert(make_atom("data", vec![compound.clone()]), &mut storage)
            .unwrap();

        // Query should return the compound term
        let result = evaluate(&[], &[], db, &mut storage).unwrap();
        assert!(result.contains(&make_atom("data", vec![compound]), &storage));
    }

    #[test]
    fn test_compound_term_in_rule_head() {
        // Test deriving compound terms via rules
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Base fact: value(foo)
        db.insert(make_atom("value", vec![atom_term("foo")]), &mut storage)
            .unwrap();

        // Rule: wrapped(wrap(X)) :- value(X).
        let rules = vec![Rule {
            head: make_atom("wrapped", vec![compound_term("wrap", vec![var_term("X")])]),
            body: vec![Literal::Positive(make_atom("value", vec![var_term("X")]))],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should derive: wrapped(wrap(foo))
        let expected = compound_term("wrap", vec![atom_term("foo")]);
        assert!(result.contains(&make_atom("wrapped", vec![expected]), &storage));
    }

    #[test]
    fn test_nested_compound_term_derivation() {
        // Test deriving deeply nested compound terms
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Base fact: base(a)
        db.insert(make_atom("base", vec![atom_term("a")]), &mut storage)
            .unwrap();

        // Rules that progressively wrap:
        // level0(X) :- base(X).
        // level1(wrap(X)) :- level0(X).
        // level2(wrap(X)) :- level1(X).
        // level3(wrap(X)) :- level2(X).
        let rules = vec![
            Rule {
                head: make_atom("level0", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom("base", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level1", vec![compound_term("wrap", vec![var_term("X")])]),
                body: vec![Literal::Positive(make_atom("level0", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level2", vec![compound_term("wrap", vec![var_term("X")])]),
                body: vec![Literal::Positive(make_atom("level1", vec![var_term("X")]))],
            },
            Rule {
                head: make_atom("level3", vec![compound_term("wrap", vec![var_term("X")])]),
                body: vec![Literal::Positive(make_atom("level2", vec![var_term("X")]))],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should derive:
        // level0(a)
        // level1(wrap(a))
        // level2(wrap(wrap(a)))
        // level3(wrap(wrap(wrap(a))))

        assert!(result.contains(&make_atom("level0", vec![atom_term("a")]), &storage));

        let wrap_a = compound_term("wrap", vec![atom_term("a")]);
        assert!(result.contains(&make_atom("level1", vec![wrap_a.clone()]), &storage));

        let wrap_wrap_a = compound_term("wrap", vec![wrap_a.clone()]);
        assert!(result.contains(&make_atom("level2", vec![wrap_wrap_a.clone()]), &storage));

        let wrap_wrap_wrap_a = compound_term("wrap", vec![wrap_wrap_a]);
        assert!(result.contains(&make_atom("level3", vec![wrap_wrap_wrap_a]), &storage));
    }

    #[test]
    fn test_compound_term_unification_in_body() {
        // Test matching compound terms in rule bodies
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Facts with compound terms
        db.insert(
            make_atom(
                "data",
                vec![compound_term("pair", vec![atom_term("a"), atom_term("b")])],
            ),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom(
                "data",
                vec![compound_term("pair", vec![atom_term("c"), atom_term("d")])],
            ),
            &mut storage,
        )
        .unwrap();

        // Rule: first(X) :- data(pair(X, _)).
        // This extracts the first element of pairs
        let rules = vec![Rule {
            head: make_atom("first", vec![var_term("X")]),
            body: vec![Literal::Positive(make_atom(
                "data",
                vec![compound_term("pair", vec![var_term("X"), var_term("Y")])],
            ))],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should derive first(a) and first(c)
        assert!(result.contains(&make_atom("first", vec![atom_term("a")]), &storage));
        assert!(result.contains(&make_atom("first", vec![atom_term("c")]), &storage));
    }

    #[test]
    fn test_multiple_arg_compound_term() {
        // Test compound terms with multiple arguments
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("x", vec![atom_term("a")]), &mut storage)
            .unwrap();
        db.insert(make_atom("y", vec![atom_term("b")]), &mut storage)
            .unwrap();

        // Rule: combined(pair(X, Y)) :- x(X), y(Y).
        let rules = vec![Rule {
            head: make_atom(
                "combined",
                vec![compound_term("pair", vec![var_term("X"), var_term("Y")])],
            ),
            body: vec![
                Literal::Positive(make_atom("x", vec![var_term("X")])),
                Literal::Positive(make_atom("y", vec![var_term("Y")])),
            ],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        let expected = compound_term("pair", vec![atom_term("a"), atom_term("b")]);
        assert!(result.contains(&make_atom("combined", vec![expected]), &storage));
    }

    // ===== Facts-Only Program Tests =====

    #[test]
    fn test_facts_only_no_rules() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("fact1", vec![atom_term("a")]), &mut storage)
            .unwrap();
        db.insert(make_atom("fact2", vec![atom_term("b")]), &mut storage)
            .unwrap();
        db.insert(make_atom("fact3", vec![atom_term("c")]), &mut storage)
            .unwrap();

        let rules = vec![]; // No rules!

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Database should remain unchanged - only original facts
        assert!(result.contains(&make_atom("fact1", vec![atom_term("a")]), &storage));
        assert!(result.contains(&make_atom("fact2", vec![atom_term("b")]), &storage));
        assert!(result.contains(&make_atom("fact3", vec![atom_term("c")]), &storage));
    }

    #[test]
    fn test_facts_only_with_query() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("mary")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("john"), atom_term("bob")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("mary"), atom_term("alice")]),
            &mut storage,
        )
        .unwrap();

        // Query: parent(john, X)?
        let query = make_atom("parent", vec![atom_term("john"), var_term("X")]);
        let results = db.query(&query, &storage);

        assert_eq!(results.len(), 2); // john is parent of mary and bob
    }

    // ===== Duplicate Facts/Rules Tests =====

    #[test]
    fn test_duplicate_facts() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("fact", vec![atom_term("a")]), &mut storage)
            .unwrap();
        db.insert(make_atom("fact", vec![atom_term("a")]), &mut storage)
            .unwrap(); // Duplicate!
        db.insert(make_atom("fact", vec![atom_term("a")]), &mut storage)
            .unwrap(); // Another duplicate!

        // Query should return only one result (deduplication)
        let query = make_atom("fact", vec![var_term("X")]);
        let results = db.query(&query, &storage);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_duplicate_rules() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();

        // Same rule twice
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should work fine, duplicate derivations get deduplicated
        assert!(result.contains(
            &make_atom("path", vec![atom_term("a"), atom_term("b")]),
            &storage
        ));
    }

    // ===== Query Tests =====

    #[test]
    fn test_query_ground_true() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        let fact = make_atom("parent", vec![atom_term("john"), atom_term("mary")]);
        db.insert(fact.clone(), &mut storage).unwrap();

        // Ground query should match
        let results = db.query(&fact, &storage);
        assert_eq!(results.len(), 1);
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

        // Query for non-existent fact
        let query = make_atom("parent", vec![atom_term("alice"), atom_term("bob")]);
        let results = db.query(&query, &storage);
        assert_eq!(results.len(), 0);
    }

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

        // Derive grandparent relationships
        let rules = vec![Rule {
            head: make_atom("grandparent", vec![var_term("X"), var_term("Z")]),
            body: vec![
                Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
            ],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should find: john is grandparent of sue
        assert!(result.contains(
            &make_atom("grandparent", vec![atom_term("john"), atom_term("sue")]),
            &storage
        ));
    }

    #[test]
    fn test_query_multiple_variables() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("c"), atom_term("d")]),
            &mut storage,
        )
        .unwrap();

        // Query: edge(X, Y)?
        let query = make_atom("edge", vec![var_term("X"), var_term("Y")]);
        let results = db.query(&query, &storage);

        assert_eq!(results.len(), 3);
    }

    // ===== All Datatypes Evaluation Tests =====

    fn int_term(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn float_term(f: f64) -> Term {
        Term::Constant(Value::Float(f))
    }

    fn bool_term(b: bool) -> Term {
        Term::Constant(Value::Boolean(b))
    }

    fn string_term(s: &str) -> Term {
        Term::Constant(Value::String(sym(s)))
    }

    #[test]
    fn test_all_datatypes_in_evaluation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Integer facts
        db.insert(
            make_atom("health", vec![atom_term("player"), int_term(100)]),
            &mut storage,
        )
        .unwrap();

        // Float facts
        db.insert(
            make_atom("position", vec![atom_term("player"), float_term(3.14)]),
            &mut storage,
        )
        .unwrap();

        // Boolean facts
        db.insert(
            make_atom("is_alive", vec![atom_term("player"), bool_term(true)]),
            &mut storage,
        )
        .unwrap();

        // String facts
        db.insert(
            make_atom("name", vec![atom_term("player"), string_term("Alice")]),
            &mut storage,
        )
        .unwrap();

        // Compound term facts
        db.insert(
            make_atom(
                "inventory",
                vec![
                    atom_term("player"),
                    compound_term("item", vec![atom_term("sword"), int_term(10)]),
                ],
            ),
            &mut storage,
        )
        .unwrap();

        // Rule: has_weapon(P) :- inventory(P, item(sword, Qty)).
        let rules = vec![Rule {
            head: make_atom("has_weapon", vec![var_term("P")]),
            body: vec![Literal::Positive(make_atom(
                "inventory",
                vec![
                    var_term("P"),
                    compound_term("item", vec![atom_term("sword"), var_term("Qty")]),
                ],
            ))],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Verify compound term matching worked
        assert!(result.contains(
            &make_atom("has_weapon", vec![atom_term("player")]),
            &storage
        ));

        // Query with integer
        let health_query = make_atom("health", vec![var_term("Who"), int_term(100)]);
        let health_results = result.query(&health_query, &storage);
        assert_eq!(health_results.len(), 1);

        // Query with boolean
        let alive_query = make_atom("is_alive", vec![var_term("Who"), bool_term(true)]);
        let alive_results = result.query(&alive_query, &storage);
        assert_eq!(alive_results.len(), 1);
    }

    #[test]
    fn test_mixed_datatypes_in_rules() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom(
                "player",
                vec![atom_term("alice"), int_term(100), bool_term(true)],
            ),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom(
                "player",
                vec![atom_term("bob"), int_term(0), bool_term(false)],
            ),
            &mut storage,
        )
        .unwrap();

        // Rule: alive_player(Name) :- player(Name, Health, true).
        let rules = vec![Rule {
            head: make_atom("alive_player", vec![var_term("Name")]),
            body: vec![Literal::Positive(make_atom(
                "player",
                vec![var_term("Name"), var_term("Health"), bool_term(true)],
            ))],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Only alice should be alive
        assert!(result.contains(
            &make_atom("alive_player", vec![atom_term("alice")]),
            &storage
        ));

        let query = make_atom("alive_player", vec![var_term("X")]);
        let results = result.query(&query, &storage);
        assert_eq!(results.len(), 1);
    }

    // ===== Multiple Constraint Tests =====

    #[test]
    fn test_constraint_no_violation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("safe", vec![atom_term("a")]), &mut storage)
            .unwrap();
        db.insert(make_atom("safe", vec![atom_term("b")]), &mut storage)
            .unwrap();

        // Constraint: :- unsafe(X).
        let constraints = vec![Constraint {
            body: vec![Literal::Positive(make_atom("unsafe", vec![var_term("X")]))],
        }];

        // Should pass - no unsafe facts
        let result = evaluate(&[], &constraints, db, &mut storage);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constraint_multiple_violations() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("unsafe", vec![atom_term("a")]), &mut storage)
            .unwrap();
        db.insert(make_atom("unsafe", vec![atom_term("b")]), &mut storage)
            .unwrap();
        db.insert(make_atom("unsafe", vec![atom_term("c")]), &mut storage)
            .unwrap();

        // Constraint: :- unsafe(X).
        let constraints = vec![Constraint {
            body: vec![Literal::Positive(make_atom("unsafe", vec![var_term("X")]))],
        }];

        let result = evaluate(&[], &constraints, db, &mut storage);
        assert!(result.is_err());

        if let Err(EvaluationError::ConstraintViolation {
            violation_count, ..
        }) = result
        {
            assert_eq!(violation_count, 3);
        } else {
            panic!("Expected ConstraintViolation error");
        }
    }

    #[test]
    fn test_constraint_with_conjunction() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("player", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(make_atom("player", vec![atom_term("bob")]), &mut storage)
            .unwrap();
        db.insert(make_atom("dead", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom("has_weapon", vec![atom_term("alice")]),
            &mut storage,
        )
        .unwrap();

        // Constraint: :- dead(X), has_weapon(X).
        // (Dead players shouldn't have weapons)
        let constraints = vec![Constraint {
            body: vec![
                Literal::Positive(make_atom("dead", vec![var_term("X")])),
                Literal::Positive(make_atom("has_weapon", vec![var_term("X")])),
            ],
        }];

        let result = evaluate(&[], &constraints, db, &mut storage);
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_with_negation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("player", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(make_atom("player", vec![atom_term("bob")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom("has_health", vec![atom_term("alice")]),
            &mut storage,
        )
        .unwrap();
        // bob has no health

        // Constraint: :- player(X), not has_health(X).
        // (All players must have health)
        let constraints = vec![Constraint {
            body: vec![
                Literal::Positive(make_atom("player", vec![var_term("X")])),
                Literal::Negative(make_atom("has_health", vec![var_term("X")])),
            ],
        }];

        let result = evaluate(&[], &constraints, db, &mut storage);
        assert!(result.is_err());

        if let Err(EvaluationError::ConstraintViolation {
            violation_count, ..
        }) = result
        {
            assert_eq!(violation_count, 1); // bob violates
        }
    }

    #[test]
    fn test_constraint_after_derivation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();
        db.insert(make_atom("blocked", vec![atom_term("a")]), &mut storage)
            .unwrap();

        // path(X, Y) :- edge(X, Y).
        // path(X, Z) :- path(X, Y), edge(Y, Z).
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        // Constraint: :- path(X, Y), blocked(X).
        // (No paths from blocked nodes)
        let constraints = vec![Constraint {
            body: vec![
                Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                Literal::Positive(make_atom("blocked", vec![var_term("X")])),
            ],
        }];

        // Should fail - paths exist from blocked node 'a'
        let result = evaluate(&rules, &constraints, db, &mut storage);
        assert!(result.is_err());
    }

    // ===== Deep Nesting Stress Tests =====

    #[test]
    fn test_very_deep_nested_compound() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Create deeply nested term: nest(nest(nest(nest(nest(value)))))
        let mut deep_term = atom_term("value");
        for _ in 0..10 {
            deep_term = compound_term("nest", vec![deep_term]);
        }

        db.insert(make_atom("deep", vec![deep_term.clone()]), &mut storage)
            .unwrap();

        // Query for it
        let query = make_atom("deep", vec![var_term("X")]);
        let results = db.query(&query, &storage);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_deep_nesting_in_rules() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("value", vec![atom_term("a")]), &mut storage)
            .unwrap();

        // Create rules that progressively wrap the value
        // level0(X) :- value(X).
        // level1(wrap(X)) :- level0(X).
        // level2(wrap(X)) :- level1(X).
        // ...
        let mut rules = vec![Rule {
            head: make_atom("level0", vec![var_term("X")]),
            body: vec![Literal::Positive(make_atom("value", vec![var_term("X")]))],
        }];

        for i in 0..5 {
            let predicate = format!("level{}", i);
            let next_predicate = format!("level{}", i + 1);

            rules.push(Rule {
                head: make_atom(
                    &next_predicate,
                    vec![compound_term("wrap", vec![var_term("X")])],
                ),
                body: vec![Literal::Positive(make_atom(
                    &predicate,
                    vec![var_term("X")],
                ))],
            });
        }

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should derive level5(wrap(wrap(wrap(wrap(wrap(a))))))
        let mut expected = atom_term("a");
        for _ in 0..5 {
            expected = compound_term("wrap", vec![expected]);
        }

        assert!(result.contains(&make_atom("level5", vec![expected]), &storage));
    }

    // ===== Negated Compound Term Tests =====

    #[test]
    fn test_negated_compound_term() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("item", vec![atom_term("sword")]), &mut storage)
            .unwrap();
        db.insert(make_atom("item", vec![atom_term("shield")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom(
                "dangerous",
                vec![compound_term(
                    "property",
                    vec![atom_term("sword"), atom_term("sharp")],
                )],
            ),
            &mut storage,
        )
        .unwrap();

        // safe_item(X) :- item(X), not dangerous(property(X, sharp)).
        let rules = vec![Rule {
            head: make_atom("safe_item", vec![var_term("X")]),
            body: vec![
                Literal::Positive(make_atom("item", vec![var_term("X")])),
                Literal::Negative(make_atom(
                    "dangerous",
                    vec![compound_term(
                        "property",
                        vec![var_term("X"), atom_term("sharp")],
                    )],
                )),
            ],
        }];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // shield is safe (no dangerous(property(shield, sharp)))
        assert!(result.contains(&make_atom("safe_item", vec![atom_term("shield")]), &storage));
        // sword is not safe (dangerous(property(sword, sharp)) exists)
        assert!(!result.contains(&make_atom("safe_item", vec![atom_term("sword")]), &storage));
    }

    #[test]
    fn test_nested_compound_in_negation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(make_atom("player", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(make_atom("player", vec![atom_term("bob")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom(
                "has_item",
                vec![
                    atom_term("alice"),
                    compound_term(
                        "item",
                        vec![
                            atom_term("weapon"),
                            compound_term("stats", vec![int_term(10), int_term(5)]),
                        ],
                    ),
                ],
            ),
            &mut storage,
        )
        .unwrap();

        // First, derive who has weapons
        // has_weapon(P) :- has_item(P, item(weapon, stats(D, W))).
        // Then: unarmed(P) :- player(P), not has_weapon(P).
        let rules = vec![
            Rule {
                head: make_atom("has_weapon", vec![var_term("P")]),
                body: vec![Literal::Positive(make_atom(
                    "has_item",
                    vec![
                        var_term("P"),
                        compound_term(
                            "item",
                            vec![
                                atom_term("weapon"),
                                compound_term("stats", vec![var_term("D"), var_term("W")]),
                            ],
                        ),
                    ],
                ))],
            },
            Rule {
                head: make_atom("unarmed", vec![var_term("P")]),
                body: vec![
                    Literal::Positive(make_atom("player", vec![var_term("P")])),
                    Literal::Negative(make_atom("has_weapon", vec![var_term("P")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // bob is unarmed (no weapon)
        assert!(result.contains(&make_atom("unarmed", vec![atom_term("bob")]), &storage));
        // alice is NOT unarmed (has weapon)
        assert!(!result.contains(&make_atom("unarmed", vec![atom_term("alice")]), &storage));
    }

    // NOTE: Zero-arity predicates are not supported by the storage layer
    // (requires at least one column for indexing). Tests for those are skipped.

    // ===== Eligibility/Complex Stratification Tests =====

    #[test]
    fn test_stratified_eligibility() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Facts about students
        db.insert(make_atom("student", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(make_atom("student", vec![atom_term("bob")]), &mut storage)
            .unwrap();
        db.insert(
            make_atom("student", vec![atom_term("charlie")]),
            &mut storage,
        )
        .unwrap();

        // Some students have scholarships
        db.insert(
            make_atom("has_scholarship", vec![atom_term("alice")]),
            &mut storage,
        )
        .unwrap();

        // Some students have jobs
        db.insert(make_atom("has_job", vec![atom_term("bob")]), &mut storage)
            .unwrap();

        // Rules:
        // needs_financial_aid(X) :- student(X), not has_scholarship(X), not has_job(X).
        // priority_candidate(X) :- needs_financial_aid(X).
        let rules = vec![
            Rule {
                head: make_atom("needs_financial_aid", vec![var_term("X")]),
                body: vec![
                    Literal::Positive(make_atom("student", vec![var_term("X")])),
                    Literal::Negative(make_atom("has_scholarship", vec![var_term("X")])),
                    Literal::Negative(make_atom("has_job", vec![var_term("X")])),
                ],
            },
            Rule {
                head: make_atom("priority_candidate", vec![var_term("X")]),
                body: vec![Literal::Positive(make_atom(
                    "needs_financial_aid",
                    vec![var_term("X")],
                ))],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // charlie needs financial aid (no scholarship, no job)
        assert!(result.contains(
            &make_atom("needs_financial_aid", vec![atom_term("charlie")]),
            &storage
        ));

        // alice doesn't need aid (has scholarship)
        assert!(!result.contains(
            &make_atom("needs_financial_aid", vec![atom_term("alice")]),
            &storage
        ));

        // bob doesn't need aid (has job)
        assert!(!result.contains(
            &make_atom("needs_financial_aid", vec![atom_term("bob")]),
            &storage
        ));

        // charlie is a priority candidate
        assert!(result.contains(
            &make_atom("priority_candidate", vec![atom_term("charlie")]),
            &storage
        ));
    }

    #[test]
    fn test_semi_naive_delta_at_different_positions() {
        // This test verifies that delta is tried at EACH position in multi-literal rules
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // A chain: a -> b -> c
        db.insert(
            make_atom("edge", vec![atom_term("a"), atom_term("b")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("edge", vec![atom_term("b"), atom_term("c")]),
            &mut storage,
        )
        .unwrap();

        // A separate chain: x -> y
        db.insert(
            make_atom("edge", vec![atom_term("x"), atom_term("y")]),
            &mut storage,
        )
        .unwrap();

        // Rules for transitive closure
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should derive path(a,c) by combining path(a,b) + edge(b,c)
        assert!(result.contains(
            &make_atom("path", vec![atom_term("a"), atom_term("c")]),
            &storage
        ));

        // Should have: path(a,b), path(b,c), path(x,y), path(a,c)
        let path_count = result.count_facts("path", &storage);
        assert_eq!(path_count, 4);
    }

    #[test]
    fn test_multiple_recursive_predicates_with_negation() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        db.insert(
            make_atom("parent", vec![atom_term("alice"), atom_term("bob")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("bob"), atom_term("charlie")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("charlie"), atom_term("dave")]),
            &mut storage,
        )
        .unwrap();

        let rules = vec![
            // ancestor(X,Y) :- parent(X,Y)
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "parent",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            // ancestor(X,Z) :- ancestor(X,Y), parent(Y,Z)
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("ancestor", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
                ],
            },
            // ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z) - different position for recursion
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("ancestor", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let (result, stats) = evaluate_instrumented(&rules, &[], db, &mut storage).unwrap();

        // Should derive all ancestor relationships
        let ancestor_count = result.count_facts("ancestor", &storage);
        assert_eq!(ancestor_count, 6);

        // Verify alice is ancestor of dave
        assert!(result.contains(
            &make_atom("ancestor", vec![atom_term("alice"), atom_term("dave")]),
            &storage
        ));

        // Verify stats - semi-naive should be efficient
        assert!(stats.iterations > 0);
        assert!(stats.facts_derived > 0);
    }

    #[test]
    fn test_ancestor_with_exclusions() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Family tree
        db.insert(
            make_atom("parent", vec![atom_term("grandpa"), atom_term("dad")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("dad"), atom_term("alice")]),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom("parent", vec![atom_term("alice"), atom_term("charlie")]),
            &mut storage,
        )
        .unwrap();

        // Estrangement (family dispute)
        db.insert(make_atom("estranged", vec![atom_term("dad")]), &mut storage)
            .unwrap();

        let rules = vec![
            // ancestor(X, Y) :- parent(X, Y).
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "parent",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
            Rule {
                head: make_atom("ancestor", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("ancestor", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("parent", vec![var_term("Y"), var_term("Z")])),
                ],
            },
            // recognized_family(X, Y) :- parent(X, Y), not estranged(X), not estranged(Y).
            Rule {
                head: make_atom("recognized_family", vec![var_term("X"), var_term("Y")]),
                body: vec![
                    Literal::Positive(make_atom("parent", vec![var_term("X"), var_term("Y")])),
                    Literal::Negative(make_atom("estranged", vec![var_term("X")])),
                    Literal::Negative(make_atom("estranged", vec![var_term("Y")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Grandpa -> dad -> alice should exist in ancestor
        assert!(result.contains(
            &make_atom("ancestor", vec![atom_term("grandpa"), atom_term("alice")]),
            &storage
        ));

        // But grandpa -> dad should NOT be in recognized_family (dad is estranged)
        assert!(!result.contains(
            &make_atom(
                "recognized_family",
                vec![atom_term("grandpa"), atom_term("dad")]
            ),
            &storage
        ));

        // Alice -> charlie should be in recognized_family (neither estranged)
        assert!(result.contains(
            &make_atom(
                "recognized_family",
                vec![atom_term("alice"), atom_term("charlie")]
            ),
            &storage
        ));
    }

    #[test]
    fn test_negation_with_different_datatypes() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Player stats
        db.insert(make_atom("player", vec![atom_term("alice")]), &mut storage)
            .unwrap();
        db.insert(make_atom("player", vec![atom_term("bob")]), &mut storage)
            .unwrap();

        // Alice has a shield (with boolean)
        db.insert(
            make_atom("has_shield", vec![atom_term("alice"), bool_term(true)]),
            &mut storage,
        )
        .unwrap();

        // Bob does not have a shield (no fact)

        // Damage values
        db.insert(make_atom("base_damage", vec![int_term(10)]), &mut storage)
            .unwrap();
        db.insert(make_atom("base_damage", vec![int_term(20)]), &mut storage)
            .unwrap();

        let rules = vec![
            // vulnerable(P) :- player(P), not has_shield(P, true).
            Rule {
                head: make_atom("vulnerable", vec![var_term("P")]),
                body: vec![
                    Literal::Positive(make_atom("player", vec![var_term("P")])),
                    Literal::Negative(make_atom(
                        "has_shield",
                        vec![var_term("P"), bool_term(true)],
                    )),
                ],
            },
            // will_take_damage(P, D) :- vulnerable(P), base_damage(D).
            Rule {
                head: make_atom("will_take_damage", vec![var_term("P"), var_term("D")]),
                body: vec![
                    Literal::Positive(make_atom("vulnerable", vec![var_term("P")])),
                    Literal::Positive(make_atom("base_damage", vec![var_term("D")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Bob is vulnerable (no shield)
        assert!(result.contains(&make_atom("vulnerable", vec![atom_term("bob")]), &storage));

        // Alice is NOT vulnerable (has shield)
        assert!(!result.contains(&make_atom("vulnerable", vec![atom_term("alice")]), &storage));

        // Bob will take damage (vulnerable)
        assert!(result.contains(
            &make_atom("will_take_damage", vec![atom_term("bob"), int_term(10)]),
            &storage
        ));
        assert!(result.contains(
            &make_atom("will_take_damage", vec![atom_term("bob"), int_term(20)]),
            &storage
        ));

        // Alice will NOT take damage (not vulnerable)
        assert!(!result.contains(
            &make_atom("will_take_damage", vec![atom_term("alice"), int_term(10)]),
            &storage
        ));
    }

    #[test]
    fn test_compound_terms_in_recursion() {
        let mut db = DatalogContext::new();
        let mut storage = MemoryEngine::new();

        // Graph with compound term labels
        db.insert(
            make_atom(
                "edge",
                vec![
                    compound_term("node", vec![atom_term("a"), int_term(1)]),
                    compound_term("node", vec![atom_term("b"), int_term(2)]),
                ],
            ),
            &mut storage,
        )
        .unwrap();
        db.insert(
            make_atom(
                "edge",
                vec![
                    compound_term("node", vec![atom_term("b"), int_term(2)]),
                    compound_term("node", vec![atom_term("c"), int_term(3)]),
                ],
            ),
            &mut storage,
        )
        .unwrap();

        // Transitive closure with compound terms
        let rules = vec![
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Y")]),
                body: vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var_term("X"), var_term("Y")],
                ))],
            },
            Rule {
                head: make_atom("path", vec![var_term("X"), var_term("Z")]),
                body: vec![
                    Literal::Positive(make_atom("path", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(make_atom("edge", vec![var_term("Y"), var_term("Z")])),
                ],
            },
        ];

        let result = evaluate(&rules, &[], db, &mut storage).unwrap();

        // Should derive path from node(a,1) to node(c,3)
        assert!(result.contains(
            &make_atom(
                "path",
                vec![
                    compound_term("node", vec![atom_term("a"), int_term(1)]),
                    compound_term("node", vec![atom_term("c"), int_term(3)])
                ]
            ),
            &storage
        ));

        // Should have: (a,1)->(b,2), (b,2)->(c,3), (a,1)->(c,3)
        let path_count = result.count_facts("path", &storage);
        assert_eq!(path_count, 3);
    }
}
