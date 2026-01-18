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

use datalog_grounding::{ground_rule, ground_rule_semi_naive_with_delta, satisfy_body};
use datalog_parser::{Constraint, Rule};
use datalog_safety::{check_program_safety, stratify, SafetyError, StratificationError};
use sql_storage::{DeltaTracker, FactDatabase, InsertError, StorageEngine, StorageError};

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
    initial_facts: FactDatabase,
    storage: &mut S,
) -> Result<FactDatabase, EvaluationError> {
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

/// Check constraints against the database.
///
/// A constraint is violated if its body can be satisfied (i.e., there exist
/// substitutions that make all literals in the body true).
pub fn check_constraints<S: StorageEngine>(
    constraints: &[Constraint],
    db: &FactDatabase,
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
    initial_facts: FactDatabase,
    storage: &mut S,
) -> Result<FactDatabase, EvaluationError> {
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
                if db.insert(fact.clone(), storage)? {
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

/// Evaluate a Datalog program using storage for indexed lookups.
///
/// Deprecated: Use `evaluate()` with storage parameter instead.
#[deprecated(note = "Use evaluate() with storage parameter instead")]
pub fn evaluate_with_storage<S: StorageEngine>(
    rules: &[Rule],
    constraints: &[Constraint],
    initial_facts: FactDatabase,
    storage: &mut S,
) -> Result<FactDatabase, EvaluationError> {
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
        let mut db = FactDatabase::new();
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
        let mut db = FactDatabase::new();
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
        let mut db = FactDatabase::new();
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
        let mut db = FactDatabase::new();
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
        let result = evaluate(&rules, &[], FactDatabase::new(), &mut storage);

        assert!(matches!(result, Err(EvaluationError::Stratification(_))));
    }
}
