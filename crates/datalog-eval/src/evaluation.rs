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

use datalog_ast::{Constraint, Rule};
use datalog_core::{FactDatabase, InsertError};
use datalog_grounding::{ground_rule, ground_rule_semi_naive, satisfy_body};
use datalog_safety::{check_program_safety, stratify, SafetyError, StratificationError};

/// Errors that can occur during evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// let result = evaluate(&rules, &[], facts)?;
/// ```
pub fn evaluate(
    rules: &[Rule],
    constraints: &[Constraint],
    initial_facts: FactDatabase,
) -> Result<FactDatabase, EvaluationError> {
    // Check safety first (variables in negation must appear in positive literals, etc.)
    check_program_safety(rules)?;

    // Stratify the program to handle negation correctly
    let stratification = stratify(rules)?;

    let mut db = initial_facts;

    // Evaluate each stratum to fixed point before moving to the next
    for stratum_rules in &stratification.rules_by_stratum {
        db = semi_naive_evaluate(stratum_rules, db)?;
    }

    // Check constraints after evaluation
    check_constraints(constraints, &db)?;

    Ok(db)
}

/// Check constraints against the database.
///
/// A constraint is violated if its body can be satisfied (i.e., there exist
/// substitutions that make all literals in the body true).
pub fn check_constraints(
    constraints: &[Constraint],
    db: &FactDatabase,
) -> Result<(), EvaluationError> {
    for constraint in constraints {
        let violations = satisfy_body(&constraint.body, db);

        if !violations.is_empty() {
            return Err(EvaluationError::ConstraintViolation {
                constraint: format!("{:?}", constraint.body),
                violation_count: violations.len(),
            });
        }
    }
    Ok(())
}

/// Semi-naive evaluation for a set of rules within a single stratum.
///
/// This is an internal function used by `evaluate()`. It processes only
/// new facts (delta) each iteration to avoid redundant derivations.
fn semi_naive_evaluate(
    rules: &[Rule],
    initial_facts: FactDatabase,
) -> Result<FactDatabase, EvaluationError> {
    let mut db = initial_facts.clone();
    let mut delta = initial_facts;

    loop {
        let mut new_delta = FactDatabase::new();

        for rule in rules {
            let derived = if rule.body.is_empty() {
                // No body - always evaluate (fact-like rule)
                ground_rule(rule, &db)
            } else if rule.body.len() == 1 {
                // Single literal - just use delta
                ground_rule(rule, &delta)
            } else {
                // Multi-literal: use semi-naive grounding
                ground_rule_semi_naive(rule, &delta, &db)
            };

            for fact in derived {
                if !db.contains(&fact) && !new_delta.contains(&fact) {
                    new_delta.insert(fact)?;
                }
            }
        }

        // Fixed point reached when no new facts are derived
        if new_delta.is_empty() {
            break;
        }

        let delta_next = new_delta.clone();
        db.absorb(new_delta);
        delta = delta_next;
    }

    Ok(db)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_ast::{Atom, Literal, Symbol, Term, Value};

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
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("mary"), atom_term("jane")],
        ))
        .unwrap();

        let rules = vec![Rule {
            head: make_atom("ancestor", vec![var_term("X"), var_term("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var_term("X"), var_term("Y")],
            ))],
        }];

        let result = evaluate(&rules, &[], db).unwrap();

        // Should have 2 parent facts + 2 derived ancestor facts = 4
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_transitive_closure() {
        // ancestor(X, Y) :- parent(X, Y).
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_term("a"), atom_term("b")]))
            .unwrap();
        db.insert(make_atom("parent", vec![atom_term("b"), atom_term("c")]))
            .unwrap();
        db.insert(make_atom("parent", vec![atom_term("c"), atom_term("d")]))
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

        let result = evaluate(&rules, &[], db).unwrap();

        // 3 parent facts + 6 ancestor facts (a->b, b->c, c->d, a->c, b->d, a->d) = 9
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_negation_with_stratification() {
        // person(alice). person(bob). parent(alice, charlie).
        // childless(X) :- person(X), not parent(X, _).
        let mut db = FactDatabase::new();
        db.insert(make_atom("person", vec![atom_term("alice")]))
            .unwrap();
        db.insert(make_atom("person", vec![atom_term("bob")]))
            .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("alice"), atom_term("charlie")],
        ))
        .unwrap();

        let rules = vec![Rule {
            head: make_atom("childless", vec![var_term("X")]),
            body: vec![
                Literal::Positive(make_atom("person", vec![var_term("X")])),
                Literal::Negative(make_atom("parent", vec![var_term("X"), var_term("_Y")])),
            ],
        }];

        let result = evaluate(&rules, &[], db).unwrap();

        // Should derive childless(bob) but not childless(alice)
        // 2 person + 1 parent + 1 childless = 4
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_constraint_violation() {
        // Constraint: :- dangerous(X).
        let mut db = FactDatabase::new();
        db.insert(make_atom("dangerous", vec![atom_term("bomb")]))
            .unwrap();

        let constraints = vec![Constraint {
            body: vec![Literal::Positive(make_atom(
                "dangerous",
                vec![var_term("X")],
            ))],
        }];

        let result = evaluate(&[], &constraints, db);

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

        let result = evaluate(&rules, &[], FactDatabase::new());

        assert!(matches!(result, Err(EvaluationError::Stratification(_))));
    }
}
