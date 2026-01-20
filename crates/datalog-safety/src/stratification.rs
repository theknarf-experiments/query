//! Stratification analysis for programs with negation
//!
//! This module analyzes programs to determine if they can be safely evaluated
//! with negation. A program is stratifiable if there are no cycles through negation.
//!
//! # Stratification
//!
//! Stratification assigns each predicate to a stratum (layer). Predicates in higher
//! strata can depend on negated predicates from lower strata, but not vice versa.
//!
//! # Algorithm
//!
//! 1. Build dependency graph of predicates
//! 2. Detect negative cycles (cycles through negated edges)
//! 3. Assign strata using topological sort
//!
//! # Example
//!
//! ```ignore
//! let stratification = stratify(&rules)?;
//! for (stratum, rules_in_stratum) in stratification.iter() {
//!     // Evaluate rules stratum by stratum
//! }
//! ```

use datalog_parser::{Literal, Rule, Symbol};
use std::collections::{HashMap, HashSet};

/// Result of stratification analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stratification {
    /// Map from predicate name to stratum number (0 = bottom stratum)
    pub predicate_strata: HashMap<Symbol, usize>,
    /// Rules organized by stratum
    pub rules_by_stratum: Vec<Vec<Rule>>,
    /// Total number of strata
    pub num_strata: usize,
}

/// Error during stratification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratificationError {
    /// Program has a cycle through negation (not stratifiable)
    CycleThroughNegation(Vec<Symbol>),
}

impl std::fmt::Display for StratificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StratificationError::CycleThroughNegation(cycle) => {
                if cycle.is_empty() {
                    write!(
                        f,
                        "Cycle through negation detected (cycle could not be reconstructed)"
                    )
                } else {
                    let cycle_sequence: Vec<String> = cycle
                        .iter()
                        .map(|symbol| symbol.as_ref().to_string())
                        .collect();
                    let mut display_cycle = cycle_sequence.clone();
                    if let Some(first) = cycle_sequence.first() {
                        display_cycle.push(first.clone());
                    }

                    write!(
                        f,
                        "Cycle through negation detected: {}",
                        display_cycle.join(" -> ")
                    )
                }
            }
        }
    }
}

/// Dependency between predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyType {
    Positive, // p depends positively on q
    Negative, // p depends negatively on q (through negation)
}

/// Dependency graph for stratification analysis
#[derive(Debug, Clone)]
struct DependencyGraph {
    /// Map from predicate to its dependencies
    dependencies: HashMap<Symbol, Vec<(Symbol, DependencyType)>>,
    /// All predicates in the program
    predicates: HashSet<Symbol>,
}

impl DependencyGraph {
    fn new() -> Self {
        DependencyGraph {
            dependencies: HashMap::new(),
            predicates: HashSet::new(),
        }
    }

    /// Add a dependency edge
    fn add_dependency(&mut self, from: Symbol, to: Symbol, dep_type: DependencyType) {
        self.predicates.insert(from);
        self.predicates.insert(to);

        self.dependencies
            .entry(from)
            .or_default()
            .push((to, dep_type));
    }

    /// Get all dependencies of a predicate
    fn get_dependencies(&self, pred: &Symbol) -> Vec<(Symbol, DependencyType)> {
        self.dependencies.get(pred).cloned().unwrap_or_default()
    }
}

/// Build dependency graph from rules
fn build_dependency_graph(rules: &[Rule]) -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    for rule in rules {
        let head_pred = rule.head.predicate;

        for literal in &rule.body {
            match literal {
                Literal::Positive(atom) => {
                    graph.add_dependency(head_pred, atom.predicate, DependencyType::Positive);
                }
                Literal::Negative(atom) => {
                    graph.add_dependency(head_pred, atom.predicate, DependencyType::Negative);
                }
                Literal::Comparison(_) => {
                    // Comparisons don't create predicate dependencies
                }
            }
        }
    }

    graph
}

/// Check if there's a path from 'from' to 'to' through a negative edge
fn has_cycle_through_negation(
    graph: &DependencyGraph,
    from: &Symbol,
    to: &Symbol,
    visited: &mut HashSet<Symbol>,
    has_negative: bool,
) -> bool {
    if from == to && has_negative {
        return true;
    }

    if visited.contains(from) {
        return false;
    }

    visited.insert(*from);

    for (dep, dep_type) in graph.get_dependencies(from) {
        let is_negative = has_negative || matches!(dep_type, DependencyType::Negative);

        if has_cycle_through_negation(graph, &dep, to, visited, is_negative) {
            return true;
        }
    }

    visited.remove(from);
    false
}

/// Detect cycles through negation in the dependency graph
fn detect_negative_cycles(graph: &DependencyGraph) -> Option<Vec<Symbol>> {
    for pred in &graph.predicates {
        let mut visited = HashSet::new();
        if has_cycle_through_negation(graph, pred, pred, &mut visited, false) {
            return Some(vec![*pred]);
        }
    }
    None
}

/// Compute stratum for each predicate using iterative algorithm
/// A predicate's stratum is the maximum stratum of its negated dependencies + 1
fn compute_strata(graph: &DependencyGraph) -> HashMap<Symbol, usize> {
    let mut strata: HashMap<Symbol, usize> = HashMap::new();

    // Initialize all predicates to stratum 0
    for pred in &graph.predicates {
        strata.insert(*pred, 0);
    }

    // Iterate until fixed point
    let mut changed = true;
    while changed {
        changed = false;

        for pred in &graph.predicates {
            let mut max_stratum = 0;

            for (dep, dep_type) in graph.get_dependencies(pred) {
                let dep_stratum = *strata.get(&dep).unwrap_or(&0);

                let required_stratum = match dep_type {
                    DependencyType::Positive => dep_stratum,
                    DependencyType::Negative => dep_stratum + 1, // Must be after negated predicate
                };

                max_stratum = max_stratum.max(required_stratum);
            }

            if max_stratum > *strata.get(pred).unwrap() {
                strata.insert(*pred, max_stratum);
                changed = true;
            }
        }
    }

    strata
}

/// Stratify a program
pub fn stratify(rules: &[Rule]) -> Result<Stratification, StratificationError> {
    if rules.is_empty() {
        return Ok(Stratification {
            predicate_strata: HashMap::new(),
            rules_by_stratum: vec![],
            num_strata: 0,
        });
    }

    // Build dependency graph
    let graph = build_dependency_graph(rules);

    // Check for cycles through negation
    if let Some(cycle) = detect_negative_cycles(&graph) {
        return Err(StratificationError::CycleThroughNegation(cycle));
    }

    // Compute strata
    let predicate_strata = compute_strata(&graph);

    // Find maximum stratum
    let num_strata = predicate_strata.values().max().copied().unwrap_or(0) + 1;

    // Organize rules by stratum
    let mut rules_by_stratum: Vec<Vec<Rule>> = vec![Vec::new(); num_strata];

    for rule in rules {
        let stratum = *predicate_strata.get(&rule.head.predicate).unwrap_or(&0);
        rules_by_stratum[stratum].push(rule.clone());
    }

    Ok(Stratification {
        predicate_strata,
        rules_by_stratum,
        num_strata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::{Atom, Literal, Rule, Value};

    fn sym(s: &str) -> Symbol {
        Symbol::new(s.to_string())
    }

    fn var_term(name: &str) -> datalog_parser::Term {
        datalog_parser::Term::Variable(sym(name))
    }

    fn atom_term(name: &str) -> datalog_parser::Term {
        datalog_parser::Term::Constant(Value::Atom(sym(name)))
    }

    fn atom(pred: &str, terms: Vec<datalog_parser::Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    fn make_rule(head: Atom, body: Vec<Literal>) -> Rule {
        Rule { head, body }
    }

    // ===== Basic Stratification Tests =====

    #[test]
    fn test_no_negation_single_stratum() {
        // ancestor(X, Y) :- parent(X, Y).
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let rules = vec![
            make_rule(
                atom("ancestor", vec![var_term("X"), var_term("Y")]),
                vec![Literal::Positive(atom(
                    "parent",
                    vec![var_term("X"), var_term("Y")],
                ))],
            ),
            make_rule(
                atom("ancestor", vec![var_term("X"), var_term("Z")]),
                vec![
                    Literal::Positive(atom("ancestor", vec![var_term("X"), var_term("Y")])),
                    Literal::Positive(atom("parent", vec![var_term("Y"), var_term("Z")])),
                ],
            ),
        ];

        let result = stratify(&rules).unwrap();
        // All predicates should be in stratum 0 (no negation)
        assert_eq!(result.num_strata, 1);
    }

    #[test]
    fn test_negation_two_strata() {
        // not_parent(X) :- person(X), not parent(X, _).
        let rules = vec![make_rule(
            atom("not_parent", vec![var_term("X")]),
            vec![
                Literal::Positive(atom("person", vec![var_term("X")])),
                Literal::Negative(atom("parent", vec![var_term("X"), var_term("_Y")])),
            ],
        )];

        let result = stratify(&rules).unwrap();
        // parent should be in stratum 0, not_parent in stratum 1
        assert_eq!(result.num_strata, 2);
        assert_eq!(*result.predicate_strata.get(&sym("parent")).unwrap(), 0);
        assert_eq!(*result.predicate_strata.get(&sym("not_parent")).unwrap(), 1);
    }

    // ===== Cycle Detection Tests =====

    #[test]
    fn test_cycle_through_negation_error() {
        // p(X) :- not q(X).
        // q(X) :- not p(X).
        let rules = vec![
            make_rule(
                atom("p", vec![var_term("X")]),
                vec![Literal::Negative(atom("q", vec![var_term("X")]))],
            ),
            make_rule(
                atom("q", vec![var_term("X")]),
                vec![Literal::Negative(atom("p", vec![var_term("X")]))],
            ),
        ];

        let result = stratify(&rules);
        assert!(result.is_err());
        match result {
            Err(StratificationError::CycleThroughNegation(_)) => {}
            _ => panic!("Expected CycleThroughNegation error"),
        }
    }

    #[test]
    fn test_positive_cycle_ok() {
        // p(X) :- q(X).
        // q(X) :- p(X).
        // Positive cycles are allowed
        let rules = vec![
            make_rule(
                atom("p", vec![var_term("X")]),
                vec![Literal::Positive(atom("q", vec![var_term("X")]))],
            ),
            make_rule(
                atom("q", vec![var_term("X")]),
                vec![Literal::Positive(atom("p", vec![var_term("X")]))],
            ),
        ];

        let result = stratify(&rules);
        assert!(result.is_ok());
    }

    #[test]
    fn test_self_negation_error() {
        // UNSTRATIFIABLE: p(X) :- not p(X).
        let rules = vec![make_rule(
            atom("p", vec![var_term("X")]),
            vec![Literal::Negative(atom("p", vec![var_term("X")]))],
        )];

        let result = stratify(&rules);
        assert!(result.is_err());
    }

    // ===== Chain of Negations Tests =====

    #[test]
    fn test_chain_of_negations() {
        // a(X) :- base(X).
        // b(X) :- a(X), not c(X).
        // c(X) :- base(X).
        // d(X) :- b(X), not e(X).
        // e(X) :- base(X).
        let rules = vec![
            make_rule(
                atom("a", vec![var_term("X")]),
                vec![Literal::Positive(atom("base", vec![var_term("X")]))],
            ),
            make_rule(
                atom("b", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("a", vec![var_term("X")])),
                    Literal::Negative(atom("c", vec![var_term("X")])),
                ],
            ),
            make_rule(
                atom("c", vec![var_term("X")]),
                vec![Literal::Positive(atom("base", vec![var_term("X")]))],
            ),
            make_rule(
                atom("d", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("b", vec![var_term("X")])),
                    Literal::Negative(atom("e", vec![var_term("X")])),
                ],
            ),
            make_rule(
                atom("e", vec![var_term("X")]),
                vec![Literal::Positive(atom("base", vec![var_term("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();
        // d depends on negated e (stratum 0), so d is stratum 1
        // d depends on b, b depends on negated c (stratum 0), so b is stratum 1
        // d must be >= max(b, e+1) = max(1, 1) = 1, but b needs c computed first
        assert!(result.num_strata >= 2);
    }

    #[test]
    fn test_three_level_stratification() {
        // level0(X) :- base(X).
        // level1(X) :- level0(X), not blocked0(X).
        // level2(X) :- level1(X), not blocked1(X).
        let rules = vec![
            make_rule(
                atom("level0", vec![var_term("X")]),
                vec![Literal::Positive(atom("base", vec![var_term("X")]))],
            ),
            make_rule(
                atom("level1", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("level0", vec![var_term("X")])),
                    Literal::Negative(atom("blocked0", vec![var_term("X")])),
                ],
            ),
            make_rule(
                atom("level2", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("level1", vec![var_term("X")])),
                    Literal::Negative(atom("blocked1", vec![var_term("X")])),
                ],
            ),
        ];

        let result = stratify(&rules).unwrap();
        // level0: stratum 0
        // level1: stratum 1 (depends on not blocked0)
        // level2: stratum 2 (depends on level1 and not blocked1)
        assert!(result.num_strata >= 2);
        assert!(
            result.predicate_strata.get(&sym("level2")).unwrap()
                > result.predicate_strata.get(&sym("level0")).unwrap()
        );
    }

    // ===== Rules by Stratum Tests =====

    #[test]
    fn test_rules_by_stratum() {
        // Rule in stratum 0
        // Rule in stratum 1
        let rules = vec![
            make_rule(
                atom("base", vec![var_term("X")]),
                vec![Literal::Positive(atom("input", vec![var_term("X")]))],
            ),
            make_rule(
                atom("derived", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("input", vec![var_term("X")])),
                    Literal::Negative(atom("base", vec![var_term("X")])),
                ],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // Check rules are organized by stratum
        assert_eq!(result.rules_by_stratum.len(), result.num_strata);

        // Stratum 0 should have the base rule
        let stratum0_preds: Vec<_> = result.rules_by_stratum[0]
            .iter()
            .map(|r| r.head.predicate)
            .collect();
        assert!(stratum0_preds.contains(&sym("base")));
    }

    // ===== Empty and Edge Cases =====

    #[test]
    fn test_empty_program() {
        let rules: Vec<Rule> = vec![];
        let result = stratify(&rules).unwrap();
        assert_eq!(result.num_strata, 0);
        assert!(result.rules_by_stratum.is_empty());
    }

    #[test]
    fn test_single_fact_rule() {
        // fact(a).
        let rules = vec![make_rule(atom("fact", vec![atom_term("a")]), vec![])];

        let result = stratify(&rules).unwrap();
        assert_eq!(result.num_strata, 1);
    }

    #[test]
    fn test_multiple_negations_same_rule() {
        // p(X) :- a(X), not b(X), not c(X), not d(X).
        let rules = vec![make_rule(
            atom("p", vec![var_term("X")]),
            vec![
                Literal::Positive(atom("a", vec![var_term("X")])),
                Literal::Negative(atom("b", vec![var_term("X")])),
                Literal::Negative(atom("c", vec![var_term("X")])),
                Literal::Negative(atom("d", vec![var_term("X")])),
            ],
        )];

        let result = stratify(&rules).unwrap();
        // p depends on negations of b, c, d - all at stratum 0
        // So p is at stratum 1
        assert_eq!(*result.predicate_strata.get(&sym("p")).unwrap(), 1);
    }

    #[test]
    fn test_diamond_dependency() {
        // Diamond: a -> b, a -> c, b -> d, c -> d
        // With negation on the paths
        // a(X) :- base(X).
        // b(X) :- a(X), not block_b(X).
        // c(X) :- a(X), not block_c(X).
        // d(X) :- b(X), c(X).
        let rules = vec![
            make_rule(
                atom("a", vec![var_term("X")]),
                vec![Literal::Positive(atom("base", vec![var_term("X")]))],
            ),
            make_rule(
                atom("b", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("a", vec![var_term("X")])),
                    Literal::Negative(atom("block_b", vec![var_term("X")])),
                ],
            ),
            make_rule(
                atom("c", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("a", vec![var_term("X")])),
                    Literal::Negative(atom("block_c", vec![var_term("X")])),
                ],
            ),
            make_rule(
                atom("d", vec![var_term("X")]),
                vec![
                    Literal::Positive(atom("b", vec![var_term("X")])),
                    Literal::Positive(atom("c", vec![var_term("X")])),
                ],
            ),
        ];

        let result = stratify(&rules).unwrap();
        // a: stratum 0
        // b, c: stratum 1 (depend on negations)
        // d: stratum 1 (depends on b and c positively)
        assert!(result.predicate_strata.get(&sym("d")).unwrap() >= &1);
    }

    #[test]
    fn test_indirect_negative_cycle_error() {
        // Indirect cycle through negation:
        // a(X) :- not b(X).
        // b(X) :- c(X).
        // c(X) :- not a(X).
        let rules = vec![
            make_rule(
                atom("a", vec![var_term("X")]),
                vec![Literal::Negative(atom("b", vec![var_term("X")]))],
            ),
            make_rule(
                atom("b", vec![var_term("X")]),
                vec![Literal::Positive(atom("c", vec![var_term("X")]))],
            ),
            make_rule(
                atom("c", vec![var_term("X")]),
                vec![Literal::Negative(atom("a", vec![var_term("X")]))],
            ),
        ];

        let result = stratify(&rules);
        assert!(result.is_err());
    }
}
