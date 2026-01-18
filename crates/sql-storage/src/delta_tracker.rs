//! Lightweight delta tracker for semi-naive Datalog evaluation
//!
//! DeltaTracker provides efficient in-memory storage for newly derived facts
//! during a single iteration of semi-naive evaluation. Unlike DatalogContext,
//! it does not use SQL storage and operates entirely in memory.
//!
//! # Indexing
//!
//! Facts are indexed by (predicate, first_argument) for efficient queries.
//! When a query pattern has a constant first argument, only matching facts
//! are considered, avoiding a full scan.
//!
//! # Example
//!
//! ```ignore
//! let mut delta = DeltaTracker::new();
//!
//! // Add newly derived facts
//! delta.insert(ancestor_fact);
//!
//! // Query delta for matching facts
//! let matches = delta.query(&pattern_with_variables);
//!
//! // Check if empty (fixpoint reached)
//! if delta.is_empty() {
//!     break;
//! }
//! ```

use crate::datalog_unification::{unify_atoms, Substitution};
use datalog_parser::{Atom, Symbol, Term};
use std::collections::HashMap;

/// Index key: (predicate, optional first argument for indexed lookup)
type IndexKey = (Symbol, Option<Term>);

/// Lightweight in-memory tracker for newly derived facts per iteration
///
/// Used in semi-naive evaluation to track the "delta" - facts derived in
/// the current iteration that weren't present before. All operations are
/// in-memory, making them fast for the typically small delta sets.
///
/// Facts are indexed by (predicate, first_arg) when the first argument is
/// a constant, enabling O(1) filtered lookups for common query patterns.
#[derive(Debug, Clone, Default)]
pub struct DeltaTracker {
    /// Facts indexed by (predicate, first_arg) for efficient filtered lookup
    indexed_facts: HashMap<IndexKey, Vec<Atom>>,
    /// Track predicates for iteration (to handle variable first-arg queries)
    predicates: HashMap<Symbol, Vec<Option<Term>>>,
}

impl DeltaTracker {
    /// Create an empty delta tracker
    pub fn new() -> Self {
        Self {
            indexed_facts: HashMap::new(),
            predicates: HashMap::new(),
        }
    }

    /// Extract the index key for a ground fact
    fn index_key(atom: &Atom) -> IndexKey {
        let first_arg = atom.terms.first().cloned();
        (atom.predicate, first_arg)
    }

    /// Add a ground fact to the delta
    ///
    /// Does not check for duplicates - deduplication is handled by
    /// storage UNIQUE constraints when facts are persisted.
    pub fn insert(&mut self, atom: Atom) {
        let key = Self::index_key(&atom);
        let first_arg = key.1.clone();

        // Track the first_arg for this predicate (for iteration)
        self.predicates
            .entry(atom.predicate)
            .or_default()
            .push(first_arg);

        // Store in indexed structure
        self.indexed_facts.entry(key).or_default().push(atom);
    }

    /// Query for facts matching a pattern using unification
    ///
    /// Uses the first-argument index when the pattern has a constant first arg,
    /// otherwise iterates all facts for the predicate.
    pub fn query(&self, pattern: &Atom) -> Vec<Substitution> {
        let first_arg = pattern.terms.first();

        // Check if first argument is a constant (can use index)
        let use_index = first_arg.is_some_and(|t| matches!(t, Term::Constant(_)));

        let mut results = Vec::new();

        if use_index {
            // Direct index lookup
            let key = (pattern.predicate, first_arg.cloned());
            if let Some(facts) = self.indexed_facts.get(&key) {
                for fact in facts {
                    let mut subst = Substitution::new();
                    if unify_atoms(pattern, fact, &mut subst) {
                        results.push(subst);
                    }
                }
            }
        } else {
            // Variable first arg or no args - iterate all first_args for this predicate
            let Some(first_args) = self.predicates.get(&pattern.predicate) else {
                return vec![];
            };

            // Get unique first_args to avoid duplicate lookups
            let mut seen_keys = std::collections::HashSet::new();
            for first_arg in first_args {
                let key = (pattern.predicate, first_arg.clone());
                if seen_keys.insert(key.clone()) {
                    if let Some(facts) = self.indexed_facts.get(&key) {
                        for fact in facts {
                            let mut subst = Substitution::new();
                            if unify_atoms(pattern, fact, &mut subst) {
                                results.push(subst);
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Check if the delta contains a specific ground fact
    #[allow(dead_code)]
    pub fn contains(&self, atom: &Atom) -> bool {
        let key = Self::index_key(atom);
        self.indexed_facts
            .get(&key)
            .map(|facts| facts.contains(atom))
            .unwrap_or(false)
    }

    /// Check if delta is empty (no facts for any predicate)
    pub fn is_empty(&self) -> bool {
        self.indexed_facts.values().all(|v| v.is_empty())
    }

    /// Get total count of facts across all predicates
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.indexed_facts.values().map(|v| v.len()).sum()
    }

    /// Clear all facts (for reuse between iterations)
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.indexed_facts.clear();
        self.predicates.clear();
    }

    /// Iterate over all facts by reference
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &Atom> {
        self.indexed_facts.values().flat_map(|v| v.iter())
    }

    /// Get facts for a specific predicate
    #[allow(dead_code)]
    pub fn get_by_predicate(&self, predicate: &Symbol) -> Vec<&Atom> {
        let Some(first_args) = self.predicates.get(predicate) else {
            return vec![];
        };

        let mut results = Vec::new();
        let mut seen_keys = std::collections::HashSet::new();
        for first_arg in first_args {
            let key = (*predicate, first_arg.clone());
            if seen_keys.insert(key.clone()) {
                if let Some(facts) = self.indexed_facts.get(&key) {
                    results.extend(facts.iter());
                }
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::{Term, Value};

    fn sym(s: &str) -> Symbol {
        Symbol::new(s.to_string())
    }

    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(sym(name)))
    }

    fn var_term(name: &str) -> Term {
        Term::Variable(sym(name))
    }

    #[test]
    fn test_insert_and_is_empty() {
        let mut delta = DeltaTracker::new();
        assert!(delta.is_empty());

        delta.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ));
        assert!(!delta.is_empty());
        assert_eq!(delta.len(), 1);
    }

    #[test]
    fn test_query_exact_match() {
        let mut delta = DeltaTracker::new();
        delta.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ));
        delta.insert(make_atom(
            "parent",
            vec![atom_term("mary"), atom_term("jane")],
        ));

        // Query with exact match
        let results = delta.query(&make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_with_variable() {
        let mut delta = DeltaTracker::new();
        delta.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ));
        delta.insert(make_atom(
            "parent",
            vec![atom_term("mary"), atom_term("jane")],
        ));

        // Query with variable in first position
        let results = delta.query(&make_atom("parent", vec![var_term("X"), atom_term("mary")]));
        assert_eq!(results.len(), 1);
        let x_binding = results[0].get(&sym("X")).unwrap();
        assert_eq!(*x_binding, atom_term("john"));
    }

    #[test]
    fn test_query_all_facts_for_predicate() {
        let mut delta = DeltaTracker::new();
        delta.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ));
        delta.insert(make_atom(
            "parent",
            vec![atom_term("mary"), atom_term("jane")],
        ));
        delta.insert(make_atom(
            "sibling",
            vec![atom_term("bob"), atom_term("sue")],
        ));

        // Query with all variables
        let results = delta.query(&make_atom("parent", vec![var_term("X"), var_term("Y")]));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_nonexistent_predicate() {
        let delta = DeltaTracker::new();
        let results = delta.query(&make_atom("unknown", vec![var_term("X")]));
        assert!(results.is_empty());
    }

    #[test]
    fn test_contains() {
        let mut delta = DeltaTracker::new();
        let fact = make_atom("parent", vec![atom_term("john"), atom_term("mary")]);

        assert!(!delta.contains(&fact));
        delta.insert(fact.clone());
        assert!(delta.contains(&fact));
    }

    #[test]
    fn test_clear() {
        let mut delta = DeltaTracker::new();
        delta.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ));
        assert!(!delta.is_empty());

        delta.clear();
        assert!(delta.is_empty());
    }

    #[test]
    fn test_indexed_query_with_constant_first_arg() {
        let mut delta = DeltaTracker::new();

        // Insert many facts with different first arguments
        delta.insert(make_atom("edge", vec![atom_term("a"), atom_term("b")]));
        delta.insert(make_atom("edge", vec![atom_term("a"), atom_term("c")]));
        delta.insert(make_atom("edge", vec![atom_term("b"), atom_term("c")]));
        delta.insert(make_atom("edge", vec![atom_term("c"), atom_term("d")]));
        delta.insert(make_atom("edge", vec![atom_term("c"), atom_term("e")]));

        // Query with constant first arg should use index
        let results = delta.query(&make_atom("edge", vec![atom_term("a"), var_term("Y")]));
        assert_eq!(results.len(), 2); // edge(a,b) and edge(a,c)

        let results = delta.query(&make_atom("edge", vec![atom_term("c"), var_term("Y")]));
        assert_eq!(results.len(), 2); // edge(c,d) and edge(c,e)

        let results = delta.query(&make_atom("edge", vec![atom_term("b"), var_term("Y")]));
        assert_eq!(results.len(), 1); // edge(b,c)

        // Query with variable first arg should scan all
        let results = delta.query(&make_atom("edge", vec![var_term("X"), var_term("Y")]));
        assert_eq!(results.len(), 5); // all edges
    }
}
