//! Lightweight delta tracker for semi-naive Datalog evaluation
//!
//! DeltaTracker provides efficient in-memory storage for newly derived facts
//! during a single iteration of semi-naive evaluation. Unlike FactDatabase,
//! it does not use SQL storage and operates entirely in memory.
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
use datalog_parser::{Atom, Symbol};
use std::collections::HashMap;

/// Lightweight in-memory tracker for newly derived facts per iteration
///
/// Used in semi-naive evaluation to track the "delta" - facts derived in
/// the current iteration that weren't present before. All operations are
/// in-memory, making them fast for the typically small delta sets.
#[derive(Debug, Clone, Default)]
pub struct DeltaTracker {
    /// Facts organized by predicate for efficient lookup
    facts: HashMap<Symbol, Vec<Atom>>,
}

impl DeltaTracker {
    /// Create an empty delta tracker
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
        }
    }

    /// Add a ground fact to the delta
    ///
    /// Does not check for duplicates - deduplication is handled by
    /// storage UNIQUE constraints when facts are persisted.
    pub fn insert(&mut self, atom: Atom) {
        self.facts.entry(atom.predicate).or_default().push(atom);
    }

    /// Query for facts matching a pattern using unification
    ///
    /// Returns substitutions for all matching facts. This is O(n) where
    /// n is the number of facts for the pattern's predicate.
    pub fn query(&self, pattern: &Atom) -> Vec<Substitution> {
        let Some(facts) = self.facts.get(&pattern.predicate) else {
            return vec![];
        };

        let mut results = Vec::new();
        for fact in facts {
            let mut subst = Substitution::new();
            if unify_atoms(pattern, fact, &mut subst) {
                results.push(subst);
            }
        }
        results
    }

    /// Check if the delta contains a specific ground fact
    #[allow(dead_code)]
    pub fn contains(&self, atom: &Atom) -> bool {
        self.facts
            .get(&atom.predicate)
            .map(|facts| facts.contains(atom))
            .unwrap_or(false)
    }

    /// Check if delta is empty (no facts for any predicate)
    pub fn is_empty(&self) -> bool {
        self.facts.values().all(|v| v.is_empty())
    }

    /// Get total count of facts across all predicates
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.facts.values().map(|v| v.len()).sum()
    }

    /// Clear all facts (for reuse between iterations)
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.facts.clear();
    }

    /// Iterate over all facts by reference
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &Atom> {
        self.facts.values().flat_map(|v| v.iter())
    }

    /// Get facts for a specific predicate
    #[allow(dead_code)]
    pub fn get_by_predicate(&self, predicate: &Symbol) -> &[Atom] {
        self.facts
            .get(predicate)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
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
}
