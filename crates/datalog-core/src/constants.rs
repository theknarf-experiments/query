//! Constant environment for Datalog evaluation.
//!
//! Provides a way to define and substitute named constants in terms and atoms.

use datalog_ast::{Atom, Symbol, Term, Value};
use std::collections::HashMap;

/// Environment storing constant declarations
#[derive(Debug, Clone, Default)]
pub struct ConstantEnv {
    constants: HashMap<Symbol, Value>,
}

impl ConstantEnv {
    /// Create a new empty constant environment
    pub fn new() -> Self {
        ConstantEnv {
            constants: HashMap::new(),
        }
    }

    /// Add a constant to the environment
    pub fn define(&mut self, name: Symbol, value: Value) {
        self.constants.insert(name, value);
    }

    /// Get the value of a constant
    pub fn get(&self, name: &Symbol) -> Option<&Value> {
        self.constants.get(name)
    }

    /// Get an integer value from a constant
    pub fn get_int(&self, name: &Symbol) -> Option<i64> {
        self.constants.get(name).and_then(|v| match v {
            Value::Integer(n) => Some(*n),
            _ => None,
        })
    }

    /// Substitute constants in a term
    /// If a term is an atom (constant) that matches a const name, replace with its value
    pub fn substitute_term(&self, term: &Term) -> Term {
        match term {
            Term::Constant(Value::Atom(name)) => {
                // Check if this atom is actually a constant name
                if let Some(value) = self.get(name) {
                    Term::Constant(value.clone())
                } else {
                    term.clone()
                }
            }
            Term::Compound(functor, args) => {
                // Recursively substitute in compound terms
                let new_args: Vec<Term> =
                    args.iter().map(|arg| self.substitute_term(arg)).collect();
                Term::Compound(*functor, new_args)
            }
            _ => term.clone(),
        }
    }

    /// Substitute constants in an atom
    pub fn substitute_atom(&self, atom: &Atom) -> Atom {
        Atom {
            predicate: atom.predicate,
            terms: atom.terms.iter().map(|t| self.substitute_term(t)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use internment::Intern;

    #[test]
    fn test_define_and_get() {
        let mut env = ConstantEnv::new();
        let name = Intern::new("width".to_string());
        env.define(name, Value::Integer(100));

        assert_eq!(env.get_int(&name), Some(100));
    }

    #[test]
    fn test_substitute_term() {
        let mut env = ConstantEnv::new();
        let name = Intern::new("max".to_string());
        env.define(name, Value::Integer(50));

        let term = Term::Constant(Value::Atom(Intern::new("max".to_string())));
        let result = env.substitute_term(&term);

        assert_eq!(result, Term::Constant(Value::Integer(50)));
    }
}
