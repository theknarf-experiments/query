//! Constant environment for Datalog evaluation.
//!
//! Provides a way to define and substitute named constants in terms and atoms.

use datalog_parser::{Atom, Symbol, Term, Value};
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

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn int_term(value: i64) -> Term {
        Term::Constant(Value::Integer(value))
    }

    fn float_term(value: f64) -> Term {
        Term::Constant(Value::Float(value))
    }

    fn bool_term(value: bool) -> Term {
        Term::Constant(Value::Boolean(value))
    }

    fn string_term(value: &str) -> Term {
        Term::Constant(Value::String(Intern::new(value.to_string())))
    }

    fn make_atom(predicate: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: Intern::new(predicate.to_string()),
            terms,
        }
    }

    // ===== Basic ConstantEnv Tests =====

    #[test]
    fn test_constant_env_new() {
        let env = ConstantEnv::new();
        assert!(env.get(&Intern::new("width".to_string())).is_none());
    }

    #[test]
    fn test_constant_env_define_and_get() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("width".to_string()), Value::Integer(10));
        env.define(Intern::new("height".to_string()), Value::Integer(20));

        assert_eq!(env.get_int(&Intern::new("width".to_string())), Some(10));
        assert_eq!(env.get_int(&Intern::new("height".to_string())), Some(20));
        assert_eq!(env.get_int(&Intern::new("depth".to_string())), None);
    }

    // ===== Term Substitution Tests =====

    #[test]
    fn test_substitute_term_atom_to_int() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("max_enemies".to_string()), Value::Integer(5));

        let term = atom_term("max_enemies");
        let result = env.substitute_term(&term);

        assert_eq!(result, int_term(5));
    }

    #[test]
    fn test_substitute_term_no_match() {
        let env = ConstantEnv::new();

        let term = atom_term("regular_atom");
        let result = env.substitute_term(&term);

        assert_eq!(result, atom_term("regular_atom"));
    }

    #[test]
    fn test_substitute_term_in_compound() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("width".to_string()), Value::Integer(10));

        let term = Term::Compound(
            Intern::new("dims".to_string()),
            vec![atom_term("width"), int_term(20)],
        );

        let result = env.substitute_term(&term);

        let expected = Term::Compound(
            Intern::new("dims".to_string()),
            vec![int_term(10), int_term(20)],
        );

        assert_eq!(result, expected);
    }

    // ===== Atom Substitution Tests =====

    #[test]
    fn test_substitute_atom() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("max_count".to_string()), Value::Integer(100));

        let atom = make_atom("count", vec![atom_term("max_count")]);
        let result = env.substitute_atom(&atom);

        let expected = make_atom("count", vec![int_term(100)]);
        assert_eq!(result, expected);
    }

    // ===== Non-Integer Constant Type Tests =====

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_substitute_term_with_float() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("pi".to_string()), Value::Float(3.14));

        let term = atom_term("pi");
        let result = env.substitute_term(&term);

        assert_eq!(result, float_term(3.14));
    }

    #[test]
    fn test_substitute_term_with_boolean() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("enabled".to_string()), Value::Boolean(true));

        let term = atom_term("enabled");
        let result = env.substitute_term(&term);

        assert_eq!(result, bool_term(true));
    }

    #[test]
    fn test_substitute_term_with_string() {
        let mut env = ConstantEnv::new();
        env.define(
            Intern::new("message".to_string()),
            Value::String(Intern::new("hello".to_string())),
        );

        let term = atom_term("message");
        let result = env.substitute_term(&term);

        assert_eq!(result, string_term("hello"));
    }

    #[test]
    fn test_substitute_term_with_atom_value() {
        let mut env = ConstantEnv::new();
        env.define(
            Intern::new("color".to_string()),
            Value::Atom(Intern::new("red".to_string())),
        );

        let term = atom_term("color");
        let result = env.substitute_term(&term);

        assert_eq!(result, atom_term("red"));
    }

    // ===== Integration Tests =====

    #[test]
    fn test_integration_multiple_const_types_in_atoms() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("max_health".to_string()), Value::Integer(100));
        env.define(
            Intern::new("damage_multiplier".to_string()),
            Value::Float(1.5),
        );
        env.define(Intern::new("is_enabled".to_string()), Value::Boolean(true));
        env.define(
            Intern::new("default_weapon".to_string()),
            Value::Atom(Intern::new("sword".to_string())),
        );

        // Create an atom with multiple constant references
        let atom = make_atom(
            "config",
            vec![
                atom_term("max_health"),
                atom_term("damage_multiplier"),
                atom_term("is_enabled"),
                atom_term("default_weapon"),
            ],
        );

        let result = env.substitute_atom(&atom);

        let expected = make_atom(
            "config",
            vec![
                int_term(100),
                float_term(1.5),
                bool_term(true),
                atom_term("sword"),
            ],
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_integration_constants_in_nested_compounds() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("width".to_string()), Value::Integer(10));
        env.define(Intern::new("height".to_string()), Value::Integer(20));
        env.define(Intern::new("scale".to_string()), Value::Float(2.0));

        // Create a nested compound term with constants
        let term = Term::Compound(
            Intern::new("rectangle".to_string()),
            vec![
                Term::Compound(
                    Intern::new("dimensions".to_string()),
                    vec![atom_term("width"), atom_term("height")],
                ),
                atom_term("scale"),
            ],
        );

        let result = env.substitute_term(&term);

        let expected = Term::Compound(
            Intern::new("rectangle".to_string()),
            vec![
                Term::Compound(
                    Intern::new("dimensions".to_string()),
                    vec![int_term(10), int_term(20)],
                ),
                float_term(2.0),
            ],
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_integration_mixed_constants_and_regular_values() {
        let mut env = ConstantEnv::new();
        env.define(
            Intern::new("default_port".to_string()),
            Value::Integer(8080),
        );
        env.define(
            Intern::new("default_host".to_string()),
            Value::String(Intern::new("localhost".to_string())),
        );

        // Create an atom mixing constants and regular values
        let atom = make_atom(
            "server",
            vec![
                atom_term("default_host"),
                atom_term("default_port"),
                bool_term(true),   // Regular value, not a constant
                atom_term("http"), // Regular atom
            ],
        );

        let result = env.substitute_atom(&atom);

        let expected = make_atom(
            "server",
            vec![
                string_term("localhost"),
                int_term(8080),
                bool_term(true),
                atom_term("http"),
            ],
        );

        assert_eq!(result, expected);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_variable_not_substituted() {
        let env = ConstantEnv::new();

        let term = Term::Variable(Intern::new("X".to_string()));
        let result = env.substitute_term(&term);

        // Variables should not be substituted
        assert_eq!(result, Term::Variable(Intern::new("X".to_string())));
    }

    #[test]
    fn test_integer_constant_not_substituted() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("42".to_string()), Value::Integer(100));

        // Integer constants should not be substituted (only atoms are looked up)
        let term = int_term(42);
        let result = env.substitute_term(&term);

        assert_eq!(result, int_term(42));
    }

    #[test]
    fn test_empty_compound_substitution() {
        let env = ConstantEnv::new();

        let term = Term::Compound(Intern::new("empty".to_string()), vec![]);
        let result = env.substitute_term(&term);

        assert_eq!(
            result,
            Term::Compound(Intern::new("empty".to_string()), vec![])
        );
    }

    #[test]
    fn test_deeply_nested_compound_substitution() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("val".to_string()), Value::Integer(42));

        // a(b(c(d(val))))
        let term = Term::Compound(
            Intern::new("a".to_string()),
            vec![Term::Compound(
                Intern::new("b".to_string()),
                vec![Term::Compound(
                    Intern::new("c".to_string()),
                    vec![Term::Compound(
                        Intern::new("d".to_string()),
                        vec![atom_term("val")],
                    )],
                )],
            )],
        );

        let result = env.substitute_term(&term);

        let expected = Term::Compound(
            Intern::new("a".to_string()),
            vec![Term::Compound(
                Intern::new("b".to_string()),
                vec![Term::Compound(
                    Intern::new("c".to_string()),
                    vec![Term::Compound(
                        Intern::new("d".to_string()),
                        vec![int_term(42)],
                    )],
                )],
            )],
        );

        assert_eq!(result, expected);
    }
}
