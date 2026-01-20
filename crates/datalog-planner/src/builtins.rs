//! Built-in predicates and arithmetic evaluation
//!
//! This module handles built-in predicates like arithmetic operations and comparisons.
//! Built-ins are special predicates evaluated during grounding rather than via rules.
//!
//! # Supported Built-ins
//!
//! - **Arithmetic**: `+`, `-`, `*`, `/`, `mod`
//! - **Comparisons**: `=`, `!=`, `\=`, `<`, `>`, `<=`, `>=`
//!
//! # Usage
//!
//! Built-ins can appear in rule bodies:
//! ```ignore
//! // age(X, A), A > 18, Result = A + 1
//! ```
//!
//! During grounding, built-ins are evaluated to filter or compute values.

use datalog_parser::{Atom, Term, Value};
use sql_storage::Substitution;

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompOp {
    Eq,  // =
    Neq, // !=
    Lt,  // <
    Gt,  // >
    Lte, // <=
    Gte, // >=
}

/// Built-in predicates that can be evaluated directly
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    /// Arithmetic comparison: X = Y + Z, X < Y, etc.
    Comparison(CompOp, Term, Term),
    /// True (always succeeds)
    True,
    /// Fail (always fails)
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Numeric {
    Int(i64),
    Float(f64),
}

impl Numeric {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Integer(i) => Some(Numeric::Int(*i)),
            Value::Float(f) => Some(Numeric::Float(*f)),
            _ => None,
        }
    }

    fn to_value(self) -> Value {
        match self {
            Numeric::Int(i) => Value::Integer(i),
            Numeric::Float(f) => Value::Float(f),
        }
    }

    fn promote(lhs: Numeric, rhs: Numeric) -> (Numeric, Numeric) {
        match (lhs, rhs) {
            (Numeric::Int(l), Numeric::Int(r)) => (Numeric::Int(l), Numeric::Int(r)),
            (Numeric::Int(l), Numeric::Float(r)) => (Numeric::Float(l as f64), Numeric::Float(r)),
            (Numeric::Float(l), Numeric::Int(r)) => (Numeric::Float(l), Numeric::Float(r as f64)),
            (Numeric::Float(l), Numeric::Float(r)) => (Numeric::Float(l), Numeric::Float(r)),
        }
    }

    fn add(self, other: Numeric) -> Numeric {
        match (self, other) {
            (Numeric::Int(l), Numeric::Int(r)) => Numeric::Int(l + r),
            (l, r) => Numeric::Float(l.to_f64() + r.to_f64()),
        }
    }

    fn sub(self, other: Numeric) -> Numeric {
        match (self, other) {
            (Numeric::Int(l), Numeric::Int(r)) => Numeric::Int(l - r),
            (l, r) => Numeric::Float(l.to_f64() - r.to_f64()),
        }
    }

    fn mul(self, other: Numeric) -> Numeric {
        match (self, other) {
            (Numeric::Int(l), Numeric::Int(r)) => Numeric::Int(l * r),
            (l, r) => Numeric::Float(l.to_f64() * r.to_f64()),
        }
    }

    fn div(self, other: Numeric) -> Option<Numeric> {
        match (self, other) {
            (Numeric::Int(_), Numeric::Int(0)) => None,
            (Numeric::Int(l), Numeric::Int(r)) => Some(Numeric::Int(l / r)),
            (l, r) => {
                let divisor = r.to_f64();
                if divisor == 0.0 {
                    None
                } else {
                    Some(Numeric::Float(l.to_f64() / divisor))
                }
            }
        }
    }

    fn modulo(self, other: Numeric) -> Option<Numeric> {
        match (self, other) {
            (Numeric::Int(_), Numeric::Int(0)) => None,
            (Numeric::Int(l), Numeric::Int(r)) => Some(Numeric::Int(l % r)),
            (l, r) => {
                let divisor = r.to_f64();
                if divisor == 0.0 {
                    None
                } else {
                    Some(Numeric::Float(l.to_f64() % divisor))
                }
            }
        }
    }

    fn to_f64(self) -> f64 {
        match self {
            Numeric::Int(i) => i as f64,
            Numeric::Float(f) => f,
        }
    }
}

/// Evaluate an arithmetic expression to a numeric value
/// Returns None if the expression cannot be evaluated (contains unbound variables)
pub fn eval_arith(term: &Term, subst: &Substitution) -> Option<Value> {
    match term {
        Term::Constant(value) => Numeric::from_value(value).map(Numeric::to_value),

        Term::Variable(var) => {
            // Look up variable in substitution
            if let Some(bound_term) = subst.get(var) {
                eval_arith(bound_term, subst)
            } else {
                None // Unbound variable
            }
        }

        Term::Compound(functor, args) => {
            // Check if this is an arithmetic operator
            match functor.as_ref().as_str() {
                "+" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    Some(left.add(right).to_value())
                }
                "-" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    Some(left.sub(right).to_value())
                }
                "*" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    Some(left.mul(right).to_value())
                }
                "/" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    left.div(right).map(Numeric::to_value)
                }
                "mod" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    left.modulo(right).map(Numeric::to_value)
                }
                _ => None, // Not an arithmetic operator
            }
        }
    }
}

/// Evaluate a comparison built-in predicate
/// Returns true if the comparison holds, false otherwise
/// Returns None if terms cannot be evaluated
pub fn eval_comparison(
    op: &CompOp,
    left: &Term,
    right: &Term,
    subst: &Substitution,
) -> Option<bool> {
    // Try to evaluate both sides as arithmetic expressions
    let left_val = eval_arith(left, subst)?;
    let right_val = eval_arith(right, subst)?;
    let left_num = Numeric::from_value(&left_val)?;
    let right_num = Numeric::from_value(&right_val)?;
    let (left_num, right_num) = Numeric::promote(left_num, right_num);

    let result = match op {
        CompOp::Eq => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l == r,
            (l, r) => l.to_f64() == r.to_f64(),
        },
        CompOp::Neq => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l != r,
            (l, r) => l.to_f64() != r.to_f64(),
        },
        CompOp::Lt => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l < r,
            (l, r) => l.to_f64() < r.to_f64(),
        },
        CompOp::Gt => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l > r,
            (l, r) => l.to_f64() > r.to_f64(),
        },
        CompOp::Lte => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l <= r,
            (l, r) => l.to_f64() <= r.to_f64(),
        },
        CompOp::Gte => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l >= r,
            (l, r) => l.to_f64() >= r.to_f64(),
        },
    };

    Some(result)
}

/// Evaluate a built-in predicate
/// Returns true if it succeeds, false if it fails, None if it cannot be evaluated
pub fn eval_builtin(builtin: &BuiltIn, subst: &Substitution) -> Option<bool> {
    match builtin {
        BuiltIn::Comparison(op, left, right) => eval_comparison(op, left, right, subst),
        BuiltIn::True => Some(true),
        BuiltIn::Fail => Some(false),
    }
}

/// Parse an atom as a potential built-in predicate
/// Returns Some(BuiltIn) if this is a built-in, None otherwise
pub fn parse_builtin(atom: &Atom) -> Option<BuiltIn> {
    let pred = atom.predicate.as_ref().as_str();

    match pred {
        "=" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Eq,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "!=" | "\\=" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Neq,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "<" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Lt,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        ">" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Gt,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "<=" | "=<" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Lte,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        ">=" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Gte,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "true" if atom.terms.is_empty() => Some(BuiltIn::True),
        "fail" | "false" if atom.terms.is_empty() => Some(BuiltIn::Fail),
        _ => None, // Not a built-in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::Symbol;

    fn sym(s: &str) -> Symbol {
        Symbol::new(s.to_string())
    }

    fn int_term(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn var_term(name: &str) -> Term {
        Term::Variable(sym(name))
    }

    #[test]
    fn test_eval_arith_constants() {
        let subst = Substitution::new();

        // Simple integer
        assert_eq!(eval_arith(&int_term(42), &subst), Some(Value::Integer(42)));

        // Float
        assert_eq!(
            eval_arith(&Term::Constant(Value::Float(3.14)), &subst),
            Some(Value::Float(3.14))
        );
    }

    #[test]
    fn test_eval_arith_operations() {
        let subst = Substitution::new();

        // Addition
        let add = Term::Compound(sym("+"), vec![int_term(2), int_term(3)]);
        assert_eq!(eval_arith(&add, &subst), Some(Value::Integer(5)));

        // Subtraction
        let sub = Term::Compound(sym("-"), vec![int_term(10), int_term(4)]);
        assert_eq!(eval_arith(&sub, &subst), Some(Value::Integer(6)));

        // Multiplication
        let mul = Term::Compound(sym("*"), vec![int_term(3), int_term(4)]);
        assert_eq!(eval_arith(&mul, &subst), Some(Value::Integer(12)));

        // Division
        let div = Term::Compound(sym("/"), vec![int_term(15), int_term(3)]);
        assert_eq!(eval_arith(&div, &subst), Some(Value::Integer(5)));

        // Modulo
        let modulo = Term::Compound(sym("mod"), vec![int_term(17), int_term(5)]);
        assert_eq!(eval_arith(&modulo, &subst), Some(Value::Integer(2)));
    }

    #[test]
    fn test_eval_arith_with_variables() {
        let mut subst = Substitution::new();
        subst.bind(sym("X"), int_term(10));
        subst.bind(sym("Y"), int_term(3));

        let add = Term::Compound(sym("+"), vec![var_term("X"), var_term("Y")]);
        assert_eq!(eval_arith(&add, &subst), Some(Value::Integer(13)));
    }

    #[test]
    fn test_eval_comparison() {
        let subst = Substitution::new();

        assert_eq!(
            eval_comparison(&CompOp::Lt, &int_term(3), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Gt, &int_term(3), &int_term(5), &subst),
            Some(false)
        );
        assert_eq!(
            eval_comparison(&CompOp::Eq, &int_term(5), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Neq, &int_term(3), &int_term(5), &subst),
            Some(true)
        );
    }

    #[test]
    fn test_parse_builtin() {
        let lt_atom = Atom {
            predicate: sym("<"),
            terms: vec![int_term(3), int_term(5)],
        };

        let builtin = parse_builtin(&lt_atom);
        assert!(builtin.is_some());

        let non_builtin_atom = Atom {
            predicate: sym("parent"),
            terms: vec![var_term("X"), var_term("Y")],
        };

        assert!(parse_builtin(&non_builtin_atom).is_none());
    }

    // ===== Additional Arithmetic Tests =====

    fn float_term(f: f64) -> Term {
        Term::Constant(Value::Float(f))
    }

    #[test]
    fn test_eval_arith_negative_integers() {
        let subst = Substitution::new();
        assert_eq!(
            eval_arith(&int_term(-10), &subst),
            Some(Value::Integer(-10))
        );
        assert_eq!(eval_arith(&int_term(0), &subst), Some(Value::Integer(0)));
    }

    #[test]
    fn test_eval_arith_division_by_zero() {
        let subst = Substitution::new();
        let div = Term::Compound(sym("/"), vec![int_term(10), int_term(0)]);
        assert_eq!(eval_arith(&div, &subst), None);
    }

    #[test]
    fn test_eval_arith_modulo_by_zero() {
        let subst = Substitution::new();
        let modulo = Term::Compound(sym("mod"), vec![int_term(10), int_term(0)]);
        assert_eq!(eval_arith(&modulo, &subst), None);
    }

    #[test]
    fn test_eval_arith_nested_expression() {
        let subst = Substitution::new();
        // (3 + 4) * 2 = 14
        let inner = Term::Compound(sym("+"), vec![int_term(3), int_term(4)]);
        let expr = Term::Compound(sym("*"), vec![inner, int_term(2)]);
        assert_eq!(eval_arith(&expr, &subst), Some(Value::Integer(14)));
    }

    #[test]
    fn test_eval_arith_unbound_variable() {
        let subst = Substitution::new();
        let expr = Term::Compound(sym("+"), vec![var_term("X"), int_term(5)]);
        assert_eq!(eval_arith(&expr, &subst), None);
    }

    #[test]
    fn test_eval_arith_float_constants() {
        let subst = Substitution::new();
        assert_eq!(
            eval_arith(&float_term(3.5), &subst),
            Some(Value::Float(3.5))
        );
    }

    #[test]
    fn test_eval_arith_mixed_addition() {
        let subst = Substitution::new();
        let expr = Term::Compound(sym("+"), vec![int_term(2), float_term(0.5)]);
        assert_eq!(eval_arith(&expr, &subst), Some(Value::Float(2.5)));
    }

    #[test]
    fn test_eval_arith_float_division() {
        let subst = Substitution::new();
        let expr = Term::Compound(sym("/"), vec![float_term(7.5), int_term(2)]);
        assert_eq!(eval_arith(&expr, &subst), Some(Value::Float(3.75)));
    }

    #[test]
    fn test_eval_arith_float_division_by_zero() {
        let subst = Substitution::new();
        let expr = Term::Compound(sym("/"), vec![float_term(3.0), float_term(0.0)]);
        assert_eq!(eval_arith(&expr, &subst), None);
    }

    #[test]
    fn test_eval_arith_float_modulo_by_zero() {
        let subst = Substitution::new();
        let expr = Term::Compound(sym("mod"), vec![float_term(3.0), float_term(0.0)]);
        assert_eq!(eval_arith(&expr, &subst), None);
    }

    #[test]
    fn test_eval_arith_float_modulo() {
        let subst = Substitution::new();
        let expr = Term::Compound(sym("mod"), vec![float_term(5.5), int_term(2)]);
        assert_eq!(eval_arith(&expr, &subst), Some(Value::Float(1.5)));
    }

    // ===== Additional Comparison Tests =====

    #[test]
    fn test_eval_comparison_equal_integers() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Eq, &int_term(5), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Eq, &int_term(5), &int_term(3), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_less_than() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Lt, &int_term(3), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lt, &int_term(5), &int_term(3), &subst),
            Some(false)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lt, &int_term(5), &int_term(5), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_greater_than() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Gt, &int_term(5), &int_term(3), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Gt, &int_term(3), &int_term(5), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_less_or_equal() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Lte, &int_term(3), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lte, &int_term(5), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lte, &int_term(6), &int_term(5), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_greater_or_equal() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Gte, &int_term(5), &int_term(3), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Gte, &int_term(5), &int_term(5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Gte, &int_term(4), &int_term(5), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_with_arithmetic() {
        let subst = Substitution::new();
        // 3 + 4 = 7
        let left = Term::Compound(sym("+"), vec![int_term(3), int_term(4)]);
        let right = int_term(7);
        assert_eq!(
            eval_comparison(&CompOp::Eq, &left, &right, &subst),
            Some(true)
        );
    }

    #[test]
    fn test_eval_comparison_float_semantics() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Eq, &float_term(2.5), &float_term(2.5), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Gt, &float_term(3.1), &float_term(3.0), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lt, &float_term(3.1), &float_term(3.0), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_mixed_numeric_types() {
        let subst = Substitution::new();
        assert_eq!(
            eval_comparison(&CompOp::Eq, &int_term(5), &float_term(5.0), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lt, &int_term(4), &float_term(4.1), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Gt, &float_term(6.2), &int_term(6), &subst),
            Some(true)
        );
    }

    // ===== Parse Builtin Tests =====

    #[test]
    fn test_parse_builtin_comparison_operators() {
        // Equal
        let eq_atom = Atom {
            predicate: sym("="),
            terms: vec![int_term(5), int_term(5)],
        };
        assert!(parse_builtin(&eq_atom).is_some());

        // Not equal (!=)
        let neq_atom = Atom {
            predicate: sym("!="),
            terms: vec![int_term(3), int_term(5)],
        };
        assert!(parse_builtin(&neq_atom).is_some());

        // Not equal (\=)
        let neq_atom2 = Atom {
            predicate: sym("\\="),
            terms: vec![int_term(3), int_term(5)],
        };
        assert!(parse_builtin(&neq_atom2).is_some());

        // Less than or equal (<=)
        let lte_atom = Atom {
            predicate: sym("<="),
            terms: vec![int_term(3), int_term(5)],
        };
        assert!(parse_builtin(&lte_atom).is_some());

        // Less than or equal (=<)
        let lte_atom2 = Atom {
            predicate: sym("=<"),
            terms: vec![int_term(3), int_term(5)],
        };
        assert!(parse_builtin(&lte_atom2).is_some());

        // Greater than or equal
        let gte_atom = Atom {
            predicate: sym(">="),
            terms: vec![int_term(5), int_term(3)],
        };
        assert!(parse_builtin(&gte_atom).is_some());
    }

    #[test]
    fn test_parse_builtin_not_builtin() {
        let atom = Atom {
            predicate: sym("parent"),
            terms: vec![
                Term::Constant(Value::Atom(sym("john"))),
                Term::Constant(Value::Atom(sym("mary"))),
            ],
        };
        assert!(parse_builtin(&atom).is_none());
    }

    #[test]
    fn test_parse_and_eval_ground_comparison() {
        // Test that we can parse and evaluate a ground comparison like "5 > 3"
        let atom = Atom {
            predicate: sym(">"),
            terms: vec![int_term(5), int_term(3)],
        };

        let builtin = parse_builtin(&atom).expect("Should parse as built-in");
        let subst = Substitution::new();
        let result = eval_builtin(&builtin, &subst);

        assert_eq!(result, Some(true));
    }

    #[test]
    fn test_parse_and_eval_ground_comparison_false() {
        let atom = Atom {
            predicate: sym(">"),
            terms: vec![int_term(3), int_term(5)],
        };

        let builtin = parse_builtin(&atom).expect("Should parse as built-in");
        let subst = Substitution::new();
        let result = eval_builtin(&builtin, &subst);

        assert_eq!(result, Some(false));
    }

    // ===== Built-in Special Values Tests =====

    #[test]
    fn test_parse_builtin_true() {
        let atom = Atom {
            predicate: sym("true"),
            terms: vec![],
        };
        let builtin = parse_builtin(&atom);
        assert!(matches!(builtin, Some(BuiltIn::True)));

        let subst = Substitution::new();
        assert_eq!(eval_builtin(&BuiltIn::True, &subst), Some(true));
    }

    #[test]
    fn test_parse_builtin_fail() {
        let atom = Atom {
            predicate: sym("fail"),
            terms: vec![],
        };
        let builtin = parse_builtin(&atom);
        assert!(matches!(builtin, Some(BuiltIn::Fail)));

        let subst = Substitution::new();
        assert_eq!(eval_builtin(&BuiltIn::Fail, &subst), Some(false));
    }

    #[test]
    fn test_parse_builtin_false() {
        let atom = Atom {
            predicate: sym("false"),
            terms: vec![],
        };
        let builtin = parse_builtin(&atom);
        assert!(matches!(builtin, Some(BuiltIn::Fail)));
    }

    // ===== Comparison with Variables Tests =====

    #[test]
    fn test_eval_comparison_with_variables() {
        let mut subst = Substitution::new();
        subst.bind(sym("X"), int_term(10));
        subst.bind(sym("Y"), int_term(5));

        assert_eq!(
            eval_comparison(&CompOp::Gt, &var_term("X"), &var_term("Y"), &subst),
            Some(true)
        );
        assert_eq!(
            eval_comparison(&CompOp::Lt, &var_term("X"), &var_term("Y"), &subst),
            Some(false)
        );
    }

    #[test]
    fn test_eval_comparison_unbound_variable() {
        let subst = Substitution::new();
        // X is unbound
        assert_eq!(
            eval_comparison(&CompOp::Lt, &var_term("X"), &int_term(5), &subst),
            None
        );
    }
}
