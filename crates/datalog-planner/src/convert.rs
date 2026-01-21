//! Conversion functions from parser AST to planner IR
//!
//! This module converts the parser's AST types to the planner's IR types.
//! During conversion, we also perform planning tasks like stratification.

use crate::ir::{
    Atom, Comparison, ComparisonOp, Literal, PlannedConstraint, PlannedProgram, PlannedQuery,
    PlannedRule, PlannedStratum, Symbol, Term, Value,
};
use crate::safety::{check_program_safety, SafetyError};
use crate::stratification::{stratify, StratificationError};
use datalog_parser as ast;
use std::collections::HashSet;

/// Error during planning/conversion
#[derive(Debug, Clone)]
pub enum PlanError {
    /// Rule violates safety constraints
    Safety(SafetyError),
    /// Program cannot be stratified (cycle through negation)
    Stratification(StratificationError),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Safety(e) => write!(f, "Safety error: {}", e),
            PlanError::Stratification(e) => write!(f, "Stratification error: {}", e),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<SafetyError> for PlanError {
    fn from(e: SafetyError) -> Self {
        PlanError::Safety(e)
    }
}

impl From<StratificationError> for PlanError {
    fn from(e: StratificationError) -> Self {
        PlanError::Stratification(e)
    }
}

/// Plan a Datalog program: convert AST to IR with stratification
pub fn plan_program(program: &ast::Program) -> Result<PlannedProgram, PlanError> {
    // Extract and convert rules from the program
    let ir_rules: Vec<PlannedRule> = program.rules().map(convert_rule).collect();

    // Safety check on IR rules
    check_program_safety(&ir_rules)?;

    // Stratify the program
    let stratification = stratify(&ir_rules)?;

    // Convert each stratum
    let mut strata = Vec::new();
    for stratum_rules in &stratification.rules_by_stratum {
        if stratum_rules.is_empty() {
            continue;
        }

        // Find predicates defined in this stratum
        let predicates: Vec<Symbol> = stratum_rules
            .iter()
            .map(|r| r.head.predicate)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Check if this stratum is recursive
        let is_recursive = is_stratum_recursive_ir(stratum_rules);

        strata.push(PlannedStratum {
            rules: stratum_rules.clone(),
            is_recursive,
            predicates,
        });
    }

    // Convert constraints
    let constraints: Vec<PlannedConstraint> =
        program.constraints().map(convert_constraint).collect();

    // Convert queries
    let queries: Vec<PlannedQuery> = program.queries().map(convert_query).collect();

    Ok(PlannedProgram {
        strata,
        constraints,
        queries,
    })
}

/// Check if a stratum contains recursive rules (IR version)
fn is_stratum_recursive_ir(rules: &[PlannedRule]) -> bool {
    let head_predicates: HashSet<_> = rules.iter().map(|r| r.head.predicate).collect();

    for rule in rules {
        for lit in &rule.body {
            if let Literal::Positive(atom) | Literal::Negative(atom) = lit {
                if head_predicates.contains(&atom.predicate) {
                    return true;
                }
            }
        }
    }

    false
}

/// Convert a parser Rule to a PlannedRule
pub fn convert_rule(rule: &ast::Rule) -> PlannedRule {
    PlannedRule {
        head: convert_atom(&rule.head),
        body: rule.body.iter().map(convert_literal).collect(),
    }
}

/// Convert a parser Constraint to a PlannedConstraint
pub fn convert_constraint(constraint: &ast::Constraint) -> PlannedConstraint {
    PlannedConstraint {
        body: constraint.body.iter().map(convert_literal).collect(),
    }
}

/// Convert a parser Query to a PlannedQuery
pub fn convert_query(query: &ast::Query) -> PlannedQuery {
    PlannedQuery {
        body: query.body.iter().map(convert_literal).collect(),
    }
}

/// Convert a parser Literal to an IR Literal
pub fn convert_literal(lit: &ast::Literal) -> Literal {
    match lit {
        ast::Literal::Positive(atom) => Literal::Positive(convert_atom(atom)),
        ast::Literal::Negative(atom) => Literal::Negative(convert_atom(atom)),
        ast::Literal::Comparison(comp) => Literal::Comparison(convert_comparison(comp)),
    }
}

/// Convert a parser Atom to an IR Atom
pub fn convert_atom(atom: &ast::Atom) -> Atom {
    Atom {
        predicate: atom.predicate,
        terms: atom.terms.iter().map(convert_term).collect(),
    }
}

/// Convert a parser Term to an IR Term
pub fn convert_term(term: &ast::Term) -> Term {
    match term {
        ast::Term::Variable(v) => Term::Variable(*v),
        ast::Term::Constant(c) => Term::Constant(convert_value(c)),
        ast::Term::Compound(f, args) => Term::Compound(*f, args.iter().map(convert_term).collect()),
    }
}

/// Convert a parser Value to an IR Value
pub fn convert_value(val: &ast::Value) -> Value {
    match val {
        ast::Value::Integer(i) => Value::Integer(*i),
        ast::Value::Float(f) => Value::Float(*f),
        ast::Value::Boolean(b) => Value::Boolean(*b),
        ast::Value::String(s) => Value::String(*s),
        ast::Value::Atom(a) => Value::Atom(*a),
    }
}

/// Convert a parser ComparisonLiteral to an IR Comparison
pub fn convert_comparison(comp: &ast::ComparisonLiteral) -> Comparison {
    Comparison {
        left: convert_term(&comp.left),
        op: convert_comparison_op(comp.op),
        right: convert_term(&comp.right),
    }
}

/// Convert a parser ComparisonOp to an IR ComparisonOp
pub fn convert_comparison_op(op: ast::ComparisonOp) -> ComparisonOp {
    match op {
        ast::ComparisonOp::Equal => ComparisonOp::Equal,
        ast::ComparisonOp::NotEqual => ComparisonOp::NotEqual,
        ast::ComparisonOp::LessThan => ComparisonOp::LessThan,
        ast::ComparisonOp::LessOrEqual => ComparisonOp::LessOrEqual,
        ast::ComparisonOp::GreaterThan => ComparisonOp::GreaterThan,
        ast::ComparisonOp::GreaterOrEqual => ComparisonOp::GreaterOrEqual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use internment::Intern;

    fn sym(s: &str) -> Symbol {
        Intern::new(s.to_string())
    }

    fn var(name: &str) -> ast::Term {
        ast::Term::Variable(sym(name))
    }

    fn make_ast_atom(pred: &str, terms: Vec<ast::Term>) -> ast::Atom {
        ast::Atom {
            predicate: sym(pred),
            terms,
        }
    }

    #[test]
    fn test_convert_simple_rule() {
        let ast_rule = ast::Rule {
            head: make_ast_atom("ancestor", vec![var("X"), var("Y")]),
            body: vec![ast::Literal::Positive(make_ast_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        };

        let ir_rule = convert_rule(&ast_rule);

        assert_eq!(ir_rule.head.predicate, sym("ancestor"));
        assert_eq!(ir_rule.head.terms.len(), 2);
        assert_eq!(ir_rule.body.len(), 1);
        assert!(ir_rule.body[0].is_positive());
    }

    #[test]
    fn test_plan_simple_program() {
        let program = ast::Program {
            statements: vec![ast::Statement::Rule(ast::Rule {
                head: make_ast_atom("ancestor", vec![var("X"), var("Y")]),
                body: vec![ast::Literal::Positive(make_ast_atom(
                    "parent",
                    vec![var("X"), var("Y")],
                ))],
            })],
        };

        let planned = plan_program(&program).unwrap();

        assert_eq!(planned.strata.len(), 1);
        assert_eq!(planned.strata[0].rules.len(), 1);
        assert!(!planned.strata[0].is_recursive);
    }

    #[test]
    fn test_plan_recursive_program() {
        let program = ast::Program {
            statements: vec![
                // Base case: ancestor(X, Y) :- parent(X, Y).
                ast::Statement::Rule(ast::Rule {
                    head: make_ast_atom("ancestor", vec![var("X"), var("Y")]),
                    body: vec![ast::Literal::Positive(make_ast_atom(
                        "parent",
                        vec![var("X"), var("Y")],
                    ))],
                }),
                // Recursive case: ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
                ast::Statement::Rule(ast::Rule {
                    head: make_ast_atom("ancestor", vec![var("X"), var("Z")]),
                    body: vec![
                        ast::Literal::Positive(make_ast_atom("ancestor", vec![var("X"), var("Y")])),
                        ast::Literal::Positive(make_ast_atom("parent", vec![var("Y"), var("Z")])),
                    ],
                }),
            ],
        };

        let planned = plan_program(&program).unwrap();

        assert_eq!(planned.strata.len(), 1);
        assert_eq!(planned.strata[0].rules.len(), 2);
        assert!(planned.strata[0].is_recursive);
    }
}
