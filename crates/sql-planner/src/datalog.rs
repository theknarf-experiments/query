//! Datalog to LogicalPlan compilation
//!
//! This module compiles Datalog programs into LogicalPlan trees that can be
//! executed by the SQL engine. This allows Datalog queries to benefit from
//! the same optimizations as SQL queries.
//!
//! # Compilation Strategy
//!
//! 1. **Safety Check**: Verify all rules are safe (variables properly bound)
//! 2. **Stratification**: Organize rules into strata for negation handling
//! 3. **Recursion Detection**: Identify recursive predicates
//! 4. **Plan Generation**: Generate LogicalPlan nodes
//!
//! # Example
//!
//! ```ignore
//! // Datalog:
//! // ancestor(X, Y) :- parent(X, Y).
//! // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
//! //
//! // Compiles to:
//! // Recursive {
//! //     name: "ancestor",
//! //     base: Scan { table: "parent" },
//! //     step: Join { ... } -> Projection { ... }
//! // }
//! ```

use crate::LogicalPlan;
use datalog_parser::{Atom, Literal, Program, Rule, Symbol, Term};
use datalog_safety::{check_program_safety, stratify, SafetyError, StratificationError};
use sql_parser::Expr;
use std::collections::{HashMap, HashSet};

/// Errors during Datalog compilation
#[derive(Debug, Clone)]
pub enum DatalogPlanError {
    /// Rule violates safety constraints
    Safety(SafetyError),
    /// Program cannot be stratified (cycle through negation)
    Stratification(StratificationError),
    /// Empty program (no rules or facts)
    EmptyProgram,
}

impl std::fmt::Display for DatalogPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatalogPlanError::Safety(e) => write!(f, "Safety error: {}", e),
            DatalogPlanError::Stratification(e) => write!(f, "Stratification error: {}", e),
            DatalogPlanError::EmptyProgram => write!(f, "Empty Datalog program"),
        }
    }
}

impl std::error::Error for DatalogPlanError {}

impl From<SafetyError> for DatalogPlanError {
    fn from(e: SafetyError) -> Self {
        DatalogPlanError::Safety(e)
    }
}

impl From<StratificationError> for DatalogPlanError {
    fn from(e: StratificationError) -> Self {
        DatalogPlanError::Stratification(e)
    }
}

/// Compile a Datalog program to a LogicalPlan
///
/// The resulting plan uses Stratify at the top level (if negation is present)
/// and Recursive nodes for recursive predicates.
pub fn compile_datalog(program: &Program) -> Result<LogicalPlan, DatalogPlanError> {
    // Extract rules from the program
    let rules: Vec<Rule> = program.rules().cloned().collect();

    if rules.is_empty() && program.facts().next().is_none() {
        return Err(DatalogPlanError::EmptyProgram);
    }

    // Safety check
    check_program_safety(&rules)?;

    // Stratify the program
    let stratification = stratify(&rules)?;

    // If only one stratum and no recursion, generate simple plan
    if stratification.num_strata <= 1 {
        compile_stratum(&rules)
    } else {
        // Multiple strata - wrap in Stratify
        let mut strata_plans = Vec::new();
        for stratum_rules in &stratification.rules_by_stratum {
            if !stratum_rules.is_empty() {
                strata_plans.push(compile_stratum(stratum_rules)?);
            }
        }
        Ok(LogicalPlan::Stratify {
            strata: strata_plans,
        })
    }
}

/// Compile a single stratum (set of rules at the same level)
fn compile_stratum(rules: &[Rule]) -> Result<LogicalPlan, DatalogPlanError> {
    if rules.is_empty() {
        // Empty stratum - return empty scan placeholder
        return Ok(LogicalPlan::Scan {
            table: "__empty__".to_string(),
        });
    }

    // Group rules by head predicate
    let mut rules_by_predicate: HashMap<Symbol, Vec<&Rule>> = HashMap::new();
    for rule in rules {
        rules_by_predicate
            .entry(rule.head.predicate)
            .or_default()
            .push(rule);
    }

    // Detect recursive predicates (predicates that appear in their own rule bodies)
    let recursive_preds = find_recursive_predicates(&rules_by_predicate);

    // Compile each predicate
    let mut predicate_plans = Vec::new();
    for (pred, pred_rules) in &rules_by_predicate {
        let plan = if recursive_preds.contains(pred) {
            compile_recursive_predicate(*pred, pred_rules)?
        } else {
            compile_non_recursive_predicate(*pred, pred_rules)?
        };
        predicate_plans.push(plan);
    }

    // If multiple predicates, combine with SetOperation (UNION)
    if predicate_plans.len() == 1 {
        Ok(predicate_plans.into_iter().next().unwrap())
    } else {
        // For now, just return the first plan
        // In a full implementation, we'd combine them appropriately
        Ok(predicate_plans.into_iter().next().unwrap())
    }
}

/// Find predicates that are recursive (appear in their own rule bodies)
fn find_recursive_predicates(rules_by_predicate: &HashMap<Symbol, Vec<&Rule>>) -> HashSet<Symbol> {
    let mut recursive = HashSet::new();

    for (pred, rules) in rules_by_predicate {
        for rule in rules {
            for literal in &rule.body {
                if let Literal::Positive(atom) | Literal::Negative(atom) = literal {
                    if atom.predicate == *pred {
                        recursive.insert(*pred);
                        break;
                    }
                }
            }
        }
    }

    recursive
}

/// Compile a recursive predicate to a Recursive LogicalPlan node
fn compile_recursive_predicate(
    pred: Symbol,
    rules: &[&Rule],
) -> Result<LogicalPlan, DatalogPlanError> {
    // Separate base case rules (non-recursive) from recursive rules
    let mut base_rules = Vec::new();
    let mut recursive_rules = Vec::new();

    for rule in rules {
        let is_recursive = rule.body.iter().any(|lit| {
            matches!(lit, Literal::Positive(atom) | Literal::Negative(atom) if atom.predicate == pred)
        });

        if is_recursive {
            recursive_rules.push(*rule);
        } else {
            base_rules.push(*rule);
        }
    }

    // Get column names from the first rule's head
    let columns: Vec<String> = rules[0]
        .head
        .terms
        .iter()
        .enumerate()
        .map(|(i, term)| match term {
            Term::Variable(v) => v.to_string(),
            _ => format!("col{}", i),
        })
        .collect();

    // Compile base case
    let base = if base_rules.is_empty() {
        // No base case - start with empty
        LogicalPlan::Scan {
            table: "__empty__".to_string(),
        }
    } else {
        compile_rules_union(&base_rules, pred)?
    };

    // Compile recursive step
    let step = if recursive_rules.is_empty() {
        // No recursive rules - just return base
        return Ok(base);
    } else {
        compile_rules_union(&recursive_rules, pred)?
    };

    Ok(LogicalPlan::Recursive {
        name: pred.to_string(),
        columns,
        base: Box::new(base),
        step: Box::new(step),
    })
}

/// Compile a non-recursive predicate
fn compile_non_recursive_predicate(
    pred: Symbol,
    rules: &[&Rule],
) -> Result<LogicalPlan, DatalogPlanError> {
    compile_rules_union(rules, pred)
}

/// Compile multiple rules for the same predicate as a UNION
fn compile_rules_union(rules: &[&Rule], _pred: Symbol) -> Result<LogicalPlan, DatalogPlanError> {
    if rules.is_empty() {
        return Ok(LogicalPlan::Scan {
            table: "__empty__".to_string(),
        });
    }

    let mut plans: Vec<LogicalPlan> = rules
        .iter()
        .map(|rule| compile_rule(rule))
        .collect::<Result<Vec<_>, _>>()?;

    if plans.len() == 1 {
        return Ok(plans.remove(0));
    }

    // Combine with UNION ALL
    let mut result = plans.remove(0);
    for plan in plans {
        result = LogicalPlan::SetOperation {
            left: Box::new(result),
            right: Box::new(plan),
            op: sql_parser::SetOperator::Union,
            all: true,
        };
    }

    Ok(result)
}

/// Compile a single rule to a LogicalPlan
fn compile_rule(rule: &Rule) -> Result<LogicalPlan, DatalogPlanError> {
    // Extract positive literals (joins) and negative literals (filters)
    let mut positive_atoms = Vec::new();
    let mut negative_atoms = Vec::new();
    let mut comparisons = Vec::new();

    for literal in &rule.body {
        match literal {
            Literal::Positive(atom) => positive_atoms.push(atom),
            Literal::Negative(atom) => negative_atoms.push(atom),
            Literal::Comparison(comp) => comparisons.push(comp),
        }
    }

    // Build the plan from positive atoms (joins)
    let mut plan = if positive_atoms.is_empty() {
        // No positive atoms - this shouldn't happen for safe rules
        LogicalPlan::Scan {
            table: "__empty__".to_string(),
        }
    } else {
        // Start with first atom as a Scan
        let first = positive_atoms.remove(0);
        let mut current = atom_to_scan(first);

        // Join with remaining atoms
        for atom in positive_atoms {
            let right = atom_to_scan(atom);
            // Build join condition from shared variables
            let join_cond = build_join_condition(first, atom);
            current = LogicalPlan::Join {
                left: Box::new(current),
                right: Box::new(right),
                join_type: sql_parser::JoinType::Inner,
                on: join_cond,
            };
        }

        current
    };

    // Add filters for comparisons
    for comp in comparisons {
        let predicate = comparison_to_expr(comp);
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // Add anti-joins for negative literals
    for neg_atom in negative_atoms {
        // Negation is handled as a filter with NOT EXISTS
        // For simplicity, we'll use a placeholder approach
        let neg_scan = atom_to_scan(neg_atom);
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(neg_scan),
            join_type: sql_parser::JoinType::Left,
            on: None, // Would need proper condition
        };
        // Filter where right side is NULL (anti-join semantics)
    }

    // Project to head columns
    let projections = rule
        .head
        .terms
        .iter()
        .map(|term| (term_to_expr(term), None))
        .collect();

    Ok(LogicalPlan::Projection {
        input: Box::new(plan),
        exprs: projections,
    })
}

/// Convert a Datalog atom to a Scan node
fn atom_to_scan(atom: &Atom) -> LogicalPlan {
    // Check if this is a reference to a recursive relation
    // For now, just use Scan with the predicate name as table
    LogicalPlan::Scan {
        table: atom.predicate.to_string(),
    }
}

/// Build a join condition from shared variables between two atoms
fn build_join_condition(_left: &Atom, _right: &Atom) -> Option<Expr> {
    // In a full implementation, we'd find shared variables and create
    // equality conditions. For now, return None (cross join).
    None
}

/// Convert a Datalog comparison to SQL Expr
fn comparison_to_expr(comp: &datalog_parser::ComparisonLiteral) -> Expr {
    let left = term_to_expr(&comp.left);
    let right = term_to_expr(&comp.right);
    let op = match comp.op {
        datalog_parser::ComparisonOp::Equal => sql_parser::BinaryOp::Eq,
        datalog_parser::ComparisonOp::NotEqual => sql_parser::BinaryOp::NotEq,
        datalog_parser::ComparisonOp::LessThan => sql_parser::BinaryOp::Lt,
        datalog_parser::ComparisonOp::LessOrEqual => sql_parser::BinaryOp::LtEq,
        datalog_parser::ComparisonOp::GreaterThan => sql_parser::BinaryOp::Gt,
        datalog_parser::ComparisonOp::GreaterOrEqual => sql_parser::BinaryOp::GtEq,
    };
    Expr::BinaryOp {
        left: Box::new(left),
        op,
        right: Box::new(right),
    }
}

/// Convert a Datalog term to SQL Expr
fn term_to_expr(term: &Term) -> Expr {
    match term {
        Term::Variable(v) => Expr::Column(v.to_string()),
        Term::Constant(val) => match val {
            datalog_parser::Value::Integer(i) => Expr::Integer(*i),
            datalog_parser::Value::Float(f) => Expr::Float(*f),
            datalog_parser::Value::Boolean(b) => Expr::Boolean(*b),
            datalog_parser::Value::String(s) => Expr::String(s.to_string()),
            datalog_parser::Value::Atom(a) => Expr::String(a.to_string()),
        },
        Term::Compound(functor, _args) => {
            // Compound terms don't have a direct SQL equivalent
            // For now, just use the functor name
            Expr::String(functor.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datalog_parser::Statement;
    use internment::Intern;

    fn sym(s: &str) -> Symbol {
        Intern::new(s.to_string())
    }

    fn var(name: &str) -> Term {
        Term::Variable(sym(name))
    }

    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: sym(pred),
            terms,
        }
    }

    #[test]
    fn test_compile_simple_rule() {
        // ancestor(X, Y) :- parent(X, Y).
        let rule = Rule {
            head: make_atom("ancestor", vec![var("X"), var("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        };

        let program = Program {
            statements: vec![Statement::Rule(rule)],
        };

        let plan = compile_datalog(&program).unwrap();

        // Should be a Projection over a Scan
        match plan {
            LogicalPlan::Projection { input, .. } => match *input {
                LogicalPlan::Scan { table } => assert_eq!(table, "parent"),
                other => panic!("Expected Scan, got {:?}", other),
            },
            other => panic!("Expected Projection, got {:?}", other),
        }
    }

    #[test]
    fn test_compile_recursive_rule() {
        // ancestor(X, Y) :- parent(X, Y).
        // ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        let base_rule = Rule {
            head: make_atom("ancestor", vec![var("X"), var("Y")]),
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        };

        let recursive_rule = Rule {
            head: make_atom("ancestor", vec![var("X"), var("Z")]),
            body: vec![
                Literal::Positive(make_atom("ancestor", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        };

        let program = Program {
            statements: vec![Statement::Rule(base_rule), Statement::Rule(recursive_rule)],
        };

        let plan = compile_datalog(&program).unwrap();

        // Should be a Recursive node
        match plan {
            LogicalPlan::Recursive { name, .. } => {
                assert_eq!(name, "ancestor");
            }
            other => panic!("Expected Recursive, got {:?}", other),
        }
    }

    #[test]
    fn test_detect_recursive_predicates() {
        let rule = Rule {
            head: make_atom("ancestor", vec![var("X"), var("Z")]),
            body: vec![
                Literal::Positive(make_atom("ancestor", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        };

        let mut rules_by_pred = HashMap::new();
        rules_by_pred.insert(sym("ancestor"), vec![&rule]);

        let recursive = find_recursive_predicates(&rules_by_pred);
        assert!(recursive.contains(&sym("ancestor")));
    }
}
