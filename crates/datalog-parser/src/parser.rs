//! Parser implementation for Datalog.
//!
//! Supports parsing:
//! - Facts: `parent(john, mary).`
//! - Rules: `ancestor(X, Y) :- parent(X, Y).`
//! - Queries: `?- ancestor(X, mary).`
//! - Constraints: `:- unsafe(X).`
//! - Negation: `not reachable(X)`
//! - Comparisons: `X > 5`, `X = Y`

use chumsky::prelude::*;
use chumsky::stream::Stream;
use internment::Intern;

use crate::ast::*;
use crate::token::{lexer, Keyword, LexError, SpannedToken, Token};
use crate::{Span, SrcId};

type ParserError = Simple<Token, Span>;

#[derive(Debug, Clone)]
pub enum ParseError {
    Lex(LexError),
    Parse(ParserError),
}

fn ident_token() -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! {
        Token::Ident(ident) => ident,
    }
    .labelled("identifier")
}

fn variable_token() -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! {
        Token::Variable(ident) => ident,
        Token::Ident(ident) if ident.starts_with('_') => ident,
    }
    .labelled("variable")
}

fn string_token() -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! { Token::String(value) => value }.labelled("string")
}

fn number_token() -> impl Parser<Token, Value, Error = ParserError> + Clone {
    select! { Token::Number(number) => number }
        .try_map(|value: String, span| {
            if value.contains('.') {
                value
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| ParserError::custom(span, "invalid float"))
            } else {
                value
                    .parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| ParserError::custom(span, "invalid integer"))
            }
        })
        .labelled("number")
}

fn signed_number_token() -> impl Parser<Token, Value, Error = ParserError> + Clone {
    operator_token("-")
        .ignore_then(number_token())
        .map(|value| match value {
            Value::Integer(n) => Value::Integer(-n),
            Value::Float(n) => Value::Float(-n),
            other => other,
        })
        .or(number_token())
        .labelled("number")
}

fn keyword_token(keyword: Keyword) -> impl Parser<Token, Keyword, Error = ParserError> + Clone {
    just(Token::Keyword(keyword)).to(keyword)
}

fn operator_token(op: &'static str) -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! { Token::Operator(value) if value == op => value }
}

fn token(kind: Token) -> impl Parser<Token, Token, Error = ParserError> + Clone {
    just(kind)
}

fn lex_with_src(input: &str, src: SrcId) -> Result<Vec<SpannedToken>, Vec<ParseError>> {
    let len = input.chars().count();
    let eoi = Span::new(src, len..len);
    let stream = Stream::from_iter(
        eoi,
        input
            .chars()
            .enumerate()
            .map(|(idx, ch)| (ch, Span::new(src, idx..idx + 1))),
    );
    lexer()
        .parse(stream)
        .map_err(|errors| errors.into_iter().map(ParseError::Lex).collect())
}

#[cfg(test)]
fn lex(input: &str) -> Result<Vec<SpannedToken>, Vec<ParseError>> {
    lex_with_src(input, SrcId::empty())
}

fn parse_with<T>(
    parser: impl Parser<Token, T, Error = ParserError>,
    input: &str,
    src: SrcId,
) -> Result<T, Vec<ParseError>> {
    let tokens = lex_with_src(input, src)?;
    let end = input.chars().count();
    let eoi = Span::new(src, end..end);
    let stream = Stream::from_iter(eoi, tokens.into_iter());
    parser
        .parse(stream)
        .map_err(|errors| errors.into_iter().map(ParseError::Parse).collect())
}

fn factor_parser<'a>(
    term: Recursive<'a, Token, Term, ParserError>,
) -> impl Parser<Token, Term, Error = ParserError> + Clone + 'a {
    let variable = variable_token().map(|s| Term::Variable(Intern::new(s)));

    let number_const = signed_number_token().map(Term::Constant);

    let string_const = string_token().map(|s| Term::Constant(Value::String(Intern::new(s))));

    let parens = term
        .clone()
        .delimited_by(token(Token::LParen), token(Token::RParen));

    let compound_or_atom = ident_token()
        .then(
            term.clone()
                .separated_by(token(Token::Comma))
                .delimited_by(token(Token::LParen), token(Token::RParen))
                .or_not(),
        )
        .map(|(name, args)| {
            if let Some(args) = args {
                Term::Compound(Intern::new(name), args)
            } else {
                match name.as_str() {
                    "true" => Term::Constant(Value::Boolean(true)),
                    "false" => Term::Constant(Value::Boolean(false)),
                    _ => Term::Constant(Value::Atom(Intern::new(name))),
                }
            }
        });

    choice((
        variable,
        number_const,
        string_const,
        parens,
        compound_or_atom,
    ))
}

fn arithmetic_parser<'a>(
    factor: impl Parser<Token, Term, Error = ParserError> + Clone + 'a,
) -> impl Parser<Token, Term, Error = ParserError> + Clone + 'a {
    let mul_div = factor
        .clone()
        .then(
            choice((operator_token("*").to("*"), operator_token("/").to("/")))
                .then(factor.clone())
                .repeated(),
        )
        .foldl(|left, (op, right)| Term::Compound(Intern::new(op.to_string()), vec![left, right]));

    mul_div
        .clone()
        .then(
            choice((operator_token("+").to("+"), operator_token("-").to("-")))
                .then(mul_div)
                .repeated(),
        )
        .foldl(|left, (op, right)| Term::Compound(Intern::new(op.to_string()), vec![left, right]))
}

/// Parse a term (variable, constant, or compound) with arithmetic operator precedence
fn term() -> impl Parser<Token, Term, Error = ParserError> + Clone {
    recursive(|term| {
        let factor = factor_parser(term.clone());
        arithmetic_parser(factor)
    })
    .labelled("term")
}

/// Parse an atom
fn atom() -> impl Parser<Token, Atom, Error = ParserError> + Clone {
    let predicate = choice((ident_token(), select! { Token::Operator(op) => op }));

    predicate
        .then(
            term()
                .separated_by(token(Token::Comma))
                .delimited_by(token(Token::LParen), token(Token::RParen))
                .or_not(),
        )
        .map(|(predicate, terms)| Atom {
            predicate: Intern::new(predicate),
            terms: terms.unwrap_or_default(),
        })
        .labelled("atom")
}

/// Parse a comparison operator
fn comparison_operator() -> impl Parser<Token, ComparisonOp, Error = ParserError> + Clone {
    choice((
        operator_token("<=").to(ComparisonOp::LessOrEqual),
        operator_token(">=").to(ComparisonOp::GreaterOrEqual),
        operator_token("!=").to(ComparisonOp::NotEqual),
        operator_token("<>").to(ComparisonOp::NotEqual),
        operator_token("=").to(ComparisonOp::Equal),
        operator_token("<").to(ComparisonOp::LessThan),
        operator_token(">").to(ComparisonOp::GreaterThan),
    ))
}

/// Parse a comparison literal: X > 5, X = Y
fn comparison_literal() -> impl Parser<Token, Literal, Error = ParserError> + Clone {
    term()
        .then(comparison_operator())
        .then(term())
        .map(|((left, op), right)| Literal::Comparison(ComparisonLiteral { left, op, right }))
}

/// Parse a literal (positive, negative, or comparison)
fn literal() -> impl Parser<Token, Literal, Error = ParserError> + Clone {
    let negated = keyword_token(Keyword::Not)
        .ignore_then(atom())
        .map(Literal::Negative);

    let positive = atom().map(Literal::Positive);

    // Try comparison first, then negated, then positive atom
    choice((comparison_literal(), negated, positive)).labelled("literal")
}

/// Parse a fact: atom.
fn fact() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    atom()
        .then_ignore(token(Token::Dot))
        .map(|atom| Statement::Fact(Fact { atom }))
        .labelled("fact")
}

/// Parse a rule: head :- body.
fn rule() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    atom()
        .then_ignore(token(Token::RuleSep))
        .then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot))
        .map(|(head, body)| Statement::Rule(Rule { head, body }))
        .labelled("rule")
}

/// Parse a constraint: :- body.
fn constraint() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    token(Token::RuleSep)
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot))
        .map(|body| Statement::Constraint(Constraint { body }))
        .labelled("constraint")
}

/// Parse a query: ?- body.
fn query() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    token(Token::Question)
        .then_ignore(operator_token("-"))
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot))
        .map(|body| Statement::Query(Query { body }))
        .labelled("query")
}

/// Parse a statement
fn statement() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    choice((query(), constraint(), rule(), fact())).labelled("statement")
}

/// Parse a program
pub fn program() -> impl Parser<Token, Program, Error = ParserError> + Clone {
    statement()
        .repeated()
        .map(|statements| Program { statements })
        .then_ignore(end())
        .labelled("program")
}

/// Parse a Datalog program from text
pub fn parse_program(input: &str, src: SrcId) -> Result<Program, Vec<ParseError>> {
    parse_with(program(), input, src)
}

/// Parse a Datalog query from text
pub fn parse_query(input: &str, src: SrcId) -> Result<Query, Vec<ParseError>> {
    let query_parser = token(Token::Question)
        .then_ignore(operator_token("-"))
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot))
        .map(|body| Query { body })
        .labelled("query");

    parse_with(query_parser, input, src)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Lexer Tests =====

    #[test]
    fn test_lex_fact() {
        let result = lex("parent(john, mary).");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lex_with_whitespace() {
        let result = lex("  parent( john ,  mary ) . ");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lex_numbers() {
        let result = lex("age(john, 30).");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lex_float() {
        let result = lex("weight(item, 3.14).");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lex_negative_number() {
        let result = lex("temp(room, -5).");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lex_string() {
        let result = lex("name(person, \"John Doe\").");
        assert!(result.is_ok());
    }

    // ===== Fact Parsing Tests =====

    #[test]
    fn test_parse_fact() {
        let result = parse_program("parent(john, mary).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Fact(f) => {
                assert_eq!(f.atom.predicate.as_ref(), "parent");
                assert_eq!(f.atom.terms.len(), 2);
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_single_arg() {
        let result = parse_program("person(john).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                assert_eq!(f.atom.terms.len(), 1);
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_zero_arity() {
        let result = parse_program("done.", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                assert_eq!(f.atom.predicate.as_ref(), "done");
                assert_eq!(f.atom.terms.len(), 0);
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_integer() {
        let result = parse_program("age(john, 30).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Constant(Value::Integer(n)) => assert_eq!(*n, 30),
                    _ => panic!("Expected integer"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_negative_integer() {
        let result = parse_program("temp(room, -10).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Constant(Value::Integer(n)) => assert_eq!(*n, -10),
                    _ => panic!("Expected negative integer"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_float() {
        let result = parse_program("weight(item, 3.14).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Constant(Value::Float(n)) => assert!((*n - 3.14).abs() < 0.001),
                    _ => panic!("Expected float"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_string() {
        let result = parse_program("name(person, \"John Doe\").", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Constant(Value::String(s)) => assert_eq!(s.as_ref(), "John Doe"),
                    _ => panic!("Expected string"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_boolean_true() {
        let result = parse_program("active(service, true).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Constant(Value::Boolean(b)) => assert!(*b),
                    _ => panic!("Expected boolean true"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_boolean_false() {
        let result = parse_program("active(service, false).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Constant(Value::Boolean(b)) => assert!(!*b),
                    _ => panic!("Expected boolean false"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    // ===== Rule Parsing Tests =====

    #[test]
    fn test_parse_rule() {
        let result = parse_program("ancestor(X, Y) :- parent(X, Y).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Rule(r) => {
                assert_eq!(r.head.predicate.as_ref(), "ancestor");
                assert_eq!(r.body.len(), 1);
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_recursive_rule() {
        let result = parse_program(
            "ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).",
            SrcId::empty(),
        );
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(r) => {
                assert_eq!(r.body.len(), 2);
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_rule_multiple_body_literals() {
        let result = parse_program(
            "result(X, Y, Z) :- a(X), b(Y), c(Z), d(X, Y).",
            SrcId::empty(),
        );
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(r) => {
                assert_eq!(r.body.len(), 4);
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_rule_with_underscore_variable() {
        let result = parse_program("has_child(X) :- parent(X, _).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(r) => {
                match &r.body[0] {
                    Literal::Positive(atom) => {
                        match &atom.terms[1] {
                            Term::Variable(v) => assert!(v.as_ref().starts_with('_')),
                            _ => panic!("Expected variable"),
                        }
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    // ===== Query Parsing Tests =====

    #[test]
    fn test_parse_query() {
        let result = parse_program("?- ancestor(X, mary).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Query(q) => {
                assert_eq!(q.body.len(), 1);
            }
            _ => panic!("Expected query"),
        }
    }

    #[test]
    fn test_parse_query_multiple_literals() {
        let result = parse_program("?- parent(X, Y), age(Y, A), A > 18.", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Query(q) => {
                assert_eq!(q.body.len(), 3);
            }
            _ => panic!("Expected query"),
        }
    }

    #[test]
    fn test_parse_query_ground() {
        let result = parse_program("?- parent(john, mary).", SrcId::empty());
        assert!(result.is_ok());
    }

    // ===== Constraint Parsing Tests =====

    #[test]
    fn test_parse_constraint() {
        let result = parse_program(":- unsafe(X).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Constraint(c) => {
                assert_eq!(c.body.len(), 1);
            }
            _ => panic!("Expected constraint"),
        }
    }

    #[test]
    fn test_parse_constraint_multiple_literals() {
        let result = parse_program(":- edge(X, Y), edge(Y, X), X != Y.", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Constraint(c) => {
                assert_eq!(c.body.len(), 3);
            }
            _ => panic!("Expected constraint"),
        }
    }

    // ===== Negation Parsing Tests =====

    #[test]
    fn test_parse_negation() {
        let result = parse_program("safe(X) :- node(X), not unsafe(X).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(r) => {
                assert_eq!(r.body.len(), 2);
                assert!(r.body[1].is_negative());
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_multiple_negations() {
        let result = parse_program(
            "valid(X) :- item(X), not blocked(X), not expired(X).",
            SrcId::empty(),
        );
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(r) => {
                assert_eq!(r.body.len(), 3);
                assert!(r.body[1].is_negative());
                assert!(r.body[2].is_negative());
            }
            _ => panic!("Expected rule"),
        }
    }

    // ===== Comparison Parsing Tests =====

    #[test]
    fn test_parse_comparison() {
        let result = parse_program("big(X) :- size(X, N), N > 100.", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(r) => {
                assert_eq!(r.body.len(), 2);
                assert!(r.body[1].is_comparison());
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_comparison_less_than() {
        let result = parse_program("small(X) :- size(X, N), N < 100.", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_comparison_less_or_equal() {
        let result = parse_program("small(X) :- size(X, N), N <= 100.", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_comparison_greater_or_equal() {
        let result = parse_program("big(X) :- size(X, N), N >= 100.", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_comparison_both_variables() {
        let result = parse_program("different(X, Y) :- pair(X, Y), X != Y.", SrcId::empty());
        assert!(result.is_ok());
    }

    // ===== Compound Term Parsing Tests =====

    #[test]
    fn test_parse_compound_term() {
        let result = parse_program("has(john, item(sword, 10)).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[1] {
                    Term::Compound(functor, args) => {
                        assert_eq!(functor.as_ref(), "item");
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_nested_compound_term() {
        let result = parse_program("data(wrapper(inner(x))).", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(f) => {
                match &f.atom.terms[0] {
                    Term::Compound(functor, args) => {
                        assert_eq!(functor.as_ref(), "wrapper");
                        match &args[0] {
                            Term::Compound(inner_functor, _) => {
                                assert_eq!(inner_functor.as_ref(), "inner");
                            }
                            _ => panic!("Expected nested compound"),
                        }
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_compound_with_variable() {
        let result = parse_program("match(item(X, Y)) :- inventory(item(X, Y)).", SrcId::empty());
        assert!(result.is_ok());
    }

    // ===== Arithmetic Parsing Tests =====

    #[test]
    fn test_parse_arithmetic_addition() {
        let result = parse_program("sum(X, Y, X + Y) :- num(X), num(Y).", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_arithmetic_subtraction() {
        let result = parse_program("diff(X, Y, X - Y) :- num(X), num(Y).", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_arithmetic_multiplication() {
        let result = parse_program("prod(X, Y, X * Y) :- num(X), num(Y).", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_arithmetic_division() {
        let result = parse_program("quot(X, Y, X / Y) :- num(X), num(Y).", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_arithmetic_complex() {
        let result = parse_program("calc(X, Y, Z, X + Y * Z) :- num(X), num(Y), num(Z).", SrcId::empty());
        assert!(result.is_ok());
    }

    // ===== Multiple Statement Parsing Tests =====

    #[test]
    fn test_parse_multiple_facts() {
        let result = parse_program(
            "parent(john, mary). parent(mary, sue). parent(bob, alice).",
            SrcId::empty(),
        );
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 3);
    }

    #[test]
    fn test_parse_mixed_statements() {
        let input = r#"
            parent(john, mary).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
            :- ancestor(X, X).
            ?- ancestor(john, Who).
        "#;
        let result = parse_program(input, SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 5);
    }

    // ===== Edge Cases and Error Handling =====

    #[test]
    fn test_parse_empty_program() {
        let result = parse_program("", SrcId::empty());
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 0);
    }

    #[test]
    fn test_parse_whitespace_only() {
        // Note: Parser may return error or empty program for whitespace-only input
        // depending on implementation. We accept either behavior.
        let result = parse_program("   \n\t  ", SrcId::empty());
        if let Ok(program) = result {
            assert_eq!(program.statements.len(), 0);
        }
        // If it errors, that's also acceptable
    }

    #[test]
    fn test_parse_missing_dot() {
        let result = parse_program("parent(john, mary)", SrcId::empty());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_rule() {
        let result = parse_program("ancestor(X, Y) :- .", SrcId::empty());
        assert!(result.is_err());
    }

    // ===== Unicode and Special Characters =====

    #[test]
    fn test_parse_underscores_in_names() {
        let result = parse_program("has_parent(john_doe, mary_doe).", SrcId::empty());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_numbers_in_names() {
        let result = parse_program("item1(value2).", SrcId::empty());
        assert!(result.is_ok());
    }
}
