//! SQL Parser implementation using Chumsky 0.9

use chumsky::prelude::*;
use sql_lexer::{Keyword, Span, Token};

use crate::ast::*;

/// Parse result type
pub type ParseResult = Result<Statement, Vec<ParseError>>;

/// A parse error with location information
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

/// Parse a SQL string into a Statement
pub fn parse(input: &str) -> ParseResult {
    // First, lex the input
    let tokens = sql_lexer::lexer(input).map_err(|errs| {
        errs.into_iter()
            .map(|e| ParseError {
                message: e.message,
                span: e.span,
            })
            .collect::<Vec<_>>()
    })?;

    // Then parse the tokens
    parse_tokens(tokens)
}

/// Parse a token stream into a Statement
fn parse_tokens(tokens: Vec<(Token, Span)>) -> ParseResult {
    let len = tokens.last().map(|(_, s)| s.end).unwrap_or(0);

    // Convert to range-based stream
    let token_stream: Vec<(Token, std::ops::Range<usize>)> = tokens
        .into_iter()
        .map(|(t, s)| (t, s.start..s.end))
        .collect();

    let stream = chumsky::Stream::from_iter(len..len + 1, token_stream.into_iter());

    statement_parser().parse(stream).map_err(|errs| {
        errs.into_iter()
            .map(|e| ParseError {
                message: format!("Unexpected token: {:?}", e.found()),
                span: Span::new(e.span().start, e.span().end),
            })
            .collect()
    })
}

/// Build the statement parser
fn statement_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    select_parser()
        .map(Statement::Select)
        .or(insert_parser().map(Statement::Insert))
        .or(update_parser().map(Statement::Update))
        .or(delete_parser().map(Statement::Delete))
        .or(create_table_parser().map(Statement::CreateTable))
        .then_ignore(just(Token::Semicolon).or_not())
}

/// Parse a SELECT statement
fn select_parser() -> impl Parser<Token, SelectStatement, Error = Simple<Token>> {
    let select_kw = just(Token::Keyword(Keyword::Select));
    let from_kw = just(Token::Keyword(Keyword::From));
    let where_kw = just(Token::Keyword(Keyword::Where));
    let order_kw = just(Token::Keyword(Keyword::Order));
    let by_kw = just(Token::Keyword(Keyword::By));
    let limit_kw = just(Token::Keyword(Keyword::Limit));
    let offset_kw = just(Token::Keyword(Keyword::Offset));

    let columns = select_column_parser()
        .separated_by(just(Token::Comma))
        .at_least(1);

    let from_clause = from_kw.ignore_then(table_ref_parser()).or_not();

    let joins = join_parser().repeated();

    let where_clause = where_kw.ignore_then(expr_parser()).or_not();

    let order_by_clause = order_kw
        .ignore_then(by_kw)
        .ignore_then(
            order_by_item_parser()
                .separated_by(just(Token::Comma))
                .at_least(1),
        )
        .or_not()
        .map(|o| o.unwrap_or_default());

    let limit_clause = limit_kw.ignore_then(expr_parser()).or_not();

    let offset_clause = offset_kw.ignore_then(expr_parser()).or_not();

    select_kw
        .ignore_then(columns)
        .then(from_clause)
        .then(joins)
        .then(where_clause)
        .then(order_by_clause)
        .then(limit_clause)
        .then(offset_clause)
        .map(
            |((((((columns, from), joins), where_clause), order_by), limit), offset)| {
                SelectStatement {
                    columns,
                    from,
                    joins,
                    where_clause,
                    order_by,
                    limit,
                    offset,
                }
            },
        )
}

/// Parse a SELECT column (either * or expression with optional alias)
fn select_column_parser() -> impl Parser<Token, SelectColumn, Error = Simple<Token>> + Clone {
    just(Token::Star).to(SelectColumn::Star).or(expr_parser()
        .then(
            just(Token::Keyword(Keyword::As))
                .ignore_then(identifier())
                .or_not(),
        )
        .map(|(expr, alias)| SelectColumn::Expr { expr, alias }))
}

/// Parse a table reference
fn table_ref_parser() -> impl Parser<Token, TableRef, Error = Simple<Token>> + Clone {
    identifier()
        .then(
            just(Token::Keyword(Keyword::As))
                .or_not()
                .ignore_then(identifier())
                .or_not(),
        )
        .map(|(name, alias)| TableRef { name, alias })
}

/// Parse a JOIN clause
fn join_parser() -> impl Parser<Token, Join, Error = Simple<Token>> + Clone {
    let on_kw = just(Token::Keyword(Keyword::On));

    // Parse join type
    let join_type = just(Token::Keyword(Keyword::Inner))
        .ignore_then(just(Token::Keyword(Keyword::Join)))
        .to(JoinType::Inner)
        .or(just(Token::Keyword(Keyword::Left))
            .ignore_then(just(Token::Keyword(Keyword::Outer)).or_not())
            .ignore_then(just(Token::Keyword(Keyword::Join)))
            .to(JoinType::Left))
        .or(just(Token::Keyword(Keyword::Right))
            .ignore_then(just(Token::Keyword(Keyword::Outer)).or_not())
            .ignore_then(just(Token::Keyword(Keyword::Join)))
            .to(JoinType::Right))
        .or(just(Token::Keyword(Keyword::Full))
            .ignore_then(just(Token::Keyword(Keyword::Outer)).or_not())
            .ignore_then(just(Token::Keyword(Keyword::Join)))
            .to(JoinType::Full))
        .or(just(Token::Keyword(Keyword::Cross))
            .ignore_then(just(Token::Keyword(Keyword::Join)))
            .to(JoinType::Cross))
        .or(just(Token::Keyword(Keyword::Join)).to(JoinType::Inner)); // Default JOIN is INNER JOIN

    let on_clause = on_kw.ignore_then(expr_parser()).or_not();

    join_type
        .then(table_ref_parser())
        .then(on_clause)
        .map(|((join_type, table), on)| Join {
            join_type,
            table,
            on,
        })
}

/// Parse an ORDER BY item
fn order_by_item_parser() -> impl Parser<Token, OrderBy, Error = Simple<Token>> + Clone {
    expr_parser()
        .then(
            just(Token::Keyword(Keyword::Desc))
                .to(true)
                .or(just(Token::Keyword(Keyword::Asc)).to(false))
                .or_not()
                .map(|o| o.unwrap_or(false)),
        )
        .map(|(expr, desc)| OrderBy { expr, desc })
}

/// Parse an expression with proper precedence
fn expr_parser() -> impl Parser<Token, Expr, Error = Simple<Token>> + Clone {
    recursive(|expr| {
        let literal = select! {
            Token::Integer(n) => Expr::Integer(n),
            Token::Float(f) => Expr::Float(f.value()),
            Token::String(s) => Expr::String(s),
            Token::Keyword(Keyword::True) => Expr::Boolean(true),
            Token::Keyword(Keyword::False) => Expr::Boolean(false),
            Token::Keyword(Keyword::Null) => Expr::Null,
        };

        // Support both simple columns (name) and qualified columns (table.name)
        let column = identifier()
            .then(just(Token::Dot).ignore_then(identifier()).or_not())
            .map(|(first, second)| match second {
                Some(col) => Expr::Column(format!("{}.{}", first, col)),
                None => Expr::Column(first),
            });

        // Aggregate functions: COUNT(*), COUNT(col), SUM(col), AVG(col), MIN(col), MAX(col)
        let aggregate_func = select! {
            Token::Keyword(Keyword::Count) => AggregateFunc::Count,
            Token::Keyword(Keyword::Sum) => AggregateFunc::Sum,
            Token::Keyword(Keyword::Avg) => AggregateFunc::Avg,
            Token::Keyword(Keyword::Min) => AggregateFunc::Min,
            Token::Keyword(Keyword::Max) => AggregateFunc::Max,
        };

        // Parse aggregate argument - either * (for COUNT) or an expression
        let aggregate_arg = just(Token::Star)
            .to(Expr::Column("*".to_string()))
            .or(expr.clone());

        let aggregate = aggregate_func
            .then(aggregate_arg.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|(func, arg)| Expr::Aggregate {
                func,
                arg: Box::new(arg),
            });

        let paren_expr = expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        let atom = literal.or(aggregate).or(column).or(paren_expr);

        // Unary operators
        let unary = just(Token::Minus)
            .to(UnaryOp::Neg)
            .or(just(Token::Keyword(Keyword::Not)).to(UnaryOp::Not))
            .repeated()
            .then(atom)
            .foldr(|op, expr| Expr::UnaryOp {
                op,
                expr: Box::new(expr),
            });

        // Multiplication and division (highest precedence)
        let mul_div = unary
            .clone()
            .then(
                just(Token::Star)
                    .to(BinaryOp::Mul)
                    .or(just(Token::Slash).to(BinaryOp::Div))
                    .or(just(Token::Percent).to(BinaryOp::Mod))
                    .then(unary)
                    .repeated(),
            )
            .foldl(|left, (op, right)| Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });

        // Addition and subtraction
        let add_sub = mul_div
            .clone()
            .then(
                just(Token::Plus)
                    .to(BinaryOp::Add)
                    .or(just(Token::Minus).to(BinaryOp::Sub))
                    .then(mul_div)
                    .repeated(),
            )
            .foldl(|left, (op, right)| Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });

        // Comparison operators
        let comparison = add_sub
            .clone()
            .then(
                just(Token::Eq)
                    .to(BinaryOp::Eq)
                    .or(just(Token::NotEq).to(BinaryOp::NotEq))
                    .or(just(Token::Lt).to(BinaryOp::Lt))
                    .or(just(Token::Gt).to(BinaryOp::Gt))
                    .or(just(Token::LtEq).to(BinaryOp::LtEq))
                    .or(just(Token::GtEq).to(BinaryOp::GtEq))
                    .then(add_sub)
                    .or_not(),
            )
            .map(|(left, right)| match right {
                Some((op, right)) => Expr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                None => left,
            });

        // AND operator
        let and_expr = comparison
            .clone()
            .then(
                just(Token::Keyword(Keyword::And))
                    .ignore_then(comparison)
                    .repeated(),
            )
            .foldl(|left, right| Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            });

        // OR operator (lowest precedence)
        and_expr
            .clone()
            .then(
                just(Token::Keyword(Keyword::Or))
                    .ignore_then(and_expr)
                    .repeated(),
            )
            .foldl(|left, right| Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
            })
    })
}

/// Parse an identifier
fn identifier() -> impl Parser<Token, String, Error = Simple<Token>> + Clone {
    select! {
        Token::Identifier(name) => name,
    }
}

/// Parse an INSERT statement
fn insert_parser() -> impl Parser<Token, InsertStatement, Error = Simple<Token>> {
    let insert_kw = just(Token::Keyword(Keyword::Insert));
    let into_kw = just(Token::Keyword(Keyword::Into));
    let values_kw = just(Token::Keyword(Keyword::Values));

    let columns = identifier()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or_not();

    let value_row = expr_parser()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen));

    let values = value_row.separated_by(just(Token::Comma)).at_least(1);

    insert_kw
        .ignore_then(into_kw)
        .ignore_then(identifier())
        .then(columns)
        .then_ignore(values_kw)
        .then(values)
        .map(|((table, columns), values)| InsertStatement {
            table,
            columns,
            values,
        })
}

/// Parse an UPDATE statement
fn update_parser() -> impl Parser<Token, UpdateStatement, Error = Simple<Token>> {
    let update_kw = just(Token::Keyword(Keyword::Update));
    let set_kw = just(Token::Keyword(Keyword::Set));
    let where_kw = just(Token::Keyword(Keyword::Where));

    let assignment = identifier()
        .then_ignore(just(Token::Eq))
        .then(expr_parser())
        .map(|(column, value)| Assignment { column, value });

    let assignments = assignment.separated_by(just(Token::Comma)).at_least(1);

    let where_clause = where_kw.ignore_then(expr_parser()).or_not();

    update_kw
        .ignore_then(identifier())
        .then_ignore(set_kw)
        .then(assignments)
        .then(where_clause)
        .map(|((table, assignments), where_clause)| UpdateStatement {
            table,
            assignments,
            where_clause,
        })
}

/// Parse a DELETE statement
fn delete_parser() -> impl Parser<Token, DeleteStatement, Error = Simple<Token>> {
    let delete_kw = just(Token::Keyword(Keyword::Delete));
    let from_kw = just(Token::Keyword(Keyword::From));
    let where_kw = just(Token::Keyword(Keyword::Where));

    let where_clause = where_kw.ignore_then(expr_parser()).or_not();

    delete_kw
        .ignore_then(from_kw)
        .ignore_then(identifier())
        .then(where_clause)
        .map(|(table, where_clause)| DeleteStatement {
            table,
            where_clause,
        })
}

/// Parse a CREATE TABLE statement
fn create_table_parser() -> impl Parser<Token, CreateTableStatement, Error = Simple<Token>> {
    let create_kw = just(Token::Keyword(Keyword::Create));
    let table_kw = just(Token::Keyword(Keyword::Table));

    let column_defs = column_def_parser()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen));

    create_kw
        .ignore_then(table_kw)
        .ignore_then(identifier())
        .then(column_defs)
        .map(|(name, columns)| CreateTableStatement { name, columns })
}

/// Parse a column definition
fn column_def_parser() -> impl Parser<Token, ColumnDef, Error = Simple<Token>> + Clone {
    let data_type = select! {
        Token::Keyword(Keyword::Int) => DataType::Int,
        Token::Keyword(Keyword::Integer) => DataType::Int,
        Token::Keyword(Keyword::Text) => DataType::Text,
        Token::Keyword(Keyword::Varchar) => DataType::Text,
        Token::Keyword(Keyword::Float) => DataType::Float,
        Token::Keyword(Keyword::Double) => DataType::Float,
        Token::Keyword(Keyword::Bool) => DataType::Bool,
        Token::Keyword(Keyword::Boolean) => DataType::Bool,
    };

    let not_null = just(Token::Keyword(Keyword::Not))
        .then(just(Token::Keyword(Keyword::Null)))
        .to(false)
        .or_not()
        .map(|o| o.unwrap_or(true)); // default is nullable

    let primary_key = just(Token::Keyword(Keyword::Primary))
        .then(just(Token::Keyword(Keyword::Key)))
        .to(true)
        .or_not()
        .map(|o| o.unwrap_or(false));

    identifier()
        .then(data_type)
        .then(not_null)
        .then(primary_key)
        .map(|(((name, data_type), nullable), primary_key)| ColumnDef {
            name,
            data_type,
            nullable,
            primary_key,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_star() {
        let result = parse("SELECT * FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns, vec![SelectColumn::Star]);
                assert_eq!(
                    s.from,
                    Some(TableRef {
                        name: "users".to_string(),
                        alias: None
                    })
                );
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_columns() {
        let result = parse("SELECT id, name FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 2);
                assert!(matches!(
                    &s.columns[0],
                    SelectColumn::Expr { expr: Expr::Column(n), alias: None } if n == "id"
                ));
                assert!(matches!(
                    &s.columns[1],
                    SelectColumn::Expr { expr: Expr::Column(n), alias: None } if n == "name"
                ));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_with_where() {
        let result = parse("SELECT * FROM users WHERE id = 1");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(s.where_clause.is_some());
                let where_clause = s.where_clause.unwrap();
                assert!(matches!(
                    where_clause,
                    Expr::BinaryOp {
                        op: BinaryOp::Eq,
                        ..
                    }
                ));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_with_order_by() {
        let result = parse("SELECT * FROM users ORDER BY name DESC");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.order_by.len(), 1);
                assert!(s.order_by[0].desc);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_with_limit() {
        let result = parse("SELECT * FROM users LIMIT 10");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(s.limit, Some(Expr::Integer(10))));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_expression_precedence() {
        let result = parse("SELECT 1 + 2 * 3");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 1);
                // Should be 1 + (2 * 3), not (1 + 2) * 3
                match &s.columns[0] {
                    SelectColumn::Expr { expr, .. } => match expr {
                        Expr::BinaryOp { left, op, right } => {
                            assert_eq!(*op, BinaryOp::Add);
                            assert!(matches!(**left, Expr::Integer(1)));
                            assert!(matches!(
                                **right,
                                Expr::BinaryOp {
                                    op: BinaryOp::Mul,
                                    ..
                                }
                            ));
                        }
                        _ => panic!("Expected binary op"),
                    },
                    _ => panic!("Expected expression"),
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_insert() {
        let result = parse("INSERT INTO users (id, name) VALUES (1, 'john')");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "users");
                assert_eq!(i.columns, Some(vec!["id".to_string(), "name".to_string()]));
                assert_eq!(i.values.len(), 1);
                assert_eq!(i.values[0].len(), 2);
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_parse_create_table() {
        let result = parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(c) => {
                assert_eq!(c.name, "users");
                assert_eq!(c.columns.len(), 2);
                assert_eq!(c.columns[0].name, "id");
                assert_eq!(c.columns[0].data_type, DataType::Int);
                assert!(c.columns[0].primary_key);
                assert_eq!(c.columns[1].name, "name");
                assert_eq!(c.columns[1].data_type, DataType::Text);
                assert!(!c.columns[1].nullable);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_and_or_precedence() {
        let result = parse("SELECT * FROM t WHERE a = 1 AND b = 2 OR c = 3");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                // Should be (a = 1 AND b = 2) OR c = 3
                match s.where_clause.unwrap() {
                    Expr::BinaryOp { op, .. } => {
                        assert_eq!(op, BinaryOp::Or);
                    }
                    _ => panic!("Expected OR at top level"),
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_table_alias() {
        let result = parse("SELECT id FROM users AS u");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(
                    s.from,
                    Some(TableRef {
                        name: "users".to_string(),
                        alias: Some("u".to_string())
                    })
                );
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_column_alias() {
        let result = parse("SELECT id AS user_id FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    &s.columns[0],
                    SelectColumn::Expr { alias: Some(a), .. } if a == "user_id"
                ));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_parenthesized_expr() {
        let result = parse("SELECT (1 + 2) * 3");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expr { expr, .. } => match expr {
                    Expr::BinaryOp { left, op, right } => {
                        assert_eq!(*op, BinaryOp::Mul);
                        assert!(matches!(
                            **left,
                            Expr::BinaryOp {
                                op: BinaryOp::Add,
                                ..
                            }
                        ));
                        assert!(matches!(**right, Expr::Integer(3)));
                    }
                    _ => panic!("Expected binary op"),
                },
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_negative_number() {
        let result = parse("SELECT -42");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expr { expr, .. } => {
                    assert!(matches!(
                        expr,
                        Expr::UnaryOp {
                            op: UnaryOp::Neg,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_update() {
        let result = parse("UPDATE users SET name = 'bob' WHERE id = 1");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Update(u) => {
                assert_eq!(u.table, "users");
                assert_eq!(u.assignments.len(), 1);
                assert_eq!(u.assignments[0].column, "name");
                assert!(u.where_clause.is_some());
            }
            _ => panic!("Expected UPDATE statement"),
        }
    }

    #[test]
    fn test_parse_update_multiple_columns() {
        let result = parse("UPDATE users SET name = 'bob', age = 30");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Update(u) => {
                assert_eq!(u.table, "users");
                assert_eq!(u.assignments.len(), 2);
                assert_eq!(u.assignments[0].column, "name");
                assert_eq!(u.assignments[1].column, "age");
                assert!(u.where_clause.is_none());
            }
            _ => panic!("Expected UPDATE statement"),
        }
    }

    #[test]
    fn test_parse_delete() {
        let result = parse("DELETE FROM users WHERE id = 1");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Delete(d) => {
                assert_eq!(d.table, "users");
                assert!(d.where_clause.is_some());
            }
            _ => panic!("Expected DELETE statement"),
        }
    }

    #[test]
    fn test_parse_delete_all() {
        let result = parse("DELETE FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Delete(d) => {
                assert_eq!(d.table, "users");
                assert!(d.where_clause.is_none());
            }
            _ => panic!("Expected DELETE statement"),
        }
    }

    #[test]
    fn test_parse_inner_join() {
        let result = parse("SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.joins.len(), 1);
                assert_eq!(s.joins[0].join_type, JoinType::Inner);
                assert_eq!(s.joins[0].table.name, "orders");
                assert!(s.joins[0].on.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_left_join() {
        let result = parse("SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.joins.len(), 1);
                assert_eq!(s.joins[0].join_type, JoinType::Left);
                assert_eq!(s.joins[0].table.name, "orders");
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_left_outer_join() {
        let result =
            parse("SELECT * FROM users LEFT OUTER JOIN orders ON users.id = orders.user_id");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.joins.len(), 1);
                assert_eq!(s.joins[0].join_type, JoinType::Left);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_cross_join() {
        let result = parse("SELECT * FROM users CROSS JOIN orders");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.joins.len(), 1);
                assert_eq!(s.joins[0].join_type, JoinType::Cross);
                assert!(s.joins[0].on.is_none());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_multiple_joins() {
        let result = parse(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id JOIN items ON orders.id = items.order_id"
        );
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.joins.len(), 2);
                assert_eq!(s.joins[0].table.name, "orders");
                assert_eq!(s.joins[1].table.name, "items");
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_join_with_alias() {
        let result = parse("SELECT * FROM users u JOIN orders o ON u.id = o.user_id");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.joins.len(), 1);
                assert_eq!(s.joins[0].table.name, "orders");
                assert_eq!(s.joins[0].table.alias, Some("o".to_string()));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_count_star() {
        let result = parse("SELECT COUNT(*) FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 1);
                match &s.columns[0] {
                    SelectColumn::Expr { expr, .. } => {
                        assert!(matches!(
                            expr,
                            Expr::Aggregate {
                                func: AggregateFunc::Count,
                                ..
                            }
                        ));
                    }
                    _ => panic!("Expected aggregate expression"),
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_sum_column() {
        let result = parse("SELECT SUM(amount) FROM orders");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expr { expr, .. } => match expr {
                    Expr::Aggregate { func, arg } => {
                        assert_eq!(*func, AggregateFunc::Sum);
                        assert!(matches!(**arg, Expr::Column(ref c) if c == "amount"));
                    }
                    _ => panic!("Expected Aggregate"),
                },
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_multiple_aggregates() {
        let result = parse("SELECT COUNT(*), AVG(price), MAX(quantity) FROM products");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 3);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_aggregate_with_alias() {
        let result = parse("SELECT COUNT(*) AS total FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expr { alias, .. } => {
                    assert_eq!(alias, &Some("total".to_string()));
                }
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }
}
