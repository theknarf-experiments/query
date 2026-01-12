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
    select_or_set_op_parser()
        .or(insert_parser().map(Statement::Insert))
        .or(update_parser().map(Statement::Update))
        .or(delete_parser().map(Statement::Delete))
        .or(create_table_parser().map(Statement::CreateTable))
        .or(create_trigger_parser().map(Statement::CreateTrigger))
        .or(create_index_parser().map(Statement::CreateIndex))
        .or(create_view_parser().map(Statement::CreateView))
        .or(drop_trigger_parser().map(Statement::DropTrigger))
        .or(drop_index_parser().map(Statement::DropIndex))
        .or(drop_view_parser().map(Statement::DropView))
        .or(drop_table_parser().map(Statement::DropTable))
        .or(alter_table_parser().map(Statement::AlterTable))
        .or(create_procedure_parser().map(Statement::CreateProcedure))
        .or(drop_procedure_parser().map(Statement::DropProcedure))
        .or(call_procedure_parser().map(Statement::CallProcedure))
        .or(begin_parser())
        .or(commit_parser())
        .or(rollback_parser())
        .or(savepoint_parser())
        .or(release_savepoint_parser())
        .then_ignore(just(Token::Semicolon).or_not())
}

/// Parse a single CTE definition: name [(columns)] AS (SELECT ...)
fn cte_parser() -> impl Parser<Token, Cte, Error = Simple<Token>> {
    let column_names = identifier()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or_not();

    identifier()
        .then(column_names)
        .then_ignore(just(Token::Keyword(Keyword::As)))
        .then(select_parser().delimited_by(just(Token::LParen), just(Token::RParen)))
        .map(|((name, columns), query)| Cte {
            name,
            columns,
            query: Box::new(query),
        })
}

/// Parse WITH clause: WITH [RECURSIVE] cte1, cte2, ...
fn with_clause_parser() -> impl Parser<Token, WithClause, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::With))
        .ignore_then(just(Token::Keyword(Keyword::Recursive)).or_not())
        .then(cte_parser().separated_by(just(Token::Comma)).at_least(1))
        .map(|(recursive, ctes)| WithClause {
            recursive: recursive.is_some(),
            ctes,
        })
}

/// Parse SELECT possibly followed by set operations (UNION, INTERSECT, EXCEPT)
fn select_or_set_op_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    // Parse set operator (UNION, INTERSECT, EXCEPT) with optional ALL
    let set_op = just(Token::Keyword(Keyword::Union))
        .to(SetOperator::Union)
        .or(just(Token::Keyword(Keyword::Intersect)).to(SetOperator::Intersect))
        .or(just(Token::Keyword(Keyword::Except)).to(SetOperator::Except))
        .then(just(Token::Keyword(Keyword::All)).or_not())
        .map(|(op, all)| (op, all.is_some()));

    // Parse optional WITH clause followed by SELECT
    with_clause_parser()
        .or_not()
        .then(select_parser())
        .then(set_op.then(select_parser()).repeated())
        .map(|((with_clause, first), set_ops)| {
            // Attach WITH clause to the first SELECT
            let first = SelectStatement {
                with_clause,
                ..first
            };

            if set_ops.is_empty() {
                Statement::Select(Box::new(first))
            } else {
                // Build the set operation chain
                let mut current = SelectOrSet::Select(Box::new(first));
                for ((op, all), select) in set_ops {
                    current = SelectOrSet::SetOp(SetOperationStatement {
                        left: Box::new(current),
                        right: Box::new(SelectOrSet::Select(Box::new(select))),
                        op,
                        all,
                    });
                }
                match current {
                    SelectOrSet::Select(s) => Statement::Select(s),
                    SelectOrSet::SetOp(s) => Statement::SetOperation(s),
                }
            }
        })
}

/// Parse BEGIN [TRANSACTION]
fn begin_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Begin))
        .then_ignore(just(Token::Keyword(Keyword::Transaction)).or_not())
        .to(Statement::Begin)
}

/// Parse COMMIT [TRANSACTION]
fn commit_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Commit))
        .then_ignore(just(Token::Keyword(Keyword::Transaction)).or_not())
        .to(Statement::Commit)
}

/// Parse ROLLBACK [TRANSACTION] or ROLLBACK TO [SAVEPOINT] name
fn rollback_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    let rollback_to = just(Token::Keyword(Keyword::Rollback))
        .ignore_then(
            just(Token::Keyword(Keyword::To))
                .ignore_then(just(Token::Keyword(Keyword::Savepoint)).or_not())
                .ignore_then(identifier()),
        )
        .map(Statement::RollbackTo);

    let rollback_full = just(Token::Keyword(Keyword::Rollback))
        .then_ignore(just(Token::Keyword(Keyword::Transaction)).or_not())
        .to(Statement::Rollback);

    rollback_to.or(rollback_full)
}

/// Parse SAVEPOINT name
fn savepoint_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Savepoint))
        .ignore_then(identifier())
        .map(Statement::Savepoint)
}

/// Parse RELEASE [SAVEPOINT] name
fn release_savepoint_parser() -> impl Parser<Token, Statement, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Release))
        .ignore_then(just(Token::Keyword(Keyword::Savepoint)).or_not())
        .ignore_then(identifier())
        .map(Statement::ReleaseSavepoint)
}

/// Parse CREATE TRIGGER
/// CREATE TRIGGER name (BEFORE|AFTER) (INSERT|UPDATE|DELETE) ON table
/// FOR EACH ROW (SET column = value | RAISE ERROR 'message'), ...
fn create_trigger_parser() -> impl Parser<Token, CreateTriggerStatement, Error = Simple<Token>> {
    let timing = just(Token::Keyword(Keyword::Before))
        .to(TriggerTiming::Before)
        .or(just(Token::Keyword(Keyword::After)).to(TriggerTiming::After));

    let event = just(Token::Keyword(Keyword::Insert))
        .to(TriggerEvent::Insert)
        .or(just(Token::Keyword(Keyword::Update)).to(TriggerEvent::Update))
        .or(just(Token::Keyword(Keyword::Delete)).to(TriggerEvent::Delete));

    let set_action = just(Token::Keyword(Keyword::Set))
        .ignore_then(identifier())
        .then_ignore(just(Token::Eq))
        .then(expr_parser())
        .map(|(column, value)| TriggerAction::SetColumn { column, value });

    let raise_action = just(Token::Keyword(Keyword::Raise))
        .ignore_then(just(Token::Keyword(Keyword::Error)))
        .ignore_then(string_literal())
        .map(TriggerAction::RaiseError);

    let action = set_action.or(raise_action);

    just(Token::Keyword(Keyword::Create))
        .ignore_then(just(Token::Keyword(Keyword::Trigger)))
        .ignore_then(identifier())
        .then(timing)
        .then(event)
        .then_ignore(just(Token::Keyword(Keyword::On)))
        .then(identifier())
        .then_ignore(just(Token::Keyword(Keyword::For)))
        .then_ignore(just(Token::Keyword(Keyword::Each)))
        .then_ignore(just(Token::Keyword(Keyword::Row)))
        .then(action.separated_by(just(Token::Comma)).at_least(1))
        .map(
            |((((name, timing), event), table), body)| CreateTriggerStatement {
                name,
                timing,
                event,
                table,
                body,
            },
        )
}

/// Parse DROP TRIGGER name
fn drop_trigger_parser() -> impl Parser<Token, String, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Drop))
        .ignore_then(just(Token::Keyword(Keyword::Trigger)))
        .ignore_then(identifier())
}

/// Parse DROP TABLE name
fn drop_table_parser() -> impl Parser<Token, String, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Drop))
        .ignore_then(just(Token::Keyword(Keyword::Table)))
        .ignore_then(identifier())
}

/// Parse CREATE INDEX name ON table(column)
fn create_index_parser() -> impl Parser<Token, CreateIndexStatement, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Create))
        .ignore_then(just(Token::Keyword(Keyword::Index)))
        .ignore_then(identifier())
        .then_ignore(just(Token::Keyword(Keyword::On)))
        .then(identifier())
        .then(identifier().delimited_by(just(Token::LParen), just(Token::RParen)))
        .map(|((name, table), column)| CreateIndexStatement {
            name,
            table,
            column,
        })
}

/// Parse DROP INDEX name
fn drop_index_parser() -> impl Parser<Token, String, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Drop))
        .ignore_then(just(Token::Keyword(Keyword::Index)))
        .ignore_then(identifier())
}

/// Parse CREATE VIEW name [(columns)] AS SELECT ...
fn create_view_parser() -> impl Parser<Token, CreateViewStatement, Error = Simple<Token>> {
    let column_names = identifier()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or_not();

    just(Token::Keyword(Keyword::Create))
        .ignore_then(just(Token::Keyword(Keyword::View)))
        .ignore_then(identifier())
        .then(column_names)
        .then_ignore(just(Token::Keyword(Keyword::As)))
        .then(select_parser())
        .map(|((name, columns), query)| CreateViewStatement {
            name,
            columns,
            query: Box::new(query),
        })
}

/// Parse DROP VIEW name
fn drop_view_parser() -> impl Parser<Token, String, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Drop))
        .ignore_then(just(Token::Keyword(Keyword::View)))
        .ignore_then(identifier())
}

/// Parse a procedure parameter: name TYPE
fn procedure_param_parser() -> impl Parser<Token, ProcedureParam, Error = Simple<Token>> {
    identifier()
        .then(data_type_parser())
        .map(|(name, data_type)| ProcedureParam { name, data_type })
}

/// Parse a procedure body statement (simplified - only supports basic DML)
fn procedure_body_statement_parser() -> impl Parser<Token, ProcedureStatement, Error = Simple<Token>>
{
    // DECLARE @name TYPE
    let declare_stmt = just(Token::Keyword(Keyword::Declare))
        .ignore_then(identifier())
        .then(data_type_parser())
        .map(|(name, data_type)| ProcedureStatement::Declare { name, data_type });

    // SET @name = expr
    let set_var_stmt = just(Token::Keyword(Keyword::Set))
        .ignore_then(identifier())
        .then_ignore(just(Token::Eq))
        .then(expr_parser())
        .map(|(name, value)| ProcedureStatement::SetVar { name, value });

    // RETURN [expr]
    let return_stmt = just(Token::Keyword(Keyword::Return))
        .ignore_then(expr_parser().or_not())
        .map(ProcedureStatement::Return);

    // INSERT statement
    let insert_stmt =
        insert_parser().map(|s| ProcedureStatement::Sql(Box::new(Statement::Insert(s))));

    // UPDATE statement
    let update_stmt =
        update_parser().map(|s| ProcedureStatement::Sql(Box::new(Statement::Update(s))));

    // DELETE statement
    let delete_stmt =
        delete_parser().map(|s| ProcedureStatement::Sql(Box::new(Statement::Delete(s))));

    // SELECT statement (simplified for procedure bodies)
    let select_stmt =
        select_parser().map(|s| ProcedureStatement::Sql(Box::new(Statement::Select(Box::new(s)))));

    declare_stmt
        .or(set_var_stmt)
        .or(return_stmt)
        .or(insert_stmt)
        .or(update_stmt)
        .or(delete_stmt)
        .or(select_stmt)
        .then_ignore(just(Token::Semicolon).or_not())
}

/// Parse CREATE PROCEDURE name (params) AS BEGIN ... END
fn create_procedure_parser() -> impl Parser<Token, CreateProcedureStatement, Error = Simple<Token>>
{
    let params = procedure_param_parser()
        .separated_by(just(Token::Comma))
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or_not()
        .map(|p| p.unwrap_or_default());

    let body = procedure_body_statement_parser().repeated();

    just(Token::Keyword(Keyword::Create))
        .ignore_then(just(Token::Keyword(Keyword::Procedure)))
        .ignore_then(identifier())
        .then(params)
        .then_ignore(just(Token::Keyword(Keyword::As)))
        .then_ignore(just(Token::Keyword(Keyword::Begin)))
        .then(body)
        .then_ignore(just(Token::Keyword(Keyword::End)))
        .map(|((name, params), body)| CreateProcedureStatement { name, params, body })
}

/// Parse DROP PROCEDURE name
fn drop_procedure_parser() -> impl Parser<Token, String, Error = Simple<Token>> {
    just(Token::Keyword(Keyword::Drop))
        .ignore_then(just(Token::Keyword(Keyword::Procedure)))
        .ignore_then(identifier())
}

/// Parse CALL/EXEC procedure_name(args)
fn call_procedure_parser() -> impl Parser<Token, CallProcedureStatement, Error = Simple<Token>> {
    let args = expr_parser()
        .separated_by(just(Token::Comma))
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or_not()
        .map(|a| a.unwrap_or_default());

    just(Token::Keyword(Keyword::Call))
        .or(just(Token::Keyword(Keyword::Exec)))
        .or(just(Token::Keyword(Keyword::Execute)))
        .ignore_then(identifier())
        .then(args)
        .map(|(name, args)| CallProcedureStatement { name, args })
}

/// Parse ALTER TABLE statement
fn alter_table_parser() -> impl Parser<Token, AlterTableStatement, Error = Simple<Token>> {
    let add_column = just(Token::Keyword(Keyword::Add))
        .ignore_then(just(Token::Keyword(Keyword::Column)).or_not())
        .ignore_then(column_def_parser())
        .map(AlterAction::AddColumn);

    let drop_column = just(Token::Keyword(Keyword::Drop))
        .ignore_then(just(Token::Keyword(Keyword::Column)).or_not())
        .ignore_then(identifier())
        .map(AlterAction::DropColumn);

    let rename_column = just(Token::Keyword(Keyword::Rename))
        .ignore_then(just(Token::Keyword(Keyword::Column)).or_not())
        .ignore_then(identifier())
        .then_ignore(just(Token::Keyword(Keyword::To)))
        .then(identifier())
        .map(|(old_name, new_name)| AlterAction::RenameColumn { old_name, new_name });

    let rename_table = just(Token::Keyword(Keyword::Rename))
        .ignore_then(just(Token::Keyword(Keyword::To)))
        .ignore_then(identifier())
        .map(AlterAction::RenameTable);

    let action = add_column
        .or(drop_column)
        .or(rename_column)
        .or(rename_table);

    just(Token::Keyword(Keyword::Alter))
        .ignore_then(just(Token::Keyword(Keyword::Table)))
        .ignore_then(identifier())
        .then(action)
        .map(|(table, action)| AlterTableStatement { table, action })
}

/// Parse a string literal
fn string_literal() -> impl Parser<Token, String, Error = Simple<Token>> + Clone {
    select! {
        Token::String(s) => s,
    }
}

/// Parse a SELECT statement
fn select_parser() -> impl Parser<Token, SelectStatement, Error = Simple<Token>> {
    select_parser_impl(None)
}

fn select_parser_impl(
    with_clause: Option<WithClause>,
) -> impl Parser<Token, SelectStatement, Error = Simple<Token>> {
    let select_kw = just(Token::Keyword(Keyword::Select));
    let distinct_kw = just(Token::Keyword(Keyword::Distinct));
    let from_kw = just(Token::Keyword(Keyword::From));
    let where_kw = just(Token::Keyword(Keyword::Where));
    let group_kw = just(Token::Keyword(Keyword::Group));
    let having_kw = just(Token::Keyword(Keyword::Having));
    let order_kw = just(Token::Keyword(Keyword::Order));
    let by_kw = just(Token::Keyword(Keyword::By));
    let limit_kw = just(Token::Keyword(Keyword::Limit));
    let offset_kw = just(Token::Keyword(Keyword::Offset));

    let distinct = distinct_kw.or_not().map(|d| d.is_some());

    let columns = select_column_parser()
        .separated_by(just(Token::Comma))
        .at_least(1);

    let from_clause = from_kw.ignore_then(table_ref_parser()).or_not();

    let joins = join_parser().repeated();

    let where_clause = where_kw.ignore_then(expr_parser()).or_not();

    let group_by_clause = group_kw
        .ignore_then(by_kw.clone())
        .ignore_then(expr_parser().separated_by(just(Token::Comma)).at_least(1))
        .or_not()
        .map(|g| g.unwrap_or_default());

    let having_clause = having_kw.ignore_then(expr_parser()).or_not();

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
        .ignore_then(distinct)
        .then(columns)
        .then(from_clause)
        .then(joins)
        .then(where_clause)
        .then(group_by_clause)
        .then(having_clause)
        .then(order_by_clause)
        .then(limit_clause)
        .then(offset_clause)
        .map(
            move |(
                (
                    (
                        ((((((distinct, columns), from), joins), where_clause), group_by), having),
                        order_by,
                    ),
                    limit,
                ),
                offset,
            )| {
                SelectStatement {
                    with_clause: with_clause.clone(),
                    distinct,
                    columns,
                    from,
                    joins,
                    where_clause,
                    group_by,
                    having,
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

        // Window functions: ROW_NUMBER() OVER (...), RANK() OVER (...), DENSE_RANK() OVER (...)
        let window_func = select! {
            Token::Keyword(Keyword::RowNumber) => WindowFunc::RowNumber,
            Token::Keyword(Keyword::Rank) => WindowFunc::Rank,
            Token::Keyword(Keyword::DenseRank) => WindowFunc::DenseRank,
        };

        // Parse ORDER BY for window
        let order_by_item = expr
            .clone()
            .then(
                just(Token::Keyword(Keyword::Desc))
                    .to(true)
                    .or(just(Token::Keyword(Keyword::Asc)).to(false))
                    .or_not(),
            )
            .map(|(expr, desc)| OrderBy {
                expr,
                desc: desc.unwrap_or(false),
            });

        // Parse PARTITION BY clause
        let partition_by = just(Token::Keyword(Keyword::Partition))
            .ignore_then(just(Token::Keyword(Keyword::By)))
            .ignore_then(expr.clone().separated_by(just(Token::Comma)).at_least(1))
            .or_not()
            .map(|opt| opt.unwrap_or_default());

        // Parse ORDER BY clause for window
        let window_order_by = just(Token::Keyword(Keyword::Order))
            .ignore_then(just(Token::Keyword(Keyword::By)))
            .ignore_then(order_by_item.separated_by(just(Token::Comma)).at_least(1))
            .or_not()
            .map(|opt| opt.unwrap_or_default());

        // Parse OVER clause
        let over_clause = just(Token::Keyword(Keyword::Over)).ignore_then(
            partition_by
                .then(window_order_by)
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        );

        let window_function = window_func
            .then_ignore(just(Token::LParen))
            .then_ignore(just(Token::RParen))
            .then(over_clause)
            .map(|(func, (partition_by, order_by))| Expr::WindowFunction {
                func,
                partition_by,
                order_by,
            });

        // EXISTS (SELECT ...)
        let exists_subquery = just(Token::Keyword(Keyword::Exists))
            .ignore_then(
                subquery_select_parser().delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|select| Expr::Exists(Box::new(select)));

        // Scalar subquery: (SELECT ...)
        let scalar_subquery = subquery_select_parser()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(|select| Expr::Subquery(Box::new(select)));

        let paren_expr = expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // CASE WHEN expression
        // Supports both:
        // - CASE WHEN cond THEN result ... [ELSE result] END
        // - CASE expr WHEN val THEN result ... [ELSE result] END
        let when_clause = just(Token::Keyword(Keyword::When))
            .ignore_then(expr.clone())
            .then_ignore(just(Token::Keyword(Keyword::Then)))
            .then(expr.clone());

        let case_expr = just(Token::Keyword(Keyword::Case))
            .ignore_then(
                // Optional operand for simple CASE
                expr.clone()
                    .then(when_clause.clone().repeated().at_least(1))
                    .map(|(operand, when_clauses)| (Some(Box::new(operand)), when_clauses))
                    .or(when_clause
                        .repeated()
                        .at_least(1)
                        .map(|when_clauses| (None, when_clauses))),
            )
            .then(
                just(Token::Keyword(Keyword::Else))
                    .ignore_then(expr.clone())
                    .or_not(),
            )
            .then_ignore(just(Token::Keyword(Keyword::End)))
            .map(|((operand, when_clauses), else_result)| Expr::Case {
                operand,
                when_clauses,
                else_result: else_result.map(Box::new),
            });

        let atom = literal
            .or(aggregate)
            .or(window_function)
            .or(exists_subquery)
            .or(case_expr)
            .or(column)
            .or(scalar_subquery)
            .or(paren_expr);

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

        // IS NULL / IS NOT NULL
        let is_null = add_sub
            .clone()
            .then(
                just(Token::Keyword(Keyword::Is))
                    .ignore_then(just(Token::Keyword(Keyword::Not)).or_not())
                    .then_ignore(just(Token::Keyword(Keyword::Null)))
                    .or_not(),
            )
            .map(|(left, is_clause)| match is_clause {
                Some(negated) => Expr::IsNull {
                    expr: Box::new(left),
                    negated: negated.is_some(),
                },
                None => left,
            });

        // LIKE pattern matching: expr [NOT] LIKE pattern
        let like_expr = is_null
            .clone()
            .then(
                just(Token::Keyword(Keyword::Not))
                    .or_not()
                    .then_ignore(just(Token::Keyword(Keyword::Like)))
                    .then(is_null.clone())
                    .or_not(),
            )
            .map(|(left, like_clause)| match like_clause {
                Some((negated, pattern)) => Expr::Like {
                    expr: Box::new(left),
                    pattern: Box::new(pattern),
                    negated: negated.is_some(),
                },
                None => left,
            });

        // BETWEEN: expr [NOT] BETWEEN low AND high
        let between_expr = like_expr
            .clone()
            .then(
                just(Token::Keyword(Keyword::Not))
                    .or_not()
                    .then_ignore(just(Token::Keyword(Keyword::Between)))
                    .then(like_expr.clone())
                    .then_ignore(just(Token::Keyword(Keyword::And)))
                    .then(like_expr.clone())
                    .or_not(),
            )
            .map(|(left, between_clause)| match between_clause {
                Some(((negated, low), high)) => Expr::Between {
                    expr: Box::new(left),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated: negated.is_some(),
                },
                None => left,
            });

        // IN subquery: expr [NOT] IN (SELECT ...)
        let in_subquery = between_expr
            .clone()
            .then(
                just(Token::Keyword(Keyword::Not))
                    .or_not()
                    .then_ignore(just(Token::Keyword(Keyword::In)))
                    .then(
                        subquery_select_parser()
                            .delimited_by(just(Token::LParen), just(Token::RParen)),
                    )
                    .or_not(),
            )
            .map(|(left, in_clause)| match in_clause {
                Some((negated, subquery)) => Expr::InSubquery {
                    expr: Box::new(left),
                    subquery: Box::new(subquery),
                    negated: negated.is_some(),
                },
                None => left,
            });

        // Comparison operators
        let comparison = in_subquery
            .clone()
            .then(
                just(Token::Eq)
                    .to(BinaryOp::Eq)
                    .or(just(Token::NotEq).to(BinaryOp::NotEq))
                    .or(just(Token::Lt).to(BinaryOp::Lt))
                    .or(just(Token::Gt).to(BinaryOp::Gt))
                    .or(just(Token::LtEq).to(BinaryOp::LtEq))
                    .or(just(Token::GtEq).to(BinaryOp::GtEq))
                    .then(in_subquery)
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

/// Parse a simplified SELECT statement for use in subqueries
/// This avoids infinite recursion by using simple_expr_parser instead of expr_parser
fn subquery_select_parser() -> impl Parser<Token, SelectStatement, Error = Simple<Token>> + Clone {
    let select_kw = just(Token::Keyword(Keyword::Select));
    let from_kw = just(Token::Keyword(Keyword::From));
    let where_kw = just(Token::Keyword(Keyword::Where));

    // Simple expression parser that doesn't allow nested subqueries
    let simple_literal = select! {
        Token::Integer(n) => Expr::Integer(n),
        Token::Float(f) => Expr::Float(f.value()),
        Token::String(s) => Expr::String(s),
        Token::Keyword(Keyword::True) => Expr::Boolean(true),
        Token::Keyword(Keyword::False) => Expr::Boolean(false),
        Token::Keyword(Keyword::Null) => Expr::Null,
    };

    let simple_column = identifier()
        .then(just(Token::Dot).ignore_then(identifier()).or_not())
        .map(|(first, second)| match second {
            Some(col) => Expr::Column(format!("{}.{}", first, col)),
            None => Expr::Column(first),
        });

    let simple_atom = simple_literal.or(simple_column.clone());

    // Comparison for WHERE clause in subquery
    let simple_comparison = simple_atom
        .clone()
        .then(
            just(Token::Eq)
                .to(BinaryOp::Eq)
                .or(just(Token::NotEq).to(BinaryOp::NotEq))
                .or(just(Token::Lt).to(BinaryOp::Lt))
                .or(just(Token::Gt).to(BinaryOp::Gt))
                .or(just(Token::LtEq).to(BinaryOp::LtEq))
                .or(just(Token::GtEq).to(BinaryOp::GtEq))
                .then(simple_atom.clone())
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

    // AND/OR for WHERE clause in subquery
    let simple_and = simple_comparison
        .clone()
        .then(
            just(Token::Keyword(Keyword::And))
                .ignore_then(simple_comparison)
                .repeated(),
        )
        .foldl(|left, right| Expr::BinaryOp {
            left: Box::new(left),
            op: BinaryOp::And,
            right: Box::new(right),
        });

    let simple_expr = simple_and
        .clone()
        .then(
            just(Token::Keyword(Keyword::Or))
                .ignore_then(simple_and)
                .repeated(),
        )
        .foldl(|left, right| Expr::BinaryOp {
            left: Box::new(left),
            op: BinaryOp::Or,
            right: Box::new(right),
        });

    let columns = just(Token::Star)
        .map(|_| vec![SelectColumn::Star])
        .or(simple_column
            .map(|e| SelectColumn::Expr {
                expr: e,
                alias: None,
            })
            .separated_by(just(Token::Comma))
            .at_least(1));

    let from_clause = from_kw.ignore_then(table_ref_parser()).or_not();

    let where_clause = where_kw.ignore_then(simple_expr).or_not();

    select_kw
        .ignore_then(columns)
        .then(from_clause)
        .then(where_clause)
        .map(|((columns, from), where_clause)| SelectStatement {
            with_clause: None,
            distinct: false,
            columns,
            from,
            joins: Vec::new(),
            where_clause,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
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

    // Column definitions and table constraints can be mixed
    let column_or_constraint = column_def_parser()
        .map(Either::Left)
        .or(table_constraint_parser().map(Either::Right));

    let definitions = column_or_constraint
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen));

    create_kw
        .ignore_then(table_kw)
        .ignore_then(identifier())
        .then(definitions)
        .map(|(name, defs)| {
            let mut columns = Vec::new();
            let mut constraints = Vec::new();
            for def in defs {
                match def {
                    Either::Left(col) => columns.push(col),
                    Either::Right(constraint) => constraints.push(constraint),
                }
            }
            CreateTableStatement {
                name,
                columns,
                constraints,
            }
        })
}

/// Helper enum for parsing mixed column defs and table constraints
enum Either<L, R> {
    Left(L),
    Right(R),
}

/// Parse a data type (INT, TEXT, FLOAT, etc.)
fn data_type_parser() -> impl Parser<Token, DataType, Error = Simple<Token>> + Clone {
    select! {
        Token::Keyword(Keyword::Int) => DataType::Int,
        Token::Keyword(Keyword::Integer) => DataType::Int,
        Token::Keyword(Keyword::Text) => DataType::Text,
        Token::Keyword(Keyword::Varchar) => DataType::Text,
        Token::Keyword(Keyword::Float) => DataType::Float,
        Token::Keyword(Keyword::Double) => DataType::Float,
        Token::Keyword(Keyword::Bool) => DataType::Bool,
        Token::Keyword(Keyword::Boolean) => DataType::Bool,
        Token::Keyword(Keyword::Date) => DataType::Date,
        Token::Keyword(Keyword::Time) => DataType::Time,
        Token::Keyword(Keyword::Timestamp) => DataType::Timestamp,
    }
}

/// Parse a column definition
fn column_def_parser() -> impl Parser<Token, ColumnDef, Error = Simple<Token>> + Clone {
    let data_type = data_type_parser();

    // Column constraints can appear in any order after the data type
    let not_null = just(Token::Keyword(Keyword::Not))
        .then(just(Token::Keyword(Keyword::Null)))
        .to(());

    let primary_key = just(Token::Keyword(Keyword::Primary))
        .then(just(Token::Keyword(Keyword::Key)))
        .to(());

    let unique = just(Token::Keyword(Keyword::Unique)).to(());

    let default_val = just(Token::Keyword(Keyword::Default)).ignore_then(literal_expr());

    let references = just(Token::Keyword(Keyword::References))
        .ignore_then(identifier())
        .then(
            identifier()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .then(referential_action_parser().or_not())
        .map(|((table, column), actions)| {
            let (on_delete, on_update) = actions.unwrap_or_default();
            ForeignKeyRef {
                table,
                column: column.unwrap_or_else(|| "id".to_string()),
                on_delete,
                on_update,
            }
        });

    // Parse column constraints in any order
    let constraint = not_null
        .map(|_| ColumnConstraint::NotNull)
        .or(primary_key.map(|_| ColumnConstraint::PrimaryKey))
        .or(unique.map(|_| ColumnConstraint::Unique))
        .or(default_val.map(ColumnConstraint::Default))
        .or(references.map(ColumnConstraint::References));

    identifier()
        .then(data_type)
        .then(constraint.repeated())
        .map(|((name, data_type), constraints)| {
            let mut nullable = true;
            let mut primary_key = false;
            let mut unique = false;
            let mut default = None;
            let mut references = None;

            for c in constraints {
                match c {
                    ColumnConstraint::NotNull => nullable = false,
                    ColumnConstraint::PrimaryKey => {
                        primary_key = true;
                        nullable = false; // Primary keys are implicitly NOT NULL
                    }
                    ColumnConstraint::Unique => unique = true,
                    ColumnConstraint::Default(expr) => default = Some(expr),
                    ColumnConstraint::References(fk) => references = Some(fk),
                }
            }

            ColumnDef {
                name,
                data_type,
                nullable,
                primary_key,
                unique,
                default,
                references,
            }
        })
}

/// Column constraints during parsing
enum ColumnConstraint {
    NotNull,
    PrimaryKey,
    Unique,
    Default(Expr),
    References(ForeignKeyRef),
}

/// Parse a literal expression (for DEFAULT values)
fn literal_expr() -> impl Parser<Token, Expr, Error = Simple<Token>> + Clone {
    select! {
        Token::Integer(n) => Expr::Integer(n),
        Token::Float(f) => Expr::Float(f.value()),
        Token::String(s) => Expr::String(s),
        Token::Keyword(Keyword::True) => Expr::Boolean(true),
        Token::Keyword(Keyword::False) => Expr::Boolean(false),
        Token::Keyword(Keyword::Null) => Expr::Null,
    }
}

/// Parse referential actions (ON DELETE/UPDATE)
fn referential_action_parser(
) -> impl Parser<Token, (ReferentialAction, ReferentialAction), Error = Simple<Token>> + Clone {
    let action = just(Token::Keyword(Keyword::Cascade))
        .to(ReferentialAction::Cascade)
        .or(just(Token::Keyword(Keyword::Restrict)).to(ReferentialAction::Restrict))
        .or(just(Token::Keyword(Keyword::NoAction))
            .then(just(Token::Keyword(Keyword::Action)))
            .to(ReferentialAction::NoAction))
        .or(just(Token::Keyword(Keyword::Set))
            .then(just(Token::Keyword(Keyword::Null)))
            .to(ReferentialAction::SetNull))
        .or(just(Token::Keyword(Keyword::Set))
            .then(just(Token::Keyword(Keyword::Default)))
            .to(ReferentialAction::SetDefault));

    let on_delete = just(Token::Keyword(Keyword::On))
        .then(just(Token::Keyword(Keyword::Delete)))
        .ignore_then(action.clone());

    let on_update = just(Token::Keyword(Keyword::On))
        .then(just(Token::Keyword(Keyword::Update)))
        .ignore_then(action);

    on_delete
        .or_not()
        .then(on_update.or_not())
        .map(|(del, upd)| {
            (
                del.unwrap_or(ReferentialAction::NoAction),
                upd.unwrap_or(ReferentialAction::NoAction),
            )
        })
}

/// Parse a table-level constraint
fn table_constraint_parser() -> impl Parser<Token, TableConstraint, Error = Simple<Token>> + Clone {
    let constraint_name = just(Token::Keyword(Keyword::Constraint))
        .ignore_then(identifier())
        .or_not();

    let column_list = identifier()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen));

    let primary_key = just(Token::Keyword(Keyword::Primary))
        .then(just(Token::Keyword(Keyword::Key)))
        .ignore_then(column_list.clone())
        .map(|columns| TableConstraint::PrimaryKey {
            name: None,
            columns,
        });

    let unique = just(Token::Keyword(Keyword::Unique))
        .ignore_then(column_list.clone())
        .map(|columns| TableConstraint::Unique {
            name: None,
            columns,
        });

    let foreign_key = just(Token::Keyword(Keyword::Foreign))
        .then(just(Token::Keyword(Keyword::Key)))
        .ignore_then(column_list.clone())
        .then_ignore(just(Token::Keyword(Keyword::References)))
        .then(identifier())
        .then(column_list.clone())
        .then(referential_action_parser().or_not())
        .map(|(((columns, ref_table), ref_cols), actions)| {
            let (on_delete, on_update) = actions.unwrap_or_default();
            TableConstraint::ForeignKey {
                name: None,
                columns,
                references_table: ref_table,
                references_columns: ref_cols,
                on_delete,
                on_update,
            }
        });

    constraint_name.ignore_then(primary_key.or(unique).or(foreign_key))
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

    #[test]
    fn test_parse_primary_key_constraint() {
        let result = parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert!(ct.columns[0].primary_key);
                assert!(!ct.columns[0].nullable); // PK implies NOT NULL
                assert!(!ct.columns[1].primary_key);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_not_null_constraint() {
        let result = parse("CREATE TABLE users (id INT NOT NULL, name TEXT)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(!ct.columns[0].nullable);
                assert!(ct.columns[1].nullable);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_unique_constraint() {
        let result = parse("CREATE TABLE users (id INT, email TEXT UNIQUE)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(!ct.columns[0].unique);
                assert!(ct.columns[1].unique);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_default_value() {
        let result = parse("CREATE TABLE users (id INT, active BOOL DEFAULT TRUE)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(ct.columns[0].default.is_none());
                assert_eq!(ct.columns[1].default, Some(Expr::Boolean(true)));
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_foreign_key_inline() {
        let result =
            parse("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert!(ct.columns[1].references.is_some());
                let fk = ct.columns[1].references.as_ref().unwrap();
                assert_eq!(fk.table, "users");
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_combined_constraints() {
        let result =
            parse("CREATE TABLE users (id INT PRIMARY KEY NOT NULL UNIQUE, name TEXT NOT NULL)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(ct.columns[0].primary_key);
                assert!(!ct.columns[0].nullable);
                assert!(ct.columns[0].unique);
                assert!(!ct.columns[1].nullable);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_table_level_primary_key() {
        let result = parse("CREATE TABLE users (id INT, name TEXT, PRIMARY KEY (id))");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert_eq!(ct.constraints.len(), 1);
                match &ct.constraints[0] {
                    TableConstraint::PrimaryKey { columns, .. } => {
                        assert_eq!(columns, &vec!["id".to_string()]);
                    }
                    _ => panic!("Expected PrimaryKey constraint"),
                }
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_table_level_foreign_key() {
        let result = parse(
            "CREATE TABLE orders (id INT, user_id INT, FOREIGN KEY (user_id) REFERENCES users (id))",
        );
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert_eq!(ct.constraints.len(), 1);
                match &ct.constraints[0] {
                    TableConstraint::ForeignKey {
                        columns,
                        references_table,
                        references_columns,
                        ..
                    } => {
                        assert_eq!(columns, &vec!["user_id".to_string()]);
                        assert_eq!(references_table, "users");
                        assert_eq!(references_columns, &vec!["id".to_string()]);
                    }
                    _ => panic!("Expected ForeignKey constraint"),
                }
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_group_by() {
        let result = parse("SELECT category, COUNT(*) FROM orders GROUP BY category");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.group_by.len(), 1);
                assert!(matches!(&s.group_by[0], Expr::Column(c) if c == "category"));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_group_by_multiple_columns() {
        let result = parse("SELECT a, b, SUM(c) FROM t GROUP BY a, b");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.group_by.len(), 2);
                assert!(matches!(&s.group_by[0], Expr::Column(c) if c == "a"));
                assert!(matches!(&s.group_by[1], Expr::Column(c) if c == "b"));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_having() {
        let result = parse(
            "SELECT category, SUM(amount) FROM orders GROUP BY category HAVING SUM(amount) > 100",
        );
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.group_by.len(), 1);
                assert!(s.having.is_some());
                // HAVING should be a comparison expression
                match s.having.unwrap() {
                    Expr::BinaryOp { op, left, .. } => {
                        assert_eq!(op, BinaryOp::Gt);
                        assert!(matches!(*left, Expr::Aggregate { .. }));
                    }
                    _ => panic!("Expected binary op in HAVING"),
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_full_query_with_group_by() {
        let result = parse(
            "SELECT category, SUM(amount) AS total FROM orders WHERE active = 1 GROUP BY category HAVING SUM(amount) > 100 ORDER BY total DESC LIMIT 10"
        );
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(s.from.is_some());
                assert!(s.where_clause.is_some());
                assert_eq!(s.group_by.len(), 1);
                assert!(s.having.is_some());
                assert_eq!(s.order_by.len(), 1);
                assert!(matches!(s.limit, Some(Expr::Integer(10))));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_in_subquery() {
        let result = parse("SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(s.where_clause.is_some());
                match s.where_clause.unwrap() {
                    Expr::InSubquery {
                        expr,
                        subquery,
                        negated,
                    } => {
                        assert!(!negated);
                        assert!(matches!(*expr, Expr::Column(c) if c == "id"));
                        assert_eq!(subquery.columns.len(), 1);
                    }
                    _ => panic!("Expected InSubquery"),
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_not_in_subquery() {
        let result = parse("SELECT name FROM users WHERE id NOT IN (SELECT banned_id FROM bans)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::InSubquery { negated, .. } => {
                    assert!(negated);
                }
                _ => panic!("Expected InSubquery"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_exists_subquery() {
        let result = parse("SELECT name FROM customers WHERE EXISTS (SELECT * FROM orders)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::Exists(subquery) => {
                    assert!(subquery.from.is_some());
                }
                _ => panic!("Expected Exists"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_create_index() {
        let result = parse("CREATE INDEX idx_name ON users (name)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateIndex(idx) => {
                assert_eq!(idx.name, "idx_name");
                assert_eq!(idx.table, "users");
                assert_eq!(idx.column, "name");
            }
            _ => panic!("Expected CREATE INDEX statement"),
        }
    }

    #[test]
    fn test_parse_drop_index() {
        let result = parse("DROP INDEX idx_name");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::DropIndex(name) => {
                assert_eq!(name, "idx_name");
            }
            _ => panic!("Expected DROP INDEX statement"),
        }
    }

    #[test]
    fn test_parse_select_distinct() {
        let result = parse("SELECT DISTINCT name FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(s.distinct);
                assert_eq!(s.columns.len(), 1);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_without_distinct() {
        let result = parse("SELECT name FROM users");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(!s.distinct);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_like() {
        let result = parse("SELECT * FROM users WHERE name LIKE '%john%'");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::Like { negated, .. } => {
                    assert!(!negated);
                }
                _ => panic!("Expected LIKE expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_not_like() {
        let result = parse("SELECT * FROM users WHERE name NOT LIKE '%admin%'");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::Like { negated, .. } => {
                    assert!(negated);
                }
                _ => panic!("Expected NOT LIKE expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_is_null() {
        let result = parse("SELECT * FROM users WHERE email IS NULL");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::IsNull { negated, .. } => {
                    assert!(!negated);
                }
                _ => panic!("Expected IS NULL expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_is_not_null() {
        let result = parse("SELECT * FROM users WHERE email IS NOT NULL");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::IsNull { negated, .. } => {
                    assert!(negated);
                }
                _ => panic!("Expected IS NOT NULL expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_case_when() {
        let result = parse("SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END FROM t");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expr { expr, .. } => match expr {
                    Expr::Case {
                        operand,
                        when_clauses,
                        else_result,
                    } => {
                        assert!(operand.is_none());
                        assert_eq!(when_clauses.len(), 1);
                        assert!(else_result.is_some());
                    }
                    _ => panic!("Expected CASE expression"),
                },
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_case_when_multiple() {
        let result = parse("SELECT CASE WHEN a THEN 1 WHEN b THEN 2 WHEN c THEN 3 END FROM t");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expr { expr, .. } => match expr {
                    Expr::Case { when_clauses, .. } => {
                        assert_eq!(when_clauses.len(), 3);
                    }
                    _ => panic!("Expected CASE expression"),
                },
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_between() {
        let result = parse("SELECT * FROM products WHERE price BETWEEN 10 AND 100");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::Between { negated, .. } => {
                    assert!(!negated);
                }
                _ => panic!("Expected BETWEEN expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_not_between() {
        let result = parse("SELECT * FROM products WHERE price NOT BETWEEN 10 AND 100");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(s) => match s.where_clause.unwrap() {
                Expr::Between { negated, .. } => {
                    assert!(negated);
                }
                _ => panic!("Expected BETWEEN expression"),
            },
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_union() {
        let result = parse("SELECT id FROM t1 UNION SELECT id FROM t2");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::SetOperation(s) => {
                assert!(!s.all);
                assert_eq!(s.op, SetOperator::Union);
            }
            _ => panic!("Expected SetOperation statement"),
        }
    }

    #[test]
    fn test_parse_union_all() {
        let result = parse("SELECT id FROM t1 UNION ALL SELECT id FROM t2");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::SetOperation(s) => {
                assert!(s.all);
                assert_eq!(s.op, SetOperator::Union);
            }
            _ => panic!("Expected SetOperation statement"),
        }
    }

    #[test]
    fn test_parse_chained_union() {
        let result = parse("SELECT a FROM t1 UNION SELECT b FROM t2 UNION SELECT c FROM t3");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::SetOperation(_) => {}
            _ => panic!("Expected SetOperation statement"),
        }
    }

    #[test]
    fn test_parse_intersect() {
        let result = parse("SELECT id FROM t1 INTERSECT SELECT id FROM t2");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::SetOperation(s) => {
                assert!(!s.all);
                assert_eq!(s.op, SetOperator::Intersect);
            }
            _ => panic!("Expected SetOperation statement"),
        }
    }

    #[test]
    fn test_parse_except() {
        let result = parse("SELECT id FROM t1 EXCEPT SELECT id FROM t2");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::SetOperation(s) => {
                assert!(!s.all);
                assert_eq!(s.op, SetOperator::Except);
            }
            _ => panic!("Expected SetOperation statement"),
        }
    }

    #[test]
    fn test_parse_create_procedure() {
        let result = parse("CREATE PROCEDURE my_proc (x INT, y TEXT) AS BEGIN SELECT * FROM t END");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateProcedure(proc) => {
                assert_eq!(proc.name, "my_proc");
                assert_eq!(proc.params.len(), 2);
                assert_eq!(proc.params[0].name, "x");
                assert_eq!(proc.params[0].data_type, DataType::Int);
                assert_eq!(proc.params[1].name, "y");
                assert_eq!(proc.params[1].data_type, DataType::Text);
                assert_eq!(proc.body.len(), 1);
            }
            _ => panic!("Expected CreateProcedure statement"),
        }
    }

    #[test]
    fn test_parse_create_procedure_no_params() {
        let result = parse("CREATE PROCEDURE simple AS BEGIN INSERT INTO t (id) VALUES (1) END");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CreateProcedure(proc) => {
                assert_eq!(proc.name, "simple");
                assert!(proc.params.is_empty());
                assert_eq!(proc.body.len(), 1);
            }
            _ => panic!("Expected CreateProcedure statement"),
        }
    }

    #[test]
    fn test_parse_drop_procedure() {
        let result = parse("DROP PROCEDURE my_proc");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::DropProcedure(name) => {
                assert_eq!(name, "my_proc");
            }
            _ => panic!("Expected DropProcedure statement"),
        }
    }

    #[test]
    fn test_parse_call_procedure() {
        let result = parse("CALL my_proc (1, 'hello')");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CallProcedure(call) => {
                assert_eq!(call.name, "my_proc");
                assert_eq!(call.args.len(), 2);
            }
            _ => panic!("Expected CallProcedure statement"),
        }
    }

    #[test]
    fn test_parse_exec_procedure() {
        let result = parse("EXEC my_proc (42)");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CallProcedure(call) => {
                assert_eq!(call.name, "my_proc");
                assert_eq!(call.args.len(), 1);
            }
            _ => panic!("Expected CallProcedure statement"),
        }
    }

    #[test]
    fn test_parse_execute_procedure() {
        let result = parse("EXECUTE my_proc");
        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::CallProcedure(call) => {
                assert_eq!(call.name, "my_proc");
                assert!(call.args.is_empty());
            }
            _ => panic!("Expected CallProcedure statement"),
        }
    }
}
