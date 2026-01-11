//! SQL Lexer implementation using Chumsky 0.9

use crate::{Keyword, LexError, LexResult, Span, Token};
use chumsky::prelude::*;

/// Create the SQL lexer
pub fn lexer(input: &str) -> LexResult {
    let result = lexer_parser().parse(input);

    match result {
        Ok(tokens) => Ok(tokens),
        Err(errors) => Err(errors
            .into_iter()
            .map(|e| LexError {
                message: format!("Unexpected character: {:?}", e.found()),
                span: Span::new(e.span().start, e.span().end),
            })
            .collect()),
    }
}

/// Build the lexer parser
fn lexer_parser() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    // Integer literal
    let integer = text::int(10).map(|s: String| Token::Integer(s.parse().unwrap()));

    // Float literal (integer followed by . and digits)
    let float = text::int(10)
        .chain::<char, _, _>(just('.'))
        .chain::<char, _, _>(text::digits(10))
        .collect::<String>()
        .map(|s| Token::Float(s.parse().unwrap()));

    // String literal (single-quoted)
    let string = just('\'')
        .ignore_then(filter(|c| *c != '\'').repeated())
        .then_ignore(just('\''))
        .collect::<String>()
        .map(Token::String);

    // Identifier or keyword
    let ident_or_keyword = text::ident().map(|s: String| {
        if let Some(kw) = Keyword::from_str(&s) {
            Token::Keyword(kw)
        } else {
            Token::Identifier(s)
        }
    });

    // Multi-character operators (must be tried before single-char ones)
    let not_eq = just('<')
        .then(just('>'))
        .to(Token::NotEq)
        .or(just('!').then(just('=')).to(Token::NotEq));
    let lt_eq = just('<').then(just('=')).to(Token::LtEq);
    let gt_eq = just('>').then(just('=')).to(Token::GtEq);

    // Single-character tokens
    let single_char = choice((
        just('(').to(Token::LParen),
        just(')').to(Token::RParen),
        just(',').to(Token::Comma),
        just(';').to(Token::Semicolon),
        just('.').to(Token::Dot),
        just('*').to(Token::Star),
        just('+').to(Token::Plus),
        just('-').to(Token::Minus),
        just('/').to(Token::Slash),
        just('%').to(Token::Percent),
        just('=').to(Token::Eq),
        just('<').to(Token::Lt),
        just('>').to(Token::Gt),
    ));

    // Comments (-- to end of line)
    let comment = just('-')
        .then(just('-'))
        .then(take_until(text::newline().or(end())))
        .padded();

    let token = choice((
        not_eq,
        lt_eq,
        gt_eq,
        float,
        integer,
        string,
        ident_or_keyword,
        single_char,
    ));

    token
        .map_with_span(|tok, span: std::ops::Range<usize>| (tok, Span::new(span.start, span.end)))
        .padded_by(comment.repeated())
        .padded()
        .repeated()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comments_are_ignored() {
        let input = "SELECT -- this is a comment\n* FROM users";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Star,
                Token::Keyword(Keyword::From),
                Token::Identifier("users".to_string()),
            ]
        );
    }

    #[test]
    fn test_negative_integer() {
        // Note: negative numbers are parsed as minus followed by integer
        // The parser will handle combining them
        let input = "-42";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![Token::Minus, Token::Integer(42),]);
    }

    #[test]
    fn test_arithmetic_operators() {
        let input = "1 + 2 - 3 * 4 / 5 % 6";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Integer(1),
                Token::Plus,
                Token::Integer(2),
                Token::Minus,
                Token::Integer(3),
                Token::Star,
                Token::Integer(4),
                Token::Slash,
                Token::Integer(5),
                Token::Percent,
                Token::Integer(6),
            ]
        );
    }
}
