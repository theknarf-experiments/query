//! SQL Lexer using Chumsky
//!
//! This crate provides lexical analysis for SQL statements,
//! converting raw SQL text into a stream of tokens.

mod lexer;
mod span;
mod token;

pub use lexer::lexer;
pub use span::Span;
pub use token::{FloatBits, Keyword, Token};

/// Result of lexing - either a list of spanned tokens or errors
pub type LexResult = Result<Vec<(Token, Span)>, Vec<LexError>>;

/// A lexical error with location information
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_select_star() {
        let input = "SELECT * FROM users";
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
    fn test_lex_select_columns() {
        let input = "SELECT id, name FROM users";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Identifier("id".to_string()),
                Token::Comma,
                Token::Identifier("name".to_string()),
                Token::Keyword(Keyword::From),
                Token::Identifier("users".to_string()),
            ]
        );
    }

    #[test]
    fn test_lex_string_literal() {
        let input = "SELECT 'hello world'";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::String("hello world".to_string()),
            ]
        );
    }

    #[test]
    fn test_lex_integer() {
        let input = "SELECT 42";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![Token::Keyword(Keyword::Select), Token::Integer(42),]
        );
    }

    #[test]
    fn test_lex_float() {
        let input = "SELECT 3.14";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Float(FloatBits::new(3.14)),
            ]
        );
    }

    #[test]
    fn test_lex_where_clause() {
        let input = "SELECT * FROM users WHERE id = 1";
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
                Token::Keyword(Keyword::Where),
                Token::Identifier("id".to_string()),
                Token::Eq,
                Token::Integer(1),
            ]
        );
    }

    #[test]
    fn test_lex_comparison_operators() {
        let input = "a < b > c <= d >= e <> f != g";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::Lt,
                Token::Identifier("b".to_string()),
                Token::Gt,
                Token::Identifier("c".to_string()),
                Token::LtEq,
                Token::Identifier("d".to_string()),
                Token::GtEq,
                Token::Identifier("e".to_string()),
                Token::NotEq,
                Token::Identifier("f".to_string()),
                Token::NotEq,
                Token::Identifier("g".to_string()),
            ]
        );
    }

    #[test]
    fn test_lex_insert() {
        let input = "INSERT INTO users (id, name) VALUES (1, 'john')";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Insert),
                Token::Keyword(Keyword::Into),
                Token::Identifier("users".to_string()),
                Token::LParen,
                Token::Identifier("id".to_string()),
                Token::Comma,
                Token::Identifier("name".to_string()),
                Token::RParen,
                Token::Keyword(Keyword::Values),
                Token::LParen,
                Token::Integer(1),
                Token::Comma,
                Token::String("john".to_string()),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn test_lex_create_table() {
        let input = "CREATE TABLE users (id INT, name TEXT)";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Create),
                Token::Keyword(Keyword::Table),
                Token::Identifier("users".to_string()),
                Token::LParen,
                Token::Identifier("id".to_string()),
                Token::Keyword(Keyword::Int),
                Token::Comma,
                Token::Identifier("name".to_string()),
                Token::Keyword(Keyword::Text),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn test_lex_case_insensitive_keywords() {
        let input = "select FROM Where";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Keyword(Keyword::From),
                Token::Keyword(Keyword::Where),
            ]
        );
    }

    #[test]
    fn test_lex_semicolon() {
        let input = "SELECT 1; SELECT 2;";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens: Vec<_> = result.unwrap().into_iter().map(|(t, _)| t).collect();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Integer(1),
                Token::Semicolon,
                Token::Keyword(Keyword::Select),
                Token::Integer(2),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn test_span_tracking() {
        let input = "SELECT";
        let result = lexer(input);
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].1, Span::new(0, 6));
    }
}
