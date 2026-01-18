//! SQL Value types

use serde::{Deserialize, Serialize};

/// Date value (year, month, day)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DateValue {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

impl DateValue {
    /// Create a new date value
    pub fn new(year: i32, month: u8, day: u8) -> Self {
        Self { year, month, day }
    }

    /// Parse from YYYY-MM-DD format
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let year = parts[0].parse().ok()?;
        let month = parts[1].parse().ok()?;
        let day = parts[2].parse().ok()?;
        Some(Self { year, month, day })
    }
}

impl std::fmt::Display for DateValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Time value (hour, minute, second, microseconds)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TimeValue {
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub microsecond: u32,
}

impl TimeValue {
    /// Create a new time value
    pub fn new(hour: u8, minute: u8, second: u8, microsecond: u32) -> Self {
        Self {
            hour,
            minute,
            second,
            microsecond,
        }
    }

    /// Parse from HH:MM:SS or HH:MM:SS.ffffff format
    pub fn parse(s: &str) -> Option<Self> {
        let (time_part, micro) = if let Some(idx) = s.find('.') {
            let micro_str = &s[idx + 1..];
            let micro = micro_str.parse().ok()?;
            (&s[..idx], micro)
        } else {
            (s, 0)
        };

        let parts: Vec<&str> = time_part.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        let hour = parts[0].parse().ok()?;
        let minute = parts[1].parse().ok()?;
        let second = parts[2].parse().ok()?;
        Some(Self {
            hour,
            minute,
            second,
            microsecond: micro,
        })
    }
}

impl std::fmt::Display for TimeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.microsecond > 0 {
            write!(
                f,
                "{:02}:{:02}:{:02}.{:06}",
                self.hour, self.minute, self.second, self.microsecond
            )
        } else {
            write!(f, "{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
        }
    }
}

/// Timestamp value (date + time)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TimestampValue {
    pub date: DateValue,
    pub time: TimeValue,
}

impl TimestampValue {
    /// Create a new timestamp value
    pub fn new(date: DateValue, time: TimeValue) -> Self {
        Self { date, time }
    }

    /// Parse from YYYY-MM-DD HH:MM:SS format
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.splitn(2, [' ', 'T']).collect();
        if parts.len() != 2 {
            return None;
        }
        let date = DateValue::parse(parts[0])?;
        let time = TimeValue::parse(parts[1])?;
        Some(Self { date, time })
    }
}

impl std::fmt::Display for TimestampValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.date, self.time)
    }
}

/// JSON value for structured data storage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>), // Ordered map for deterministic serialization
}

impl JsonValue {
    /// Parse a JSON string into a JsonValue
    pub fn parse(s: &str) -> Result<Self, JsonParseError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(JsonParseError::UnexpectedEnd);
        }

        let mut chars = s.chars().peekable();
        Self::parse_value(&mut chars)
    }

    fn parse_value(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        Self::skip_whitespace(chars);

        match chars.peek() {
            Some('"') => Self::parse_string(chars),
            Some('[') => Self::parse_array(chars),
            Some('{') => Self::parse_object(chars),
            Some('t') | Some('f') => Self::parse_bool(chars),
            Some('n') => Self::parse_null(chars),
            Some(c) if c.is_ascii_digit() || *c == '-' => Self::parse_number(chars),
            Some(c) => Err(JsonParseError::UnexpectedChar(*c)),
            None => Err(JsonParseError::UnexpectedEnd),
        }
    }

    fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars>) {
        while let Some(c) = chars.peek() {
            if c.is_whitespace() {
                chars.next();
            } else {
                break;
            }
        }
    }

    fn parse_string(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        chars.next(); // consume opening quote
        let mut s = String::new();
        let mut escaped = false;

        for c in chars.by_ref() {
            if escaped {
                match c {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    _ => s.push(c),
                }
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                return Ok(JsonValue::String(s));
            } else {
                s.push(c);
            }
        }
        Err(JsonParseError::UnterminatedString)
    }

    fn parse_number(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        let mut num_str = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+' {
                num_str.push(c);
                chars.next();
            } else {
                break;
            }
        }
        num_str
            .parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| JsonParseError::InvalidNumber(num_str))
    }

    fn parse_bool(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        let word: String = chars.take_while(|c| c.is_alphabetic()).collect();
        match word.as_str() {
            "true" => Ok(JsonValue::Bool(true)),
            "false" => Ok(JsonValue::Bool(false)),
            _ => Err(JsonParseError::InvalidLiteral(word)),
        }
    }

    fn parse_null(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        let word: String = chars.take_while(|c| c.is_alphabetic()).collect();
        if word == "null" {
            Ok(JsonValue::Null)
        } else {
            Err(JsonParseError::InvalidLiteral(word))
        }
    }

    fn parse_array(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        chars.next(); // consume '['
        let mut items = Vec::new();

        loop {
            Self::skip_whitespace(chars);
            if let Some(']') = chars.peek() {
                chars.next();
                return Ok(JsonValue::Array(items));
            }

            if !items.is_empty() {
                Self::skip_whitespace(chars);
                if chars.next() != Some(',') {
                    return Err(JsonParseError::ExpectedComma);
                }
                Self::skip_whitespace(chars);
            }

            items.push(Self::parse_value(chars)?);
        }
    }

    fn parse_object(
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> Result<Self, JsonParseError> {
        chars.next(); // consume '{'
        let mut pairs = Vec::new();

        loop {
            Self::skip_whitespace(chars);
            if let Some('}') = chars.peek() {
                chars.next();
                return Ok(JsonValue::Object(pairs));
            }

            if !pairs.is_empty() {
                Self::skip_whitespace(chars);
                if chars.next() != Some(',') {
                    return Err(JsonParseError::ExpectedComma);
                }
                Self::skip_whitespace(chars);
            }

            // Parse key
            let key = match Self::parse_string(chars)? {
                JsonValue::String(s) => s,
                _ => return Err(JsonParseError::ExpectedString),
            };

            Self::skip_whitespace(chars);
            if chars.next() != Some(':') {
                return Err(JsonParseError::ExpectedColon);
            }

            // Parse value
            let value = Self::parse_value(chars)?;
            pairs.push((key, value));
        }
    }

    /// Format a number, using integer format when possible
    fn format_number(n: f64) -> String {
        if n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
            (n as i64).to_string()
        } else {
            n.to_string()
        }
    }

    fn escape_string(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                _ => result.push(c),
            }
        }
        result
    }

    /// Extract a value at a JSON path (e.g., "$.foo.bar[0]")
    pub fn extract(&self, path: &str) -> Option<&JsonValue> {
        let path = path.strip_prefix('$').unwrap_or(path);
        let mut current = self;

        let mut remaining = path;
        while !remaining.is_empty() {
            remaining = remaining.trim_start_matches('.');

            if remaining.starts_with('[') {
                // Array index
                let end = remaining.find(']')?;
                let index: usize = remaining[1..end].parse().ok()?;
                remaining = &remaining[end + 1..];

                if let JsonValue::Array(arr) = current {
                    current = arr.get(index)?;
                } else {
                    return None;
                }
            } else {
                // Object key
                let end = remaining.find(['.', '[']).unwrap_or(remaining.len());
                let key = &remaining[..end];
                remaining = &remaining[end..];

                if key.is_empty() {
                    continue;
                }

                if let JsonValue::Object(pairs) = current {
                    current = pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)?;
                } else {
                    return None;
                }
            }
        }

        Some(current)
    }

    /// Get array length if this is an array
    pub fn array_length(&self) -> Option<usize> {
        match self {
            JsonValue::Array(arr) => Some(arr.len()),
            _ => None,
        }
    }
}

impl std::fmt::Display for JsonValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{}", b),
            JsonValue::Number(n) => write!(f, "{}", Self::format_number(*n)),
            JsonValue::String(s) => write!(f, "\"{}\"", Self::escape_string(s)),
            JsonValue::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            JsonValue::Object(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "\"{}\":{}", Self::escape_string(k), v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Errors that can occur during JSON parsing
#[derive(Debug, Clone, PartialEq)]
pub enum JsonParseError {
    UnexpectedEnd,
    UnexpectedChar(char),
    UnterminatedString,
    InvalidNumber(String),
    InvalidLiteral(String),
    ExpectedComma,
    ExpectedColon,
    ExpectedString,
}

impl std::fmt::Display for JsonParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonParseError::UnexpectedEnd => write!(f, "unexpected end of input"),
            JsonParseError::UnexpectedChar(c) => write!(f, "unexpected character: {}", c),
            JsonParseError::UnterminatedString => write!(f, "unterminated string"),
            JsonParseError::InvalidNumber(s) => write!(f, "invalid number: {}", s),
            JsonParseError::InvalidLiteral(s) => write!(f, "invalid literal: {}", s),
            JsonParseError::ExpectedComma => write!(f, "expected comma"),
            JsonParseError::ExpectedColon => write!(f, "expected colon"),
            JsonParseError::ExpectedString => write!(f, "expected string"),
        }
    }
}

impl std::error::Error for JsonParseError {}

/// A runtime SQL value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Date(DateValue),
    Time(TimeValue),
    Timestamp(TimestampValue),
    Json(JsonValue),
}

impl Value {
    /// Check if the value is null
    pub const fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== JsonValue Parsing Tests =====

    #[test]
    fn test_parse_null() {
        assert_eq!(JsonValue::parse("null").unwrap(), JsonValue::Null);
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(JsonValue::parse("true").unwrap(), JsonValue::Bool(true));
        assert_eq!(JsonValue::parse("false").unwrap(), JsonValue::Bool(false));
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(JsonValue::parse("42").unwrap(), JsonValue::Number(42.0));
        assert_eq!(JsonValue::parse("-17").unwrap(), JsonValue::Number(-17.0));
        assert_eq!(JsonValue::parse("3.14").unwrap(), JsonValue::Number(3.14));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(
            JsonValue::parse("\"hello\"").unwrap(),
            JsonValue::String("hello".to_string())
        );
        assert_eq!(
            JsonValue::parse("\"with\\nescapes\"").unwrap(),
            JsonValue::String("with\nescapes".to_string())
        );
    }

    #[test]
    fn test_parse_array() {
        assert_eq!(JsonValue::parse("[]").unwrap(), JsonValue::Array(vec![]));
        assert_eq!(
            JsonValue::parse("[1, 2, 3]").unwrap(),
            JsonValue::Array(vec![
                JsonValue::Number(1.0),
                JsonValue::Number(2.0),
                JsonValue::Number(3.0)
            ])
        );
    }

    #[test]
    fn test_parse_object() {
        assert_eq!(JsonValue::parse("{}").unwrap(), JsonValue::Object(vec![]));
        assert_eq!(
            JsonValue::parse("{\"a\": 1}").unwrap(),
            JsonValue::Object(vec![("a".to_string(), JsonValue::Number(1.0))])
        );
    }

    #[test]
    fn test_parse_nested() {
        let json = r#"{"f": "nest", "a": [{"f": "nest", "a": [{"t": "a", "v": "value"}]}]}"#;
        let parsed = JsonValue::parse(json).unwrap();

        // Check structure
        if let JsonValue::Object(pairs) = &parsed {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].0, "f");
            assert_eq!(pairs[0].1, JsonValue::String("nest".to_string()));
        } else {
            panic!("Expected object");
        }
    }

    // ===== JsonValue Serialization Tests =====

    #[test]
    fn test_serialize_roundtrip() {
        let values = vec![
            JsonValue::Null,
            JsonValue::Bool(true),
            JsonValue::Number(42.0),
            JsonValue::String("hello".to_string()),
            JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]),
            JsonValue::Object(vec![(
                "key".to_string(),
                JsonValue::String("val".to_string()),
            )]),
        ];

        for value in values {
            let serialized = value.to_string();
            let parsed = JsonValue::parse(&serialized).unwrap();
            assert_eq!(parsed, value, "Roundtrip failed for {:?}", value);
        }
    }

    // ===== JsonValue Extract Tests =====

    #[test]
    fn test_extract_simple() {
        let json = JsonValue::parse(r#"{"a": 1, "b": "two"}"#).unwrap();

        assert_eq!(json.extract("$.a"), Some(&JsonValue::Number(1.0)));
        assert_eq!(
            json.extract("$.b"),
            Some(&JsonValue::String("two".to_string()))
        );
        assert_eq!(json.extract("$.c"), None);
    }

    #[test]
    fn test_extract_nested() {
        let json = JsonValue::parse(r#"{"outer": {"inner": 42}}"#).unwrap();

        assert_eq!(
            json.extract("$.outer.inner"),
            Some(&JsonValue::Number(42.0))
        );
    }

    #[test]
    fn test_extract_array() {
        let json = JsonValue::parse(r#"{"arr": [1, 2, 3]}"#).unwrap();

        assert_eq!(json.extract("$.arr[0]"), Some(&JsonValue::Number(1.0)));
        assert_eq!(json.extract("$.arr[2]"), Some(&JsonValue::Number(3.0)));
        assert_eq!(json.extract("$.arr[5]"), None);
    }

    #[test]
    fn test_extract_compound_term_structure() {
        // This is what a Datalog compound term would look like
        // nest(nest(value)) encoded as JSON
        let json = JsonValue::parse(
            r#"{"f": "nest", "a": [{"f": "nest", "a": [{"t": "a", "v": "value"}]}]}"#,
        )
        .unwrap();

        // Extract functor
        assert_eq!(
            json.extract("$.f"),
            Some(&JsonValue::String("nest".to_string()))
        );

        // Extract first argument's functor
        assert_eq!(
            json.extract("$.a[0].f"),
            Some(&JsonValue::String("nest".to_string()))
        );

        // Extract innermost value
        assert_eq!(
            json.extract("$.a[0].a[0].v"),
            Some(&JsonValue::String("value".to_string()))
        );
    }

    // ===== JsonValue Array Length Tests =====

    #[test]
    fn test_array_length() {
        let arr = JsonValue::Array(vec![
            JsonValue::Number(1.0),
            JsonValue::Number(2.0),
            JsonValue::Number(3.0),
        ]);
        assert_eq!(arr.array_length(), Some(3));

        let not_arr = JsonValue::String("hello".to_string());
        assert_eq!(not_arr.array_length(), None);
    }
}
