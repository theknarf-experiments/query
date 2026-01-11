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
}

impl Value {
    /// Check if the value is null
    pub const fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}
