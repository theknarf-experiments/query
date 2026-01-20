//! Metadata types for functions and triggers
//!
//! These types represent stored functions and triggers that are persisted
//! in special metadata tables (_functions and _triggers).

/// Definition of a stored function
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    /// Function name (unique identifier)
    pub name: String,
    /// Parameter definitions as JSON: [{"name": "x", "type": "INT"}, ...]
    pub params: String,
    /// Function body (SQL statements)
    pub body: String,
    /// Language identifier (e.g., "sql")
    pub language: String,
}

/// Definition of a trigger
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerDef {
    /// Trigger name (unique identifier)
    pub name: String,
    /// Table the trigger is attached to
    pub table_name: String,
    /// When the trigger fires: "BEFORE" or "AFTER"
    pub timing: TriggerTiming,
    /// Events that fire the trigger
    pub events: Vec<TriggerEvent>,
    /// Name of the function to execute
    pub function_name: String,
}

/// When the trigger fires relative to the operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
}

impl TriggerTiming {
    pub fn as_str(&self) -> &'static str {
        match self {
            TriggerTiming::Before => "BEFORE",
            TriggerTiming::After => "AFTER",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BEFORE" => Some(TriggerTiming::Before),
            "AFTER" => Some(TriggerTiming::After),
            _ => None,
        }
    }
}

/// Event that can fire a trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
}

impl TriggerEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            TriggerEvent::Insert => "INSERT",
            TriggerEvent::Update => "UPDATE",
            TriggerEvent::Delete => "DELETE",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "INSERT" => Some(TriggerEvent::Insert),
            "UPDATE" => Some(TriggerEvent::Update),
            "DELETE" => Some(TriggerEvent::Delete),
            _ => None,
        }
    }
}

/// Serialize events to JSON array string
pub fn events_to_json(events: &[TriggerEvent]) -> String {
    let event_strs: Vec<&str> = events.iter().map(|e| e.as_str()).collect();
    format!(
        "[{}]",
        event_strs
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Deserialize events from JSON array string
pub fn events_from_json(json: &str) -> Vec<TriggerEvent> {
    // Simple parser for ["INSERT", "UPDATE", "DELETE"]
    let trimmed = json.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return vec![];
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    inner
        .split(',')
        .filter_map(|s| {
            let s = s.trim().trim_matches('"');
            TriggerEvent::parse(s)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_timing_roundtrip() {
        assert_eq!(
            TriggerTiming::parse(TriggerTiming::Before.as_str()),
            Some(TriggerTiming::Before)
        );
        assert_eq!(
            TriggerTiming::parse(TriggerTiming::After.as_str()),
            Some(TriggerTiming::After)
        );
        assert_eq!(TriggerTiming::parse("before"), Some(TriggerTiming::Before));
        assert_eq!(TriggerTiming::parse("invalid"), None);
    }

    #[test]
    fn test_trigger_event_roundtrip() {
        assert_eq!(
            TriggerEvent::parse(TriggerEvent::Insert.as_str()),
            Some(TriggerEvent::Insert)
        );
        assert_eq!(
            TriggerEvent::parse(TriggerEvent::Update.as_str()),
            Some(TriggerEvent::Update)
        );
        assert_eq!(
            TriggerEvent::parse(TriggerEvent::Delete.as_str()),
            Some(TriggerEvent::Delete)
        );
        assert_eq!(TriggerEvent::parse("insert"), Some(TriggerEvent::Insert));
        assert_eq!(TriggerEvent::parse("invalid"), None);
    }

    #[test]
    fn test_events_json_roundtrip() {
        let events = vec![TriggerEvent::Insert, TriggerEvent::Update];
        let json = events_to_json(&events);
        assert_eq!(json, r#"["INSERT","UPDATE"]"#);

        let parsed = events_from_json(&json);
        assert_eq!(parsed, events);
    }

    #[test]
    fn test_events_json_all_events() {
        let events = vec![
            TriggerEvent::Insert,
            TriggerEvent::Update,
            TriggerEvent::Delete,
        ];
        let json = events_to_json(&events);
        let parsed = events_from_json(&json);
        assert_eq!(parsed, events);
    }

    #[test]
    fn test_events_json_empty() {
        let events: Vec<TriggerEvent> = vec![];
        let json = events_to_json(&events);
        assert_eq!(json, "[]");

        let parsed = events_from_json(&json);
        assert!(parsed.is_empty());
    }
}
