//! Import/Export functionality for CSV and JSON formats

use csv::{Reader, Writer};
use serde_json;
use sql_storage::Value;
use std::io::{Read, Write};

/// Export result containing column names and data
#[derive(Debug, Clone)]
pub struct ExportData {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Export data to CSV format
pub fn export_csv<W: Write>(data: &ExportData, writer: W) -> Result<(), ExportError> {
    let mut wtr = Writer::from_writer(writer);

    // Write header
    wtr.write_record(&data.columns)
        .map_err(|e| ExportError::Csv(e.to_string()))?;

    // Write data rows
    for row in &data.rows {
        let record: Vec<String> = row.iter().map(value_to_string).collect();
        wtr.write_record(&record)
            .map_err(|e| ExportError::Csv(e.to_string()))?;
    }

    wtr.flush().map_err(|e| ExportError::Csv(e.to_string()))?;
    Ok(())
}

/// Export data to JSON format
pub fn export_json<W: Write>(data: &ExportData, mut writer: W) -> Result<(), ExportError> {
    let rows: Vec<serde_json::Map<String, serde_json::Value>> = data
        .rows
        .iter()
        .map(|row| {
            data.columns
                .iter()
                .zip(row.iter())
                .map(|(col, val)| (col.clone(), value_to_json(val)))
                .collect()
        })
        .collect();

    let json = serde_json::to_string_pretty(&rows).map_err(|e| ExportError::Json(e.to_string()))?;
    writer
        .write_all(json.as_bytes())
        .map_err(|e| ExportError::Json(e.to_string()))?;
    Ok(())
}

/// Import data from CSV format
pub fn import_csv<R: Read>(reader: R, has_header: bool) -> Result<ImportData, ImportError> {
    let mut rdr = Reader::from_reader(reader);

    let columns = if has_header {
        rdr.headers()
            .map_err(|e| ImportError::Csv(e.to_string()))?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result.map_err(|e| ImportError::Csv(e.to_string()))?;
        let row: Vec<Value> = record.iter().map(string_to_value).collect();
        rows.push(row);
    }

    Ok(ImportData { columns, rows })
}

/// Import data from JSON format
pub fn import_json<R: Read>(mut reader: R) -> Result<ImportData, ImportError> {
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .map_err(|e| ImportError::Json(e.to_string()))?;

    let rows: Vec<serde_json::Map<String, serde_json::Value>> =
        serde_json::from_str(&content).map_err(|e| ImportError::Json(e.to_string()))?;

    if rows.is_empty() {
        return Ok(ImportData {
            columns: Vec::new(),
            rows: Vec::new(),
        });
    }

    // Get column names from first row
    let columns: Vec<String> = rows[0].keys().cloned().collect();

    // Convert JSON values to Value
    let data_rows: Vec<Vec<Value>> = rows
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|col| json_to_value(row.get(col)))
                .collect()
        })
        .collect();

    Ok(ImportData {
        columns,
        rows: data_rows,
    })
}

/// Imported data
#[derive(Debug, Clone)]
pub struct ImportData {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Export error
#[derive(Debug, Clone, PartialEq)]
pub enum ExportError {
    Csv(String),
    Json(String),
}

/// Import error
#[derive(Debug, Clone, PartialEq)]
pub enum ImportError {
    Csv(String),
    Json(String),
}

/// Convert Value to string for CSV
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => s.clone(),
        Value::Date(d) => d.to_string(),
        Value::Time(t) => t.to_string(),
        Value::Timestamp(ts) => ts.to_string(),
    }
}

/// Convert Value to JSON value
fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::Value::Number((*i).into()),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::Text(s) => serde_json::Value::String(s.clone()),
        Value::Date(d) => serde_json::Value::String(d.to_string()),
        Value::Time(t) => serde_json::Value::String(t.to_string()),
        Value::Timestamp(ts) => serde_json::Value::String(ts.to_string()),
    }
}

/// Convert string to Value (with type inference)
fn string_to_value(s: &str) -> Value {
    if s.is_empty() {
        return Value::Null;
    }

    // Try parsing as integer
    if let Ok(i) = s.parse::<i64>() {
        return Value::Int(i);
    }

    // Try parsing as float
    if let Ok(f) = s.parse::<f64>() {
        return Value::Float(f);
    }

    // Try parsing as boolean
    match s.to_lowercase().as_str() {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        _ => {}
    }

    // Default to text
    Value::Text(s.to_string())
}

/// Convert JSON value to Value
fn json_to_value(json: Option<&serde_json::Value>) -> Value {
    match json {
        None | Some(serde_json::Value::Null) => Value::Null,
        Some(serde_json::Value::Bool(b)) => Value::Bool(*b),
        Some(serde_json::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        Some(serde_json::Value::String(s)) => Value::Text(s.clone()),
        Some(serde_json::Value::Array(_)) | Some(serde_json::Value::Object(_)) => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_csv() {
        let data = ExportData {
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                vec![Value::Int(1), Value::Text("alice".to_string())],
                vec![Value::Int(2), Value::Text("bob".to_string())],
            ],
        };

        let mut output = Vec::new();
        export_csv(&data, &mut output).unwrap();

        let csv = String::from_utf8(output).unwrap();
        assert!(csv.contains("id,name"));
        assert!(csv.contains("1,alice"));
        assert!(csv.contains("2,bob"));
    }

    #[test]
    fn test_import_csv() {
        let csv = "id,name\n1,alice\n2,bob\n";
        let result = import_csv(csv.as_bytes(), true).unwrap();

        assert_eq!(result.columns, vec!["id", "name"]);
        assert_eq!(result.rows.len(), 2);
        assert_eq!(
            result.rows[0],
            vec![Value::Int(1), Value::Text("alice".to_string())]
        );
        assert_eq!(
            result.rows[1],
            vec![Value::Int(2), Value::Text("bob".to_string())]
        );
    }

    #[test]
    fn test_export_json() {
        let data = ExportData {
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                vec![Value::Int(1), Value::Text("alice".to_string())],
                vec![Value::Int(2), Value::Text("bob".to_string())],
            ],
        };

        let mut output = Vec::new();
        export_json(&data, &mut output).unwrap();

        let json = String::from_utf8(output).unwrap();
        assert!(json.contains("\"id\": 1"));
        assert!(json.contains("\"name\": \"alice\""));
    }

    #[test]
    fn test_import_json() {
        let json = r#"[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]"#;
        let result = import_json(json.as_bytes()).unwrap();

        assert_eq!(result.rows.len(), 2);
        // Note: JSON object keys may be in any order
        assert!(result.columns.contains(&"id".to_string()));
        assert!(result.columns.contains(&"name".to_string()));
    }

    #[test]
    fn test_csv_type_inference() {
        let csv = "int,float,bool,text,null\n42,3.14,true,hello,\n";
        let result = import_csv(csv.as_bytes(), true).unwrap();

        assert_eq!(result.rows[0][0], Value::Int(42));
        assert_eq!(result.rows[0][1], Value::Float(3.14));
        assert_eq!(result.rows[0][2], Value::Bool(true));
        assert_eq!(result.rows[0][3], Value::Text("hello".to_string()));
        assert_eq!(result.rows[0][4], Value::Null);
    }

    #[test]
    fn test_csv_null_values() {
        let data = ExportData {
            columns: vec!["a".to_string(), "b".to_string()],
            rows: vec![vec![Value::Int(1), Value::Null]],
        };

        let mut output = Vec::new();
        export_csv(&data, &mut output).unwrap();

        let csv = String::from_utf8(output).unwrap();
        assert!(csv.contains("1,"));
    }

    #[test]
    fn test_json_null_values() {
        let data = ExportData {
            columns: vec!["a".to_string(), "b".to_string()],
            rows: vec![vec![Value::Int(1), Value::Null]],
        };

        let mut output = Vec::new();
        export_json(&data, &mut output).unwrap();

        let json = String::from_utf8(output).unwrap();
        assert!(json.contains("null"));
    }
}
