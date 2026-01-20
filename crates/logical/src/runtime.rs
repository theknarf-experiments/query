//! Runtime trait for executing trigger functions
//!
//! This module defines the interface that allows the logical layer to execute
//! trigger functions without knowing the specifics of the execution engine
//! (e.g., SQL vs Datalog).

use storage::{Row, TriggerEvent, TriggerTiming};

/// Context passed to trigger functions
#[derive(Debug, Clone)]
pub struct TriggerContext<'a> {
    /// The event that fired the trigger
    pub event: TriggerEvent,
    /// Whether this is a BEFORE or AFTER trigger
    pub timing: TriggerTiming,
    /// The table the trigger is attached to
    pub table: &'a str,
    /// The row before the operation (UPDATE, DELETE only)
    pub old_row: Option<&'a Row>,
    /// The row after the operation (INSERT, UPDATE only)
    pub new_row: Option<&'a Row>,
    /// Column names for the table
    pub column_names: &'a [String],
}

/// Result of executing a trigger function
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerResult {
    /// Continue with the (possibly modified) row
    /// For BEFORE INSERT/UPDATE: the row to use
    /// For AFTER triggers or DELETE: ignored
    Proceed(Option<Row>),
    /// Skip this row (BEFORE trigger returned NULL equivalent)
    /// Only valid for BEFORE triggers
    Skip,
    /// Abort the operation with an error message
    Abort(String),
}

/// Errors that can occur during trigger function execution
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    /// The specified function was not found
    FunctionNotFound(String),
    /// Error during function execution
    ExecutionError(String),
    /// Type error in function
    TypeError(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::FunctionNotFound(name) => {
                write!(f, "Function not found: {}", name)
            }
            RuntimeError::ExecutionError(msg) => {
                write!(f, "Execution error: {}", msg)
            }
            RuntimeError::TypeError(msg) => {
                write!(f, "Type error: {}", msg)
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Runtime trait for executing trigger functions
///
/// This trait is implemented by the SQL engine (or other runtimes) to provide
/// function execution capabilities to the logical layer.
pub trait Runtime<S: storage::StorageEngine> {
    /// Execute a trigger function
    ///
    /// # Arguments
    /// * `function_name` - Name of the function to execute
    /// * `context` - Trigger context with OLD/NEW rows and metadata
    /// * `storage` - Storage engine for looking up function definitions and executing SQL
    ///
    /// # Returns
    /// * `Ok(TriggerResult)` - The result of the trigger execution
    /// * `Err(RuntimeError)` - If the function couldn't be executed
    fn execute_trigger_function(
        &self,
        function_name: &str,
        context: TriggerContext,
        storage: &mut S,
    ) -> Result<TriggerResult, RuntimeError>;
}

/// A no-op runtime that doesn't execute any functions
///
/// This is useful when triggers should be ignored (e.g., during schema migrations)
/// or when no runtime is available.
pub struct NoOpRuntime;

impl<S: storage::StorageEngine> Runtime<S> for NoOpRuntime {
    fn execute_trigger_function(
        &self,
        _function_name: &str,
        _context: TriggerContext,
        _storage: &mut S,
    ) -> Result<TriggerResult, RuntimeError> {
        // Always proceed without modification
        Ok(TriggerResult::Proceed(None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_runtime_always_proceeds() {
        let runtime = NoOpRuntime;
        let mut storage = storage::MemoryEngine::new();
        let context = TriggerContext {
            event: TriggerEvent::Insert,
            timing: TriggerTiming::Before,
            table: "test",
            old_row: None,
            new_row: None,
            column_names: &[],
        };

        let result = runtime.execute_trigger_function("any_function", context, &mut storage);
        assert_eq!(result, Ok(TriggerResult::Proceed(None)));
    }

    #[test]
    fn test_trigger_result_variants() {
        let row = vec![storage::Value::Int(42)];

        let proceed = TriggerResult::Proceed(Some(row.clone()));
        assert_eq!(proceed, TriggerResult::Proceed(Some(row)));

        let skip = TriggerResult::Skip;
        assert_eq!(skip, TriggerResult::Skip);

        let abort = TriggerResult::Abort("error".to_string());
        assert_eq!(abort, TriggerResult::Abort("error".to_string()));
    }

    #[test]
    fn test_runtime_error_display() {
        let err = RuntimeError::FunctionNotFound("my_func".to_string());
        assert_eq!(format!("{}", err), "Function not found: my_func");

        let err = RuntimeError::ExecutionError("bad stuff".to_string());
        assert_eq!(format!("{}", err), "Execution error: bad stuff");

        let err = RuntimeError::TypeError("int vs string".to_string());
        assert_eq!(format!("{}", err), "Type error: int vs string");
    }
}
