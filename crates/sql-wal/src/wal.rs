//! Write-Ahead Log implementation

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Result type for WAL operations
pub type WalResult<T> = Result<T, WalError>;

/// WAL error types
#[derive(Debug, Clone, PartialEq)]
pub enum WalError {
    /// I/O error
    Io(String),
    /// Serialization error
    Serialization(String),
    /// Corrupted log entry
    Corrupted(String),
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalError::Io(msg) => write!(f, "I/O error: {}", msg),
            WalError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            WalError::Corrupted(msg) => write!(f, "Corrupted log: {}", msg),
        }
    }
}

impl std::error::Error for WalError {}

/// Database operation to be logged
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// Insert a row into a table
    Insert { table: String, row_data: Vec<u8> },
    /// Update rows in a table
    Update {
        table: String,
        old_data: Vec<u8>,
        new_data: Vec<u8>,
    },
    /// Delete rows from a table
    Delete { table: String, row_data: Vec<u8> },
    /// Create a table
    CreateTable { name: String, schema: Vec<u8> },
    /// Drop a table
    DropTable { name: String },
    /// Checkpoint marker
    Checkpoint { lsn: u64 },
}

/// A single log entry in the WAL
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log sequence number
    pub lsn: u64,
    /// The operation performed
    pub operation: Operation,
    /// CRC32 checksum for integrity
    pub checksum: u32,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(lsn: u64, operation: Operation) -> Self {
        let mut entry = Self {
            lsn,
            operation,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32 checksum of the entry (excluding checksum field)
    fn compute_checksum(&self) -> u32 {
        let data = bincode::serialize(&(&self.lsn, &self.operation)).unwrap_or_default();
        crc32(&data)
    }

    /// Verify the entry's checksum
    pub fn verify(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

/// Simple CRC32 implementation
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Write-Ahead Log
pub struct Wal {
    /// Path to the WAL file
    path: std::path::PathBuf,
    /// Current log sequence number
    current_lsn: u64,
    /// File writer
    writer: Option<BufWriter<File>>,
}

impl Wal {
    /// Create a new WAL at the given path
    pub fn new<P: AsRef<Path>>(path: P) -> WalResult<Self> {
        let path = path.as_ref().to_path_buf();

        // Open or create the WAL file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| WalError::Io(e.to_string()))?;

        let writer = BufWriter::new(file);

        let mut wal = Self {
            path,
            current_lsn: 0,
            writer: Some(writer),
        };

        // Recover the last LSN from existing entries
        wal.recover_lsn()?;

        Ok(wal)
    }

    /// Recover the last LSN from the WAL file
    fn recover_lsn(&mut self) -> WalResult<()> {
        let entries = self.read_all()?;
        if let Some(last) = entries.last() {
            self.current_lsn = last.lsn;
        }
        Ok(())
    }

    /// Append an operation to the WAL
    pub fn append(&mut self, operation: Operation) -> WalResult<u64> {
        self.current_lsn += 1;
        let entry = LogEntry::new(self.current_lsn, operation);

        let data =
            bincode::serialize(&entry).map_err(|e| WalError::Serialization(e.to_string()))?;

        // Write length prefix followed by data
        let len = data.len() as u32;
        let len_bytes = len.to_le_bytes();

        if let Some(writer) = &mut self.writer {
            writer
                .write_all(&len_bytes)
                .map_err(|e| WalError::Io(e.to_string()))?;
            writer
                .write_all(&data)
                .map_err(|e| WalError::Io(e.to_string()))?;
            writer.flush().map_err(|e| WalError::Io(e.to_string()))?;
        }

        Ok(self.current_lsn)
    }

    /// Sync the WAL to disk
    pub fn sync(&mut self) -> WalResult<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush().map_err(|e| WalError::Io(e.to_string()))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| WalError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Read all entries from the WAL
    pub fn read_all(&self) -> WalResult<Vec<LogEntry>> {
        let file =
            File::open(&self.path).map_err(|e| WalError::Io(format!("Cannot open WAL: {}", e)))?;

        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut len_buf = [0u8; 4];

        loop {
            // Read length prefix
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(WalError::Io(e.to_string())),
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];

            reader
                .read_exact(&mut data)
                .map_err(|e| WalError::Io(e.to_string()))?;

            let entry: LogEntry =
                bincode::deserialize(&data).map_err(|e| WalError::Serialization(e.to_string()))?;

            // Verify checksum
            if !entry.verify() {
                return Err(WalError::Corrupted(format!(
                    "Checksum mismatch for LSN {}",
                    entry.lsn
                )));
            }

            entries.push(entry);
        }

        Ok(entries)
    }

    /// Read entries from a specific LSN onwards
    pub fn read_from(&self, from_lsn: u64) -> WalResult<Vec<LogEntry>> {
        let all = self.read_all()?;
        Ok(all.into_iter().filter(|e| e.lsn >= from_lsn).collect())
    }

    /// Write a checkpoint marker
    pub fn checkpoint(&mut self) -> WalResult<u64> {
        let checkpoint_lsn = self.current_lsn;
        self.append(Operation::Checkpoint {
            lsn: checkpoint_lsn,
        })
    }

    /// Truncate the WAL up to a checkpoint
    pub fn truncate(&mut self, up_to_lsn: u64) -> WalResult<()> {
        // Read all entries
        let entries: Vec<LogEntry> = self
            .read_all()?
            .into_iter()
            .filter(|e| e.lsn > up_to_lsn)
            .collect();

        // Close the writer
        self.writer = None;

        // Truncate and rewrite
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| WalError::Io(e.to_string()))?;

        let mut writer = BufWriter::new(file);

        for entry in &entries {
            let data =
                bincode::serialize(entry).map_err(|e| WalError::Serialization(e.to_string()))?;
            let len = data.len() as u32;
            writer
                .write_all(&len.to_le_bytes())
                .map_err(|e| WalError::Io(e.to_string()))?;
            writer
                .write_all(&data)
                .map_err(|e| WalError::Io(e.to_string()))?;
        }

        writer.flush().map_err(|e| WalError::Io(e.to_string()))?;

        // Reopen for appending
        let file = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| WalError::Io(e.to_string()))?;

        self.writer = Some(BufWriter::new(file));

        Ok(())
    }

    /// Get the current LSN
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_wal() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let wal = Wal::new(&wal_path);
        assert!(wal.is_ok());
        assert_eq!(wal.unwrap().current_lsn(), 0);
    }

    #[test]
    fn test_append_and_read() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = Wal::new(&wal_path).unwrap();

        // Append some operations
        let lsn1 = wal
            .append(Operation::Insert {
                table: "users".to_string(),
                row_data: vec![1, 2, 3],
            })
            .unwrap();

        let lsn2 = wal
            .append(Operation::Insert {
                table: "users".to_string(),
                row_data: vec![4, 5, 6],
            })
            .unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);

        // Read all entries
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lsn, 1);
        assert_eq!(entries[1].lsn, 2);
    }

    #[test]
    fn test_recovery() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Create and write to WAL
        {
            let mut wal = Wal::new(&wal_path).unwrap();
            wal.append(Operation::Insert {
                table: "t".to_string(),
                row_data: vec![1],
            })
            .unwrap();
            wal.append(Operation::Insert {
                table: "t".to_string(),
                row_data: vec![2],
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Reopen and verify LSN recovery
        let wal = Wal::new(&wal_path).unwrap();
        assert_eq!(wal.current_lsn(), 2);

        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_checksum_verification() {
        let entry = LogEntry::new(
            1,
            Operation::Insert {
                table: "test".to_string(),
                row_data: vec![1, 2, 3],
            },
        );
        assert!(entry.verify());

        // Corrupt the entry
        let mut corrupted = entry.clone();
        corrupted.lsn = 999;
        assert!(!corrupted.verify());
    }

    #[test]
    fn test_checkpoint_and_truncate() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = Wal::new(&wal_path).unwrap();

        // Write some entries
        wal.append(Operation::Insert {
            table: "t".to_string(),
            row_data: vec![1],
        })
        .unwrap();
        wal.append(Operation::Insert {
            table: "t".to_string(),
            row_data: vec![2],
        })
        .unwrap();

        // Checkpoint
        let checkpoint_lsn = wal.checkpoint().unwrap();
        assert_eq!(checkpoint_lsn, 3);

        // More entries
        wal.append(Operation::Insert {
            table: "t".to_string(),
            row_data: vec![3],
        })
        .unwrap();

        // Truncate up to checkpoint (entries 1 and 2 are before checkpoint)
        wal.truncate(2).unwrap();

        // Only checkpoint and entry 4 should remain
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].lsn >= 3);
    }

    #[test]
    fn test_read_from_lsn() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = Wal::new(&wal_path).unwrap();

        for i in 1..=5 {
            wal.append(Operation::Insert {
                table: "t".to_string(),
                row_data: vec![i],
            })
            .unwrap();
        }

        let entries = wal.read_from(3).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].lsn, 3);
        assert_eq!(entries[1].lsn, 4);
        assert_eq!(entries[2].lsn, 5);
    }

    #[test]
    fn test_various_operations() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = Wal::new(&wal_path).unwrap();

        wal.append(Operation::CreateTable {
            name: "users".to_string(),
            schema: vec![1, 2, 3],
        })
        .unwrap();

        wal.append(Operation::Insert {
            table: "users".to_string(),
            row_data: vec![4, 5, 6],
        })
        .unwrap();

        wal.append(Operation::Update {
            table: "users".to_string(),
            old_data: vec![4, 5, 6],
            new_data: vec![7, 8, 9],
        })
        .unwrap();

        wal.append(Operation::Delete {
            table: "users".to_string(),
            row_data: vec![7, 8, 9],
        })
        .unwrap();

        wal.append(Operation::DropTable {
            name: "users".to_string(),
        })
        .unwrap();

        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 5);

        assert!(matches!(
            entries[0].operation,
            Operation::CreateTable { .. }
        ));
        assert!(matches!(entries[1].operation, Operation::Insert { .. }));
        assert!(matches!(entries[2].operation, Operation::Update { .. }));
        assert!(matches!(entries[3].operation, Operation::Delete { .. }));
        assert!(matches!(entries[4].operation, Operation::DropTable { .. }));
    }
}
