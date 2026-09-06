//! Atomic replacement of small application records.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Write beside the destination, then rename so readers never see partial JSON
/// or YAML. Unique scratch names also isolate concurrent writers.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path.file_name().ok_or_else(|| io::Error::other("missing filename"))?;
    let mut scratch_name = name.to_os_string();
    scratch_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let scratch = path.with_file_name(scratch_name);
    let mut file = OpenOptions::new().create_new(true).write(true).open(&scratch)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&scratch, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&scratch);
    }
    result
}

pub fn write_if_changed(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if fs::read(path).is_ok_and(|existing| existing == bytes) {
        return Ok(());
    }
    write_atomic(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_replacement_preserves_destination_and_cleans_scratch() {
        let root = std::env::temp_dir().join(format!("llmctl-atomic-{}", std::process::id()));
        fs::create_dir_all(root.join("record")).unwrap();
        fs::write(root.join("record/keep"), b"original").unwrap();
        assert!(write_atomic(&root.join("record"), b"replacement").is_err());
        assert_eq!(fs::read(root.join("record/keep")).unwrap(), b"original");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
