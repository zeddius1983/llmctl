//! Minimal GGUF header reader.
//!
//! GGUF stores model metadata in a key/value section right after a small
//! header, before the tensor data. We read only that section (a few KB to a
//! few MB of token tables, skipped without allocating) to extract architecture,
//! context length, quantization, and whether a chat template is embedded.
//!
//! Layout (little-endian, GGUF v2/v3):
//! ```text
//!   magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
//!   kv_count × { key: string | value_type: u32 | value }
//!   string := len u64 | bytes
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use anyhow::{Result, bail};

const MAGIC: &[u8; 4] = b"GGUF";

// GGUF value type tags.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// Metadata extracted from a GGUF header.
#[derive(Debug, Clone, Default)]
pub struct GgufInfo {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    /// Quantization label derived from `general.file_type`, if present.
    pub file_type_label: Option<String>,
    pub has_chat_template: bool,
    pub has_mtp: bool,
}

/// Scalar metadata value (arrays are skipped, not stored).
enum Scalar {
    U(u64),
    S(String),
}

impl Scalar {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Scalar::U(v) => Some(*v),
            Scalar::S(_) => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Scalar::S(s) => Some(s),
            Scalar::U(_) => None,
        }
    }
}

/// Read and parse the GGUF metadata section of `path`.
pub fn read_gguf_info(path: &Path) -> Result<GgufInfo> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not a GGUF file");
    }
    let version = read_u32(&mut r)?;
    if version < 2 {
        // v1 used 32-bit lengths/counts; not worth supporting.
        bail!("unsupported GGUF version {version}");
    }
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;

    let mut kv: HashMap<String, Scalar> = HashMap::new();
    let mut has_chat_template = false;
    let mut has_mtp = false;

    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let vtype = read_u32(&mut r)?;
        match read_value(&mut r, vtype)? {
            Some(scalar) => {
                if key == "tokenizer.chat_template" {
                    has_chat_template = true;
                }
                if key.ends_with(".nextn_predict_layers")
                    && scalar.as_u64().is_some_and(|layers| layers > 0)
                {
                    has_mtp = true;
                }
                // Only retain keys we actually consult.
                if key == "general.architecture"
                    || key == "general.file_type"
                    || key.ends_with(".context_length")
                {
                    kv.insert(key, scalar);
                }
            }
            None => {
                // Arrays are skipped; a chat template is never an array.
            }
        }
    }

    let architecture = kv.get("general.architecture").and_then(|v| v.as_str()).map(String::from);
    let context_length = architecture
        .as_ref()
        .and_then(|arch| kv.get(&format!("{arch}.context_length")))
        .or_else(|| kv.iter().find(|(k, _)| k.ends_with(".context_length")).map(|(_, v)| v))
        .and_then(|v| v.as_u64());
    let file_type_label =
        kv.get("general.file_type").and_then(|v| v.as_u64()).and_then(file_type_label);

    Ok(GgufInfo { architecture, context_length, file_type_label, has_chat_template, has_mtp })
}

/// Read one value of the given type. Returns `None` for arrays (skipped) and
/// for scalar types we don't keep but still must consume.
fn read_value<R: Read>(r: &mut R, vtype: u32) -> Result<Option<Scalar>> {
    let scalar = match vtype {
        T_UINT8 => Scalar::U(read_n::<R, 1>(r)?[0] as u64),
        T_INT8 => Scalar::U(read_n::<R, 1>(r)?[0] as i8 as u64),
        T_BOOL => Scalar::U((read_n::<R, 1>(r)?[0] != 0) as u64),
        T_UINT16 => Scalar::U(u16::from_le_bytes(read_n(r)?) as u64),
        T_INT16 => Scalar::U(i16::from_le_bytes(read_n(r)?) as u64),
        T_UINT32 => Scalar::U(read_u32(r)? as u64),
        T_INT32 => Scalar::U(i32::from_le_bytes(read_n(r)?) as u64),
        T_FLOAT32 => {
            let _ = read_u32(r)?;
            return Ok(None);
        }
        T_UINT64 => Scalar::U(read_u64(r)?),
        T_INT64 => Scalar::U(i64::from_le_bytes(read_n(r)?) as u64),
        T_FLOAT64 => {
            let _ = read_u64(r)?;
            return Ok(None);
        }
        T_STRING => Scalar::S(read_string(r)?),
        T_ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            skip_array(r, elem_type, count)?;
            return Ok(None);
        }
        other => bail!("unknown GGUF value type {other}"),
    };
    Ok(Some(scalar))
}

/// Skip the elements of an array without materializing them.
fn skip_array<R: Read>(r: &mut R, elem_type: u32, count: u64) -> Result<()> {
    let elem_size = match elem_type {
        T_UINT8 | T_INT8 | T_BOOL => 1,
        T_UINT16 | T_INT16 => 2,
        T_UINT32 | T_INT32 | T_FLOAT32 => 4,
        T_UINT64 | T_INT64 | T_FLOAT64 => 8,
        T_STRING => {
            for _ in 0..count {
                let len = read_u64(r)?;
                skip_bytes(r, len)?;
            }
            return Ok(());
        }
        T_ARRAY => bail!("nested GGUF arrays are not supported"),
        other => bail!("unknown GGUF array element type {other}"),
    };
    skip_bytes(r, count.saturating_mul(elem_size))
}

fn skip_bytes<R: Read>(r: &mut R, n: u64) -> Result<()> {
    // Read through the buffer (rather than seeking) so BufReader stays warm.
    let copied = io::copy(&mut r.by_ref().take(n), &mut io::sink())?;
    if copied != n {
        bail!("unexpected EOF skipping {n} bytes (got {copied})");
    }
    Ok(())
}

fn read_n<R: Read, const N: usize>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_n(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_n(r)?))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    // Guard against absurd lengths from a corrupt/non-GGUF file.
    if len > 64 * 1024 * 1024 {
        bail!("implausible GGUF string length {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Map `general.file_type` (LLAMA_FTYPE enum) to a short quant label. Covers the
/// common values; unknown ones fall back to the filename heuristic upstream.
fn file_type_label(ft: u64) -> Option<String> {
    let label = match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        32 => "BF16",
        _ => return None,
    };
    Some(label.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn detects_integrated_mtp_metadata() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("llmctl-mtp-header-{nonce}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&1_u64.to_le_bytes()); // metadata count
        push_string(&mut bytes, "gemma4-assistant.nextn_predict_layers");
        bytes.extend_from_slice(&T_UINT32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let info = read_gguf_info(&path).unwrap();
        assert!(info.has_mtp);

        std::fs::remove_file(path).unwrap();
    }
}
