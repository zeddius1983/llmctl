//! FastFlowLM runtime and authoritative model-catalogue discovery.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::config::FastFlowLmConfig;
use crate::domain::{FastFlowModel, Model, Runtime, RuntimeId};

const DEFAULT_PORT: u16 = 52625;
const CACHE_FILE: &str = "fastflowlm-models.json";

#[derive(Debug, Deserialize)]
struct Catalogue {
    #[serde(default)]
    models: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    installed: bool,
    #[serde(default)]
    default_context_length: Option<u64>,
    #[serde(default)]
    max_prefill_len: Option<u64>,
    #[serde(default)]
    flm_min_version: Option<String>,
    #[serde(default)]
    footprint: Option<f64>,
    #[serde(default)]
    label: Vec<String>,
    #[serde(default)]
    vlm: bool,
    #[serde(default)]
    details: Details,
}

#[derive(Debug, Default, Deserialize)]
struct Details {
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    parameter_size: Option<String>,
    #[serde(default)]
    quantization_level: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Version {
    version: String,
}

#[derive(Debug, Default, Deserialize)]
struct Validation {
    #[serde(default)]
    ready: bool,
    #[serde(default)]
    devices: Vec<ValidationDevice>,
}

#[derive(Debug, Deserialize)]
struct ValidationDevice {
    device: String,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema: u8,
    runtime: &'static str,
    tag: &'a str,
    installed: bool,
    min_version: Option<&'a str>,
}

/// Discover the runtime, its current catalogue, and configured default port.
pub fn discover(
    cfg: &FastFlowLmConfig,
    cache_dir: &Path,
    models_dir: &Path,
) -> (Runtime, Vec<Model>, u16) {
    let binary_path = super::runtimes::resolve_binary(&cfg.binary);
    let command =
        binary_path.as_ref().map(|path| vec![path.display().to_string()]).unwrap_or_default();

    let version = binary_path
        .as_ref()
        .and_then(|_| run_output(&command, &["version", "--json"]))
        .and_then(|output| parse_json::<Version>(&output.stdout).ok())
        .map(|value| value.version);
    let validation = binary_path
        .as_ref()
        .and_then(|_| run_output(&command, &["validate", "--json"]))
        .and_then(|output| parse_json::<Validation>(&output.stdout).ok());
    let devices = validation
        .as_ref()
        .map(|value| value.devices.iter().map(|device| device.device.clone()).collect())
        .unwrap_or_default();
    let ready = validation.as_ref().is_some_and(|value| value.ready);
    let port = binary_path
        .as_ref()
        .and_then(|_| run_output(&command, &["port"]))
        .and_then(|output| parse_port(&output.stdout))
        .unwrap_or(DEFAULT_PORT);

    let cache_path = cache_dir.join(CACHE_FILE);
    let catalogue = binary_path
        .as_ref()
        .and_then(|_| run_output(&command, &["list", "--json", "--quiet"]))
        .and_then(|output| {
            let json = json_payload(&output.stdout)?;
            let catalogue = serde_json::from_slice::<Catalogue>(json).ok()?;
            if let Err(error) = fs::write(&cache_path, json) {
                debug!(%error, path = %cache_path.display(), "could not cache FastFlowLM catalogue");
            }
            Some(catalogue)
        })
        .or_else(|| {
            fs::read(&cache_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Catalogue>(&bytes).ok())
        });

    if binary_path.is_none() {
        warn!(binary = %cfg.binary, "FastFlowLM binary not found");
    }
    let mut models = catalogue
        .map(|catalogue| build_models(catalogue, version.as_deref(), models_dir))
        .unwrap_or_default();
    sort_models(&mut models);

    let description =
        if ready { "AMD Ryzen AI NPU inference (NPU ready)" } else { "AMD Ryzen AI NPU inference" };
    (
        Runtime {
            id: RuntimeId::FastFlowLm,
            name: "FastFlowLM".into(),
            description: description.into(),
            version,
            binary_path,
            command,
            default_port: Some(port),
            bench_path: None,
            formats: vec!["NPU2".into(), "Q4NX".into()],
            devices,
        },
        models,
        port,
    )
}

/// Group installed models first while keeping each group alphabetical.
pub fn sort_models(models: &mut [Model]) {
    models.sort_by(|a, b| {
        let ai = a.fastflow.as_ref().is_some_and(|model| model.installed);
        let bi = b.fastflow.as_ref().is_some_and(|model| model.installed);
        bi.cmp(&ai).then_with(|| a.name.to_ascii_lowercase().cmp(&b.name.to_ascii_lowercase()))
    });
}

fn run_output(command: &[String], args: &[&str]) -> Option<Output> {
    let (program, prefix) = command.split_first()?;
    let output = Command::new(program).args(prefix).args(args).output().ok()?;
    if output.status.success() {
        Some(output)
    } else {
        debug!(program, ?args, status = ?output.status.code(), "FastFlowLM command failed");
        None
    }
}

/// FLM currently writes an informational line before JSON, even with --quiet.
fn json_payload(output: &[u8]) -> Option<&[u8]> {
    let start = output.iter().position(|byte| *byte == b'{')?;
    Some(&output[start..])
}

fn parse_json<T: for<'de> Deserialize<'de>>(output: &[u8]) -> Result<T, serde_json::Error> {
    let payload = json_payload(output).unwrap_or(output);
    serde_json::from_slice(payload)
}

fn parse_port(output: &[u8]) -> Option<u16> {
    String::from_utf8_lossy(output)
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .next_back()
}

fn build_models(catalogue: Catalogue, version: Option<&str>, models_dir: &Path) -> Vec<Model> {
    catalogue
        .models
        .into_iter()
        .filter_map(|entry| {
            let tag = if entry.model.is_empty() { entry.name.clone() } else { entry.model.clone() };
            if tag.is_empty() {
                return None;
            }
            let embedding = entry.label.iter().any(|label| label == "embeddings");
            let whisper =
                entry.details.family.as_deref().is_some_and(|family| family.starts_with("whisper"));
            let version_compatible = entry
                .flm_min_version
                .as_deref()
                .and_then(|required| version.map(|actual| version_at_least(actual, required)));
            // Unknown version information must not masquerade as an
            // incompatibility. FLM will still validate the model on first use.
            let supported = !embedding && !whisper && version_compatible != Some(false);
            let leaf = models_dir.join("fastflowlm").join("catalogue").join(safe_tag(&tag));
            if let Err(error) = fs::create_dir_all(leaf.join("profiles")) {
                warn!(%error, path = %leaf.display(), "could not create FastFlowLM catalogue leaf");
            } else {
                let manifest = Manifest {
                    schema: 1,
                    runtime: "fastflowlm",
                    tag: &tag,
                    installed: entry.installed,
                    min_version: entry.flm_min_version.as_deref(),
                };
                if let Ok(body) = serde_yaml::to_string(&manifest) {
                    let _ = fs::write(leaf.join(".llmctl.yml"), body);
                }
            }
            let size_bytes =
                entry.footprint.map(|gb| (gb * 1_000_000_000.0).round() as u64).unwrap_or_default();
            Some(Model {
                id: format!("flm:{tag}"),
                name: tag.clone(),
                path: PathBuf::new(),
                shard_paths: Vec::new(),
                mtp_path: None,
                projector_path: None,
                has_mtp: false,
                // FastFlowLM already is the catalogue boundary in the runtime
                // pane, so expose model leaves directly beneath it.
                catalog_path: vec![tag.clone()],
                catalog_dir: leaf,
                size_bytes,
                quantization: entry.details.quantization_level.clone(),
                architecture: entry.details.family.clone(),
                context_length: entry.default_context_length,
                modified: None,
                has_chat_template: true,
                remote: None,
                fastflow: Some(FastFlowModel {
                    tag,
                    installed: entry.installed,
                    min_version: entry.flm_min_version,
                    version_compatible,
                    footprint_gb: entry.footprint,
                    parameter_size: entry.details.parameter_size,
                    labels: entry.label,
                    vision: entry.vlm,
                    default_context_length: entry.default_context_length,
                    max_prefill_len: entry.max_prefill_len,
                    supported,
                }),
            })
        })
        .collect()
}

fn safe_tag(tag: &str) -> String {
    tag.chars()
        .map(|character| if matches!(character, '/' | '\0') { '_' } else { character })
        .collect()
}

fn version_at_least(actual: &str, required: &str) -> bool {
    fn parts(version: &str) -> Vec<u64> {
        version.split('.').map(|part| part.parse().unwrap_or(0)).collect()
    }
    let mut actual = parts(actual);
    let mut required = parts(required);
    let length = actual.len().max(required.len());
    actual.resize(length, 0);
    required.resize(length, 0);
    actual >= required
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_after_flm_banner() {
        let output = b"[FLM] Fetching models from: /opt/model_list.json\n{\"version\":\"0.9.45\"}";
        assert_eq!(parse_json::<Version>(output).unwrap().version, "0.9.45");
    }

    #[test]
    fn compares_dotted_versions_numerically() {
        assert!(version_at_least("0.9.45", "0.9.9"));
        assert!(version_at_least("1.0", "0.9.45"));
        assert!(!version_at_least("0.9.22", "0.9.43"));
    }

    #[test]
    fn unknown_runtime_version_does_not_reject_catalogue_models() {
        let fixture = br#"{
          "models": [{
            "model": "qwen3.5:4b",
            "installed": false,
            "flm_min_version": "0.9.45"
          }]
        }"#;
        let catalogue: Catalogue = serde_json::from_slice(fixture).unwrap();
        let root =
            std::env::temp_dir().join(format!("llmctl-fastflow-unknown-{}", std::process::id()));
        let models = build_models(catalogue, None, &root);
        let flm = models[0].fastflow.as_ref().unwrap();
        assert!(flm.supported);
        assert_eq!(flm.version_compatible, None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn parses_port_after_status_banner() {
        assert_eq!(parse_port(b"[FLM] status\nServer Port: 52625\n"), Some(52625));
    }

    #[test]
    fn catalogue_entry_becomes_a_runtime_owned_model_leaf() {
        let fixture = br#"{
          "models": [{
            "model": "qwen3:4b",
            "installed": false,
            "default_context_length": 32768,
            "max_prefill_len": 4096,
            "flm_min_version": "0.9.22",
            "footprint": 3.1,
            "label": ["reasoning", "tool-calling"],
            "details": {
              "family": "qwen3",
              "parameter_size": "4B",
              "quantization_level": "Q4_1"
            },
            "future_field": true
          }]
        }"#;
        let catalogue: Catalogue = serde_json::from_slice(fixture).unwrap();
        let root =
            std::env::temp_dir().join(format!("llmctl-fastflow-test-{}", std::process::id()));
        let models = build_models(catalogue, Some("0.9.45"), &root);
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.profile_key(), "flm:qwen3:4b");
        assert_eq!(model.catalog_path, ["qwen3:4b"]);
        assert!(model.is_model());
        assert!(model.fastflow.as_ref().unwrap().supported);
        assert!(model.catalog_dir.join("profiles").is_dir());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn installed_models_sort_before_available_models() {
        let fixture = br#"{
          "models": [
            {"model": "available-a", "installed": false},
            {"model": "installed-z", "installed": true},
            {"model": "installed-b", "installed": true},
            {"model": "available-c", "installed": false}
          ]
        }"#;
        let catalogue: Catalogue = serde_json::from_slice(fixture).unwrap();
        let root =
            std::env::temp_dir().join(format!("llmctl-fastflow-sort-{}", std::process::id()));
        let mut models = build_models(catalogue, None, &root);

        sort_models(&mut models);

        assert_eq!(
            models.iter().map(|model| model.name.as_str()).collect::<Vec<_>>(),
            vec!["installed-b", "installed-z", "available-a", "available-c"]
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
