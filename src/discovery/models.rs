//! GGUF model discovery: recursive scan of configured directories with a
//! mtime/size-keyed cache so re-scans (and the `F5` refresh) are cheap.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tracing::{debug, warn};
use walkdir::WalkDir;

use crate::domain::Model;

use super::{catalog, catalog::ModelSource, gguf};

const CACHE_SCHEMA: u8 = 2;

/// On-disk scan cache: keyed by absolute path string.
#[derive(Default, Serialize, Deserialize)]
struct Cache {
    #[serde(default)]
    schema: u8,
    models: HashMap<String, Model>,
}

/// Scan every configured directory recursively for GGUF models.
///
/// Results are de-duplicated by first shard, sorted by name, and cached to
/// `cache_path`. Unchanged files (same size + mtime) reuse cached metadata
/// instead of re-parsing the GGUF header.
pub fn scan(sources: &[ModelSource], cache_path: &Path) -> Vec<Model> {
    let cache = load_cache(cache_path);
    let mut fresh: HashMap<String, Model> = HashMap::new();
    let mut models: Vec<Model> = Vec::new();
    let mut mtp_sidecars: HashMap<PathBuf, PathBuf> = HashMap::new();
    let mut projectors: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

    let mut catalog_paths: HashMap<Vec<String>, PathBuf> = HashMap::new();
    for source in sources {
        let base = &source.root;
        if !base.exists() {
            debug!(path = %base.display(), "configured model path does not exist");
            continue;
        }
        for entry in WalkDir::new(base).follow_links(true).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if !entry.file_type().is_file() || !is_gguf(path) || !is_first_shard(path) {
                continue;
            }
            if is_projector(path) {
                if let Some(parent) = path.parent() {
                    projectors.entry(parent.to_path_buf()).or_default().push(path.to_path_buf());
                }
                continue;
            }
            if let Some(base_path) = mtp_base_path(path) {
                mtp_sidecars.insert(base_path, path.to_path_buf());
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            // For multi-part models, sum every shard's size, not just the first.
            let size = aggregate_size(path, meta.len());
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());

            let key = path.to_string_lossy().into_owned();
            if fresh.contains_key(&key) {
                continue; // overlapping configured roots must not duplicate a model
            }
            let mut model = match cache.models.get(&key) {
                Some(c) if c.size_bytes == size && c.modified == modified => c.clone(),
                _ => build_model(path, size, modified),
            };
            model.shard_paths = shard_paths(path);
            model.id = format!("{}:{}", source.name, catalog::short_hash(path));
            model.catalog_path = catalog::catalog_path(source, path, &model.name);
            if let Some(existing) = catalog_paths.get(&model.catalog_path) {
                if existing != path {
                    if catalog::resolved_layout(source, path)
                        == crate::config::ModelLayout::HuggingFace
                    {
                        let old_modified = models
                            .iter()
                            .find(|m| m.catalog_path == model.catalog_path)
                            .and_then(|m| m.modified)
                            .unwrap_or(0);
                        if !prefer_huggingface_candidate(
                            source,
                            path,
                            modified,
                            existing,
                            Some(old_modified),
                        ) {
                            fresh.insert(key, model);
                            continue;
                        }
                        models.retain(|m| m.catalog_path != model.catalog_path);
                    } else {
                        if let Some(last) = model.catalog_path.last_mut() {
                            last.push('-');
                            last.push_str(&catalog::short_hash(path));
                        }
                    }
                }
            }
            catalog_paths.insert(model.catalog_path.clone(), path.to_path_buf());
            model.catalog_dir = PathBuf::new();
            fresh.insert(key, model.clone());
            models.push(model);
        }
    }

    // MTP GGUFs are sidecars named `mtp-<base filename>`, per the GGUF naming
    // convention. Repositories may omit the base artifact's quantization suffix
    // from that name (for example `mtp-gemma-4-12b-it.gguf` beside
    // `gemma-4-12b-it-Q4_K_M.gguf`). Walk order is unspecified, so associate
    // them only after all sources have been scanned. Orphan sidecars stay hidden:
    // they cannot be launched without their matching base model.
    let mut model_families: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    for model in &models {
        if let (Some(parent), Some(family)) = (model.path.parent(), model_family(&model.path)) {
            model_families.entry(parent.to_path_buf()).or_default().insert(family);
        }
    }
    for model in &mut models {
        model.mtp_path = matching_mtp_sidecar(&model.path, &mtp_sidecars);
        model.projector_path = matching_projector(&model.path, &projectors, &model_families);
    }
    for model in fresh.values_mut() {
        model.mtp_path = matching_mtp_sidecar(&model.path, &mtp_sidecars);
        model.projector_path = matching_projector(&model.path, &projectors, &model_families);
    }

    resolve_prefix_collisions(&mut models);
    save_cache(cache_path, &Cache { schema: CACHE_SCHEMA, models: fresh });
    models.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    models
}

fn prefer_huggingface_candidate(
    source: &ModelSource,
    candidate: &Path,
    candidate_modified: Option<u64>,
    current: &Path,
    current_modified: Option<u64>,
) -> bool {
    hf_snapshot_rank(source, candidate, candidate_modified)
        > hf_snapshot_rank(source, current, current_modified)
}

fn hf_snapshot_rank(
    source: &ModelSource,
    path: &Path,
    modified: Option<u64>,
) -> (bool, u64, String, String) {
    let relative = path.strip_prefix(&source.root).unwrap_or(path);
    let components: Vec<_> = relative.components().collect();
    let snapshot = components
        .iter()
        .position(|c| matches!(c, std::path::Component::Normal(s) if *s == "snapshots"));
    let revision = snapshot
        .and_then(|i| components.get(i + 1))
        .and_then(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .unwrap_or_default();
    let is_main = snapshot.is_some_and(|i| {
        let model_dir = components[..i]
            .iter()
            .fold(source.root.clone(), |path, component| path.join(component.as_os_str()));
        std::fs::read_to_string(model_dir.join("refs/main"))
            .is_ok_and(|main| main.trim() == revision)
    });
    (is_main, modified.unwrap_or(0), revision, path.to_string_lossy().into_owned())
}

/// A catalog leaf is itself a directory (`model.gguf` + `profiles/`), so it
/// cannot also act as a browser directory for a longer logical model path.
/// Give only the leaf a stable suffix when one path prefixes another.
fn resolve_prefix_collisions(models: &mut [Model]) {
    let collisions: Vec<bool> = models
        .iter()
        .map(|candidate| {
            models.iter().any(|other| {
                other.id != candidate.id
                    && other.catalog_path.len() > candidate.catalog_path.len()
                    && other.catalog_path.starts_with(&candidate.catalog_path)
            })
        })
        .collect();
    for (model, collision) in models.iter_mut().zip(collisions) {
        if collision && let Some(last) = model.catalog_path.last_mut() {
            last.push('-');
            last.push_str(&catalog::short_hash(&model.path));
        }
    }
}

/// Read a model's metadata from its GGUF header, falling back to the filename
/// for quantization when the header doesn't carry a usable file-type.
fn build_model(path: &Path, size: u64, modified: Option<u64>) -> Model {
    let raw_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let name = display_name(&raw_name);
    let info = match gguf::read_gguf_info(path) {
        Ok(info) => Some(info),
        Err(err) => {
            warn!(path = %path.display(), %err, "failed to read GGUF header");
            None
        }
    };

    // Prefer the filename label — it captures exact variants (e.g. Unsloth
    // `Q4_K_XL`, `MXFP4`) that the header's coarse `file_type` enum misses.
    let quantization = quant_from_filename(&name)
        .or_else(|| info.as_ref().and_then(|i| i.file_type_label.clone()));

    Model {
        id: String::new(),
        name,
        path: path.to_path_buf(),
        shard_paths: shard_paths(path),
        mtp_path: None,
        projector_path: None,
        has_mtp: info.as_ref().is_some_and(|info| info.has_mtp) || has_mtp_filename(path),
        catalog_path: Vec::new(),
        catalog_dir: PathBuf::new(),
        size_bytes: size,
        quantization,
        architecture: info.as_ref().and_then(|i| i.architecture.clone()),
        context_length: info.as_ref().and_then(|i| i.context_length),
        modified,
        has_chat_template: info.as_ref().map(|i| i.has_chat_template).unwrap_or(false),
        remote: None,
    }
}

/// Total on-disk size of a model. For a first shard (`-00001-of-000NN.gguf`),
/// sum every shard in the same directory; otherwise return the single size.
fn aggregate_size(path: &Path, first_size: u64) -> u64 {
    let shards = shard_paths(path);
    if shards.len() == 1 {
        return first_size;
    }
    let sum =
        shards.iter().filter_map(|path| std::fs::metadata(path).ok()).map(|meta| meta.len()).sum();
    if sum == 0 { first_size } else { sum }
}

fn shard_paths(path: &Path) -> Vec<PathBuf> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap());

    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(caps) = re.captures(&name) else {
        return vec![path.to_path_buf()];
    };
    let total: u32 = caps[2].parse().unwrap_or(1);
    let idx = caps.get(1).unwrap();
    let (head, tail) = (&name[..idx.start()], &name[idx.end()..]);
    let dir = path.parent().unwrap_or(Path::new("."));

    (1..=total).map(|i| dir.join(format!("{head}{i:05}{tail}"))).collect()
}

/// Drop the `-00001-of-00005` shard suffix from a multi-part model's display
/// name (the on-disk path is kept intact for launching).
fn display_name(file: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-\d{5}-of-\d{5}(\.gguf)$").unwrap());
    re.replace(file, "$1").into_owned()
}

fn is_gguf(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// Multi-part GGUF files are named `...-00001-of-00005.gguf`; show only the
/// first shard so a split model appears once.
fn is_first_shard(path: &Path) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-\d{5}\.gguf$").unwrap());
    let name = path.to_string_lossy();
    match re.captures(&name) {
        Some(caps) => &caps[1] == "00001",
        None => true,
    }
}

/// Projector/companion files (`mmproj-*.gguf`) are not standalone models.
fn is_projector(path: &Path) -> bool {
    path.file_name()
        .map(|n| n.to_string_lossy().to_lowercase().starts_with("mmproj"))
        .unwrap_or(false)
}

/// Select one same-directory projector for a base model. Publishers commonly
/// provide several precision variants with generic names such as
/// `mmproj-BF16.gguf`; an unqualified publisher default wins, followed by the
/// compact high-fidelity variants. The path order is the deterministic tie
/// breaker.
fn matching_projector(
    model_path: &Path,
    projectors: &HashMap<PathBuf, Vec<PathBuf>>,
    model_families: &HashMap<PathBuf, HashSet<String>>,
) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    if model_families.get(parent)?.len() != 1 {
        return None;
    }
    let candidates = projectors.get(parent)?;
    candidates
        .iter()
        .max_by(|a, b| projector_rank(a).cmp(&projector_rank(b)).then_with(|| b.cmp(a)))
        .cloned()
}

fn model_family(path: &Path) -> Option<String> {
    let stem = gguf_stem(path)?;
    let family = match quant_from_filename(&stem) {
        Some(quant) => stem
            .get(..stem.len().checked_sub(quant.len())?)?
            .trim_end_matches('-')
            .trim_end_matches("UD")
            .trim_end_matches('-'),
        None => &stem,
    };
    Some(family.to_ascii_lowercase())
}

fn projector_rank(path: &Path) -> u8 {
    match path.file_name().and_then(|name| quant_from_filename(&name.to_string_lossy())) {
        None => 6,
        Some(quant) if quant.eq_ignore_ascii_case("BF16") => 5,
        Some(quant) if quant.eq_ignore_ascii_case("F16") || quant.eq_ignore_ascii_case("FP16") => 4,
        Some(quant) if quant.eq_ignore_ascii_case("Q8_0") => 3,
        Some(quant) if quant.eq_ignore_ascii_case("Q4_0") => 2,
        Some(_) => 1,
    }
}

/// Return the base-model path named by an `mtp-*.gguf` sidecar. The prefix is
/// case-insensitive for discovery robustness, while the remainder is kept
/// byte-for-byte so it still resolves correctly on case-sensitive filesystems.
fn mtp_base_path(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_string_lossy();
    let prefix = name.get(..4)?;
    if !prefix.eq_ignore_ascii_case("mtp-") || name.len() == 4 {
        return None;
    }
    Some(path.with_file_name(&name[4..]))
}

/// Match an MTP sidecar to a base GGUF. Prefer the exact official filename
/// relationship, then accept a same-directory base whose stem extends the
/// sidecar's stem at a `-` boundary (normally with a quantization suffix). When
/// several prefixes match, the longest/most-specific one wins.
fn matching_mtp_sidecar(
    model_path: &Path,
    sidecars: &HashMap<PathBuf, PathBuf>,
) -> Option<PathBuf> {
    if let Some(exact) = sidecars.get(model_path) {
        return Some(exact.clone());
    }
    let model_stem = gguf_stem(model_path)?.to_ascii_lowercase();
    let mut candidates = sidecars
        .iter()
        .filter_map(|(base_path, sidecar_path)| {
            if base_path.parent() != model_path.parent() {
                return None;
            }
            let base_stem = gguf_stem(base_path)?.to_ascii_lowercase();
            let suffix = model_stem.strip_prefix(&base_stem)?;
            suffix.starts_with('-').then_some((base_stem.len(), sidecar_path))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    candidates.first().map(|(_, path)| (*path).clone())
}

fn gguf_stem(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    let display = display_name(&name);
    Path::new(&display).file_stem().map(|stem| stem.to_string_lossy().into_owned())
}

/// Integrated MTP releases commonly mark the base GGUF with an `MTP` filename
/// token. Header metadata is authoritative when present; this fallback keeps
/// discovery useful for older converters that only advertised it in the name.
fn has_mtp_filename(path: &Path) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)(?:^|[-_.])mtp(?:[-_.]|$)").unwrap());
    path.file_name().is_some_and(|name| re.is_match(&name.to_string_lossy()))
}

/// Heuristic quantization label from the filename (e.g. `Q4_K_XL`, `IQ3_XXS`,
/// `MXFP4`). Longer/more-specific patterns are matched first.
pub fn quant_from_filename(name: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|Q\d+|MXFP\d+|BF16|FP16|FP8|F16|F32)\b",
        )
        .unwrap()
    });
    re.find(name).map(|m| m.as_str().to_uppercase())
}

fn load_cache(path: &Path) -> Cache {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .ok()
            .filter(|cache: &Cache| cache.schema == CACHE_SCHEMA)
            .unwrap_or_default(),
        Err(_) => Cache::default(),
    }
}

fn save_cache(path: &Path, cache: &Cache) {
    match serde_json::to_vec(cache) {
        Ok(bytes) => {
            if let Err(err) = std::fs::write(path, bytes) {
                warn!(path = %path.display(), %err, "failed to write model cache");
            }
        }
        Err(err) => warn!(%err, "failed to serialize model cache"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelLayout;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn model(id: &str, path: &str, catalog_path: &[&str]) -> Model {
        Model {
            id: id.into(),
            name: Path::new(path).file_name().unwrap().to_string_lossy().into_owned(),
            path: path.into(),
            shard_paths: vec![path.into()],
            mtp_path: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: catalog_path.iter().map(|s| (*s).into()).collect(),
            catalog_dir: PathBuf::new(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
            remote: None,
        }
    }

    #[test]
    fn suffixes_a_leaf_that_would_hide_a_directory() {
        let mut models = vec![
            model("leaf", "/models/foo.gguf", &["src", "foo"]),
            model("nested", "/models/foo/bar.gguf", &["src", "foo", "bar"]),
        ];
        resolve_prefix_collisions(&mut models);
        assert!(models[0].catalog_path[1].starts_with("foo-"));
        assert_eq!(models[1].catalog_path, ["src", "foo", "bar"]);
    }

    #[test]
    fn huggingface_main_ref_wins_even_with_an_older_mtime() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-hf-rank-{nonce}"));
        let repo = root.join("models--org--repo");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "main-revision\n").unwrap();
        let source = ModelSource {
            name: "huggingface".into(),
            root: root.clone(),
            layout: ModelLayout::HuggingFace,
        };
        let main = repo.join("snapshots/main-revision/model.gguf");
        let other = repo.join("snapshots/other-revision/model.gguf");
        assert!(prefer_huggingface_candidate(&source, &main, Some(1), &other, Some(99)));
        assert!(!prefer_huggingface_candidate(&source, &other, Some(99), &main, Some(1)));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn huggingface_ties_have_a_deterministic_path_order() {
        let source = ModelSource {
            name: "huggingface".into(),
            root: "/cache".into(),
            layout: ModelLayout::HuggingFace,
        };
        let a = Path::new("/cache/models--org--repo/snapshots/aaa/model.gguf");
        let b = Path::new("/cache/models--org--repo/snapshots/bbb/model.gguf");
        assert!(prefer_huggingface_candidate(&source, b, None, a, None));
        assert!(!prefer_huggingface_candidate(&source, a, None, b, None));
    }

    #[test]
    fn scan_pairs_mtp_sidecar_and_hides_it_as_a_standalone_model() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-mtp-scan-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        for name in [
            "Qwen.gguf",
            "mtp-Qwen.gguf",
            "mtp-Orphan.gguf",
            "gemma-4-12b-it-Q4_K_M.gguf",
            "mtp-gemma-4-12b-it.gguf",
            // MTP in an ordinary model name is not a sidecar prefix.
            "Qwen-MTP-Q4_K_M.gguf",
        ] {
            std::fs::write(root.join(name), b"not a real GGUF header").unwrap();
        }
        let source = ModelSource {
            name: "models".into(),
            root: root.clone(),
            layout: ModelLayout::Directory,
        };

        let models = scan(&[source], &root.join("models.json"));
        assert_eq!(models.len(), 3);
        let base = models.iter().find(|model| model.name == "Qwen.gguf").unwrap();
        assert_eq!(base.mtp_path.as_deref(), Some(root.join("mtp-Qwen.gguf").as_path()));
        let gemma = models.iter().find(|model| model.name == "gemma-4-12b-it-Q4_K_M.gguf").unwrap();
        assert_eq!(gemma.mtp_path.as_deref(), Some(root.join("mtp-gemma-4-12b-it.gguf").as_path()));
        assert!(models.iter().any(|model| model.name == "Qwen-MTP-Q4_K_M.gguf"));
        assert!(models.iter().find(|model| model.name == "Qwen-MTP-Q4_K_M.gguf").unwrap().has_mtp);
        assert!(models.iter().all(|model| !model.name.starts_with("mtp-")));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scan_pairs_one_projector_and_hides_all_projector_variants() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-mmproj-scan-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        for name in ["gemma.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"] {
            std::fs::write(root.join(name), b"not a real GGUF header").unwrap();
        }
        let source = ModelSource {
            name: "models".into(),
            root: root.clone(),
            layout: ModelLayout::Directory,
        };

        let models = scan(&[source], &root.join("models.json"));
        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].projector_path.as_deref(),
            Some(root.join("mmproj-BF16.gguf").as_path())
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generic_projector_is_not_attached_in_a_mixed_model_directory() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-mmproj-mixed-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        for name in ["gemma-Q4_K_M.gguf", "qwen-Q4_K_M.gguf", "mmproj-BF16.gguf"] {
            std::fs::write(root.join(name), b"not a real GGUF header").unwrap();
        }
        let source = ModelSource {
            name: "models".into(),
            root: root.clone(),
            layout: ModelLayout::Directory,
        };

        let models = scan(&[source], &root.join("models.json"));
        assert_eq!(models.len(), 2);
        assert!(models.iter().all(|model| model.projector_path.is_none()));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mtp_sidecar_prefix_is_case_insensitive_and_preserves_shard_name() {
        assert_eq!(
            mtp_base_path(Path::new("/models/MTP-Qwen-00001-of-00002.gguf")),
            Some(PathBuf::from("/models/Qwen-00001-of-00002.gguf"))
        );
        assert_eq!(mtp_base_path(Path::new("/models/model-MTP.gguf")), None);
    }
}
