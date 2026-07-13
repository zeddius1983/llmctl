//! Hugging Face REST API client: model search and GGUF file listings.
//!
//! Search is pre-filtered to models llama.cpp can actually run
//! (`pipeline_tag=text-generation&library=gguf&apps=llama.cpp`), sorted by
//! trending score like the hub's own browse page. HTTP is blocking (`ureq`)
//! and always runs on a worker thread (see [`crate::hub`]).

use std::io::Read;
use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;
use serde::Deserialize;

use crate::discovery::quant_from_filename;

const BASE: &str = "https://huggingface.co";
/// Results per search — one popup page; the hub ranks well within this.
const SEARCH_LIMIT: usize = 30;
/// Upper bound when buffering an API response body.
const MAX_BODY_BYTES: u64 = 32 * 1024 * 1024;

/// A model repository as returned by the search endpoint.
#[derive(Debug, Clone)]
pub struct HubModel {
    /// Repository id, e.g. `unsloth/Qwen3-4B-Instruct-2507-GGUF`.
    pub id: String,
    pub likes: u64,
    pub downloads: u64,
    /// Creation date (`YYYY-MM-DD`), when reported.
    pub created: Option<String>,
}

/// A repository's GGUF contents (the file listing of a hub repo folder).
#[derive(Debug, Clone)]
pub struct HubRepo {
    /// Gated repos need an accepted license + `HF_TOKEN` to download.
    pub gated: bool,
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub files: Vec<HubFile>,
}

/// One downloadable GGUF artifact — a single file, or a multi-part shard set
/// grouped under its logical name.
#[derive(Debug, Clone)]
pub struct HubFile {
    /// Repo-relative display name with any `-00001-of-000NN` suffix stripped.
    pub name: String,
    pub quant: Option<String>,
    /// Total size across all parts (0 when the API omits sizes).
    pub size_bytes: u64,
    /// Physical files to fetch, in shard order.
    pub parts: Vec<FilePart>,
}

/// One physical file within an artifact.
#[derive(Debug, Clone)]
pub struct FilePart {
    /// Repo-relative filename (may contain subdirectories).
    pub rfilename: String,
    pub size: u64,
}

pub fn search_models(query: &str) -> Result<Vec<HubModel>, String> {
    let mut url = format!(
        "{BASE}/api/models?pipeline_tag=text-generation&library=gguf&apps=llama.cpp\
         &sort=trendingScore&limit={SEARCH_LIMIT}"
    );
    let query = query.trim();
    if !query.is_empty() {
        url.push_str("&search=");
        url.push_str(&urlencode(query));
    }
    parse_models(&http_get(&url)?)
}

pub fn repo_files(repo_id: &str) -> Result<HubRepo, String> {
    let encoded: Vec<String> = repo_id.split('/').map(urlencode).collect();
    let url = format!("{BASE}/api/models/{}?blobs=true", encoded.join("/"));
    parse_repo(&http_get(&url)?)
}

/// The `resolve` URL a repo-relative file downloads from.
pub fn resolve_url(repo_id: &str, rfilename: &str) -> String {
    let repo: Vec<String> = repo_id.split('/').map(urlencode).collect();
    let file: Vec<String> = rfilename.split('/').map(urlencode).collect();
    format!("{BASE}/{}/resolve/main/{}", repo.join("/"), file.join("/"))
}

// --- HTTP -------------------------------------------------------------------

/// Shared agent for API calls (connection reuse, bounded timeouts).
fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(30))
            .build()
    })
}

/// Attach the standard headers; `HF_TOKEN` (when set) unlocks gated repos.
pub(super) fn prepare(request: ureq::Request) -> ureq::Request {
    let request = request.set("User-Agent", concat!("llmctl/", env!("CARGO_PKG_VERSION")));
    match std::env::var("HF_TOKEN") {
        Ok(token) if !token.trim().is_empty() => {
            request.set("Authorization", &format!("Bearer {}", token.trim()))
        }
        _ => request,
    }
}

pub(super) fn describe(err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(401, _) | ureq::Error::Status(403, _) => {
            "access denied — gated model? set HF_TOKEN to a token with access".into()
        }
        ureq::Error::Status(code, _) => format!("Hugging Face returned HTTP {code}"),
        ureq::Error::Transport(t) => format!("network error: {t}"),
    }
}

fn http_get(url: &str) -> Result<String, String> {
    let response = prepare(agent().get(url)).call().map_err(describe)?;
    let mut body = String::new();
    response
        .into_reader()
        .take(MAX_BODY_BYTES)
        .read_to_string(&mut body)
        .map_err(|e| format!("reading response: {e}"))?;
    Ok(body)
}

/// Percent-encode everything outside the RFC 3986 unreserved set.
fn urlencode(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for byte in raw.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

// --- JSON parsing ------------------------------------------------------------

#[derive(Deserialize)]
struct ApiModel {
    id: String,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    downloads: u64,
    #[serde(default, rename = "createdAt")]
    created_at: Option<String>,
}

#[derive(Deserialize)]
struct ApiRepo {
    /// `false` for open repos, or a mode string (`"auto"`/`"manual"`).
    #[serde(default)]
    gated: Option<serde_json::Value>,
    #[serde(default)]
    siblings: Vec<ApiSibling>,
    #[serde(default)]
    gguf: Option<ApiGguf>,
}

#[derive(Deserialize)]
struct ApiSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Deserialize)]
struct ApiGguf {
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
}

fn parse_models(json: &str) -> Result<Vec<HubModel>, String> {
    let models: Vec<ApiModel> =
        serde_json::from_str(json).map_err(|e| format!("unexpected search response: {e}"))?;
    Ok(models
        .into_iter()
        .map(|m| HubModel {
            id: m.id,
            likes: m.likes,
            downloads: m.downloads,
            created: m.created_at.map(|d| d.chars().take(10).collect()),
        })
        .collect())
}

fn parse_repo(json: &str) -> Result<HubRepo, String> {
    let repo: ApiRepo =
        serde_json::from_str(json).map_err(|e| format!("unexpected model response: {e}"))?;
    let gated = match repo.gated {
        Some(serde_json::Value::Bool(flag)) => flag,
        Some(serde_json::Value::String(_)) => true,
        _ => false,
    };
    Ok(HubRepo {
        gated,
        architecture: repo.gguf.as_ref().and_then(|g| g.architecture.clone()),
        context_length: repo.gguf.as_ref().and_then(|g| g.context_length),
        files: group_files(&repo.siblings),
    })
}

/// Collapse the sibling listing into downloadable artifacts: keep GGUF files,
/// drop `mmproj` projectors, and fold `-000NN-of-000NN` shards into one entry.
fn group_files(siblings: &[ApiSibling]) -> Vec<HubFile> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap());

    let mut files: Vec<HubFile> = Vec::new();
    for sibling in siblings {
        let name = sibling.rfilename.as_str();
        if !name.to_lowercase().ends_with(".gguf") || is_projector(name) {
            continue;
        }
        let part = FilePart { rfilename: name.to_string(), size: sibling.size.unwrap_or(0) };
        let logical = re.replace(name, ".gguf").into_owned();
        match files.iter_mut().find(|f| f.name == logical) {
            Some(file) => {
                file.size_bytes += part.size;
                file.parts.push(part);
            }
            None => files.push(HubFile {
                quant: quant_from_filename(&logical),
                name: logical,
                size_bytes: part.size,
                parts: vec![part],
            }),
        }
    }
    for file in &mut files {
        file.parts.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
    }
    // Smallest first reads naturally when picking a quant to fit in memory.
    files.sort_by(|a, b| a.size_bytes.cmp(&b.size_bytes).then_with(|| a.name.cmp(&b.name)));
    files
}

/// Projector/companion files are not standalone models (same rule as discovery).
fn is_projector(rfilename: &str) -> bool {
    rfilename.rsplit('/').next().map(|f| f.to_lowercase().starts_with("mmproj")).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_search_results_with_missing_optional_fields() {
        let json = r#"[
            {"id":"unsloth/Qwen3-GGUF","likes":184,"downloads":86454,
             "trendingScore":12.4,"createdAt":"2025-08-06T07:41:24.000Z"},
            {"id":"org/bare-model"}
        ]"#;
        let models = parse_models(json).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "unsloth/Qwen3-GGUF");
        assert_eq!(models[0].likes, 184);
        assert_eq!(models[0].created.as_deref(), Some("2025-08-06"));
        assert_eq!(models[1].likes, 0);
        assert_eq!(models[1].created, None);
    }

    #[test]
    fn parses_repo_and_groups_shards_into_one_artifact() {
        let json = r#"{
            "id":"org/model-GGUF",
            "gated":false,
            "gguf":{"architecture":"qwen3","context_length":262144},
            "siblings":[
                {"rfilename":"README.md"},
                {"rfilename":"mmproj-F16.gguf","size":10},
                {"rfilename":"model-Q4_K_M.gguf","size":90},
                {"rfilename":"UD-Q2_K_XL/model-UD-Q2_K_XL-00002-of-00002.gguf","size":30},
                {"rfilename":"UD-Q2_K_XL/model-UD-Q2_K_XL-00001-of-00002.gguf","size":70}
            ]
        }"#;
        let repo = parse_repo(json).unwrap();
        assert!(!repo.gated);
        assert_eq!(repo.architecture.as_deref(), Some("qwen3"));
        assert_eq!(repo.context_length, Some(262144));
        assert_eq!(repo.files.len(), 2);

        // Sorted smallest-first: the single Q4 file, then the sharded set.
        assert_eq!(repo.files[0].name, "model-Q4_K_M.gguf");
        assert_eq!(repo.files[0].quant.as_deref(), Some("Q4_K_M"));
        let sharded = &repo.files[1];
        assert_eq!(sharded.name, "UD-Q2_K_XL/model-UD-Q2_K_XL.gguf");
        assert_eq!(sharded.size_bytes, 100);
        assert_eq!(
            sharded.parts.iter().map(|p| p.rfilename.as_str()).collect::<Vec<_>>(),
            [
                "UD-Q2_K_XL/model-UD-Q2_K_XL-00001-of-00002.gguf",
                "UD-Q2_K_XL/model-UD-Q2_K_XL-00002-of-00002.gguf"
            ]
        );
    }

    #[test]
    fn gated_mode_strings_count_as_gated() {
        let repo = parse_repo(r#"{"id":"org/m","gated":"manual","siblings":[]}"#).unwrap();
        assert!(repo.gated);
    }

    #[test]
    fn resolve_url_percent_encodes_but_keeps_path_separators() {
        assert_eq!(
            resolve_url("org/repo", "UD Q2/model.gguf"),
            "https://huggingface.co/org/repo/resolve/main/UD%20Q2/model.gguf"
        );
    }

    #[test]
    fn urlencode_covers_query_metacharacters() {
        assert_eq!(urlencode("qwen coder+7b&x=1"), "qwen%20coder%2B7b%26x%3D1");
    }
}
