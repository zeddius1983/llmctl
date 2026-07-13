//! Lazy Hugging Face catalog discovery backed by the managed model tree.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::domain::{Model, RemoteModel};

const API: &str = "https://huggingface.co/api/models";
const SOURCE: [&str; 2] = ["online", "huggingface"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Sort {
    #[default]
    Trending,
    Popular,
    Downloads,
}

impl Sort {
    pub fn label(self) -> &'static str {
        match self {
            Self::Trending => "Trending",
            Self::Popular => "Popular",
            Self::Downloads => "Downloads",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Trending => Self::Popular,
            Self::Popular => Self::Downloads,
            Self::Downloads => Self::Trending,
        }
    }

    fn api_value(self) -> &'static str {
        match self {
            Self::Trending => "trendingScore",
            Self::Popular => "likes",
            Self::Downloads => "downloads",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Repositories(Sort),
    Repository(String),
    Search { query: String, author: Option<String>, sort: Sort },
}

pub struct Response {
    pub epoch: u64,
    pub request: Request,
    pub result: Result<Vec<Model>, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Repository {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    sha: Option<String>,
    #[serde(default, deserialize_with = "deserialize_gated")]
    gated: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RepositoryList {
    schema: u8,
    fetched_at: u64,
    #[serde(default)]
    sort: Sort,
    repositories: Vec<Repository>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RepositoryDetail {
    schema: u8,
    fetched_at: u64,
    repository: Repository,
    #[serde(default)]
    siblings: Vec<Sibling>,
    #[serde(default)]
    gguf: Option<GgufInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GgufInfo {
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    chat_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sibling {
    #[serde(alias = "path")]
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Lfs {
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Serialize)]
struct ArtifactManifest<'a> {
    schema: u8,
    id: String,
    source: &'static str,
    repo: &'a str,
    revision: Option<&'a str>,
    file: &'a str,
    size_bytes: u64,
    cached_path: Option<&'a Path>,
}

impl Sibling {
    fn bytes(&self) -> u64 {
        self.size.or_else(|| self.lfs.as_ref().and_then(|lfs| lfs.size)).unwrap_or(0)
    }
}

/// The permanent virtual root is present even before the first network request.
pub fn load_cached(root: &Path) -> Vec<Model> {
    ensure_root(root);
    let mut models = vec![directory(&SOURCE)];
    let Ok(bytes) = fs::read(repository_list_path(root)) else {
        return models;
    };
    let Ok(list) = serde_json::from_slice::<RepositoryList>(&bytes) else {
        return models;
    };
    for repository in list.repositories {
        models.push(repository_directory(root, &repository));
        if let Ok(bytes) = fs::read(repository_detail_path(root, &repository.id))
            && let Ok(detail) = serde_json::from_slice::<RepositoryDetail>(&bytes)
        {
            models.extend(artifacts(root, &detail));
        }
    }
    models
}

pub fn cached_sort(root: &Path) -> Sort {
    fs::read(repository_list_path(root))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<RepositoryList>(&bytes).ok())
        .map(|list| list.sort)
        .unwrap_or_default()
}

/// Remove generated online catalog metadata while preserving user profile
/// YAML and the actual Hugging Face cache. The next fetch therefore builds a
/// clean logical layout without risking user configuration or model data.
pub fn clear_cached_layout(root: &Path) -> Result<()> {
    let online = root.join(SOURCE[0]).join(SOURCE[1]);
    if !online.exists() {
        return Ok(());
    }
    for entry in walkdir::WalkDir::new(&online).into_iter().filter_map(|entry| entry.ok()) {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy();
        let generated = matches!(
            name.as_ref(),
            ".repositories.json"
                | ".repository.json"
                | ".llmctl-online.yml"
                | ".repositories.json.tmp"
                | ".repository.json.tmp"
                | ".llmctl-online.yml.tmp"
        ) || (name == "model.gguf" && path.is_symlink());
        if generated {
            fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
        }
    }
    Ok(())
}

pub fn fetch(root: &Path, request: &Request) -> Result<Vec<Model>> {
    match request {
        Request::Repositories(sort) => {
            let fresh = request_repositories(root, None, None, *sort)?;
            save_repositories(root, fresh, *sort)?;
            Ok(load_cached(root))
        }
        Request::Repository(repo) => {
            fetch_repository(root, repo)?;
            Ok(load_cached(root))
        }
        Request::Search { query, author, sort } => {
            let repositories = request_repositories(root, Some(query), author.as_deref(), *sort)?;
            Ok(repositories
                .iter()
                .map(|repository| repository_directory(root, repository))
                .collect())
        }
    }
}

fn request_repositories(
    root: &Path,
    search: Option<&str>,
    author: Option<&str>,
    sort: Sort,
) -> Result<Vec<Repository>> {
    ensure_root(root);
    let mut request = agent()
        .get(API)
        .query("filter", "gguf")
        .query("apps", "llama.cpp")
        .query("sort", sort.api_value())
        .query("direction", "-1")
        .query("limit", "20");
    if let Some(search) = search {
        request = request.query("search", search);
    }
    if let Some(author) = author {
        request = request.query("author", author);
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    request
        .call()
        .context("requesting trending Hugging Face models")?
        .into_json()
        .context("parsing Hugging Face model list")
}

fn save_repositories(root: &Path, fresh: Vec<Repository>, sort: Sort) -> Result<()> {
    let mut repositories = fresh;
    if let Ok(bytes) = fs::read(repository_list_path(root))
        && let Ok(cached) = serde_json::from_slice::<RepositoryList>(&bytes)
    {
        for repository in cached.repositories {
            if !repositories.iter().any(|candidate| candidate.id == repository.id) {
                repositories.push(repository);
            }
        }
    }
    let list = RepositoryList { schema: 1, fetched_at: now(), sort, repositories };
    write_json(&repository_list_path(root), &list)
}

/// Promote one transient search result into the persistent online catalogue.
pub fn save_selected_repository(root: &Path, model: &Model, sort: Sort) -> Result<()> {
    ensure_root(root);
    let remote = model.remote.as_ref().context("selected model is not a Hub repository")?;
    if remote.file.is_some() {
        anyhow::bail!("selected model is a Hub artifact, not a repository");
    }
    let selected = Repository {
        id: remote.repo.clone(),
        downloads: remote.downloads,
        likes: remote.likes,
        sha: remote.revision.clone(),
        gated: remote.gated,
    };
    let mut repositories = fs::read(repository_list_path(root))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<RepositoryList>(&bytes).ok())
        .map(|list| list.repositories)
        .unwrap_or_default();
    if let Some(existing) = repositories.iter_mut().find(|repository| repository.id == selected.id)
    {
        *existing = selected;
    } else {
        repositories.push(selected);
    }
    let list = RepositoryList { schema: 1, fetched_at: now(), sort, repositories };
    write_json(&repository_list_path(root), &list)
}

fn fetch_repository(root: &Path, repo: &str) -> Result<()> {
    let mut request = agent().get(&format!("{API}/{repo}")).query("blobs", "true");
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let value: serde_json::Value = request
        .call()
        .with_context(|| format!("requesting Hugging Face repository {repo}"))?
        .into_json()
        .context("parsing Hugging Face repository")?;
    let repository = serde_json::from_value::<Repository>(value.clone())
        .context("parsing repository metadata")?;
    let siblings = serde_json::from_value::<Vec<Sibling>>(
        value.get("siblings").cloned().unwrap_or_else(|| serde_json::Value::Array(Vec::new())),
    )
    .context("parsing repository files")?;
    let gguf = serde_json::from_value::<GgufInfo>(
        value.get("gguf").cloned().unwrap_or(serde_json::Value::Null),
    )
    .ok();
    let detail = RepositoryDetail { schema: 1, fetched_at: now(), repository, siblings, gguf };
    let path = repository_detail_path(root, repo);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(&path, &detail)
}

fn artifacts(root: &Path, detail: &RepositoryDetail) -> Vec<Model> {
    let shard = Regex::new(r"(?i)-([0-9]{5})-of-([0-9]{5})\.gguf$").unwrap();
    let mut result = Vec::new();
    for file in &detail.siblings {
        let lower = file.rfilename.to_ascii_lowercase();
        if !lower.ends_with(".gguf")
            || file
                .rfilename
                .rsplit('/')
                .next()
                .is_some_and(|name| name.to_ascii_lowercase().starts_with("mmproj"))
        {
            continue;
        }
        if shard.captures(&file.rfilename).is_some_and(|captures| &captures[1] != "00001") {
            continue;
        }
        let display = shard.replace(&file.rfilename, ".gguf").into_owned();
        let bytes = if let Some(captures) = shard.captures(&file.rfilename) {
            let prefix = &file.rfilename[..captures.get(1).unwrap().start()];
            let suffix = &file.rfilename[captures.get(1).unwrap().end()..];
            detail
                .siblings
                .iter()
                .filter(|candidate| {
                    candidate.rfilename.starts_with(prefix) && candidate.rfilename.ends_with(suffix)
                })
                .map(Sibling::bytes)
                .sum()
        } else {
            file.bytes()
        };
        let leaf = sanitize(&display);
        let catalog_dir = repository_dir(root, &detail.repository.id).join(&leaf);
        let _ = fs::create_dir_all(catalog_dir.join("profiles"));
        let local_path = cached_file(&detail.repository.id, &file.rfilename).unwrap_or_default();
        let manifest = ArtifactManifest {
            schema: 1,
            id: format!("hf:{}/{}", detail.repository.id, file.rfilename),
            source: "hugging-face-online",
            repo: &detail.repository.id,
            revision: detail.repository.sha.as_deref(),
            file: &file.rfilename,
            size_bytes: bytes,
            cached_path: (!local_path.as_os_str().is_empty()).then_some(local_path.as_path()),
        };
        if let Ok(yaml) = serde_yaml::to_string(&manifest) {
            let _ = write_if_changed(&catalog_dir.join(".llmctl-online.yml"), yaml.as_bytes());
        }
        if !local_path.as_os_str().is_empty() {
            reconcile_link(&catalog_dir.join("model.gguf"), &local_path);
        }
        result.push(Model {
            id: format!("hf:{}/{}", detail.repository.id, file.rfilename),
            name: display,
            path: local_path,
            shard_paths: Vec::new(),
            catalog_path: vec![
                SOURCE[0].into(),
                SOURCE[1].into(),
                detail.repository.id.clone(),
                leaf,
            ],
            catalog_dir,
            size_bytes: bytes,
            quantization: super::models::quant_from_filename(&file.rfilename),
            architecture: detail.gguf.as_ref().and_then(|gguf| gguf.architecture.clone()),
            context_length: detail.gguf.as_ref().and_then(|gguf| gguf.context_length),
            modified: Some(detail.fetched_at),
            has_chat_template: detail
                .gguf
                .as_ref()
                .and_then(|gguf| gguf.chat_template.as_ref())
                .is_some(),
            remote: Some(RemoteModel {
                repo: detail.repository.id.clone(),
                revision: detail.repository.sha.clone(),
                file: Some(file.rfilename.clone()),
                downloads: detail.repository.downloads,
                likes: detail.repository.likes,
                gated: detail.repository.gated,
            }),
        });
    }
    result
}

fn repository_directory(root: &Path, repository: &Repository) -> Model {
    Model {
        id: format!("hf:{}", repository.id),
        name: repository.id.clone(),
        path: PathBuf::new(),
        shard_paths: Vec::new(),
        catalog_path: vec![SOURCE[0].into(), SOURCE[1].into(), repository.id.clone()],
        catalog_dir: repository_dir(root, &repository.id),
        size_bytes: 0,
        quantization: None,
        architecture: None,
        context_length: None,
        modified: None,
        has_chat_template: false,
        remote: Some(RemoteModel {
            repo: repository.id.clone(),
            revision: repository.sha.clone(),
            file: None,
            downloads: repository.downloads,
            likes: repository.likes,
            gated: repository.gated,
        }),
    }
}

fn directory(path: &[&str]) -> Model {
    Model {
        id: String::new(),
        name: path.last().copied().unwrap_or_default().into(),
        path: PathBuf::new(),
        shard_paths: Vec::new(),
        catalog_path: path.iter().map(|part| (*part).into()).collect(),
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

fn cached_file(repo: &str, file: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE").map(PathBuf::from).or_else(|| {
        std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")).or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
        })
    })?;
    let repo_dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let revision = fs::read_to_string(repo_dir.join("refs/main")).ok()?;
    let path = repo_dir.join("snapshots").join(revision.trim()).join(file);
    path.is_file().then_some(path)
}

fn ensure_root(root: &Path) {
    let _ = fs::create_dir_all(root.join(SOURCE[0]).join(SOURCE[1]));
}

fn repository_list_path(root: &Path) -> PathBuf {
    root.join(SOURCE[0]).join(SOURCE[1]).join(".repositories.json")
}

fn repository_dir(root: &Path, repo: &str) -> PathBuf {
    repo.split('/')
        .fold(root.join(SOURCE[0]).join(SOURCE[1]), |path, part| path.join(sanitize(part)))
}

fn repository_detail_path(root: &Path, repo: &str) -> PathBuf {
    repository_dir(root, repo).join(".repository.json")
}

fn sanitize(raw: &str) -> String {
    let clean: String = raw
        .chars()
        .map(|character| if character == '/' || character == '\0' { '_' } else { character })
        .collect();
    match clean.as_str() {
        "" | "." | ".." => "_".into(),
        _ => clean,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if fs::read(path).is_ok_and(|current| current == bytes) {
        return Ok(());
    }
    let tmp = path.with_extension("yml.tmp");
    fs::write(&tmp, bytes).and_then(|_| fs::rename(&tmp, path))
}

#[cfg(unix)]
fn reconcile_link(link: &Path, target: &Path) {
    if fs::read_link(link).is_ok_and(|current| current == target) {
        return;
    }
    if link.is_symlink() {
        let _ = fs::remove_file(link);
    } else if link.exists() {
        return;
    }
    let _ = std::os::unix::fs::symlink(target, link);
}

#[cfg(not(unix))]
fn reconcile_link(_link: &Path, _target: &Path) {}

fn deserialize_gated<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(!matches!(value, serde_json::Value::Bool(false) | serde_json::Value::Null))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_secs()).unwrap_or(0)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(10))
        .timeout_read(std::time::Duration::from_secs(30))
        .build()
}

pub fn repository_for_path(path: &[String]) -> Option<String> {
    let tail = path.strip_prefix(&[SOURCE[0].to_string(), SOURCE[1].to_string()])?;
    tail.first().cloned()
}

pub fn is_online_path(path: &[String]) -> bool {
    path.first().is_some_and(|part| part == SOURCE[0])
}

pub fn request_for_path(path: &[String], sort: Sort) -> Option<Request> {
    if !is_online_path(path) {
        return None;
    }
    repository_for_path(path).map(Request::Repository).or(Some(Request::Repositories(sort)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_accepts_boolean_and_string_api_shapes() {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(deserialize_with = "deserialize_gated")]
            gated: bool,
        }
        assert!(!serde_json::from_str::<Wrapper>(r#"{"gated":false}"#).unwrap().gated);
        assert!(serde_json::from_str::<Wrapper>(r#"{"gated":"manual"}"#).unwrap().gated);
    }

    #[test]
    fn request_scope_follows_online_catalog_depth() {
        assert_eq!(
            request_for_path(&["online".into()], Sort::Popular),
            Some(Request::Repositories(Sort::Popular))
        );
        assert_eq!(
            request_for_path(
                &["online".into(), "huggingface".into(), "owner/repo".into()],
                Sort::Downloads,
            ),
            Some(Request::Repository("owner/repo".into()))
        );
        assert_eq!(request_for_path(&["models".into()], Sort::Trending), None);
    }

    #[test]
    fn cached_file_path_uses_standard_hub_layout() {
        assert_eq!(sanitize("folder/model.gguf"), "folder_model.gguf");
        assert_eq!(sanitize(".."), "_");
    }

    #[test]
    fn sort_cycles_and_uses_hub_sort_names() {
        assert_eq!(Sort::Trending.next(), Sort::Popular);
        assert_eq!(Sort::Popular.next(), Sort::Downloads);
        assert_eq!(Sort::Downloads.next(), Sort::Trending);
        assert_eq!(Sort::Trending.api_value(), "trendingScore");
        assert_eq!(Sort::Popular.api_value(), "likes");
        assert_eq!(Sort::Downloads.api_value(), "downloads");
    }

    #[test]
    fn clearing_layout_preserves_profiles_and_records_cached_sort() {
        let root = std::env::temp_dir().join(format!("llmctl-online-clear-{}", now()));
        let artifact = root.join("online/huggingface/owner/repo/model");
        fs::create_dir_all(artifact.join("profiles")).unwrap();
        fs::write(artifact.join("profiles/Chat.yml"), b"profile").unwrap();
        fs::write(artifact.join(".llmctl-online.yml"), b"generated").unwrap();
        fs::write(root.join("online/huggingface/owner/repo/.repository.json"), b"generated")
            .unwrap();
        let list = RepositoryList {
            schema: 1,
            fetched_at: 1,
            sort: Sort::Downloads,
            repositories: Vec::new(),
        };
        write_json(&repository_list_path(&root), &list).unwrap();
        assert_eq!(cached_sort(&root), Sort::Downloads);

        clear_cached_layout(&root).unwrap();
        assert!(artifact.join("profiles/Chat.yml").is_file());
        assert!(!artifact.join(".llmctl-online.yml").exists());
        assert!(!root.join("online/huggingface/owner/repo/.repository.json").exists());
        assert!(!repository_list_path(&root).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn only_the_selected_search_repository_is_persisted() {
        let root = std::env::temp_dir().join(format!("llmctl-online-selected-{}", now()));
        let first = Repository {
            id: "owner/first".into(),
            downloads: 10,
            likes: 1,
            sha: Some("first-sha".into()),
            gated: false,
        };
        let selected = Repository {
            id: "owner/selected".into(),
            downloads: 20,
            likes: 2,
            sha: Some("selected-sha".into()),
            gated: false,
        };
        let ignored = Repository {
            id: "owner/ignored".into(),
            downloads: 30,
            likes: 3,
            sha: Some("ignored-sha".into()),
            gated: false,
        };
        let results = [first, selected, ignored]
            .iter()
            .map(|repository| repository_directory(&root, repository))
            .collect::<Vec<_>>();

        save_selected_repository(&root, &results[1], Sort::Popular).unwrap();

        let cached = load_cached(&root);
        let repositories = cached
            .iter()
            .filter_map(|model| model.remote.as_ref().map(|remote| remote.repo.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(repositories, vec!["owner/selected"]);
        assert_eq!(cached_sort(&root), Sort::Popular);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifacts_group_shards_and_materialize_profile_leaf() {
        let root = std::env::temp_dir().join(format!("llmctl-online-{}", now()));
        let detail = RepositoryDetail {
            schema: 1,
            fetched_at: 42,
            repository: Repository {
                id: "owner/repo".into(),
                downloads: 12,
                likes: 3,
                sha: Some("abc".into()),
                gated: false,
            },
            siblings: vec![
                Sibling {
                    rfilename: "model-Q4_K_M-00001-of-00002.gguf".into(),
                    size: Some(10),
                    lfs: None,
                },
                Sibling {
                    rfilename: "model-Q4_K_M-00002-of-00002.gguf".into(),
                    size: Some(11),
                    lfs: None,
                },
                Sibling { rfilename: "mmproj-model.gguf".into(), size: Some(5), lfs: None },
            ],
            gguf: Some(GgufInfo {
                architecture: Some("llama".into()),
                context_length: Some(32768),
                chat_template: Some("template".into()),
            }),
        };
        let models = artifacts(&root, &detail);
        let repository = repository_directory(&root, &detail.repository);
        assert_eq!(repository.catalog_path, vec!["online", "huggingface", "owner/repo"]);
        assert_eq!(repository.display_label(), "owner/repo");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].size_bytes, 21);
        assert_eq!(models[0].name, "model-Q4_K_M.gguf");
        assert_eq!(models[0].context_length, Some(32768));
        assert_eq!(
            models[0].catalog_path,
            vec!["online", "huggingface", "owner/repo", "model-Q4_K_M.gguf"]
        );
        assert_eq!(models[0].profile_key(), "hf:owner/repo/model-Q4_K_M-00001-of-00002.gguf");
        assert!(models[0].catalog_dir.join("profiles").is_dir());
        assert!(models[0].catalog_dir.join(".llmctl-online.yml").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore = "uses the live Hugging Face API"]
    fn live_hugging_face_fetch_materializes_catalog() {
        let root = std::env::temp_dir().join(format!("llmctl-online-live-{}", now()));
        let repositories = fetch(&root, &Request::Repositories(Sort::Trending)).unwrap();
        let repo = repositories
            .iter()
            .find_map(|model| model.remote.as_ref().map(|remote| remote.repo.clone()))
            .expect("trending repository");
        let details = fetch(&root, &Request::Repository(repo.clone())).unwrap();
        assert!(details.iter().any(|model| {
            model.remote.as_ref().is_some_and(|remote| remote.repo == repo && remote.file.is_some())
        }));
        let searched = fetch(
            &root,
            &Request::Search { query: "Qwen".into(), author: None, sort: Sort::Trending },
        )
        .unwrap();
        assert!(searched.iter().any(|model| {
            model
                .remote
                .as_ref()
                .is_some_and(|remote| remote.repo.to_ascii_lowercase().contains("qwen"))
        }));
        let author_search = fetch(
            &root,
            &Request::Search {
                query: "gemma-4".into(),
                author: Some("unsloth".into()),
                sort: Sort::Trending,
            },
        )
        .unwrap();
        assert!(author_search.iter().any(|model| {
            model.remote.as_ref().is_some_and(|remote| {
                remote.repo.starts_with("unsloth/")
                    && remote.repo.to_ascii_lowercase().contains("gemma-4")
            })
        }));

        clear_cached_layout(&root).unwrap();
        let popular = fetch(&root, &Request::Repositories(Sort::Popular)).unwrap();
        let likes: Vec<u64> = popular
            .iter()
            .filter_map(|model| model.remote.as_ref().map(|remote| remote.likes))
            .collect();
        assert!(likes.windows(2).all(|pair| pair[0] >= pair[1]));
        assert_eq!(cached_sort(&root), Sort::Popular);

        clear_cached_layout(&root).unwrap();
        let downloaded = fetch(&root, &Request::Repositories(Sort::Downloads)).unwrap();
        let downloads: Vec<u64> = downloaded
            .iter()
            .filter_map(|model| model.remote.as_ref().map(|remote| remote.downloads))
            .collect();
        assert!(downloads.windows(2).all(|pair| pair[0] >= pair[1]));
        assert_eq!(cached_sort(&root), Sort::Downloads);
        let _ = fs::remove_dir_all(root);
    }
}
