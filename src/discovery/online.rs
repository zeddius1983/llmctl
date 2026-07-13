//! Lazy Hugging Face catalog discovery backed by the managed model tree.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::domain::{Model, RemoteBlob, RemoteModel};

const API: &str = "https://huggingface.co/api/models";
const SOURCE: [&str; 2] = ["online", "huggingface"];
const DOWNLOAD_SCHEMA: u8 = 1;

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
            Self::Popular => "Most likes",
            Self::Downloads => "Most downloads",
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
pub struct DownloadJobRecord {
    schema: u8,
    pub model_id: String,
    pub model: String,
    pub remote: RemoteModel,
}

impl DownloadJobRecord {
    pub fn new(model_id: String, model: String, remote: RemoteModel) -> Self {
        Self { schema: DOWNLOAD_SCHEMA, model_id, model, remote }
    }
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
    #[serde(default, alias = "sha256")]
    oid: Option<String>,
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
            && detail.schema >= 2
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
    for entry in walkdir::WalkDir::new(&online)
        .into_iter()
        .filter_entry(|entry| {
            entry.depth() == 0
                || !matches!(entry.file_name().to_str(), Some(".downloads" | "profiles"))
        })
        .filter_map(|entry| entry.ok())
    {
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
    let detail = RepositoryDetail { schema: 2, fetched_at: now(), repository, siblings, gguf };
    let path = repository_detail_path(root, repo);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(&path, &detail)
}

fn artifacts(root: &Path, detail: &RepositoryDetail) -> Vec<Model> {
    let cache = hugging_face_cache();
    artifacts_with_cache(root, detail, cache.as_deref())
}

fn artifacts_with_cache(
    root: &Path,
    detail: &RepositoryDetail,
    cache: Option<&Path>,
) -> Vec<Model> {
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
        let files: Vec<&Sibling> = if let Some(captures) = shard.captures(&file.rfilename) {
            let prefix = &file.rfilename[..captures.get(1).unwrap().start()];
            let suffix = &file.rfilename[captures.get(1).unwrap().end()..];
            detail
                .siblings
                .iter()
                .filter(|candidate| {
                    candidate.rfilename.starts_with(prefix) && candidate.rfilename.ends_with(suffix)
                })
                .collect()
        } else {
            vec![file]
        };
        let bytes = files.iter().map(|file| file.bytes()).sum();
        let blobs = files
            .iter()
            .filter_map(|file| {
                let lfs = file.lfs.as_ref()?;
                Some(RemoteBlob {
                    oid: lfs.oid.clone()?,
                    size_bytes: file.bytes(),
                    file: file.rfilename.clone(),
                })
            })
            .collect();
        let leaf = sanitize(&display);
        let catalog_dir = repository_dir(root, &detail.repository.id).join(&leaf);
        let _ = fs::create_dir_all(catalog_dir.join("profiles"));
        let cached_paths = files
            .iter()
            .map(|candidate| {
                cache.and_then(|hub| {
                    cached_file_in(hub, &detail.repository.id, &candidate.rfilename)
                })
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();
        let local_path = (cached_paths.len() == files.len())
            .then(|| cached_paths.first().cloned())
            .flatten()
            .unwrap_or_default();
        let shard_paths = if local_path.as_os_str().is_empty() { Vec::new() } else { cached_paths };
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
            shard_paths,
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
                blobs,
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
            blobs: Vec::new(),
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
    let hub = hugging_face_cache()?;
    cached_file_in(&hub, repo, file)
}

fn cached_file_in(hub: &Path, repo: &str, file: &str) -> Option<PathBuf> {
    let repo_dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let revision = fs::read_to_string(repo_dir.join("refs/main")).ok()?;
    let path = repo_dir.join("snapshots").join(revision.trim()).join(file);
    path.is_file().then_some(path)
}

fn hugging_face_cache() -> Option<PathBuf> {
    std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")).or_else(|| {
                std::env::var_os("HOME")
                    .map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
            })
        })
}

/// Paths used by the standard Hugging Face cache while a known LFS blob is
/// transferring and after it has completed.
pub fn cache_blob_paths(repo: &str, oid: &str) -> Option<(PathBuf, PathBuf)> {
    cache_blob_paths_in(&hugging_face_cache()?, repo, oid)
}

pub fn load_download_records(root: &Path) -> Vec<DownloadJobRecord> {
    let Ok(entries) = fs::read_dir(download_records_dir(root)) else {
        return Vec::new();
    };
    let mut paths = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "json"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|path| {
            let record = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<DownloadJobRecord>(&bytes).ok());
            match record {
                Some(record) if record.schema == DOWNLOAD_SCHEMA => Some(record),
                _ => {
                    tracing::warn!(path = %path.display(), "ignoring invalid download record");
                    None
                }
            }
        })
        .collect()
}

pub fn save_download_record(root: &Path, record: &DownloadJobRecord) -> Result<()> {
    fs::create_dir_all(download_records_dir(root)).context("creating download record directory")?;
    write_json(&download_record_path(root, &record.model_id), record)
}

pub fn delete_download_record(root: &Path, model_id: &str) -> Result<()> {
    let path = download_record_path(root, model_id);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("removing {}", path.display())),
    }
}

fn download_records_dir(root: &Path) -> PathBuf {
    root.join(SOURCE[0]).join(SOURCE[1]).join(".downloads")
}

fn download_record_path(root: &Path, model_id: &str) -> PathBuf {
    download_records_dir(root).join(format!("{:016x}.json", stable_hash(model_id.as_bytes())))
}

fn stable_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

/// Download one online GGUF artifact (including every shard) into the standard
/// Hugging Face cache. Existing partial blobs are resumed when the server
/// accepts byte ranges. `progress` receives aggregate downloaded and expected
/// bytes often enough for the TUI without flooding its event channel.
pub enum DownloadResult {
    Downloaded(PathBuf),
    Cancelled,
}

pub fn download_model(
    remote: &RemoteModel,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<DownloadResult> {
    let primary = remote.file.as_deref().context("Hub artifact has no filename")?;
    if remote.blobs.is_empty() {
        anyhow::bail!("Hub artifact has no downloadable blob metadata; refresh its repository");
    }
    if remote.blobs.iter().any(|blob| blob.file.is_empty()) {
        anyhow::bail!("Hub artifact has incomplete shard metadata; refresh its repository");
    }
    let total = remote.blobs.iter().map(|blob| blob.size_bytes).sum::<u64>();
    if total == 0 {
        anyhow::bail!("Hub artifact reports an unknown download size");
    }

    progress(cached_downloaded_bytes(remote), total);
    for blob in &remote.blobs {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(DownloadResult::Cancelled);
        }
        let (incomplete, complete) = cache_blob_paths(&remote.repo, &blob.oid)
            .context("Hugging Face cache directory is unavailable")?;
        if complete.metadata().is_ok_and(|metadata| metadata.len() == blob.size_bytes) {
            materialize_snapshot_file(remote, &blob.file, &complete)?;
            continue;
        }
        if let Some(parent) = incomplete.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating cache directory {}", parent.display()))?;
        }
        if incomplete.metadata().is_ok_and(|metadata| metadata.len() == blob.size_bytes) {
            fs::rename(&incomplete, &complete)
                .with_context(|| format!("completing cached blob {}", complete.display()))?;
            materialize_snapshot_file(remote, &blob.file, &complete)?;
            progress(cached_downloaded_bytes(remote), total);
            continue;
        }

        if !download_blob(remote, blob, &incomplete, cancelled, |_, _| {
            progress(cached_downloaded_bytes(remote), total)
        })? {
            return Ok(DownloadResult::Cancelled);
        }
        fs::rename(&incomplete, &complete)
            .with_context(|| format!("completing cached blob {}", complete.display()))?;
        materialize_snapshot_file(remote, &blob.file, &complete)?;
        progress(cached_downloaded_bytes(remote), total);
    }

    cached_file(&remote.repo, primary)
        .map(DownloadResult::Downloaded)
        .context("download completed but cache link is unavailable")
}

fn download_blob(
    remote: &RemoteModel,
    blob: &RemoteBlob,
    incomplete: &Path,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<bool> {
    if cancelled.load(Ordering::Relaxed) {
        return Ok(false);
    }
    let existing = incomplete.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    let revision = remote.revision.as_deref().unwrap_or("main");
    let url = resolve_url(&remote.repo, revision, &blob.file);
    let mut request = agent().get(&url);
    if existing > 0 {
        request = request.set("Range", &format!("bytes={existing}-"));
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .with_context(|| format!("downloading hf://{}/{}", remote.repo, blob.file))?;
    let resumed = existing > 0 && response.status() == 206;
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if resumed {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut output = options
        .open(incomplete)
        .with_context(|| format!("opening partial download {}", incomplete.display()))?;
    let mut reader = response.into_reader();
    let mut downloaded = if resumed { existing } else { 0 };
    let mut reported = downloaded;
    let mut buffer = [0_u8; 256 * 1024];
    loop {
        if cancelled.load(Ordering::Relaxed) {
            output.flush().context("flushing cancelled Hugging Face download")?;
            return Ok(false);
        }
        let read = reader.read(&mut buffer).context("reading Hugging Face response")?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read]).context("writing Hugging Face cache blob")?;
        downloaded = downloaded.saturating_add(read as u64);
        if downloaded.saturating_sub(reported) >= 8 * 1024 * 1024 {
            progress(downloaded.min(blob.size_bytes), blob.size_bytes);
            reported = downloaded;
        }
    }
    output.flush().context("flushing Hugging Face cache blob")?;
    if downloaded != blob.size_bytes {
        anyhow::bail!(
            "incomplete Hugging Face download for {}: received {} of {} bytes",
            blob.file,
            downloaded,
            blob.size_bytes
        );
    }
    progress(downloaded, blob.size_bytes);
    Ok(true)
}

pub fn cached_downloaded_bytes(remote: &RemoteModel) -> u64 {
    remote
        .blobs
        .iter()
        .map(|blob| {
            let Some((incomplete, complete)) = cache_blob_paths(&remote.repo, &blob.oid) else {
                return 0;
            };
            if complete.metadata().is_ok_and(|metadata| metadata.len() == blob.size_bytes) {
                blob.size_bytes
            } else {
                incomplete
                    .metadata()
                    .map(|metadata| metadata.len().min(blob.size_bytes))
                    .unwrap_or(0)
            }
        })
        .sum()
}

/// Finish cache bookkeeping without network access when every expected blob is
/// already present (including a fully-written partial file left by a crash).
pub fn finalize_cached_download(remote: &RemoteModel) -> Result<PathBuf> {
    let primary = remote.file.as_deref().context("Hub artifact has no filename")?;
    if remote.blobs.is_empty() {
        anyhow::bail!("Hub artifact has no blob metadata");
    }
    for blob in &remote.blobs {
        let (incomplete, complete) = cache_blob_paths(&remote.repo, &blob.oid)
            .context("Hugging Face cache directory is unavailable")?;
        if !complete.metadata().is_ok_and(|metadata| metadata.len() == blob.size_bytes) {
            if incomplete.metadata().is_ok_and(|metadata| metadata.len() == blob.size_bytes) {
                fs::rename(&incomplete, &complete)
                    .with_context(|| format!("completing cached blob {}", complete.display()))?;
            } else {
                anyhow::bail!("download is still incomplete");
            }
        }
        materialize_snapshot_file(remote, &blob.file, &complete)?;
    }
    cached_file(&remote.repo, primary).context("completed cache link is unavailable")
}

fn materialize_snapshot_file(remote: &RemoteModel, file: &str, blob: &Path) -> Result<PathBuf> {
    let hub = hugging_face_cache().context("Hugging Face cache directory is unavailable")?;
    let revision = remote.revision.as_deref().unwrap_or("main");
    let repo = hub.join(format!("models--{}", remote.repo.replace('/', "--")));
    let reference = repo.join("refs/main");
    if let Some(parent) = reference.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&reference, revision)
        .with_context(|| format!("writing Hugging Face reference {}", reference.display()))?;

    let snapshot = repo.join("snapshots").join(revision).join(file);
    if let Some(parent) = snapshot.parent() {
        fs::create_dir_all(parent)?;
    }
    reconcile_cache_link(&snapshot, blob)?;
    Ok(snapshot)
}

#[cfg(unix)]
fn reconcile_cache_link(link: &Path, target: &Path) -> Result<()> {
    if fs::read_link(link).is_ok_and(|current| current == target) {
        return Ok(());
    }
    if link.is_file() {
        return Ok(());
    }
    if link.is_symlink() {
        fs::remove_file(link)
            .with_context(|| format!("replacing cache link {}", link.display()))?;
    }
    std::os::unix::fs::symlink(target, link)
        .with_context(|| format!("linking cached model {}", link.display()))
}

#[cfg(not(unix))]
fn reconcile_cache_link(link: &Path, target: &Path) -> Result<()> {
    fs::copy(target, link)
        .map(|_| ())
        .with_context(|| format!("copying cached model {}", link.display()))
}

fn resolve_url(repo: &str, revision: &str, file: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        encode_url_path(repo),
        encode_url_path(revision),
        encode_url_path(file)
    )
}

fn encode_url_path(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn cache_blob_paths_in(hub: &Path, repo: &str, oid: &str) -> Option<(PathBuf, PathBuf)> {
    let oid = oid.strip_prefix("sha256:").unwrap_or(oid);
    if oid.is_empty() || !oid.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let blob = hub.join(format!("models--{}", repo.replace('/', "--"))).join("blobs").join(oid);
    let incomplete = blob.with_file_name(format!("{oid}.downloadInProgress"));
    Some((incomplete, blob))
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
        assert_eq!(
            cache_blob_paths_in(Path::new("/cache"), "owner/repo", "aabb").unwrap(),
            (
                PathBuf::from("/cache/models--owner--repo/blobs/aabb.downloadInProgress"),
                PathBuf::from("/cache/models--owner--repo/blobs/aabb")
            )
        );
    }

    #[test]
    fn resolve_url_preserves_paths_and_encodes_unsafe_bytes() {
        assert_eq!(
            resolve_url("owner/model", "main", "nested/model Q4_K_M.gguf"),
            "https://huggingface.co/owner/model/resolve/main/nested/model%20Q4_K_M.gguf"
        );
    }

    #[test]
    fn download_honors_cancellation_before_starting_network_io() {
        let remote = RemoteModel {
            repo: "owner/cancel-test".into(),
            revision: Some("abc".into()),
            file: Some("model.gguf".into()),
            blobs: vec![RemoteBlob {
                oid: "ff".repeat(32),
                size_bytes: 1_000,
                file: "model.gguf".into(),
            }],
            downloads: 0,
            likes: 0,
            gated: false,
        };
        let cancelled = AtomicBool::new(true);
        let result = download_model(&remote, &cancelled, |_, _| {}).unwrap();
        assert!(matches!(result, DownloadResult::Cancelled));
    }

    #[test]
    fn sort_cycles_and_uses_hub_sort_names() {
        assert_eq!(Sort::Trending.next(), Sort::Popular);
        assert_eq!(Sort::Popular.next(), Sort::Downloads);
        assert_eq!(Sort::Downloads.next(), Sort::Trending);
        assert_eq!(Sort::Trending.label(), "Trending");
        assert_eq!(Sort::Popular.label(), "Most likes");
        assert_eq!(Sort::Downloads.label(), "Most downloads");
        assert_eq!(Sort::Trending.api_value(), "trendingScore");
        assert_eq!(Sort::Popular.api_value(), "likes");
        assert_eq!(Sort::Downloads.api_value(), "downloads");
    }

    #[test]
    fn clearing_layout_preserves_profiles_and_records_cached_sort() {
        let root = std::env::temp_dir().join(format!("llmctl-online-clear-{}", now()));
        let artifact = root.join("online/huggingface/owner/repo/model");
        let download_record = root.join("online/huggingface/.downloads/job.json");
        fs::create_dir_all(artifact.join("profiles")).unwrap();
        fs::create_dir_all(download_record.parent().unwrap()).unwrap();
        fs::write(artifact.join("profiles/Chat.yml"), b"profile").unwrap();
        fs::write(&download_record, b"download").unwrap();
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
        assert_eq!(fs::read(download_record).unwrap(), b"download");
        assert!(!artifact.join(".llmctl-online.yml").exists());
        assert!(!root.join("online/huggingface/owner/repo/.repository.json").exists());
        assert!(!repository_list_path(&root).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn download_records_round_trip_in_managed_online_storage() {
        let root = std::env::temp_dir().join(format!("llmctl-download-record-{}", now()));
        let remote = RemoteModel {
            repo: "owner/repo".into(),
            revision: Some("revision".into()),
            file: Some("model-Q4_K_M.gguf".into()),
            blobs: vec![RemoteBlob {
                oid: "ab".repeat(32),
                size_bytes: 42,
                file: "model-Q4_K_M.gguf".into(),
            }],
            downloads: 10,
            likes: 2,
            gated: false,
        };
        let record = DownloadJobRecord::new(
            "online:huggingface:owner/repo:model-Q4_K_M.gguf".into(),
            "owner/repo/model-Q4_K_M.gguf".into(),
            remote,
        );

        save_download_record(&root, &record).unwrap();
        let path = download_record_path(&root, &record.model_id);
        assert_eq!(path.parent(), Some(download_records_dir(&root).as_path()));
        assert!(path.is_file());

        let loaded = load_download_records(&root);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].model_id, record.model_id);
        assert_eq!(loaded[0].model, record.model);
        assert_eq!(loaded[0].remote.repo, "owner/repo");
        assert_eq!(loaded[0].remote.blobs[0].size_bytes, 42);

        delete_download_record(&root, &record.model_id).unwrap();
        assert!(load_download_records(&root).is_empty());
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
                    lfs: Some(Lfs { oid: Some("aa".repeat(32)), size: Some(10) }),
                },
                Sibling {
                    rfilename: "model-Q4_K_M-00002-of-00002.gguf".into(),
                    size: Some(11),
                    lfs: Some(Lfs { oid: Some("bb".repeat(32)), size: Some(11) }),
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
        assert_eq!(models[0].remote.as_ref().unwrap().blobs.len(), 2);
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
    fn split_artifact_is_cached_only_after_every_shard_exists() {
        let root = std::env::temp_dir().join(format!("llmctl-online-split-{}", now()));
        let hub = root.join("hub");
        let detail = RepositoryDetail {
            schema: 1,
            fetched_at: 42,
            repository: Repository {
                id: "owner/repo".into(),
                downloads: 0,
                likes: 0,
                sha: Some("abc".into()),
                gated: false,
            },
            siblings: vec![
                Sibling {
                    rfilename: "model-00001-of-00002.gguf".into(),
                    size: Some(10),
                    lfs: Some(Lfs { oid: Some("aa".repeat(32)), size: Some(10) }),
                },
                Sibling {
                    rfilename: "model-00002-of-00002.gguf".into(),
                    size: Some(11),
                    lfs: Some(Lfs { oid: Some("bb".repeat(32)), size: Some(11) }),
                },
            ],
            gguf: None,
        };
        let repo = hub.join("models--owner--repo");
        fs::create_dir_all(repo.join("refs")).unwrap();
        fs::write(repo.join("refs/main"), "abc").unwrap();
        let first = repo.join("snapshots/abc/model-00001-of-00002.gguf");
        let second = repo.join("snapshots/abc/model-00002-of-00002.gguf");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::write(&first, vec![0; 10]).unwrap();

        let partial = artifacts_with_cache(&root, &detail, Some(&hub));
        assert!(partial[0].path.as_os_str().is_empty());
        assert!(partial[0].shard_paths.is_empty());

        fs::write(&second, vec![0; 11]).unwrap();
        let complete = artifacts_with_cache(&root, &detail, Some(&hub));
        assert_eq!(complete[0].path, first);
        assert_eq!(complete[0].shard_paths, vec![first, second]);
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
        assert!(details.iter().any(|model| {
            model.remote.as_ref().is_some_and(|remote| {
                remote.repo == repo && remote.file.is_some() && !remote.blobs.is_empty()
            })
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
