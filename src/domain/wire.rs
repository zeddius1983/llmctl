//! Compatibility with the original flat model cache format.

use super::*;

// Compatibility DTO: only this boundary interprets the old optional fields and
// empty-path convention. Runtime code matches the explicit variants above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ModelFile {
    /// Stable catalog identity, distinct from the display filename.
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    /// All physical shards (one entry for a non-split model).
    #[serde(default)]
    pub shard_paths: Vec<PathBuf>,
    /// A same-directory `mtp-*.gguf` sidecar paired with this base model.
    /// llama.cpp loads it as a speculative draft model, never as a standalone
    /// model.
    #[serde(default)]
    pub mtp_path: Option<PathBuf>,
    /// A same-directory `dflash-*.gguf` sidecar paired with this base model.
    /// llama.cpp loads it as the draft model for `--spec-type draft-dflash`,
    /// never as a standalone model.
    #[serde(default)]
    pub dflash_path: Option<PathBuf>,
    /// The drafter's trained block size (`dflash.block_size`): the most tokens
    /// it can emit per pass, and the value llama.cpp clamps
    /// `--spec-draft-n-max` to. Known only once the drafter is on disk.
    #[serde(default)]
    pub dflash_block_size: Option<u64>,
    /// A compatible multimodal projector sidecar. The projector encodes image
    /// or audio inputs for the base model and is never launched by itself.
    #[serde(default)]
    pub projector_path: Option<PathBuf>,
    /// Whether the base GGUF itself contains Multi-Token Prediction heads.
    #[serde(default)]
    pub has_mtp: bool,
    /// Source/provider/repository/artifact path shown by the model browser.
    #[serde(default)]
    pub catalog_path: Vec<String>,
    /// Managed catalog leaf containing the manifest, symlink, and profiles.
    #[serde(default)]
    pub catalog_dir: PathBuf,
    pub size_bytes: u64,
    pub quantization: Option<String>,
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    /// Last-modified time, seconds since the Unix epoch (cache invalidation).
    pub modified: Option<u64>,
    pub has_chat_template: bool,
    /// Hugging Face identity for a lazily-discovered online entry. A missing
    /// `file` represents a repository directory; a file makes this a
    /// launchable remote GGUF leaf even before it has a local cache path.
    #[serde(default)]
    pub remote: Option<RemoteModel>,
    /// FastFlowLM catalog identity. Present for every entry the `flm` catalog
    /// knows about, installed or not.
    #[serde(default)]
    pub flm: Option<FlmModel>,
    /// Which runtime owns this model. Determines the profile-store namespace,
    /// so profiles never leak between runtimes.
    #[serde(default = "default_runtime")]
    pub runtime: String,
}

impl TryFrom<ModelFile> for Model {
    type Error = String;

    fn try_from(file: ModelFile) -> Result<Self, Self::Error> {
        let entry = match (file.flm, file.remote) {
            (Some(_), Some(_)) => {
                return Err("a model cannot have both FastFlowLM and GGUF identities".into());
            }
            (Some(flm), None) => CatalogEntry::Model(ModelSource::FastFlowLm(flm)),
            (None, remote)
                if file.path.as_os_str().is_empty()
                    && remote.as_ref().and_then(|remote| remote.file.as_ref()).is_none() =>
            {
                CatalogEntry::Directory { repository: remote }
            }
            (None, remote) => CatalogEntry::Model(ModelSource::Gguf { remote }),
        };
        Ok(Self {
            entry,
            id: file.id,
            name: file.name,
            path: file.path,
            shard_paths: file.shard_paths,
            mtp_path: file.mtp_path,
            dflash_path: file.dflash_path,
            dflash_block_size: file.dflash_block_size,
            projector_path: file.projector_path,
            has_mtp: file.has_mtp,
            catalog_path: file.catalog_path,
            catalog_dir: file.catalog_dir,
            size_bytes: file.size_bytes,
            quantization: file.quantization,
            architecture: file.architecture,
            context_length: file.context_length,
            modified: file.modified,
            has_chat_template: file.has_chat_template,
            runtime: file.runtime,
        })
    }
}

impl From<Model> for ModelFile {
    fn from(model: Model) -> Self {
        let (remote, flm) = match model.entry {
            CatalogEntry::Directory { repository } => (repository, None),
            CatalogEntry::Model(ModelSource::Gguf { remote }) => (remote, None),
            CatalogEntry::Model(ModelSource::FastFlowLm(flm)) => (None, Some(flm)),
        };
        Self {
            remote,
            flm,
            id: model.id,
            name: model.name,
            path: model.path,
            shard_paths: model.shard_paths,
            mtp_path: model.mtp_path,
            dflash_path: model.dflash_path,
            dflash_block_size: model.dflash_block_size,
            projector_path: model.projector_path,
            has_mtp: model.has_mtp,
            catalog_path: model.catalog_path,
            catalog_dir: model.catalog_dir,
            size_bytes: model.size_bytes,
            quantization: model.quantization,
            architecture: model.architecture,
            context_length: model.context_length,
            modified: model.modified,
            has_chat_template: model.has_chat_template,
            runtime: model.runtime,
        }
    }
}

fn default_runtime() -> String {
    crate::runtime::llama_cpp::NAME.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy() -> serde_json::Value {
        serde_json::json!({
            "name": "model", "path": "", "size_bytes": 0,
            "quantization": null, "architecture": null, "context_length": null,
            "modified": null, "has_chat_template": false
        })
    }

    #[test]
    fn an_uncached_gguf_is_a_model_and_keeps_its_identity_after_download() {
        let mut file = legacy();
        file["remote"] =
            serde_json::json!({"repo": "org/repo", "file": "model.gguf", "revision": "main"});
        let mut model: Model = serde_json::from_value(file).unwrap();
        assert!(matches!(model.entry, CatalogEntry::Model(ModelSource::Gguf { .. })));
        let key = model.profile_key();
        assert!(crate::runtime::LaunchContext::new("server", &model, &[]).is_ok());
        model.path = "/cache/model.gguf".into();
        assert_eq!(model.profile_key(), key);
        let saved = serde_json::to_value(&model).unwrap();
        assert!(saved.get("entry").is_none(), "keep the flat cache format");
        let restored: Model = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.profile_key(), key);
        assert!(restored.is_model());
    }

    #[test]
    fn directories_cannot_construct_a_launch_context() {
        let mut directory: Model = serde_json::from_value(legacy()).unwrap();
        assert!(directory.is_catalog_dir());
        // Kind no longer changes incidentally when metadata is updated.
        directory.path = "/catalog/directory".into();
        assert!(directory.is_catalog_dir());
        assert!(crate::runtime::LaunchContext::new("server", &directory, &[]).is_err());
    }

    #[test]
    fn legacy_fastflowlm_round_trips_and_mixed_identities_are_rejected() {
        let mut file = legacy();
        file["flm"] = serde_json::json!({
            "tag": "qwen:1b", "installed": false, "repo": "org/repo",
            "labels": [], "vlm": false, "asr": false, "max_prefill_len": null
        });
        let model: Model = serde_json::from_value(file.clone()).unwrap();
        assert!(matches!(model.entry, CatalogEntry::Model(ModelSource::FastFlowLm(_))));
        assert_eq!(model.profile_key(), "flm:qwen:1b");
        let restored: Model =
            serde_json::from_value(serde_json::to_value(&model).unwrap()).unwrap();
        assert!(restored.flm().is_some());
        file["remote"] = serde_json::json!({"repo": "org/repo", "file": "model.gguf"});
        assert!(serde_json::from_value::<Model>(file).is_err());
    }
}
