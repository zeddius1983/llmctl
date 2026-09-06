//! Catalog scans run independently of the terminal loop, one writer per runtime.
use crate::discovery::ModelSource;
use crate::domain::Model;
use crate::runtime::{CatalogCtx, RuntimeBackend, RuntimeId};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

struct Request {
    backend: Arc<dyn RuntimeBackend>,
    sources: Vec<ModelSource>,
    cache_path: PathBuf,
    models_dir: PathBuf,
    view: usize,
    reload: bool,
}

pub(super) struct CatalogResult {
    pub runtime: RuntimeId,
    pub models: Vec<Model>,
}

pub(super) struct CatalogJobs {
    active: std::collections::HashSet<RuntimeId>,
    queued: HashMap<RuntimeId, Request>,
    tx: mpsc::Sender<CatalogResult>,
    rx: mpsc::Receiver<CatalogResult>,
}

impl Default for CatalogJobs {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { active: Default::default(), queued: Default::default(), tx, rx }
    }
}

impl CatalogJobs {
    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    pub fn request(&mut self, backend: Arc<dyn RuntimeBackend>, ctx: &CatalogCtx) {
        let mut request = Request {
            backend,
            sources: ctx.sources.to_vec(),
            cache_path: ctx.cache_path.into(),
            models_dir: ctx.models_dir.into(),
            view: ctx.view,
            reload: ctx.reload,
        };
        let id = request.backend.id();
        if self.active.contains(&id) {
            if let Some(previous) = self.queued.get(&id) {
                request.reload |= previous.reload;
            }
            self.queued.insert(id, request);
        } else {
            self.start(request);
        }
    }

    fn start(&mut self, request: Request) {
        self.active.insert(request.backend.id());
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let ctx = CatalogCtx {
                sources: &request.sources,
                cache_path: &request.cache_path,
                models_dir: &request.models_dir,
                view: request.view,
                reload: request.reload,
            };
            let models = request.backend.models(&ctx);
            let _ = tx.send(CatalogResult { runtime: request.backend.id(), models });
        });
    }

    pub fn poll(&mut self) -> Vec<CatalogResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.rx.try_recv() {
            self.active.remove(&result.runtime);
            if let Some(next) = self.queued.remove(&result.runtime) {
                // The latest request supersedes the old result, and starts only
                // after its writer has stopped touching the cache.
                self.start(next);
            } else {
                results.push(result);
            }
        }
        results
    }
}
