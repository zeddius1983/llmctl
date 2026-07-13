# llmctl v0.3.0 — online Hugging Face catalog

v0.3.0 brings Hugging Face GGUF discovery and downloading into llmctl's normal
directory-style model workflow. Browse compatible repositories, search the
Hub, configure the usual model profiles, launch through llama.cpp, or download
artifacts without starting a server.

## Highlights

- **Virtual online source** — enter `online ▸ huggingface` under llama.cpp to
  browse the first 20 compatible repositories without maintaining a separate
  model screen.
- **Hub ranking views** — press `s` to cycle through Trending, Most likes, and
  Most downloads. Repository rows retain the Hub's ranking and show muted likes
  and download counts.
- **Scoped search** — `/` searches Hugging Face globally from the repository
  list and narrows to the current repository after entering it. Local searches
  remain recursive only below the current local directory.
- **Lazy, selection-only cataloguing** — repository details and GGUF metadata
  are fetched only when needed. Search results remain transient until selected,
  preventing broad searches from permanently expanding the local catalogue.
- **Consistent artifact browsing** — local and online GGUF files show aligned
  quantization, aggregate size, and filename columns, sorted by size.
- **Native remote launch** — uncached artifacts launch with llama.cpp's
  `--hf-repo` and `--hf-file` support. Sessions report download percentage
  before transitioning to model-loading `Starting`.
- **Download without launching** — press `d` on an online GGUF to download it
  into the standard Hugging Face cache. Multiple downloads can run at once,
  including aggregate progress for split artifacts.
- **Cancellation, resume, and restart recovery** — `x` safely cancels a
  download while preserving partial blobs; `R` or another `d` resumes it.
  Incomplete jobs return as `Interrupted` after restarting llmctl, with progress
  reconstructed from cached bytes.
- **Unified job management** — the Session Manager now stacks Sessions and
  Downloads in a 70/30 split with continuous keyboard navigation and a shared
  Detail pane.

## Upgrade notes

- Existing local models and profiles require no migration.
- Online catalog metadata and profiles live below the managed
  `~/.config/llmctl/models/online/huggingface` tree. Sort changes and online
  `F5` rebuild generated metadata while preserving profiles, download records,
  partial files, and completed model data.
- A recent llama.cpp build advertising `--hf-repo` and `--hf-file` is required
  for remote launch. Set `HF_TOKEN` in the environment for gated or private
  repositories; llmctl never persists the token.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

- The online repository view currently fetches one 20-model page; pagination,
  recent sorting, and structured size/quantization filters remain follow-ups.
- Automatic repository companion discovery (for example MTP draft models and
  multimodal projectors) is not yet preserved when an online artifact is later
  launched strictly by its cached local path.
- Linux only; llama.cpp/GGUF only. vLLM remains a navigation stub.
