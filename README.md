# llmctl

A keyboard-driven terminal UI (TUI) for discovering, configuring, launching, and
managing local LLM inference servers — in the style of [Yazi](https://github.com/sxyazi/yazi),
[Lazygit](https://github.com/jesseduffield/lazygit), and `systemctl`.

The goal: **never hand-type a complex inference-server command again.** Browse
your models, tune launch options with live validation, start detached servers,
and watch them from a built-in session manager.

> **Status:** v0.4.0. Two runtimes ship on **Linux**: **llama.cpp + GGUF**
> (CPU/GPU) and **FastFlowLM** (`flm`, AMD XDNA2 NPU). Others (vLLM, Ollama, …)
> are future work behind the `RuntimeBackend` trait.

## Features

- **Two runtimes, one workflow** — llama.cpp for CPU/GPU GGUF inference, and
  FastFlowLM (`flm`) for models on an AMD XDNA2 NPU. Everything runtime-specific
  — binary discovery, option vocabulary, templates, catalog, launch/chat/benchmark
  commands, readiness checks — sits behind a `RuntimeBackend` trait, so both
  runtimes browse, configure, launch, and monitor identically.
- **Yazi-style navigation** — a sliding three-column view over the hierarchy
  `Runtime ▸ source ▸ provider/repository ▸ Model ▸ Profile ▸ Options`, driven entirely from the keyboard
  (`hjkl`, `g`/`G`, drill in / back out).
- **Model discovery** — recursively scans your configured directories, or (when
  none are configured) well-known locations (llama.cpp cache, HuggingFace hub,
  LM Studio, `~/models`).
  Reads GGUF headers for architecture, context length, quantization, and embedded
  chat template; detects integrated MTP heads and pairs `mtp-*.gguf` and
  `mmproj-*.gguf` companions with their base models; dedupes multi-shard models
  and sums their sizes. `F5` to rescan.
- **Physical model catalog** — mirrors discovery below
  `~/.config/llmctl/models` using source-aware folders, safe manifests, model
  symlinks, and per-model YAML profiles. Press `/` to search recursively from
  the current local catalog directory.
- **Online Hugging Face catalog** — browse `online ▸ huggingface` like a local
  directory. It lazily caches 30 trending llama.cpp-compatible GGUF repositories
  and their artifacts; MTP/projector companions remain attributes of their base
  model, and starting a remote model lets llama.cpp download it into the standard
  Hugging Face cache.
- **Profiles & options** — built-in, read-only templates (Default, Chat, Coding,
  Long Context, Server) that fork into per-model editable instances on first edit.
  Edit options with live validation, cycle enums/flags in place, and adjust
  numerics with `+`/`-`/`[`/`]` or jump to default/min/max with `Home`/`End`.
  All edits auto-save, scoped per **runtime + model**.
- **Launch command builder** — assembles the exact `llama-server` invocation from
  the resolved options. `y` previews and yanks the command to your clipboard
  (OSC 52); options left at their default are omitted so llama.cpp's own defaults
  apply.
- **FastFlowLM on the NPU** — the `flm` catalog, grouped by capability label
  (reasoning, vision, tool-calling, audio, embeddings) under the same
  `local`/`online` split. NPU-specific options (power mode, prefill chunk,
  queue length) and templates, `flm serve` sessions, `flm run` chat, and `flm
  bench` benchmarking. Downloads are llmctl's own, so they resume. The XDNA
  driver grants a single hardware context, so llmctl runs one FastFlowLM model
  at a time and says which session holds the device.
- **Detached sessions** — `s` launches a server in its own process group
  (`setsid`), with stdout/stderr redirected to a per-session log file and
  automatic port-conflict resolution. Sessions are rediscovered across restarts.
- **Session manager** (`t`) — live status (Downloading / Starting / Running /
  Crashed), PID,
  port, uptime, and `/proc`-sampled CPU & memory; a `/health` probe promotes
  Downloading → Starting → Running. Stop (`x`), kill (`K`), restart (`R`), copy endpoint (`c`),
  and tail logs (`L`).

## Requirements

- **Linux** (`setsid`, `/proc` sampling, and POSIX signals).
- **At least one runtime.** Each is optional and discovered independently; a
  runtime that is missing or unusable is still listed, with the reason shown in
  the status line instead of failing at launch.
  - **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — `llama-server` on
    your `$PATH` (or an absolute path in the config). `llama-cli` beside it
    enables the chat shortcut (`C`), `llama-bench` the benchmark shortcut (`b`).
  - **[FastFlowLM](https://github.com/FastFlowLM/FastFlowLM)** — `flm` on your
    `$PATH`, plus an AMD XDNA2 NPU with a working driver stack. `flm validate`
    gates launching, so a missing driver or too low a memlock limit is reported
    up front. `flm` may be a wrapper script (a distrobox entry point, say);
    llmctl re-acquires the real server process either way.
- **Rust** (edition 2024) to build — install via [rustup](https://rustup.rs).

## Install

Build a release binary from source:

```sh
git clone https://github.com/zeddius1983/llmctl.git
cd llmctl
cargo build --release
```

The binary lands at `target/release/llmctl`. Copy it onto your `$PATH`, e.g.:

```sh
install -Dm755 target/release/llmctl ~/.local/bin/llmctl
```

Or install straight from the checkout:

```sh
cargo install --path .
```

## Usage

Just run it:

```sh
llmctl
```

Navigate `Runtime ▸ Model ▸ Profile ▸ Options`, tune a profile, then press `s`
to launch (or `y` to copy the command). Press `?` at any time for the keybinding
overlay.

### Keybindings

| Key | Action |
|-----|--------|
| `j` / `k` | Move down / up |
| `l` / `→` | Drill into selection |
| `h` / `←` | Back up a level |
| `g` / `G` | First / last item |
| `/` | Search recursively in the current catalog directory |
| `s` | Sort online models: Trending / Most likes / Most downloads |
| `d` | Download the selected online GGUF file |
| **Profiles** | |
| `a` | Create profile |
| `r` | Rename (custom profiles only) |
| `D` | Duplicate profile |
| `d` | Delete custom / reset built-in profile |
| `f` | Toggle favorite |
| **Options** | |
| `e` | Edit / cycle value |
| `-` / `+`, `[` / `]` | Decrement / increment |
| `Home` / `End` | Default·min / max |
| **Launch & sessions** | |
| `s` | Start server |
| `C` | Chat in terminal (`llama-cli` / `flm run`) |
| `b` | Benchmark the selected model (`llama-bench` / `flm bench`, when available) |
| `y` | Yank launch command |
| `t` | Session manager |
| `x` / `K` | Stop / kill a server; cancel a download |
| `R` | Restart a server or resume a download |
| `L` | View logs |
| `c` | Copy endpoint |
| **General** | |
| `F5` | Rescan / reload |
| `?` / `q` | Help / quit |

### Launch options

Each runtime exposes its own curated option set; the sections below cover
llama.cpp, then FastFlowLM.

#### llama.cpp

A curated set of `llama-server` flags, including context size,
GPU layers, device selection (`--device`, with a selector populated by
`llama-server --list-devices`), sampling (`temperature`, `top-p`, `top-k`,
`min-p`, `repeat-penalty`),
threads, batch size, flash attention, reasoning, KV cache types (`--cache-type-k`
/ `--cache-type-v`), `--no-mmap` (handy for ROCm/AMD GPUs), host/port, and
speculative decoding (`--spec-type`, `--spec-draft-n-max`, `--spec-draft-n-min`).
Local models with integrated MTP heads default to `draft-mtp`; a same-directory
`mtp-<base>.gguf` sidecar is passed through `--spec-draft-model` automatically,
including repositories that omit the base artifact's quantization suffix from
the sidecar name. A compatible local `mmproj-*.gguf` is passed through
`--mmproj`. For uncached Hub models, recent llama.cpp builds auto-discover root
MTP and projector companions from `-hf`; downloaded companions become explicit
local paths on subsequent launches.
Any option left at its default value is omitted from the command line.

#### FastFlowLM

An NPU has no GPU layers or flash attention, so `flm` gets its own vocabulary:
power mode (`--pmode`: powersaver / balanced / performance / turbo), context
length, prefill chunk length, NPU queue length, socket limit, preemption, CORS,
the ASR and embedding companion models, vision pre-resize, and host/port.
Templates (Default, Chat, Long Context, Server, Low Power) mirror llama.cpp's
intent in FastFlowLM's dialect. Options are clamped to what the selected model
actually supports — its trained context length and maximum prefill.
`--host`/`--port` are always emitted, because llmctl needs the concrete endpoint
to health-check the server and re-acquire it after a restart.

## Configuration

llmctl follows the XDG base-directory spec and runs with **zero setup**. On the
first run it creates `~/.config/llmctl/config.toml` with the llama.cpp cache,
Hugging Face, LM Studio, and `~/models` sources. Edit that file to add a source:

```toml
[[models.sources]]
name = "nas"
path = "/mnt/nas/llms"
layout = "directory" # auto, directory, flat, lm-studio, or hugging-face

[runtime.llama_cpp]
# Binary name (resolved on $PATH) or an absolute path.
binary = "llama-server"

# FastFlowLM runs models on an AMD XDNA2 NPU. Its catalog comes from `flm`
# itself, so the model sources above do not apply to it.
[runtime.fastflowlm]
binary = "flm"

[defaults]
host = "127.0.0.1"
port = 8000
```

### On-disk locations

| Path | Purpose |
|------|---------|
| `~/.config/llmctl/config.toml` | Configuration |
| `~/.config/llmctl/config.yaml` | Ignored legacy configuration; archive after migrating anything useful |
| `~/.config/llmctl/models/` | Managed source tree, symlinks, and YAML profiles |
| `~/.local/state/llmctl/` | Session records, logs, and profile migration fallback |
| `~/.cache/llmctl/` | Model & runtime scan cache |
| `~/.config/flm/models/` | FastFlowLM's own models, owned by `flm` (or `$FLM_MODEL_PATH`) |

The generated file explicitly lists the standard locations so they are easy to
inspect and extend. Older `[models].paths` arrays remain supported, but named
`[[models.sources]]` entries provide stable catalog names and layout control.
Your `$HOME` is never scanned wholesale.

### Online models

Under the llama.cpp runtime, enter `online`, then `huggingface`. Selecting the
source fetches 30 trending llama.cpp-compatible GGUF repositories. Repositories
appear as flat `provider/repository` rows with likes and download counts;
entering one fetches its GGUF variants. Choose an artifact, configure a normal
profile, and press `s`. llmctl launches llama.cpp with `--hf-repo` and
`--hf-file`, then links the downloaded file from the standard Hugging Face
cache into the managed catalog. `mtp-*` and `mmproj-*` files stay hidden as
standalone artifacts; the publisher's root MTP default and a compatible
projector are attached to each base model. Direct downloads include those
companions, while native `-hf` launches use llama.cpp's companion auto-discovery.

Press `d` on an uncached online GGUF to download it without starting a server.
Multiple models can download concurrently in a Downloads pane below Sessions.
The left jobs column is split 70/30 between servers and downloads, with one
continuous up/down selection across both panes. Each download row shows
aggregate shard progress; `x` cancels the selected transfer while preserving
its partial files, and `R` (or `d` on the model) resumes it. Completed rows
retain the final cache path in their detail pane. Incomplete jobs are recorded
under `online/huggingface/.downloads` in the managed model directory. After an
llmctl restart they return as `Interrupted`, with progress reconstructed from
the Hub cache; press `R` to continue them.

The online repository pane is titled `Trending`, `Most likes`, or `Most downloads`.
Inside a repository, the GGUF files pane uses the standard `Model` title, like
local repositories. Press `s` to cycle between Hub trending score, most likes,
and most downloads. A view change or online `F5` discards generated online layout
metadata and fetches a clean first page for the active view; profile YAML and
download records and model data are preserved. `/` performs debounced server-side search
across Hugging Face.
Hub search results remain transient until Enter saves the selected repository;
closing the search does not expand the local catalogue. Inside a repository it
searches only fetched GGUF artifacts. Online results never mix into local
folder searches. Set `HF_TOKEN` in the environment for gated/private models;
llmctl never persists the token.

## Roadmap

Done: TUI skeleton, model/runtime discovery, profiles & options, and launch &
session management. Planned next: log search & startup-failure classification,
incremental search/filters, and polish. See [docs/roadmap.md](docs/roadmap.md)
for the full picture and [docs/decisions.md](docs/decisions.md) for the
architectural decision records.

## License

[MIT](LICENSE)
