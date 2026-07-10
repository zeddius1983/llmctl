# llmctl

A keyboard-driven terminal UI (TUI) for discovering, configuring, launching, and
managing local LLM inference servers — in the style of [Yazi](https://github.com/sxyazi/yazi),
[Lazygit](https://github.com/jesseduffield/lazygit), and `systemctl`.

The goal: **never hand-type a complex `llama-server` command again.** Browse your
GGUF models, tune launch options with live validation, start detached servers,
and watch them from a built-in session manager.

> **Status:** v0.2.0. Targets **llama.cpp + GGUF on Linux**. vLLM support is in
> development; its configured binary is detected, while model discovery and
> launching remain unavailable until the integration is complete.

## Features

- **Yazi-style navigation** — a sliding three-column view over the hierarchy
  `Runtime ▸ source ▸ provider/repository ▸ Model ▸ Profile ▸ Options`, driven entirely from the keyboard
  (`hjkl`, `g`/`G`, drill in / back out).
- **Model discovery** — recursively scans your configured directories, or (when
  none are configured) well-known locations (llama.cpp cache, HuggingFace hub,
  LM Studio, `~/models`).
  Reads GGUF headers for architecture, context length, quantization, and embedded
  chat template; dedupes multi-shard models and sums their sizes. `F5` to rescan.
- **Physical model catalog** — mirrors discovery below
  `~/.config/llmctl/models` using source-aware folders, safe manifests, model
  symlinks, and per-model YAML profiles. Press `/` for global model search.
- **Profiles & options** — built-in, read-only templates (Default, Chat, Coding,
  Long Context, Server) that fork into per-model editable instances on first edit.
  Edit options with live validation, cycle enums/flags in place, and adjust
  numerics with `+`/`-`/`[`/`]` or jump to default/min/max with `Home`/`End`.
  All edits auto-save, scoped per **runtime + model**.
- **Launch command builder** — assembles the exact `llama-server` invocation from
  the resolved options. `y` previews and yanks the command to your clipboard
  (OSC 52); options left at their default are omitted so llama.cpp's own defaults
  apply.
- **Detached sessions** — `s` launches a server in its own process group
  (`setsid`), with stdout/stderr redirected to a per-session log file and
  automatic port-conflict resolution. Sessions are rediscovered across restarts.
- **Session manager** (`t`) — live status (Starting / Running / Crashed), PID,
  port, uptime, and `/proc`-sampled CPU & memory; a `/health` probe promotes
  Starting → Running. Stop (`x`), kill (`K`), restart (`R`), copy endpoint (`c`),
  and tail logs (`L`).

## Requirements

- **Linux** (the MVP uses `setsid`, `/proc` sampling, and POSIX signals).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — `llama-server` must be
  on your `$PATH` (or set its path in the config). `llama-cli` next to it enables
  the in-terminal chat shortcut (`C`).
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
| `/` | Search all models and jump to a result |
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
| `C` | Chat in terminal (`llama-cli`) |
| `y` | Yank launch command |
| `t` | Session manager |
| `x` / `K` | Stop / kill |
| `R` | Restart |
| `L` | View logs |
| `c` | Copy endpoint |
| **General** | |
| `F5` | Rescan / reload |
| `?` / `q` | Help / quit |

### Launch options

The MVP exposes a curated set of `llama-server` flags, including context size,
GPU layers, sampling (`temperature`, `top-p`, `top-k`, `min-p`, `repeat-penalty`),
threads, batch size, flash attention, reasoning, KV cache types (`--cache-type-k`
/ `--cache-type-v`), `--no-mmap` (handy for ROCm/AMD GPUs), host/port, and
speculative decoding (`--spec-type`, `--spec-draft-n-max`, `--spec-draft-n-min`).
Any option left at its default value is omitted from the command line.

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

[runtime.vllm]
binary = "vllm"

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

The generated file explicitly lists the standard locations so they are easy to
inspect and extend. Older `[models].paths` arrays remain supported, but named
`[[models.sources]]` entries provide stable catalog names and layout control.
Your `$HOME` is never scanned wholesale.

## Roadmap

Done: TUI skeleton, model/runtime discovery, profiles & options, and launch &
session management. Planned next: log search & startup-failure classification,
incremental search/filters, and polish. See [docs/roadmap.md](docs/roadmap.md)
for the full picture and [docs/decisions.md](docs/decisions.md) for the
architectural decision records.

## License

[MIT](LICENSE)
