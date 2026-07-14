# llmctl - Functional Requirements (MVP)

## Overview

llmctl is a keyboard-driven terminal user interface for discovering, configuring, launching, monitoring, and managing local Large Language Models.

The user experience is inspired by Yazi, Lazygit, and systemctl.

The first release focuses exclusively on **llama.cpp** support and GGUF models.

Future versions may support:

* vLLM
* Ollama
* LM Studio
* SGLang
* ExLlamaV2
* Additional inference providers

---

# Goals

The primary goal is to eliminate the need to remember and manually type complex llama.cpp commands.

Users should be able to:

1. Discover GGUF models.
2. Configure model launch parameters.
3. Save reusable profiles.
4. Launch llama.cpp servers.
5. Manage running sessions.
6. Inspect logs.
7. Reuse previous configurations.
8. Restart or stop servers.
9. Browse compatible Hugging Face GGUF repositories as an online catalog and
   launch selected artifacts through llama.cpp's native downloader.

Everything should be available from a single terminal application.

## Online model catalog

The llama.cpp hierarchy contains the virtual source `online ▸ huggingface`.
Selecting it loads 30 trending models filtered for GGUF and llama.cpp
compatibility, including text-only and multimodal pipelines. Repository
contents are fetched lazily, cached into the managed catalog, and use the same
Model → Profile → Options workflow as local models. Authentication is inherited
from `HF_TOKEN` and is never persisted by llmctl. `/` searches the Hub
server-side when invoked from the online hierarchy. Search results are
transient; only the repository selected with Enter is added to the persistent
online catalogue.

Repository artifacts named `mtp-*.gguf` or `mmproj-*.gguf` are companions, not
standalone models. The preferred MTP drafter and multimodal projector are
associated with each compatible base artifact. Native `-hf` launches use
llama.cpp's root-MTP and projector auto-discovery; direct downloads include the
selected companions, and cached launches pass their local paths explicitly.

The online repository pane title reflects its active view: `Trending`,
`Most likes`, or `Most downloads`. A repository's GGUF files pane uses the
standard `Model` title. `s` cycles through Hub trending score, likes, and
download count. Changing view or pressing `F5` anywhere in
the online hierarchy resets all generated online catalog metadata and fetches a
clean first page. User profiles and the standard Hugging Face model cache are
never deleted by this reset.

Pressing `d` on an uncached online GGUF downloads the artifact immediately
without launching a server. Multiple downloads may run concurrently in a
Downloads pane below the server Sessions pane. The jobs column uses a 70/30
vertical split and one continuous up/down selection across both panes. Each
split artifact reports aggregate byte and percentage progress on one line. `x`
cancels the selected transfer without deleting partial blobs; `R`, or `d` on
the model again, resumes it. Completion refreshes the model as a cached local
file while retaining its online identity and profiles. Incomplete jobs persist
as JSON records in the managed online catalogue and are restored as
`Interrupted` after restart. Restored progress is derived from actual Hub-cache
blob sizes, jobs never auto-resume, and online refresh/sort operations preserve
the records.

For local catalog directories, `/` searches recursively only below the current
directory. Local searches exclude the virtual online catalog; selecting or
entering `online ▸ huggingface` switches `/` to an isolated Hub-wide search.
Online repositories are displayed as flat `provider/repository` rows, followed
by their GGUF artifacts when entered. Inside a repository, search is limited to
its fetched artifacts.

---

# User Interface

The application consists of five permanent panes.

```text
┌ Runtime ──────┬ Model ────────┬ Profile ───────┬ Options ───────┬ Info ─────────────┐
│ llama.cpp     │ qwen3.gguf    │ Default        │ ctx-size       │ Preview           │
│               │ gemma.gguf    │ Coding         │ gpu-layers     │                   │
│               │ mistral.gguf  │ Chat           │ temperature    │                   │
└───────────────┴───────────────┴────────────────┴───────────────┴───────────────────┘
```

The Info pane is always visible and always located on the far right.

Navigation progresses from left to right.

```text
Runtime
  → Model
      → Profile
          → Options
```

Each entity behaves similarly to a file manager item.

---

# Entity Model

## Runtime

Represents an inference backend.

MVP:

```text
llama.cpp
```

Future:

```text
vLLM
Ollama
LM Studio
SGLang
ExLlamaV2
```

Runtime behaves like a directory.

Selecting a runtime enters the Model pane.

### Runtime Preview

The Info pane displays:

* Runtime name
* Description
* Installed version
* Executable path
* Supported model formats

The preview should also display:

```bash
llama-server --help
```

captured and cached.

---

## Model

Represents a discovered GGUF model.

Examples:

```text
Qwen3-32B-Q6_K.gguf
Gemma-27B-Q4_K_M.gguf
GPT-OSS-20B-Q8.gguf
```

Models behave like directories.

Selecting a model enters the Profile pane.

### Model Preview

The Info pane displays:

* Name
* Full path
* File size
* Architecture
* Quantization
* Context length
* Chat template information
* Integrated or sidecar MTP availability
* Multimodal projector availability
* Last modified date

when detectable.

---

## Profile

Profiles represent reusable launch configurations.

Built-in profiles:

```text
Default
Chat
Coding
Long Context
Server
```

Profiles behave like directories.

Selecting a profile enters the Options pane.

### Profile Preview

The Info pane displays resolved option values.

Example:

```text
ctx-size: 32768
gpu-layers: 999
temperature: 0.7
top-p: 0.95
top-k: 40
flash-attn: true
```

### Profile Management

Users can:

* Create profile
* Rename profile
* Duplicate profile
* Delete profile
* Favorite profile

Profiles are scoped to:

```text
runtime + model
```

### Profile Persistence

Profiles are automatically saved.

No explicit save action is required.

---

## Options

Options represent editable configuration values.

Examples:

```text
ctx-size
gpu-layers
temperature
top-p
top-k
min-p
repeat-penalty
threads
batch-size
flash-attn
host
port
```

Options behave like editable files.

Editing a value automatically updates the selected profile.

### Option Preview

The Info pane displays:

* Current value
* Default value
* Allowed range
* Description
* Equivalent CLI argument

Example:

```text
ctx-size

Current: 32768
Default: 4096

CLI:
--ctx-size

Maximum context window size.
```

---

# Launch Workflow

A launch requires:

```text
Runtime
 → Model
   → Profile
     → Options
```

The user launches the selected configuration.

Example generated command:

```bash
llama-server \
  -m qwen3.gguf \
  --ctx-size 32768 \
  --temp 0.7 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8000
```

Before execution users may inspect the generated command.

---

# Session Manager

Session Manager is a dedicated screen for managing running inference processes.

Inspired by:

* Yazi Tasks
* btop
* systemctl

Accessible globally.

Shortcut:

```text
t
```

---

## Session Manager Layout

```text
┌ Sessions ─────────────────────────────────────────────────────┬ Info ───────────┐
│ ● qwen3-coding                          port:8000  12m       │ PID: 14231      │
│ ● gemma-chat                            port:8001   3h       │ Running         │
│ ✖ mistral-long-context                  port:8002           │ Crashed         │
└──────────────────────────────────────────────────────────────┴────────────────┘
```

Status indicators:

```text
● Running
⇩ Downloading (67%)
◐ Starting
✖ Crashed
■ Stopped
```

---

## Session Metadata

Each session displays:

* Session name
* Runtime
* Model
* Profile
* PID
* Port
* Status
* Uptime
* CPU usage
* Memory usage
* Log file

Example:

```text
Runtime: llama.cpp
Model: qwen3-32b-q6_k.gguf
Profile: Coding

PID: 14231
Port: 8000

Status: Running
Uptime: 2h 17m
Memory: 23.8 GB
CPU: 140%
```

---

## Session Operations

Supported actions:

* View details
* Open logs
* Stop
* Restart
* Kill
* Copy endpoint URL

Example endpoint:

```text
http://127.0.0.1:8000/v1
```

---

## Session Detail View

Displays:

* Runtime
* Model
* Profile
* Resolved options
* Generated command
* Environment
* Logs
* Resource usage

---

## Session Persistence

When llmctl restarts:

* Running sessions are rediscovered
* Session metadata is reconstructed
* Stale records are removed

The user should never lose visibility of a running server.

---

# Process Management

llmctl is responsible for lifecycle management of llama.cpp processes.

Supported process types:

```text
Server Mode
Chat Mode (future)
```

MVP focuses on Server Mode.

---

## Start Process

Shortcut:

```text
s
```

Actions:

* Validate configuration
* Generate command
* Create log file
* Launch process
* Register session

---

## Stop Process

Shortcut:

```text
x
```

Actions:

* Send SIGTERM
* Wait configurable timeout
* Update session status

---

## Restart Process

Shortcut:

```text
R
```

Actions:

* Stop existing process
* Relaunch using stored configuration

---

## Kill Process

Shortcut:

```text
K
```

Actions:

* Send SIGKILL
* Mark session terminated

---

## Session State Detection

Supported states:

```text
Running
Stopped
Downloading
Starting
Crashed
Unknown
```

---

# Logs

Every session receives a dedicated log file.

Example:

```text
~/.local/state/llmctl/logs/
```

---

## Log Features

Users can:

* Open logs
* Tail logs
* Search logs
* Copy logs

Shortcut:

```text
L
```

---

## Error Detection

Common startup failures should be highlighted.

Examples:

* Port already in use
* Model file missing
* Out of memory
* Unsupported argument
* Invalid model
* Failed GPU initialization
* Vulkan backend unavailable
* CUDA backend unavailable

---

# Model Discovery

## Supported Formats

MVP:

```text
GGUF
```

Only GGUF models are shown.

---

## Search Locations

User configurable.

Example:

```toml
[models]
paths = [
  "~/models",
  "/mnt/models",
  "/data/models"
]
```

Recursive scanning occurs only within configured directories.

The application should never recursively scan the entire home directory by default.

---

## Discovery Metadata

Collected metadata:

* Path
* File size
* Quantization
* Architecture
* Modification time
* Integrated MTP metadata and matching `mtp-*.gguf` sidecars
* Matching `mmproj-*.gguf` multimodal projector sidecars

Results should be cached.

An `mtp-<base filename>.gguf` is an auxiliary speculative-decoding model, not a
standalone catalog entry. It is paired with the same-directory base GGUF and
passed to llama.cpp with `--spec-draft-model` when `draft-mtp` is active.
Pairing accepts both an exact base filename and a base filename that extends the
sidecar stem with a quantization suffix.
Integrated MTP heads are detected from GGUF metadata, with the filename's MTP
token as a compatibility fallback.

An `mmproj-*.gguf` is likewise an auxiliary model rather than a standalone
catalog entry. A compatible same-directory projector is associated with an
unambiguous local model family and passed to llama.cpp with `--mmproj`.

Manual refresh must be supported.

---

## Runtime Discovery

The application should discover:

```bash
llama-server
llama-cli
```

Information collected:

* Executable path
* Version
* Help output

---

# Search and Filtering

Every pane supports incremental search.

Shortcut:

```text
/
```

---

## Model Filtering

Examples:

```text
name:qwen
quant:q6
size:>10GB
favorite:true
recent:true
```

---

## Profile Filtering

Examples:

```text
coding
chat
favorite
```

---

## Session Filtering

Examples:

```text
running
stopped
crashed
port:8000
```

---

# Configuration

llmctl follows the XDG specification.

## Paths

```text
~/.config/llmctl/config.toml
~/.local/state/llmctl/
~/.cache/llmctl/
```

---

## Example Configuration

```toml
[models]
paths = [
  "~/models",
  "/mnt/models"
]

[runtime.llama_cpp]
binary = "llama-server"

[defaults]
host = "127.0.0.1"
port = 8000
```

---

# Keyboard Shortcuts

## Navigation

```text
j / k          Move
h / l          Back / Enter
g / G          First / Last
```

## Search

```text
/              Search
n              Next result
N              Previous result
```

## Profile Management

```text
a              Create profile
r              Rename profile
D              Duplicate profile
d              Delete profile
```

## Options

```text
e              Edit option
```

## Launch

```text
s              Start
```

## Process Control

```text
x              Stop
R              Restart
K              Kill
```

## Logs

```text
L              Open logs
```

## Sessions

```text
t              Session Manager
```

## Refresh

```text
F5             Refresh models
```

## General

```text
?              Help
q              Quit
```

---

# Success Criteria

The MVP is successful when a user can:

1. Launch llmctl.
2. Select llama.cpp.
3. Select a discovered GGUF model.
4. Select a profile.
5. Adjust options.
6. Start a llama.cpp server.
7. Monitor logs.
8. Manage running sessions.
9. Restart or stop servers.

without manually writing or remembering llama.cpp commands.
