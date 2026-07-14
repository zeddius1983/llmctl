# llmctl v0.3.1 — MTP and multimodal companions

v0.3.1 teaches llmctl how GGUF companion files relate to their base models.
Multi-Token Prediction (MTP) heads and multimodal projectors now participate in
the normal discovery, profile, download, and llama.cpp launch workflow instead
of appearing as unrelated model entries.

## Highlights

- **Integrated MTP detection** — local GGUF headers carrying
  `nextn_predict_layers` metadata are recognized automatically, with an `MTP`
  filename-token fallback for older converters.
- **MTP sidecar pairing** — `mtp-*.gguf` files are hidden as auxiliary models
  and paired with their same-directory base GGUF. Pairing supports publisher
  names that omit the base model's quantization suffix.
- **Model-aware speculative defaults** — integrated and paired MTP models
  default `spec-type` to `draft-mtp`. Sidecars are passed to both
  `llama-server` and `llama-cli` with `--spec-draft-model`.
- **Multimodal projector companions** — compatible `mmproj-*.gguf` files are
  associated with an unambiguous local model family and passed through
  `--mmproj`, while projector files remain hidden from the model list.
- **Online companion discovery** — Hugging Face repository parsing keeps root
  MTP defaults, nested MTP precision variants, and projectors attached to each
  base artifact. Direct downloads include the selected companions.
- **Native and cached Hub launches** — uncached models use llama.cpp's root
  companion auto-discovery or `--spec-draft-hf` for nested MTP variants. Once
  cached, llmctl launches with explicit local MTP and projector paths.
- **Broader initial Hub view** — online discovery now fetches 30 repositories
  instead of 20. Hub-wide search remains available beyond the initial page.

## Upgrade notes

- Existing models and profiles require no migration. Press `F5` to rescan local
  model directories and populate newly discovered companion relationships.
- Use a recent llama.cpp build that advertises the required MTP/projector flags.
  llmctl reports a launch error when `--spec-draft-hf` or `--mmproj-auto` is
  needed but unavailable.
- Integrated MTP needs no separate draft-model path. A sidecar MTP model does;
  llmctl supplies it automatically when the matching file is present.
- llama.cpp and model-specific restrictions still apply when combining MTP,
  multimodal projectors, parallel decoding, or older GGUF conversions.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

- An uncached online artifact cannot be inspected for header-only integrated
  MTP metadata. Integrated MTP is confirmed after the GGUF is cached and scanned;
  explicit `mtp-*` companions are recognized before download.
- MTP and projector precision selection currently follows the publisher's root
  default and deterministic precision fallback; there is no per-model selector.
- The initial Hub list is one 30-model page. Pagination is deferred because
  Hub-wide search can locate repositories outside that page.
- Linux only; llama.cpp/GGUF only. vLLM remains a navigation stub.
