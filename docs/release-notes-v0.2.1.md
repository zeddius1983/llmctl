# llmctl v0.2.1 — device selection and benchmarking

v0.2.1 adds explicit llama.cpp accelerator selection and a convenient way to
benchmark the selected model using the same device and GPU-offload settings.

## Highlights

- **Runtime device discovery** — llmctl runs `llama-server --list-devices` and
  exposes identifiers such as `ROCm0` and `Vulkan0` in the profile options.
- **Profile-level device selection** — the selected device is persisted with
  each model profile and emitted as `llama-server --device <name>`. The
  `default` value leaves device selection to llama.cpp.
- **Keyboard-friendly editing** — Enter opens a filterable device selector;
  `+`/`]` and `-`/`[` cycle forward or backward through discovered devices.
- **Optional llama-bench integration** — when `llama-bench` is installed beside
  `llama-server` or available on `PATH`, press `b` to benchmark the selected
  model in the foreground.
- **Matching benchmark configuration** — concrete profile `device` and
  `gpu-layers` values are forwarded to llama-bench as `--device` and `-ngl`.

## Upgrade notes

- Existing profiles require no migration. Their device value resolves to
  `default`, so current llama.cpp behavior is preserved until explicitly
  changed.
- `llama-bench` is optional. The benchmark shortcut is shown only when the
  binary is discovered.
- Benchmark output is displayed directly in the terminal; press Enter when it
  finishes to return to llmctl.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

Linux only; llama.cpp/GGUF only. vLLM remains a navigation stub.
