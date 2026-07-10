# llmctl v0.2.0 — source-aware model catalog

v0.2.0 replaces the flat model list with a Yazi-style, source-aware catalog so
identically named GGUF files from different providers remain distinguishable.

## Highlights

- **Source-aware model tree** — browse `source ▸ provider ▸ repository ▸
  artifact` with variable-depth Miller columns. LM Studio and Hugging Face
  layouts are normalized; arbitrary configured folders preserve their relative
  directory structure.
- **Physical catalog** — llmctl mirrors discovered models under
  `~/.config/llmctl/models` using safe ownership manifests and symlinks while
  continuing to launch from the original model path.
- **Per-model YAML profiles** — model profiles now live below each catalog leaf
  instead of one flat `profiles.json`. Existing profiles migrate automatically,
  with a backup and write-failure fallback retained in the state directory.
- **Global model search** — press `/` from the Runtime or Model browser to search
  model names and their full source context, then jump directly into the tree.
- **Explicit source configuration** — first run creates
  `~/.config/llmctl/config.toml` with the llama.cpp cache, Hugging Face, LM
  Studio, and `~/models`. Named custom sources support `auto`, `directory`,
  `flat`, `lm-studio`, and `hugging-face` layouts.

## Upgrade notes

- Existing `config.toml` files are never overwritten.
- A `config.yaml` from the former implementation is ignored and retained as a
  backup because it may contain presets worth migrating manually.
- Legacy `~/.local/state/llmctl/profiles.json` entries are migrated when their
  model is discovered. `profiles.json.bak` preserves the pre-migration data.
- Catalog files are local derived state. Do not replace source model files with
  catalog symlinks or edit `.llmctl.yml` manifests by hand.

## Reliability and performance

- Handles multi-shard GGUFs, overlapping scan roots, sanitized-name and
  leaf/directory collisions, stale catalog entries, and deterministic Hugging
  Face snapshot selection (`refs/main` preferred).
- Profile mutations write only the affected YAML file, unchanged manifests are
  not rewritten, and search results are cached per query.
- Catalog write failures fall back to legacy JSON persistence rather than
  losing profile edits.

## Install

Download a prebuilt Linux binary from the GitHub release (musl is fully static),
or install from source with `cargo install --path .`. `llama-server` must be on
`PATH` or configured explicitly.

## Known limitations

Linux only; llama.cpp/GGUF only. vLLM remains a navigation stub. See the
[roadmap](https://github.com/zeddius1983/llmctl/blob/main/docs/roadmap.md) for
planned log search, structured filters, and additional runtimes.
