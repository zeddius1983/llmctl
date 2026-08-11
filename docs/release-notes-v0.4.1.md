# v0.4.1 — Muse Glimmer 30B and dFlash speculative decoding

v0.4.1 adds support for **Meta's Muse Glimmer 30B**, which llmctl now discovers,
downloads, configures, and launches end to end — with its dFlash drafter and
multimodal projector attached automatically, and speculative decoding on by
default.

Getting there meant teaching llmctl llama.cpp's `draft-dflash` speculation,
adding the launch options a published multi-GPU configuration needs, and
migrating the deprecated `--no-mmap` switch.

## Muse Glimmer 30B

`unsloth/Muse-Glimmer-30B-GGUF` publishes fourteen quantizations of the model,
one shared `dflash-kquant.gguf` drafter, and an `mmproj-*.gguf` projector. Select
any quantization and llmctl attaches both companions, downloads them with the
model, defaults `spec-type` to `draft-dflash`, and sets `spec-draft-n-max` to the
drafter's block size — the `-md`/`--spec-type`/`--mmproj` wiring that otherwise
has to be typed by hand.

That one-drafter-for-every-quantization shape is a companion layout neither of
llmctl's existing pairing rules could express, which is what the rest of this
release is about.

**Requires llama.cpp b10353 or newer.** The `muse-glimmer` architecture landed
upstream in `62bf73d2` (#26841); older builds fail at model load with
`unknown model architecture: 'muse-glimmer'`.

## Highlights

- **dFlash drafters as companions (ADR-014).** A `dflash-*.gguf` is a companion
  attribute of the model it drafts for, like an MTP sidecar or a projector —
  never its own catalog leaf. Its filename names no base model, so it is paired
  by directory locally (only where that directory publishes a single model
  family) and repository-wide on the Hub, which is what lets one drafter serve
  every quantization in a repository. It is hidden as a standalone artifact and
  downloaded with the model it belongs to.
- **A cached drafter sets the default.** With a drafter on disk, `spec-type`
  defaults to `draft-dflash`. A `Drafter` enum maps the *resolved* spec type to
  the companion that fills `--spec-draft-model`, so MTP and dFlash can coexist
  on one model and the option decides which is loaded.
- **`spec-draft-n-max` defaults to the drafter's block size**, read from the
  companion's `dflash.block_size` GGUF metadata, instead of llama.cpp's 3. See
  below for why that matters.
- **New llama.cpp options** — `split-mode` (`-sm`), `tensor-split` (`-ts`),
  `parallel` (`-np`), and `sleep-idle-seconds`, all omitted at their `default`
  sentinel. Together with the dFlash options these reproduce the published
  two-GPU Muse Glimmer configuration from a profile.
- **`load-mode` (`-lm`) replaces `--no-mmap`**, matching upstream's
  consolidation of `--mmap`, `--mlock`, and `--direct-io` into one option
  (`none`, `mmap`, `mlock`, `mmap+mlock`, `dio`).

## Why the n-max default changed

dFlash emits a whole block per forward pass, so throughput keeps climbing toward
the block size even as the acceptance *rate* falls — the opposite of the
intuition that a lower acceptance rate means a worse draft. Measured on a Radeon
8060S (llama.cpp b10353, Muse-Glimmer-30B-UD-Q4_K_XL, 256 tokens, greedy,
ctx 32768, `-np 1`):

| `--spec-draft-n-max` | Throughput | Acceptance rate | Mean accepted |
|---|---|---|---|
| no drafter | 11.2 t/s | — | — |
| 3 (llama.cpp's default) | 16.3 t/s | 0.66 | 3.0 |
| 8 | 27.4 t/s | 0.35 | 3.9 |
| 16 (this drafter's block size) | 28.5 t/s | 0.24 | 4.5 |

Roughly 2.5x undrafted, and ~12 t/s better than accepting llama.cpp's default.
llama.cpp clamps n-max above the block size regardless, so the block size is
both the best value and the ceiling.

## Refusals instead of failed launches

Two dFlash failure modes are caught before `llama-server` starts, because both
fail in ways that do not point at the cause:

- **No cached drafter.** `--spec-draft-hf` accepts only a `repo[:quant]`
  selector — there is no `--hf-file` equivalent for the draft model — so an
  unqualified `dflash-*.gguf` cannot be addressed remotely at all. It must be
  downloaded first, and `launch_blocker` says so. Worse, llama.cpp *silently*
  disables `draft-dflash` when no draft model is loaded, so the alternative is a
  server that starts, runs at undrafted speed, and never explains why.
- **A binary that does not support it.** `draft-dflash` is sniffed from the
  cached `--help`, like `--hf-repo` and `--mmproj-auto` already are. A
  model-aware default must never turn a working undrafted launch into one the
  binary rejects, so the default only applies when the configured `llama-server`
  advertises the spec type, and an explicitly saved `draft-dflash` on an older
  binary is refused with the reason.

## Upgrade notes

- **`mmap: off` profiles are migrated, not dropped.** They resolve as
  `load-mode: none` through a new `RuntimeBackend::legacy_value` hook — a
  key-level counterpart to `normalize_legacy`, consulted only when a profile
  carries no value under the current key. Saved profiles need no edit.
- **`--load-mode` requires a recent `llama-server`.** Binaries too old to
  advertise it are refused unless the option sits at its omitted default, rather
  than being launched with a flag they will reject.
- **Existing drafters are picked up on rescan** (`F5`), including for models
  already in the catalog.
- **Muse Glimmer needs llama.cpp b10353 or newer**, as above. That is a
  llama.cpp requirement rather than an llmctl one — llmctl launches the model,
  the architecture support is upstream's.

## Also in this release

The GitHub Release body is now generated from the checked-in
`docs/release-notes-v*.md` instead of being written by hand after each tag, so
the release page and the repository can no longer drift, and re-running the
workflow for a tag stops discarding its notes. The release-notes H1 convention
is now `# v<version> — <title>` for every file, which is what
`parse-changelog` accepts.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

- **dFlash aborts on llama.cpp's Vulkan backend** —
  `pre-allocated tensor (output.weight) in a buffer (Vulkan0) that cannot run
  the operation (NONE)`, with or without `--spec-draft-ngl 0`. ROCm works.
  Undrafted throughput is the same on both backends on the test machine, so the
  speedup above is dFlash rather than the backend. This is an upstream issue,
  noted here because the crash looks like an llmctl bug.
- **`draft-dspark` is selectable but untested** — it is exposed alongside
  `draft-dflash` because llama.cpp accepts it; no dSpark drafter was available
  to verify pairing or defaults against.
- **Directory-scoped pairing needs a single model family per directory.** A
  folder holding two unrelated models plus one `dflash-*.gguf` is ambiguous, so
  no pairing is made rather than guessing.
- **Linux only.** `setsid`, `/proc` sampling, and POSIX signals are load-bearing.
- Phases 4 and 5 (log search, startup-failure classification) remain deferred.

---

**Full changelog:** https://github.com/zeddius1983/llmctl/compare/v0.4.0...v0.4.1
