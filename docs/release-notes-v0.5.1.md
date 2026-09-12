# v0.5.1 — reliability, responsiveness, and safer persistence

v0.5.1 hardens the paths llmctl relies on every time it saves a profile,
refreshes a catalog, or starts and monitors a server. The visible workflows are
the same as v0.5.0, but failures are now reported instead of silently losing
state, slow runtime checks no longer stall keyboard input, and detached child
processes have an explicit owner.

This is a patch release: existing profiles, caches, downloads, and session
records remain compatible, with no migration or configuration changes needed.

## Safer persistence

Profile, download, and session-record updates now use atomic file replacement.
A failed write leaves the previous on-disk record intact instead of exposing a
partially written file, and the UI reports the error rather than pretending the
change was saved.

Profile editing also keeps the in-memory and fallback copies coherent when a
write fails. Creating, renaming, deleting, favoriting, resetting, and editing a
profile either complete as one operation or preserve the state from before the
attempt. A catalog refresh no longer discards the last usable profile fallback
while storage is temporarily unavailable.

Saved option values are validated again at the launch boundary. Invalid numeric
text, non-finite values, and values outside an option's range now stop with a
specific error before llmctl constructs or starts a runtime command. Valid
existing profiles continue to resolve as before.

## A responsive input loop

Catalog scans and readiness probes now run on bounded background workers instead
of the terminal input loop. Runtime catalog refreshes are serialized and
duplicate requests are coalesced, while readiness results are tied to the exact
session process and endpoint that requested them. Typing and navigation remain
responsive while a slow filesystem, runtime launcher, or health endpoint is
being checked.

Session log refreshes read bounded chunks, preventing a large or very active log
from monopolizing a tick. Interrupted Hugging Face downloads can be resumed and
retried without getting stuck behind their previous failed attempt.

## Explicit process ownership

The detached-session supervisor now retains and reaps only the child processes
it launched. It no longer changes SIGCHLD handling for the whole llmctl process,
which could interfere with runtime discovery, foreground commands, and failed
process launches elsewhere in the application.

Closing llmctl still leaves detached inference servers running. Any outstanding
child waits are handed to background reapers on exit, preserving the existing
server lifecycle without a process-wide signal side effect.

## Runtime and interface fixes

- FastFlowLM profiles using omitted enum defaults launch correctly instead of
  treating the omitted value as invalid.
- Runtime backends now own their catalog, refresh, transfer, deletion, command,
  and option behavior behind stable runtime identifiers.
- Catalog directories and launchable model sources use explicit typed states,
  while the compatibility layer continues to read existing cache data.
- Browser, modal, download, and session-view state have separate owners, so one
  input modal cannot accidentally leak state into another workflow.
- llama.cpp command construction now lives beside its backend and uses named
  requests, reducing positional-argument mixups without changing valid emitted
  commands.
- Session metadata and uptime once again stay padded against the right pane
  edge.

## Upgrade notes

- No profile, cache, download-record, or session-record migration is required.
- No key bindings or valid runtime command lines changed.
- A previously saved invalid numeric option may now block launch with an error;
  edit or reset that option to continue.
- Initial discovery still completes before the first frame. Later refreshes and
  readiness checks are the work moved into the background.
- CI now checks formatting, warning-free builds, the default test suite, and
  strict Clippy on every pull request and push to `main`.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

- The help overlay is still clipped on a 24-row terminal.
- The Session Manager Detail pane still repeats information already visible in
  its row.
- Linux only: detached sessions depend on `setsid`, `/proc`, and POSIX signals.
- Phases 4 and 5 (log search, startup-failure classification, and broader pane
  filtering) remain deferred.

---

**Full changelog:** https://github.com/zeddius1983/llmctl/compare/v0.5.0...v0.5.1
