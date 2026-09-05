# v0.5.0 — model storage, live throughput, and the session log

v0.5.0 is about what happens *after* a model is running. Until now llmctl could
find models, configure them, and start them; it could not tell you how fast one
was going, show you what it was doing, or take a model back off the disk.

Three things arrive together, all in the panes you already use: `D` removes a
model's files, the Session Manager reports each server's `tg` and `pp` rates in
aligned columns, and `l` swaps the Detail pane for a live tail of the selected
session's log.

## Remove a model from disk (`D`)

Deleting a model by hand is guesswork where it matters most. In the Hugging Face
cache a model is a set of hash-named blobs — `blobs/8f3a…` says nothing about
which model it belongs to, and the file you browsed to is a symlink whose bytes
are somewhere else entirely.

`D` in the Model pane plans the removal, shows it in one line — `Remove
Muse-Glimmer-30B-UD-Q4_K_XL.gguf (15.1 GB) from disk?` — and touches nothing
until you answer. `y` or Enter goes ahead; every other key backs out.

- Works on scanned GGUFs, the Hugging Face blob cache, and FastFlowLM model
  directories, following snapshot symlinks so the blob is what is counted and
  what is removed.
- **A companion another quantization still needs is kept**, and the quoted size
  excludes it. So are your profiles: re-download the model and your settings are
  still there.
- A model with a live session, or one currently downloading, is refused until
  you stop it.

The plan is anchored to files, never to names: to the snapshot the selected
artifact was found in, to what is cached on disk rather than what the last
catalog refresh recorded, and — for the live-session guard — to a session's
absolute paths, download record, and every absolute path in its argv. ADR-015
records why each of those is the question that decides the answer.

## Throughput in the Session Manager

Every session now shows how fast it is going: `tg` (token generation, decode)
and `pp` (prompt processing, prefill), in tokens per second.

Nothing to enable and no restart. Both runtimes already print their own
per-request timings to the session log, and llmctl reads them there — llama.cpp's
`prompt eval time` / `eval time` lines, FastFlowLM's usage block. `/metrics` is
off by default and could not be turned on for a session llmctl merely
rediscovered; `/slots` carries no timings at all.

What you see is the runtime's own figure for its most recent request, unaveraged.
llmctl times nothing itself, and could not usefully: wall-clock timing cannot
tell generating apart from idling, so a server that produced 20 tokens in a
second and then sat quiet would appear to slow down while doing nothing.

Session rows became columns — **model · profile · port · size · backend · tg ·
pp · uptime** — with figures right-aligned so digits stack down the pane, and
**sessions grouped under the runtime serving them**. As a pane narrows the
columns shed by worth rather than right-to-left: size and backend first, since
neither changes once a session is up. Within a row nothing is ever omitted; a
server that has answered nothing shows `tg --.-- t/s`, so it still lines up with
a busy one.

## The session log, beside the list (`l` / `→`)

`L` opened the log full screen, which hides every other session — the right
shape for reading back through a startup failure, the wrong one for "what is
this thing doing right now?"

`l` or `→` now swaps the right-hand column between the Detail pane and a live
tail of the selected session's log, and swaps back. `L` still gives the log the
whole screen.

The tail reads no files of its own. The Session Manager already polls every
session's log each tick for the rates above, and now keeps the last lines of
what it read instead of dropping them — so the pane costs one push per new line
and nothing per frame, where re-reading the file would mean tens of megabytes
every tick.

## Room, and the lack of it

- The right-hand column takes 43% of the pane and **disappears below the 44
  columns it needs to say anything**. Narrower than that it stops answering
  questions while still taking width from the list, so the list takes the whole
  width and `l` opens the log full screen instead.
- **Below 80×24, llmctl says so** — one centred line naming what it needs and
  what there is — rather than drawing a frame nothing fits in. Degradation has a
  bottom: at 24×8 the browser is three panes of six characters.
- The model name column now takes what the names in the pane actually need,
  dropping a column to seat a name whole rather than cutting one while keeping a
  backend that never changes.

## Upgrade notes

- **`l`, `→`, and Enter in the Session Manager no longer open the full-screen
  log.** They toggle the log pane beside the list. `L` is the full-screen view.
- **80×24 is now a hard minimum.** Below it llmctl draws its size requirement
  instead of an interface. Nothing else changed about how it runs.
- Sessions started by an older llmctl show no size or compute backend — both are
  recorded at launch, because neither can be recovered from a running server.
  Restart the session and both appear.
- No config, profile, or session-record format changed. Nothing to migrate.

## Also in this release

- `llama-server --version` is read from llama.cpp's own `version:` line rather
  than off the top of stderr. A `distrobox`-exported binary is a shell wrapper
  that prints `Starting container... [ OK ]` there when the container is cold,
  and that banner was being shown as the runtime's version.
- `presence-penalty` joins the llama.cpp option table.
- The help overlay's key column pads in terminal columns, so `e / Enter` and
  `Home/End` are no longer printed flush against their own descriptions, and the
  popup follows its longest row instead of cutting it.
- The README leads with three screenshots of a real session and is a third
  shorter, having handed the reference material back to the app and the docs.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

- **The help overlay is clipped on a 24-row terminal.** It is about 44 rows of
  content, and 24 rows is now a supported size, so the sections past Options are
  cut off. Either it scrolls or it lays out in two columns; filed on the roadmap.
- **The Detail pane still restates the row beside it.** Nine of its fifteen
  facts are now in the columns. Trimming it is roadmap work; the log pane does
  not depend on it.
- **Throughput needs one finished request.** A server that has answered nothing
  since llmctl started shows placeholders — there is no timing in its log to
  read yet.
- **Linux only.** `setsid`, `/proc` sampling, and POSIX signals are load-bearing.
- Phases 4 and 5 (log search, startup-failure classification) remain deferred.

---

**Full changelog:** https://github.com/zeddius1983/llmctl/compare/v0.4.1...v0.5.0
