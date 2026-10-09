# Contributing to Patina

Thanks for your interest in building the community's Substance Painter alternative.

**Current stage: Wave 0 — research & design.** The design docs are the product right now. The best ways to contribute today:

1. **Read [`SPEC.md`](SPEC.md)** — the project contract. If something's missing or wrong, open an issue.
2. **Review the research** in [`docs/research/`](docs/research/) — the Substance Painter feature inventory, competitor autopsies, Rust ecosystem survey, and brush-engine architecture report. Corrections with sources welcome.
3. **Feature requests** — open an issue with your workflow: what you paint, what tools you use, where they hurt.
4. **Docs fixes** — PRs welcome at any stage.
5. **Code** — opens when Wave 1 breaks ground (watch [`STATE.md`](STATE.md)). Until then, code PRs are parked so the spec and architecture land first.

## Ground rules

- **Rust only** for the core. FFI only where no viable crate exists (OIDN, OCIO, ufbx) and the license is compatible.
- **GPL-3.0-or-later.** By contributing you agree your work is licensed under GPL-3.0-or-later. Standalone library crates may be released under MIT.
- **Original implementations only.** No Adobe SDKs, no .sbsar/.spp format execution, no decompilation of Painter — Patina builds its own node-graph engine and its own file formats. Ideas are not copyrightable; code is.
- **Design docs live in the repo.** Decisions land in `DECISIONS.md` (append-only). If you disagree with a decision, open an issue rather than editing the log.
- **Be excellent to each other** — see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Development setup (when code opens)

```bash
git clone https://graphics.fun/repo/patina
cd patina
cargo build --workspace
cargo test --workspace
```

*(Setup docs expand when the workspace lands.)*
