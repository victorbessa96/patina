# Internationalization (i18n) — Wave-6 Design

Wave-6's fifth slice. Written 2026-10-10 09:13 against the tree at
`18e8c46`. The audit's wave-6 list names i18n without a shape. The
honest v1 for an egui app: **fluent-rs (the Mozilla Fluent
standard) + a strings catalog**, not a framework lock-in.

## The shape

- **fluent-rs** (the pure-Rust Fluent implementation — the .ftl
  format, the same files Firefox/Blender-class apps use): the
  workspace gains fluent + fluent-syntax (both small, pure).
- **The catalog**: `assets/i18n/en-US.ftl` (the source of truth)
  + `pt-BR.ftl` (the first target — Bessa's locale, the honest
  first translation with a native reviewer built in) + the loading
  infrastructure for arbitrary additional locales
  (`assets/i18n/<locale>.ftl` auto-discovered).
- **The bundle API** (umber-app, a small i18n.rs): load at startup
  (the system locale per egui's detect, overridable via a
  settings row + persisted in DisplaySettings-adjacent state),
  `tr(key, args)` for every UI string. Fallback: the key itself
  when missing (never a panic, never a blank — the honest missing-
  string behavior) + a dev-mode warning listing the missing keys
  (a test greps the .ftl files vs the tr() call sites — the
  completeness test).
- **The migration**: the panels' hardcoded strings move to keys
  INCREMENTALLY — v1 migrates the top surface (menu labels,
  panel titles, the primary buttons: ~30 strings), the long tail
  migrates as touched. The completeness test only covers the keys
  the code actually calls; absent keys in a non-en locale fall
  back to en, then to the key. Documented: full-coverage i18n is
  a sustained-effort row, not a slice.

## Tests

1. The catalog loads; `tr("menu.file")` == the en value; a
   missing key returns the key itself (asserted).
2. pt-BR loads; `tr` with the pt locale returns the pt string;
   a key absent in pt falls back to en (asserted).
3. The completeness grep: every `tr("...")` call site's key
   exists in en-US.ftl (the test parses the .ftl + scans the
   source — the can-fail core: a typo'd key fails it).
4. The locale override persists through the settings (the
   round-trip).

## Build

One lean claw slice: the deps + the bundle + the two .ftl files
(the 30-string surface) + the tr() migration on the top surface +
the four tests. No framework, no macros beyond a thin `tr!`
shorthand.
