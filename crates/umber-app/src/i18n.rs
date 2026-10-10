//! UI string catalogs (`docs/specs/i18n-design.md`): Fluent bundles over
//! `assets/i18n/<locale>.ftl`, a per-thread current locale, and the
//! [`tr`] / [`tr_args`] lookups every migrated UI string goes through.
//!
//! # Keys
//!
//! The code calls dotted keys (`group.name`). Fluent message ids cannot
//! contain dots, so a dotted key resolves to message `group`, attribute
//! `.name` (see `en-US.ftl`); an undotted key resolves to a message value.
//!
//! # Fallback (never a panic, never a blank)
//!
//! A lookup tries the current locale's bundle, then [`FALLBACK_LOCALE`],
//! then returns the key itself. Debug builds log each missing key once.
//! Only the top surface (menus, panel titles, primary buttons, two status
//! phrases) is migrated in v1; the long tail stays hardcoded and moves to
//! keys as it is touched. The completeness test below checks that every
//! key the source calls exists in en-US.
//!
//! # Where catalogs come from ([`I18n::load_all`])
//!
//! 1. The built-in catalogs, embedded at compile time: the installers do
//!    not ship `assets/`, so en-US and pt-BR always work.
//! 2. Every `*.ftl` in the discovered catalog dirs ([`catalog_dirs`]),
//!    the file stem naming the locale. A discovered file replaces the
//!    built-in bundle for its locale, so dev edits show without a rebuild
//!    and a new `<locale>.ftl` is a new picker entry.
//!
//! # The locale choice
//!
//! At startup: the persisted preference ([`locale_pref_path`]), else the
//! system locale, else en-US. The system locale is env-based in v1 —
//! `LC_ALL`, then `LC_MESSAGES`, then `LANG` (POSIX precedence) — because
//! egui exposes no OS-locale query. On Windows those vars are usually
//! unset, so the app starts in en-US until a language is picked; an OS
//! query there is the named follow-up.
//!
//! The preference is a one-line file next to the per-user brushes and
//! plugins dirs (`$XDG_DATA_HOME/umber/locale`, `%APPDATA%\umber\locale`).
//! It is an app preference, not a document property, so it does not live
//! with `DisplaySettings` in the `.umber` project.
//!
//! # Threading
//!
//! Fluent's default bundle is not `Sync`, so the current catalog is a
//! thread-local: the UI thread (where `main` installs it) is the only
//! one that renders strings. Any other thread lazily gets the built-in
//! catalogs in en-US — which is also what keeps tests deterministic
//! regardless of the machine's `LANG`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use fluent::{FluentArgs, FluentBundle, FluentResource, FluentValue};

/// The source-of-truth locale and the last resort before the bare key.
pub const FALLBACK_LOCALE: &str = "en-US";

/// The catalogs compiled into the binary.
const BUILTIN: &[(&str, &str)] = &[
    ("en-US", include_str!("../../../assets/i18n/en-US.ftl")),
    ("pt-BR", include_str!("../../../assets/i18n/pt-BR.ftl")),
];

/// The repo-relative catalog dir (correct when CWD is the repo root).
pub const CATALOG_DIR: &str = "assets/i18n";

/// The loaded catalogs plus the current locale.
pub struct I18n {
    locale: String,
    bundles: HashMap<String, FluentBundle<FluentResource>>,
    /// Keys already warned about (debug builds), so a missing key logs
    /// once instead of every frame.
    warned: RefCell<HashSet<String>>,
}

impl I18n {
    /// The built-in catalogs only, in en-US.
    pub fn builtin() -> Self {
        Self::from_sources(BUILTIN)
    }

    /// The built-in catalogs overlaid with every `*.ftl` under
    /// [`catalog_dirs`], in en-US (see the module docs).
    pub fn load_all() -> Self {
        let mut i18n = Self::builtin();
        for dir in catalog_dirs() {
            i18n.load_dir(&dir);
        }
        i18n
    }

    /// Bundles from in-memory `(locale, ftl source)` pairs, in en-US.
    pub fn from_sources(sources: &[(&str, &str)]) -> Self {
        let mut i18n = Self {
            locale: FALLBACK_LOCALE.to_owned(),
            bundles: HashMap::new(),
            warned: RefCell::new(HashSet::new()),
        };
        for (locale, source) in sources {
            i18n.add_catalog(locale, source);
        }
        i18n
    }

    /// Adds (or replaces) `locale`'s bundle from Fluent source. Syntax
    /// errors keep the entries that did parse and log the rest; an
    /// unparseable locale tag is skipped with a warning.
    pub fn add_catalog(&mut self, locale: &str, source: &str) {
        let resource = match FluentResource::try_new(source.to_owned()) {
            Ok(res) => res,
            Err((res, errors)) => {
                log::warn!("i18n: {locale}.ftl has syntax errors: {errors:?}");
                res
            }
        };
        let mut bundle = match locale.parse() {
            Ok(langid) => FluentBundle::new(vec![langid]),
            Err(err) => {
                log::warn!("i18n: skipping catalog with bad locale tag {locale:?}: {err}");
                return;
            }
        };
        // No U+2068/U+2069 isolation marks around placeables: egui draws
        // them as boxes, and none of our args are bidi-sensitive.
        bundle.set_use_isolating(false);
        if let Err(errors) = bundle.add_resource(resource) {
            log::warn!("i18n: {locale}.ftl has duplicate entries: {errors:?}");
        }
        self.bundles.insert(locale.to_owned(), bundle);
    }

    /// Loads every direct-child `*.ftl` in `dir` (file stem = locale).
    /// A missing/unreadable dir or file is a silent skip.
    pub fn load_dir(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "ftl"))
            .collect();
        paths.sort();
        for path in paths {
            let (Some(stem), Ok(source)) = (
                path.file_stem().and_then(|s| s.to_str()),
                std::fs::read_to_string(&path),
            ) else {
                continue;
            };
            self.add_catalog(stem, &source);
        }
    }

    /// The current locale tag.
    pub fn locale(&self) -> &str {
        &self.locale
    }

    /// Every loaded locale tag, sorted.
    pub fn locales(&self) -> Vec<String> {
        let mut tags: Vec<String> = self.bundles.keys().cloned().collect();
        tags.sort();
        tags
    }

    /// Switches to `wanted` (negotiated: see [`Self::negotiate`]).
    /// Returns the locale now active; an unavailable request leaves the
    /// current locale unchanged.
    pub fn set_locale(&mut self, wanted: &str) -> &str {
        if let Some(tag) = self.negotiate(wanted) {
            self.locale = tag;
        }
        &self.locale
    }

    /// The loaded locale best matching `wanted`: an exact tag match
    /// (case-insensitive), else the first loaded locale with the same
    /// language (`pt` or `pt-PT` → `pt-BR`), else `None`.
    pub fn negotiate(&self, wanted: &str) -> Option<String> {
        let tags = self.locales();
        if let Some(tag) = tags.iter().find(|t| t.eq_ignore_ascii_case(wanted)) {
            return Some(tag.clone());
        }
        let language = |t: &str| t.split('-').next().unwrap_or("").to_ascii_lowercase();
        let want = language(wanted);
        tags.into_iter()
            .find(|t| !want.is_empty() && language(t) == want)
    }

    /// The startup locale: the persisted preference, else the system
    /// locale, else en-US — each only if a catalog for it is loaded.
    pub fn choose_startup_locale(&mut self, pref: Option<&str>, system: Option<&str>) {
        let chosen = [pref, system]
            .into_iter()
            .flatten()
            .find_map(|wanted| self.negotiate(wanted))
            .unwrap_or_else(|| FALLBACK_LOCALE.to_owned());
        self.locale = chosen;
    }

    /// Whether `locale`'s bundle defines `key` (no fallback).
    #[cfg(test)]
    pub fn has_key(&self, locale: &str, key: &str) -> bool {
        self.bundles
            .get(locale)
            .is_some_and(|bundle| format_in(bundle, key, None).is_some())
    }

    /// `key` in the current locale, else en-US, else the key itself.
    pub fn lookup(&self, key: &str, args: Option<&FluentArgs>) -> String {
        for locale in [self.locale.as_str(), FALLBACK_LOCALE] {
            if let Some(text) = self
                .bundles
                .get(locale)
                .and_then(|bundle| format_in(bundle, key, args))
            {
                return text;
            }
        }
        if cfg!(debug_assertions) && self.warned.borrow_mut().insert(key.to_owned()) {
            log::warn!("i18n: missing key {key:?} (locale {})", self.locale);
        }
        key.to_owned()
    }
}

/// Formats `key` (dotted = message + attribute) from one bundle; `None`
/// when the message, attribute or value is absent.
fn format_in(
    bundle: &FluentBundle<FluentResource>,
    key: &str,
    args: Option<&FluentArgs>,
) -> Option<String> {
    let (id, attribute) = match key.split_once('.') {
        Some((id, attr)) => (id, Some(attr)),
        None => (key, None),
    };
    let message = bundle.get_message(id)?;
    let pattern = match attribute {
        Some(attr) => message.get_attribute(attr)?.value(),
        None => message.value()?,
    };
    let mut errors = Vec::new();
    let text = bundle.format_pattern(pattern, args, &mut errors);
    if !errors.is_empty() {
        log::warn!("i18n: formatting {key:?}: {errors:?}");
    }
    Some(text.into_owned())
}

thread_local! {
    static CURRENT: RefCell<I18n> = RefCell::new(I18n::builtin());
}

/// Makes `i18n` this thread's catalog (the UI thread calls this once at
/// startup via [`init`]).
pub fn install(i18n: I18n) {
    CURRENT.with(|c| *c.borrow_mut() = i18n);
}

/// Runs `f` against this thread's catalog. Do not call [`tr`] inside `f`
/// (the catalog is already borrowed).
pub fn with<R>(f: impl FnOnce(&mut I18n) -> R) -> R {
    CURRENT.with(|c| f(&mut c.borrow_mut()))
}

/// The UI string for `key` in the current locale (fallback rules in the
/// module docs). Pass a string literal so the completeness test sees it.
pub fn tr(key: &str) -> String {
    CURRENT.with(|c| c.borrow().lookup(key, None))
}

/// [`tr`] with Fluent arguments (`$name` placeables, plural selectors —
/// pass counts as numbers so plural rules apply).
pub fn tr_args<'a>(
    key: &str,
    args: impl IntoIterator<Item = (&'a str, FluentValue<'a>)>,
) -> String {
    let mut fluent_args = FluentArgs::new();
    for (name, value) in args {
        fluent_args.set(name, value);
    }
    CURRENT.with(|c| c.borrow().lookup(key, Some(&fluent_args)))
}

/// Startup: load every catalog, pick the locale (preference → system →
/// en-US) and install it on the calling (UI) thread.
pub fn init() {
    let mut i18n = I18n::load_all();
    startup_from(
        &mut i18n,
        locale_pref_path().as_deref(),
        system_locale().as_deref(),
    );
    log::info!(
        "i18n: locale {} (available: {})",
        i18n.locale(),
        i18n.locales().join(", ")
    );
    install(i18n);
}

/// Switches this thread's locale and persists the choice (a failed write
/// logs; the switch still applies for the session).
pub fn set_locale_persisted(wanted: &str) {
    let path = locale_pref_path();
    with(|i18n| persist_locale(i18n, wanted, path.as_deref()));
}

/// The startup choice against an explicit pref file (`None`: no home).
pub fn startup_from(i18n: &mut I18n, pref_path: Option<&Path>, system: Option<&str>) {
    let pref = pref_path.and_then(read_locale_pref);
    i18n.choose_startup_locale(pref.as_deref(), system);
}

/// Switches `i18n` to `wanted` and writes the locale now active (the
/// negotiated tag, not the request) to `pref_path`.
pub fn persist_locale(i18n: &mut I18n, wanted: &str, pref_path: Option<&Path>) {
    let active = i18n.set_locale(wanted);
    if let Some(path) = pref_path {
        if let Err(err) = write_locale_pref(path, active) {
            log::warn!("i18n: could not save locale to {}: {err}", path.display());
        }
    }
}

/// Catalog dirs in load order (later replaces earlier per locale): the
/// compile-time repo path, then the CWD-relative repo path, then an
/// `i18n/` dir next to the executable. Need not exist.
pub fn catalog_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(CATALOG_DIR),
        PathBuf::from(CATALOG_DIR),
    ];
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        dirs.push(exe_dir.join("i18n"));
    }
    dirs
}

/// The locale preference file: a sibling of the per-user brushes dir
/// (as the plugins dir is). `None` when no home is discoverable.
pub fn locale_pref_path() -> Option<PathBuf> {
    locale_pref_path_from(umber_brush::preset_library::user_preset_dir().as_deref())
}

/// Pure half of [`locale_pref_path`].
pub fn locale_pref_path_from(user_brush_dir: Option<&Path>) -> Option<PathBuf> {
    user_brush_dir
        .and_then(Path::parent)
        .map(|base| base.join("locale"))
}

/// The persisted locale tag, `None` if the file is missing or blank.
pub fn read_locale_pref(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let tag = text.trim();
    (!tag.is_empty()).then(|| tag.to_owned())
}

/// Persists `tag` (creating the parent dir).
pub fn write_locale_pref(path: &Path, tag: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, format!("{tag}\n"))
}

/// The system locale from the environment (`LC_ALL` → `LC_MESSAGES` →
/// `LANG`, first non-empty wins), as a BCP-47-style tag.
pub fn system_locale() -> Option<String> {
    let raw = ["LC_ALL", "LC_MESSAGES", "LANG"]
        .into_iter()
        .filter_map(|var| std::env::var(var).ok())
        .find(|v| !v.is_empty())?;
    posix_to_tag(&raw)
}

/// `pt_BR.UTF-8@euro` → `pt-BR`; `C` / `POSIX` / empty → `None`.
pub fn posix_to_tag(raw: &str) -> Option<String> {
    let base = raw.split(['.', '@']).next().unwrap_or("").trim();
    if base.is_empty() || base == "C" || base == "POSIX" {
        return None;
    }
    Some(base.replace('_', "-"))
}

/// Every key passed as a string literal to `tr(` / `tr_args(` in `source`
/// (the completeness test's scanner). The call name must not be the tail
/// of a longer identifier (`attr(` is not `tr(`).
#[cfg(test)]
pub fn scan_tr_keys(source: &str) -> Vec<String> {
    let mut keys = Vec::new();
    for call in ["tr(\"", "tr_args(\""] {
        let mut from = 0;
        while let Some(pos) = source[from..].find(call) {
            let start = from + pos;
            from = start + call.len();
            let preceded_by_ident = source[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
            if preceded_by_ident {
                continue;
            }
            if let Some(len) = source[from..].find('"') {
                keys.push(source[from..from + len].to_owned());
            }
        }
    }
    keys
}

/// The keys in `sources` (name, text) that en-US does not define, as
/// `(source name, key)` pairs.
#[cfg(test)]
pub fn missing_keys(i18n: &I18n, sources: &[(String, String)]) -> Vec<(String, String)> {
    let mut missing = Vec::new();
    for (name, text) in sources {
        for key in scan_tr_keys(text) {
            if !i18n.has_key(FALLBACK_LOCALE, &key) {
                missing.push((name.clone(), key));
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn fresh_tmpdir(tag: &str) -> PathBuf {
        let id = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "umber-app-i18n-test-{}-{}-{}",
            std::process::id(),
            id,
            tag
        ));
        std::fs::create_dir_all(&dir).expect("test tmpdir must be creatable");
        dir
    }

    /// The crate's own sources, `(file name, text)`.
    fn crate_sources() -> Vec<(String, String)> {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&src)
            .expect("src/ must be listable")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "rs"))
            .collect();
        files.sort();
        files
            .into_iter()
            .map(|p| {
                let text = std::fs::read_to_string(&p).expect("source must be readable");
                (p.file_name().unwrap().to_string_lossy().into_owned(), text)
            })
            .collect()
    }

    #[test]
    fn builtin_catalogs_parse_cleanly() {
        for (locale, source) in BUILTIN {
            if let Err((_, errors)) = FluentResource::try_new(source.to_string()) {
                panic!("{locale}.ftl: {errors:?}");
            }
        }
        assert_eq!(I18n::builtin().locales(), vec!["en-US", "pt-BR"]);
    }

    #[test]
    fn tr_resolves_en_by_default() {
        // A fresh thread's catalog is the built-in en-US one.
        assert_eq!(tr("menu.file"), "File");
        assert_eq!(tr("panel.layers"), "Layers");
        assert_eq!(tr("button.bake"), "Bake");
    }

    #[test]
    fn pt_br_returns_the_pt_string() {
        let mut i18n = I18n::builtin();
        assert_eq!(i18n.set_locale("pt-BR"), "pt-BR");
        assert_eq!(i18n.lookup("menu.file", None), "Arquivo");
        assert_eq!(i18n.lookup("panel.layers", None), "Camadas");
        // And through the thread-local `tr`.
        install(i18n);
        assert_eq!(tr("menu.view"), "Exibir");
        install(I18n::builtin());
    }

    #[test]
    fn missing_in_pt_falls_back_to_en_then_to_the_key() {
        let mut i18n = I18n::from_sources(&[
            (
                "en-US",
                "menu =\n    .file = File\n    .view = View\nplain = Plain\n",
            ),
            ("pt-BR", "menu =\n    .file = Arquivo\n"),
        ]);
        i18n.set_locale("pt-BR");
        assert_eq!(i18n.lookup("menu.file", None), "Arquivo");
        // Message present in pt, attribute only in en.
        assert_eq!(i18n.lookup("menu.view", None), "View");
        // Message only in en.
        assert_eq!(i18n.lookup("plain", None), "Plain");
        // Nowhere: the key itself.
        assert_eq!(i18n.lookup("menu.nonexistent", None), "menu.nonexistent");
        assert_eq!(i18n.lookup("no.such.key", None), "no.such.key");
        assert_eq!(i18n.lookup("", None), "");
    }

    #[test]
    fn args_and_plurals_format_without_isolation_marks() {
        let mut i18n = I18n::builtin();
        let count = |n: usize| {
            let mut a = FluentArgs::new();
            a.set("count", n);
            a
        };
        assert_eq!(
            i18n.lookup("status.baking", Some(&count(1))),
            "Baking 1 map…"
        );
        assert_eq!(
            i18n.lookup("status.baking", Some(&count(4))),
            "Baking 4 maps…"
        );
        i18n.set_locale("pt-BR");
        assert_eq!(
            i18n.lookup("status.baking", Some(&count(4))),
            "Fazendo bake de 4 mapas…"
        );
        // Through tr_args on this thread (en).
        let (loaded, failed) = (FluentValue::from(3), FluentValue::from(0));
        let line = tr_args("status.plugins", [("loaded", loaded), ("failed", failed)]);
        assert_eq!(line, "3 plugin(s) loaded, 0 failed");
    }

    #[test]
    fn every_called_key_exists_in_en() {
        let i18n = I18n::builtin();
        let sources = crate_sources();
        let called: usize = sources.iter().map(|(_, t)| scan_tr_keys(t).len()).sum();
        // The scan must actually see the migrated surface (main.rs alone
        // has ~30 call sites); a scanner that matches nothing passes
        // vacuously.
        assert!(called >= 30, "only {called} tr() call sites found");
        let missing = missing_keys(&i18n, &sources);
        assert!(
            missing.is_empty(),
            "keys missing from en-US.ftl: {missing:?}"
        );
    }

    #[test]
    fn completeness_check_catches_a_typod_key() {
        // Built at runtime so this file's own text holds no bad call site.
        let call = |key: &str| format!("ui.button({}(\"{key}\"));\n", "tr");
        let fixture = format!(
            "{}{}let s = {}(\"status.baking\", []);\nlet t = attr(\"x.y\");\n",
            call("menu.file"),
            call("menu.flie"),
            "tr_args"
        );
        assert_eq!(
            scan_tr_keys(&fixture),
            vec!["menu.file", "menu.flie", "status.baking"],
            "attr(\"…\") is not a tr call"
        );
        let missing = missing_keys(&I18n::builtin(), &[("fixture.rs".to_owned(), fixture)]);
        assert_eq!(
            missing,
            vec![("fixture.rs".to_owned(), "menu.flie".to_owned())]
        );
    }

    #[test]
    fn locale_round_trips_through_the_pref_file() {
        let dir = fresh_tmpdir("locale-pref");
        let path = locale_pref_path_from(Some(dir.join("brushes").as_path())).unwrap();
        assert_eq!(path, dir.join("locale"));
        assert_eq!(read_locale_pref(&path), None, "no file yet");
        let path = path.as_path();
        let absent = dir.join("absent");

        // The picker's path: a language-only request saves the negotiated
        // tag.
        let mut i18n = I18n::builtin();
        persist_locale(&mut i18n, "pt", Some(path));
        assert_eq!(i18n.locale(), "pt-BR");
        assert_eq!(read_locale_pref(path).as_deref(), Some("pt-BR"));

        // A fresh startup reads it back; the pref beats the system locale.
        let mut reloaded = I18n::builtin();
        startup_from(&mut reloaded, Some(path), Some("en-US"));
        assert_eq!(reloaded.locale(), "pt-BR");
        assert_eq!(reloaded.lookup("menu.file", None), "Arquivo");

        // An unavailable request keeps (and re-saves) the active locale.
        persist_locale(&mut reloaded, "fr-FR", Some(path));
        assert_eq!(read_locale_pref(path).as_deref(), Some("pt-BR"));

        // Switching back persists too.
        persist_locale(&mut reloaded, "en-US", Some(path));
        let mut again = I18n::builtin();
        startup_from(&mut again, Some(path), Some("pt-BR"));
        assert_eq!(again.locale(), "en-US");

        // No pref file: the system locale decides.
        let mut fresh = I18n::builtin();
        startup_from(&mut fresh, Some(absent.as_path()), Some("pt_BR"));
        assert_eq!(fresh.locale(), "en-US", "raw POSIX strings are not tags");
        startup_from(&mut fresh, None, Some("pt-BR"));
        assert_eq!(fresh.locale(), "pt-BR");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn startup_locale_negotiates_and_falls_back() {
        let mut i18n = I18n::builtin();
        i18n.choose_startup_locale(None, Some("pt"));
        assert_eq!(i18n.locale(), "pt-BR", "language-only match");
        i18n.choose_startup_locale(Some("de-DE"), Some("ja-JP"));
        assert_eq!(i18n.locale(), "en-US", "nothing available -> en-US");
        i18n.choose_startup_locale(Some("PT-br"), None);
        assert_eq!(i18n.locale(), "pt-BR", "case-insensitive");
        // An unavailable set_locale keeps the current one.
        assert_eq!(i18n.set_locale("fr-FR"), "pt-BR");
    }

    #[test]
    fn posix_locale_strings_map_to_tags() {
        assert_eq!(posix_to_tag("pt_BR.UTF-8").as_deref(), Some("pt-BR"));
        assert_eq!(posix_to_tag("de_DE@euro").as_deref(), Some("de-DE"));
        assert_eq!(posix_to_tag("en_US").as_deref(), Some("en-US"));
        assert_eq!(posix_to_tag("C"), None);
        assert_eq!(posix_to_tag("POSIX"), None);
        assert_eq!(posix_to_tag("C.UTF-8"), None);
        assert_eq!(posix_to_tag(""), None);
    }

    #[test]
    fn discovered_catalogs_add_and_replace_locales() {
        let dir = fresh_tmpdir("discover");
        std::fs::write(dir.join("de-DE.ftl"), "menu =\n    .file = Datei\n").unwrap();
        std::fs::write(dir.join("en-US.ftl"), "menu =\n    .file = Files!\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored").unwrap();
        let mut i18n = I18n::builtin();
        i18n.load_dir(&dir);
        assert_eq!(i18n.locales(), vec!["de-DE", "en-US", "pt-BR"]);
        assert_eq!(i18n.lookup("menu.file", None), "Files!");
        i18n.set_locale("de-DE");
        assert_eq!(i18n.lookup("menu.file", None), "Datei");
        // de lacks the key; the (replaced) en bundle is the fallback.
        assert_eq!(i18n.lookup("menu.view", None), "menu.view");
        std::fs::remove_dir_all(&dir).ok();
    }
}
