//! glui's tests, reported in TAP, as lem's are ([`crate::lem::testing`]):
//!
//! - every theme in `src/glui/themes/` gives every value, and a theme that
//!   lacks one is refused;
//! - pythia's approval HUD (`lem/approval.lem`, with `lem/host.lem`), a
//!   page of the kinds that documents need (`lem/test/page.lem`), and
//!   pythia's contexts (`lem/contexts.lem`) tick as
//!   `lem/test/approval_ticks.txt`, `lem/test/page_ticks.txt` and
//!   `lem/test/contexts_ticks.txt` say, in every theme, since rules never
//!   read one; and, in the first theme, on a runtime that forgets what its
//!   rules read before each tick, with the same facts after each;
//! - glui's files and themes print back;
//! - the three, compiled with glem's part ([`glem::compile`]) and read
//!   back, tick as written too, with each tick as JSON steps, as
//!   `pythia-ui lem --compile` gives them to the web;
//! - `glui.js`, the web look, is ASCII, writes no HTML, and runs no code
//!   that it's given;
//! - the looks' stylesheets, in `web/`, read only the variables that
//!   `glui.js` sets from a theme, which `theme.lem` names;
//! - the web look's pages, `web/demo.html` and `web/test.html`, and the
//!   contexts' sample pages, give themselves the CSP that pythia's node
//!   server sends its glui view's pages, and run only the crate's own
//!   files, and those that `make web` writes: no script or style inline;
//! - the compiled programs in `web/compiled/`, which the contexts' pages
//!   fetch, are what their files compile to now; and the contexts,
//!   compiled, go to `target/web/contexts_tests.js`, for
//!   `tools/contexts_test.js`, which tests `web/contexts.js`.
//!
//! The web look's own cases, which only JavaScript runs, go to
//! `target/web/glui_tests.js`, which `tools/glui_test.js` runs on `glui.js`
//! (`make js-test`): the HUD, the page, a form of glui's controls
//! (`lem/test/form.lem`) and the contexts, compiled, with their cases in a
//! stand-in DOM (`lem/test/*_dom.txt`); and each theme, compiled, with its
//! values as CSS (`lem/test/themes_css.txt`).

use super::{complete, load, notes, FILES, THEMES};
use crate::glem;
use crate::json::{self, object, Value as Json};
use crate::lem::host;
use crate::lem::testing::{
  cases, read, report, round_trip, tick_cases, tick_twice, unsafe_in_script, write_web,
};
use std::path::Path;

/// The themes in src/glui/themes/, glui's own and any other there: (file,
/// text).
fn themes() -> Vec<(String, String)> {
  let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/glui/themes");
  let mut names: Vec<String> = std::fs::read_dir(dir)
    .unwrap_or_else(|e| panic!("{dir}: {e}"))
    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
    .filter(|name| name.ends_with(".lem"))
    .collect();
  names.sort();
  let read = |name: &String| {
    let path = format!("{dir}/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
  };
  names
    .iter()
    .map(|name| (format!("src/glui/themes/{name}"), read(name)))
    .collect()
}

/// A runtime for glui's files, a theme, and `files` from lem/.
fn runtime(theme: (&str, &str), files: &[&str]) -> glem::Runtime {
  glem::Runtime::new(program(theme, files))
}

/// A program of glui's files, a theme, and `files` from lem/.
fn program(theme: (&str, &str), files: &[&str]) -> glem::Program {
  let texts: Vec<(String, String)> = files
    .iter()
    .map(|name| (format!("lem/{name}"), read(name)))
    .collect();
  let files: Vec<(&str, &str)> = texts
    .iter()
    .map(|(name, text)| (name.as_str(), text.as_str()))
    .collect();
  load(theme, &files).unwrap_or_else(|e| panic!("{}", e.text))
}

#[test]
fn every_theme_gives_every_value() {
  let themes = themes();
  for (_, file, _) in THEMES {
    assert!(themes.iter().any(|(name, _)| name == file), "{file}");
  }
  let mut results = Vec::new();
  for (name, text) in &themes {
    let mut runtime = runtime((name, text), &[]);
    results.push((name.clone(), complete(&mut runtime).err()));
  }
  report(results);
}

#[test]
fn a_theme_that_lacks_a_value_is_refused() {
  let short = THEMES[0].2.replace("Size(Radius, 4)", "");
  let mut runtime = runtime(("short.lem", &short), &[]);
  assert_eq!(
    complete(&mut runtime),
    Err("the theme gives no Size(Radius)".to_string())
  );
}

#[test]
fn the_programs_keep_their_keys() {
  // No rule may give one key two facts, nor one id two widgets, as far as
  // lem's checker can tell before a tick runs (src/lem/clash.rs).
  let (_, file, text) = THEMES[0];
  for files in [
    &["contexts.lem"][..],
    &["host.lem", "approval.lem"],
    &["test/page.lem"],
    &["test/form.lem"],
  ] {
    assert_eq!(
      program((file, text), files).clashes(),
      Vec::<String>::new(),
      "{files:?}"
    );
  }
}

#[test]
fn the_approval_hud_ticks_as_written_in_every_theme() {
  for (_, file, text) in THEMES {
    let mut runtime = runtime((file, text), &["host.lem", "approval.lem"]);
    tick_cases("test/approval_ticks.txt", |lines| {
      runtime.script(lines, &notes)
    });
  }
}

#[test]
fn the_page_ticks_as_written_in_every_theme() {
  for (_, file, text) in THEMES {
    let mut runtime = runtime((file, text), &["test/page.lem"]);
    tick_cases("test/page_ticks.txt", |lines| runtime.script(lines, &notes));
  }
}

#[test]
fn the_contexts_tick_as_written_in_every_theme() {
  for (_, file, text) in THEMES {
    let mut runtime = runtime((file, text), &["contexts.lem"]);
    tick_cases("test/contexts_ticks.txt", |lines| {
      runtime.script(lines, &notes)
    });
  }
}

#[test]
fn ticks_remembering_what_rules_read_are_ticks_that_forget_it() {
  let (_, file, text) = THEMES[0];
  let programs: [(&[&str], &str); 3] = [
    (&["host.lem", "approval.lem"], "test/approval_ticks.txt"),
    (&["test/page.lem"], "test/page_ticks.txt"),
    (&["contexts.lem"], "test/contexts_ticks.txt"),
  ];
  for (files, ticks) in programs {
    tick_twice(
      ticks,
      [runtime((file, text), files), runtime((file, text), files)],
      |runtime, lines| runtime.script(lines, &notes),
      |runtime| &mut runtime.lem,
    );
  }
}

#[test]
fn gluis_files_and_themes_print_back() {
  let files = FILES
    .iter()
    .map(|(name, text)| (name.to_string(), text.to_string()));
  let results = files
    .chain(themes())
    .map(|(name, text)| {
      let why = round_trip(&name, &text).err();
      (name, why)
    })
    .collect();
  report(results);
}

#[test]
fn the_hud_the_page_and_the_contexts_tick_on_their_compiled_forms() {
  let (_, file, text) = THEMES[0];
  let programs: [(&[&str], &str); 3] = [
    (&["host.lem", "approval.lem"], "test/approval_ticks.txt"),
    (&["test/page.lem"], "test/page_ticks.txt"),
    (&["contexts.lem"], "test/contexts_ticks.txt"),
  ];
  for (files, ticks) in programs {
    let form = glem::compile::compile(&program((file, text), files));
    let back = glem::compile::read(&form).unwrap_or_else(|e| panic!("{files:?}: {e}"));
    assert!(
      glem::compile::compile(&back) == form,
      "{files:?} reads back otherwise"
    );
    let mut runtime = glem::Runtime::new(back);
    tick_cases(ticks, |lines| {
      let steps = host::script_json(&mut runtime.lem.prog, lines);
      let script = host::read_steps(&mut runtime.lem, &steps)?;
      runtime.run(&script, &notes)
    });
  }
}

#[test]
fn the_web_looks_cases_for_javascript() {
  let (_, file, text) = THEMES[0];
  let programs: [(&[&str], &str); 4] = [
    (&["host.lem", "approval.lem"], "test/approval_dom.txt"),
    (&["test/page.lem"], "test/page_dom.txt"),
    (&["test/form.lem"], "test/form_dom.txt"),
    (&["contexts.lem"], "test/contexts_dom.txt"),
  ];
  let suites = programs.map(|(files, dom)| {
    let form = glem::compile::compile(&program((file, text), files));
    let cases = cases(dom).into_iter().map(|(title, lines, want)| {
      let steps = lines
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| json::parse(line).unwrap_or_else(|e| panic!("{dom}: {line}: {e}")));
      object([
        ("title", Json::from(title.as_str())),
        ("steps", Json::Array(steps.collect())),
        ("want", Json::from(want.as_str())),
      ])
    });
    object([
      ("name", Json::from(dom.trim_start_matches("test/"))),
      (
        "program",
        json::parse(&form).expect("the compiled form is JSON"),
      ),
      ("cases", Json::Array(cases.collect())),
    ])
  });
  let wants = cases("test/themes_css.txt");
  let themes = THEMES.map(|(name, file, text)| {
    let form = glem::compile::compile(&program((file, text), &[]));
    let want = wants
      .iter()
      .find(|(title, ..)| title == name)
      .map_or(String::new(), |(_, _, want)| want.clone());
    object([
      ("name", Json::from(name)),
      (
        "program",
        json::parse(&form).expect("the compiled form is JSON"),
      ),
      ("want", Json::from(want.as_str())),
    ])
  });
  let tests = object([
    ("suites", Json::Array(suites.into())),
    ("themes", Json::Array(themes.into())),
  ]);
  write_web("glui_tests.js", "GLUI_TESTS", "src/glui/tests.rs", &tests);
}

#[test]
fn glui_js_is_ascii_writes_no_html_and_runs_no_code() {
  let found = unsafe_in_script(include_str!("glui.js"));
  assert!(found.is_empty(), "glui.js has {found:?}");
}

/// The web look's stylesheets, and its pages': (file, text).
const STYLESHEETS: [(&str, &str); 5] = [
  ("web/glui.css", include_str!("../../web/glui.css")),
  ("web/native.css", include_str!("../../web/native.css")),
  ("web/log.css", include_str!("../../web/log.css")),
  ("web/pages.css", include_str!("../../web/pages.css")),
  ("web/contexts.css", include_str!("../../web/contexts.css")),
];

/// A name in CamelCase as CSS writes names, as glui.js does: PanelBack,
/// panel-back.
fn kebab(name: &str) -> String {
  let mut css = String::new();
  for (i, c) in name.chars().enumerate() {
    if c.is_ascii_uppercase() && i > 0 {
      css.push('-');
    }
    css.push(c.to_ascii_lowercase());
  }
  css
}

#[test]
fn the_stylesheets_read_only_what_glui_js_gives() {
  let (_, file, text) = THEMES[0];
  let prog = program((file, text), &[]);
  let names = |ty: &str| -> Vec<String> {
    let ctors = prog.lem.ctors.iter().filter(|c| &*c.ty == ty);
    ctors.map(|c| kebab(&c.name)).collect()
  };
  let mut given: Vec<String> = names("Paint")
    .into_iter()
    .chain(names("Metric"))
    .map(|name| format!("--glui-{name}"))
    .collect();
  given.extend(
    names("Tone")
      .iter()
      .map(|name| format!("--glui-{name}-text")),
  );
  given.sort();
  let mut results = Vec::new();
  // What theme.lem names is what glui.js gives, as themes_css.txt says.
  for (theme, _, want) in cases("test/themes_css.txt") {
    let mut set: Vec<String> = want
      .lines()
      .filter_map(|line| line.split(':').next())
      .filter(|name| name.starts_with("--glui-"))
      .map(str::to_string)
      .collect();
    set.sort();
    let why = (set != given).then(|| format!("glui.js gives\n{set:?}\nnot\n{given:?}"));
    results.push((format!("themes_css.txt: {theme}"), why));
  }
  for (name, css) in STYLESHEETS {
    let mut unknown: Vec<&str> = Vec::new();
    for (at, _) in css.match_indices("var(--") {
      let rest = &css[at + 4..];
      let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .unwrap_or(rest.len());
      let var = &rest[..end];
      let known = var.starts_with("--g-") || given.iter().any(|name| name == var);
      if !known && !unknown.contains(&var) {
        unknown.push(var);
      }
    }
    let open = css.matches('{').count();
    let why = if !css.is_ascii() {
      Some("it has a character beyond ASCII".to_string())
    } else if open != css.matches('}').count() {
      Some("its braces don't pair".to_string())
    } else if !unknown.is_empty() {
      Some(format!("it reads {unknown:?}, which glui.js doesn't set"))
    } else {
      None
    };
    results.push((name.to_string(), why));
  }
  report(results);
}

/// The web look's pages, and the contexts' sample pages; and their own
/// scripts: (file, text).
const PAGES: [(&str, &str); 4] = [
  ("web/demo.html", include_str!("../../web/demo.html")),
  ("web/test.html", include_str!("../../web/test.html")),
  (
    "web/contexts-index.html",
    include_str!("../../web/contexts-index.html"),
  ),
  (
    "web/contexts-log.html",
    include_str!("../../web/contexts-log.html"),
  ),
];
const PAGE_SCRIPTS: [(&str, &str); 3] = [
  ("web/demo.js", include_str!("../../web/demo.js")),
  ("web/test.js", include_str!("../../web/test.js")),
  ("web/contexts.js", include_str!("../../web/contexts.js")),
];

/// The CSP that pythia's node server sends with its glui view's pages
/// (pythia/interaction/node/view.py), which the web look's pages give
/// themselves, but `frame-ancestors`, which only a header can give.
const CSP: &str = "<meta http-equiv=\"Content-Security-Policy\" \
  content=\"default-src 'none'; script-src 'self'; style-src 'self'; \
  connect-src 'self'; base-uri 'none'; form-action 'none'\">";

/// What's wrong with a page: its CSP; script or style inline; and a script
/// or stylesheet that isn't a file in the crate, or under target/, which
/// `make web` writes.
fn page_faults(page: &str) -> Vec<String> {
  let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/web"));
  let lower = page.to_ascii_lowercase();
  let mut faults = Vec::new();
  if !page.is_ascii() {
    faults.push("a character beyond ASCII".to_string());
  }
  if !page.contains(CSP) {
    faults.push("not the CSP that the node server sends".to_string());
  }
  for inline in ["<style", " style=", "javascript:"] {
    if lower.contains(inline) {
      faults.push(format!("`{inline}`"));
    }
  }
  for (at, _) in lower.match_indices(" on") {
    let rest = &lower[at + 3..];
    let word = rest.len()
      - rest
        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .len();
    if word > 0 && rest[word..].starts_with('=') {
      faults.push(format!("a handler inline, on{}", &rest[..word]));
    }
  }
  let mut files = Vec::new();
  for (at, _) in lower.match_indices("<script") {
    let end = lower[at..]
      .find("</script>")
      .map_or(lower.len(), |end| at + end);
    let tag = &page[at..end];
    match tag
      .strip_prefix("<script src=\"")
      .and_then(|rest| rest.strip_suffix("\">"))
    {
      Some(src) if !src.contains('"') => files.push(src),
      _ => faults.push(format!("a script that isn't a file alone: {tag}")),
    }
  }
  let link = "<link rel=\"stylesheet\" href=\"";
  for (at, _) in lower.match_indices(link) {
    let rest = &page[at + link.len()..];
    files.push(&rest[..rest.find('"').unwrap_or(0)]);
  }
  for file in files {
    if !file.starts_with("../target/") && !dir.join(file).is_file() {
      faults.push(format!("{file} isn't a file"));
    }
  }
  faults
}

#[test]
fn the_web_pages_run_only_the_crates_files() {
  let mut results = Vec::new();
  for (name, page) in PAGES {
    let faults = page_faults(page);
    results.push((
      name.to_string(),
      (!faults.is_empty()).then(|| faults.join("\n")),
    ));
  }
  for (name, script) in PAGE_SCRIPTS {
    let found = unsafe_in_script(script);
    results.push((
      name.to_string(),
      (!found.is_empty()).then(|| format!("it has {found:?}")),
    ));
  }
  let broken = [
    "<script>alert(1)</script>",
    "<script src=\"x.js\" async></script>",
    "<script src=\"../src/nowhere.js\"></script>",
    "<link rel=\"stylesheet\" href=\"nowhere.css\">",
    "<div style=\"color:red\"></div>",
    "<button onclick=\"go()\">",
  ];
  for (i, html) in broken.iter().enumerate() {
    let page = PAGES[0].1.replace("</body>", &format!("{html}\n</body>"));
    let faults = page_faults(&page);
    let why = (faults.len() != 1).then(|| format!("{html} gives {faults:?}"));
    results.push((format!("a broken page, {}", i + 1), why));
  }
  report(results);
}

#[test]
fn the_compiled_files_are_up_to_date() {
  let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/web/compiled");
  let (_, file, text) = THEMES[0];
  let contexts = glem::compile::compile(&program((file, text), &["contexts.lem"]));
  let form = json::parse(&contexts).expect("the compiled form is JSON");
  let tests = object([("program", form)]);
  write_web(
    "contexts_tests.js",
    "CONTEXTS_TESTS",
    "src/glui/tests.rs",
    &tests,
  );
  let mut files = vec![("contexts.json".to_string(), contexts)];
  for (name, file, text) in THEMES {
    let form = glem::compile::compile(&program((file, text), &[]));
    files.push((format!("theme-{name}.json"), form));
  }
  let results = files
    .into_iter()
    .map(|(name, want)| {
      let path = format!("{dir}/{name}");
      let why = match std::fs::read_to_string(&path) {
        Err(e) => Some(format!("{path}: {e}; `make compiled` writes it")),
        Ok(got) if got != want => {
          Some("it's out of date: `make compiled` writes it anew".to_string())
        }
        Ok(_) => None,
      };
      (format!("web/compiled/{name}"), why)
    })
    .collect();
  report(results);
}
