//! glui's tests, reported in TAP, as lem's are ([`crate::lem::testing`]):
//!
//! - every theme in `src/glui/themes/` gives every value, and a theme that
//!   lacks one is refused;
//! - pythia's approval HUD (`lem/approval.lem`, with `lem/host.lem`) and a
//!   page of the kinds that documents need (`lem/test/page.lem`) tick as
//!   `lem/test/approval_ticks.txt` and `lem/test/page_ticks.txt` say, in
//!   every theme, since rules never read one;
//! - glui's files and themes print back;
//! - the approval HUD and the page, compiled with glem's part
//!   ([`glem::compile`]) and read back, tick as written too, with each tick
//!   as JSON steps, as `pythia-ui lem --compile` gives them to the web.

use super::{complete, load, notes, FILES, THEMES};
use crate::glem;
use crate::lem::host;
use crate::lem::testing::{read, report, round_trip, tick_cases};

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
fn the_hud_and_the_page_tick_on_their_compiled_forms() {
  let (_, file, text) = THEMES[0];
  let programs: [(&[&str], &str); 2] = [
    (&["host.lem", "approval.lem"], "test/approval_ticks.txt"),
    (&["test/page.lem"], "test/page_ticks.txt"),
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
