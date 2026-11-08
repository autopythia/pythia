//! Tests of glem's part of the compiled form ([`super::compile`]), reported
//! in TAP ([`crate::lem::testing`]):
//!
//! - the task list (`lem/test/tasks.lem`), compiled and read back, compiles
//!   to the same text;
//! - its ticks (`lem/test/tasks_ticks.txt`) run on the compiled form read
//!   back, with each tick as JSON steps, and give what they give on the
//!   program itself;
//! - a compiled form whose `glem` member doesn't fit lem's is refused;
//! - `glem.js` is ASCII, writes no HTML, and runs no code that it's given.
//!
//! The ticks, as JSON, with the compiled form and the facts that each
//! tick's joins visit, are what glem's runtime in JavaScript must give
//! too: they go to `target/web/glem_tests.js`, which `tools/glem_test.js`
//! runs on `glem.js` (`make js-test`); with the cases of
//! `lem/test/tasks_dom.txt`, which only JavaScript runs: the task list in a
//! stand-in DOM (`tools/stand_in_dom.js`), with clicks, focus and keys.

use super::compile::{compile, read};
use super::{load, no_notes, Runtime};
use crate::json::{self, object, Value as Json};
use crate::lem::host;
use crate::lem::testing::{cases, read as read_file, report, unsafe_in_script, write_web};

/// The task list's compiled form.
fn tasks() -> String {
  let text = read_file("test/tasks.lem");
  let prog = load(&[("lem/test/tasks.lem", &text)]).unwrap_or_else(|e| panic!("{}", e.text));
  compile(&prog)
}

#[test]
fn the_compiled_form_reads_back_as_it_was() {
  let form = tasks();
  let why = match read(&form) {
    Err(e) => Some(format!("it doesn't read back: {e}")),
    Ok(back) if compile(&back) != form => {
      Some(format!("it reads back as\n{}\nnot\n{form}", compile(&back)))
    }
    Ok(_) => None,
  };
  report(vec![("tasks".to_string(), why)]);
}

#[test]
fn ticks_run_on_the_compiled_form() {
  let form = tasks();
  let mut runtime = Runtime::new(read(&form).unwrap_or_else(|e| panic!("{e}")));
  let (mut results, mut ticks) = (Vec::new(), Vec::new());
  for (title, lines, want) in cases("test/tasks_ticks.txt") {
    let steps = host::script_json(&mut runtime.lem.prog, &lines);
    let got = host::read_steps(&mut runtime.lem, &steps)
      .and_then(|script| runtime.run(&script, &no_notes))
      .unwrap_or_else(|e| format!("error: {e}"));
    let why = (got != want).then(|| format!("the tick gives\n{got}\nnot\n{want}"));
    results.push((title.clone(), why));
    ticks.push(object([
      ("title", Json::from(title.as_str())),
      ("steps", steps),
      ("want", Json::from(want.as_str())),
      ("visits", Json::from(runtime.lem.visits())),
    ]));
  }
  let dom = cases("test/tasks_dom.txt")
    .into_iter()
    .map(|(title, lines, want)| {
      let steps = lines
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| json::parse(line).unwrap_or_else(|e| panic!("tasks_dom.txt: {line}: {e}")));
      object([
        ("title", Json::from(title.as_str())),
        ("steps", Json::Array(steps.collect())),
        ("want", Json::from(want.as_str())),
      ])
    });
  let program = json::parse(&form).expect("the compiled form is JSON");
  let tests = object([
    ("program", program),
    ("ticks", Json::Array(ticks)),
    ("dom", Json::Array(dom.collect())),
  ]);
  write_web(
    "glem_tests.js",
    "GLEM_TESTS",
    "src/glem/compile_test.rs",
    &tests,
  );
  report(results);
}

#[test]
fn a_glem_member_that_doesnt_fit_is_refused() {
  let form = tasks();
  let surface = form
    .find("\"surface\"]")
    .expect("the task list has a surface");
  let as_widget = format!("{}\"widget\"]{}", &form[..surface], &form[surface + 10..]);
  let keep = form
    .find("\"keep\":[")
    .expect("the task list keeps a relation")
    + 8;
  let end = keep + form[keep..].find(']').expect("keep's list ends");
  let keep_now = format!("{}0{}", &form[..keep], &form[end..]);
  let no_glem = form.replace(",\n\"glem\":", ",\n\"other\":");
  let cases = [
    (as_widget, "Screen isn't a widget's relation"),
    (keep_now, "Now is `keep`, but isn't a `state` relation"),
    (
      no_glem,
      "this isn't the compiled form of a glem program: it has no `glem`",
    ),
  ];
  let mut results = Vec::new();
  for (i, (text, want)) in cases.into_iter().enumerate() {
    let got = read(&text).err().unwrap_or_else(|| "it reads".to_string());
    let why = (got != want).then(|| format!("{got}\nnot\n{want}"));
    results.push((format!("broken form {}", i + 1), why));
  }
  report(results);
}

#[test]
fn glem_js_is_ascii_writes_no_html_and_runs_no_code() {
  let found = unsafe_in_script(include_str!("glem.js"));
  assert!(found.is_empty(), "glem.js has {found:?}");
}
