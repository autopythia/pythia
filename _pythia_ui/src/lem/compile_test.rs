//! Tests of the compiled form ([`super::compile`]) and of the host's JSON
//! ([`super::host`]), reported in TAP ([`super::testing`]):
//!
//! - each of lem's test programs, compiled and read back, compiles to the
//!   same text, and has the same plans;
//! - each of their tick files runs on the compiled form read back, with
//!   each tick as JSON steps, and gives what it gives on the program
//!   itself; and so do `lem/test/*_json.txt`, ticks that a host would send,
//!   as JSON, and what's wrong with them;
//! - `lem.js` is ASCII, writes no HTML, and runs no code that it's given.
//!
//! The ticks, as JSON, with the compiled forms, are what lem's runtime in
//! JavaScript must give too, down to the facts that each tick's joins
//! visit: they go to `target/web/lem_tests.js`, which `tools/lem_test.js`
//! runs on `lem.js` (`make js-test`).

use super::check::load;
use super::compile::{compile, read};
use super::eval::Runtime;
use super::host;
use super::testing::{cases, read as read_file, report, unsafe_in_script, write_web};
use crate::json::{self, object, Value as Json};

/// lem's test programs, in `lem/test/`, by name.
const PROGRAMS: [&str; 4] = ["jobs", "numbers", "recursion", "joins"];

/// Files of ticks, each on a program from its start: (program, file in
/// `lem/test/`, whether its lines are JSON steps rather than lem).
const SUITES: [(&str, &str, bool); 6] = [
  ("jobs", "jobs_ticks.txt", false),
  ("numbers", "numbers_ticks.txt", false),
  ("recursion", "recursion_ticks.txt", false),
  ("joins", "joins_ticks.txt", false),
  ("jobs", "jobs_json.txt", true),
  ("numbers", "numbers_json.txt", true),
];

/// A program's compiled form.
fn form(name: &str) -> String {
  let file = format!("lem/test/{name}.lem");
  let text = read_file(&format!("test/{name}.lem"));
  let prog = load(&[(&file, &text)]).unwrap_or_else(|e| panic!("{}", e.text));
  compile(&prog)
}

#[test]
fn the_compiled_form_reads_back_as_it_was() {
  let mut results = Vec::new();
  for name in PROGRAMS {
    let file = format!("lem/test/{name}.lem");
    let text = read_file(&format!("test/{name}.lem"));
    let prog = load(&[(&file, &text)]).unwrap_or_else(|e| panic!("{}", e.text));
    let form = compile(&prog);
    let why = match read(&form) {
      Err(e) => Some(format!("it doesn't read back: {e}")),
      Ok(back) if compile(&back) != form => {
        Some(format!("it reads back as\n{}\nnot\n{form}", compile(&back)))
      }
      Ok(back) if back.plans() != prog.plans() => Some(format!(
        "its plans are\n{}\nnot\n{}",
        back.plans(),
        prog.plans()
      )),
      Ok(_) => None,
    };
    results.push((name.to_string(), why));
  }
  report(results);
}

#[test]
fn ticks_run_on_the_compiled_form() {
  let mut results = Vec::new();
  let mut suites = Vec::new();
  for (name, file, is_json) in SUITES {
    let form = form(name);
    let prog = read(&form).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut runtime = Runtime::new(prog);
    let mut cases_json = Vec::new();
    for (title, lines, want) in cases(&format!("test/{file}")) {
      let steps = match is_json {
        false => host::script_json(&mut runtime.prog, &lines),
        true => Json::Array(
          lines
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| json::parse(line).unwrap_or_else(|e| panic!("{file}: {line}: {e}")))
            .collect(),
        ),
      };
      let got = host::run(&mut runtime, &steps).unwrap_or_else(|e| format!("error: {e}"));
      let why = (got != want).then(|| format!("the tick gives\n{got}\nnot\n{want}"));
      results.push((format!("{file}: {title}"), why));
      cases_json.push(object([
        ("title", Json::from(title.as_str())),
        ("steps", steps),
        ("want", Json::from(want.as_str())),
        ("visits", Json::from(runtime.visits())),
      ]));
    }
    let program = json::parse(&form).expect("the compiled form is JSON");
    suites.push(object([
      ("name", Json::from(file)),
      ("program", program),
      ("cases", Json::Array(cases_json)),
    ]));
  }
  let by = "src/lem/compile_test.rs";
  write_web("lem_tests.js", "LEM_TESTS", by, &Json::Array(suites));
  report(results);
}

#[test]
fn lem_js_is_ascii_writes_no_html_and_runs_no_code() {
  let found = unsafe_in_script(include_str!("lem.js"));
  assert!(found.is_empty(), "lem.js has {found:?}");
}
