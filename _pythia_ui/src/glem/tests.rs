//! glem's file-driven tests, reported in TAP, as lem's are
//! ([`crate::lem::testing`]):
//!
//! - glem's prelude and `lem/test/tasks.lem` parse and print back;
//! - each case of `lem/test/glem_errors.txt` is a file, `t.lem`, that fails
//!   to load, after glem's prelude, with its error;
//! - `lem/test/tasks_ticks.txt` runs `lem/test/tasks.lem`, a task list with
//!   toy widgets of its own, a tick per case; and again on a runtime that
//!   forgets what its rules read before each tick, with the same facts;
//! - a tick with nothing new runs no rule, though glem looks for state
//!   whose widget has gone after each tick.

use super::{load, no_notes, Runtime, PRELUDE};
use crate::lem::testing::{error_cases, read, report, round_trip, tick_cases, tick_twice};

#[test]
fn glems_files_print_back() {
  let tasks = read("test/tasks.lem");
  report(vec![
    (
      PRELUDE.0.to_string(),
      round_trip(PRELUDE.0, PRELUDE.1).err(),
    ),
    (
      "test/tasks.lem".to_string(),
      round_trip("test/tasks.lem", &tasks).err(),
    ),
  ]);
}

#[test]
fn glem_errors_read_as_written() {
  error_cases("test/glem_errors.txt", &["src/glem/", "src/lem/"], |src| {
    load(&[("t.lem", src)]).err()
  });
}

#[test]
fn ticks_of_a_task_list() {
  let src = read("test/tasks.lem");
  let prog = load(&[("lem/test/tasks.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/tasks_ticks.txt", |lines| {
    runtime.script(lines, &no_notes)
  });
}

#[test]
fn ticks_of_a_task_list_remembering_what_rules_read_are_ticks_that_forget_it() {
  let src = read("test/tasks.lem");
  let runtime = || {
    let prog = load(&[("lem/test/tasks.lem", &src)]).unwrap_or_else(|e| panic!("{}", e.text));
    Runtime::new(prog)
  };
  tick_twice(
    "test/tasks_ticks.txt",
    [runtime(), runtime()],
    |runtime, lines| runtime.script(lines, &no_notes),
    |runtime| &mut runtime.lem,
  );
}

#[test]
fn a_tick_with_nothing_new_runs_no_rule() {
  // After each tick, glem drops the state whose widget has gone: none has,
  // so the editing state keeps its version, and the next tick runs nothing.
  let src = read("test/tasks.lem");
  let prog = load(&[("lem/test/tasks.lem", &src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for lines in [
    "+ Todo(\"a\", \"first\", 10)",
    "! Open(\"a\")",
    "! Type(\"a\", \"draft\")",
  ] {
    runtime.script(lines, &no_notes).unwrap();
  }
  let shown = runtime.script("? Editing", &no_notes).unwrap();
  assert!(
    shown.contains("Line(Editor(\"a\"), text=\"draft\")"),
    "{shown}"
  );
  assert!(shown.contains("Editing(Row(\"a\"))"), "{shown}");
  assert_eq!(runtime.lem.visits(), 0);
}
