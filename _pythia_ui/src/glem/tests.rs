//! glem's file-driven tests, reported in TAP, as lem's are
//! ([`crate::lem::testing`]):
//!
//! - glem's prelude and `lem/test/tasks.lem` parse and print back;
//! - each case of `lem/test/glem_errors.txt` is a file, `t.lem`, that fails
//!   to load, after glem's prelude, with its error;
//! - `lem/test/tasks_ticks.txt` runs `lem/test/tasks.lem`, a task list with
//!   toy widgets of its own, a tick per case.

use super::{load, no_notes, Runtime, PRELUDE};
use crate::lem::testing::{error_cases, read, report, round_trip, tick_cases};

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
