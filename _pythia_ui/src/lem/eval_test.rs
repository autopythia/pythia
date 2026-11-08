//! lem's file-driven tests of the checker and the runtime, reported in TAP
//! ([`super::testing`]):
//!
//! - each case of `lem/test/check_errors.txt` is a file, `t.lem`, that fails
//!   to load, after lem's prelude alone, with its error;
//! - `lem/test/jobs_ticks.txt` runs `lem/test/jobs.lem`, a job queue, a tick
//!   per case, each on the facts that the cases before it left;
//! - `lem/test/numbers_ticks.txt` runs `lem/test/numbers.lem`: how numbers
//!   compute and print, as every runtime must.

use super::check::load;
use super::eval::Runtime;
use super::testing::{error_cases, read, tick_cases};

#[test]
fn check_errors_read_as_written() {
  error_cases("test/check_errors.txt", &["src/lem/"], |src| {
    load(&[("t.lem", src)]).err()
  });
}

#[test]
fn ticks_of_a_job_queue() {
  let src = read("test/jobs.lem");
  let prog = load(&[("lem/test/jobs.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/jobs_ticks.txt", |lines| runtime.script(lines));
}

#[test]
fn ticks_of_numbers() {
  let src = read("test/numbers.lem");
  let prog = load(&[("lem/test/numbers.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/numbers_ticks.txt", |lines| runtime.script(lines));
}
