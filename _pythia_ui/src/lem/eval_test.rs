//! lem's file-driven tests of the checker and the runtime, reported in TAP
//! ([`super::testing`]):
//!
//! - each case of `lem/test/check_errors.txt` is a file, `t.lem`, that fails
//!   to load, after lem's prelude alone, with its error;
//! - `lem/test/jobs_ticks.txt` runs `lem/test/jobs.lem`, a job queue, a tick
//!   per case, each on the facts that the cases before it left;
//! - `lem/test/numbers_ticks.txt` runs `lem/test/numbers.lem`: how numbers
//!   compute and print, as every runtime must;
//! - `lem/test/recursion_ticks.txt` runs `lem/test/recursion.lem`: recursion
//!   through one relation and through two, and why a fact holds;
//! - `lem/test/jobs_plans.txt` is the job queue's plans: the order of its
//!   components, and what each atom is looked up by;
//! - `lem/test/joins_ticks.txt` runs `lem/test/joins.lem`: bodies that the
//!   runtime joins in the order that their facts ask for.
//!
//! And a join's work, as the facts it visits, where the order matters.

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

#[test]
fn ticks_of_recursion() {
  let src = read("test/recursion.lem");
  let prog = load(&[("lem/test/recursion.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/recursion_ticks.txt", |lines| runtime.script(lines));
}

#[test]
fn ticks_of_joins() {
  let src = read("test/joins.lem");
  let prog = load(&[("lem/test/joins.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/joins_ticks.txt", |lines| runtime.script(lines));
}

#[test]
fn a_join_takes_the_smallest_bucket_first() {
  // Triangles in a star, a hub and its leaves, with edges both ways and no
  // triangles. Taken as written, the join would walk the n² paths of two
  // steps through the hub. The smallest bucket first, it walks each edge,
  // 2n, and for each the one way on from a leaf, 2n more.
  let src = "input E(a: Int, b: Int)\nrel Tri(a: Int, b: Int, c: Int)\n\
             Tri(a, b, c) :- E(a, b), E(b, c), E(a, c)\n";
  let prog = load(&[("star.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  let n = 1000;
  for i in 1..=n {
    runtime.assert(&format!("E(0, {i})")).unwrap();
    runtime.assert(&format!("E({i}, 0)")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert!(runtime.facts("Tri").is_empty());
  assert_eq!(runtime.visits(), 4 * n);
}

#[test]
fn plans_of_a_job_queue() {
  let src = read("test/jobs.lem");
  let prog = load(&[("lem/test/jobs.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let want: String = read("test/jobs_plans.txt")
    .lines()
    .filter(|line| !line.starts_with('#'))
    .map(|line| format!("{line}\n"))
    .collect();
  assert_eq!(prog.plans(), want, "\n{}", prog.plans());
}
