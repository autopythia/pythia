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
//! And a join's work, as the facts it visits, where the order matters, and
//! where an atom binds nothing; and the work that a lazy relation saves.

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
fn an_atom_that_binds_nothing_is_a_test() {
  // A block's head, such as `Window(Top) :- Page():`, gives each widget in
  // it an atom that binds no variable. The join checks that atom once, and
  // doesn't walk its facts for each solution: n facts take n + 1 visits,
  // not n² + n. And an `on` block sends its effect once, not once a fact.
  let src = "input A(n: Int)\nrel B(n: Int)\nB(n) :- A(), A(n)\n\
             msg Go\neffect Went(n: Int)\non Go:\n  Went(1) :- A()\n";
  let prog = load(&[("test.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  let n = 1000;
  for i in 0..n {
    runtime.assert(&format!("A({i})")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.facts("B").len() as u64, n);
  assert_eq!(runtime.visits(), n + 1);
  let shown = runtime.script("! Go").unwrap();
  assert_eq!(shown.matches("effect Went(1)").count(), 1, "{shown}");
}

#[test]
fn a_lazy_relation_is_derived_only_when_something_reads_it() {
  // A tick skips `Far`, which is lazy, and which nothing reads, unless the
  // host asks for it; it derives `Near`, which is lazy, for `Seen`.
  let src = "input E(a: Int, b: Int)\n\
             rel Far(a: Int, b: Int) lazy\nFar(a, c) :- E(a, b), E(b, c)\n\
             rel Near(a: Int) lazy\nNear(a) :- E(a, b)\n\
             rel Seen(a: Int)\nSeen(a) :- Near(a)\n";
  let prog = load(&[("lazy.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for i in 0..10 {
    runtime.assert(&format!("E({i}, {})", i + 1)).unwrap();
  }
  runtime.tick(None).unwrap();
  assert!(runtime.facts("Far").is_empty());
  assert_eq!(runtime.facts("Seen").len(), 10);
  let skipping = runtime.visits();
  let shown = runtime.script("? Far").unwrap();
  assert_eq!(shown.lines().count(), 9, "{shown}");
  assert!(runtime.visits() > skipping);
  runtime.tick(None).unwrap();
  assert!(runtime.facts("Far").is_empty());
  assert_eq!(runtime.visits(), skipping);
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
