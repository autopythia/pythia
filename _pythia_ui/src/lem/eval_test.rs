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
//!   runtime joins in the order that their facts ask for;
//! - `lem/test/rerun_ticks.txt` runs `lem/test/rerun.lem`: what a tick runs
//!   again, and what it keeps, through negations, a lazy relation, keys that
//!   don't hold, and a rule that fails;
//! - `lem/test/follow_ticks.txt` runs `lem/test/follow.lem`: how a tick
//!   follows what came and went, and where a rule runs whole;
//! - `lem/test/keys_ticks.txt` runs `lem/test/keys.lem`: a rule that reads
//!   a relation through its key, and a key that doesn't hold.
//!
//! Each tick file also runs on a runtime that forgets, before each tick,
//! what its rules read, so that every rule runs: the facts must be the same.
//! And a join's work, as the facts it visits, where the order matters, and
//! where an atom binds nothing; the work that a lazy relation saves; and the
//! work that a tick saves where what rules read hasn't changed, and where it
//! follows what changed, in one relation or several, skips what a change
//! leaves as it was, or starts over.

use super::check::load;
use super::eval::Runtime;
use super::testing::{error_cases, read, tick_cases, tick_twice};

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
fn ticks_of_what_runs_again() {
  let src = read("test/rerun.lem");
  let prog = load(&[("lem/test/rerun.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/rerun_ticks.txt", |lines| runtime.script(lines));
}

#[test]
fn ticks_through_keys() {
  let src = read("test/keys.lem");
  let prog = load(&[("lem/test/keys.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/keys_ticks.txt", |lines| runtime.script(lines));
}

#[test]
fn ticks_of_what_a_tick_follows() {
  let src = read("test/follow.lem");
  let prog = load(&[("lem/test/follow.lem", &src)])
    .unwrap_or_else(|e| panic!("{}\n(raised at {})", e.text, e.loc));
  let mut runtime = Runtime::new(prog);
  tick_cases("test/follow_ticks.txt", |lines| runtime.script(lines));
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
  // host asks for it; it derives `Near`, which is lazy, for `Seen`. Once the
  // host has asked, `Near` and `Seen` keep their facts, since what they read
  // hasn't changed, and the next tick's `Far` has none. Asked again, `Far`
  // is derived again, though what it reads hasn't changed either.
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
  // E's 10 facts for Near, and Near's 10 for Seen.
  assert_eq!(runtime.visits(), 20);
  let shown = runtime.script("? Far").unwrap();
  assert_eq!(shown.lines().count(), 9, "{shown}");
  // Far's alone: E's 10 facts, and the next edge from each, 9.
  assert_eq!(runtime.visits(), 19);
  runtime.tick(None).unwrap();
  assert!(runtime.facts("Far").is_empty());
  assert_eq!(runtime.facts("Seen").len(), 10);
  assert_eq!(runtime.visits(), 0);
  let shown = runtime.script("? Far").unwrap();
  assert_eq!(shown.lines().count(), 9, "{shown}");
  assert_eq!(runtime.visits(), 19);
}

#[test]
fn a_tick_runs_only_the_rules_whose_reads_changed() {
  // Two chains: B, then C, from A; and Y from X. A tick runs a component's
  // rules only if what they read has changed; and if their facts come out
  // the same, what reads them doesn't run either. Once they've run, a tick
  // follows what came and went: each such fact is a visit.
  let src = "input A(n: Int, m: Int)\nrel B(n: Int)\nB(n) :- A(n, m)\n\
             rel C(n: Int)\nC(n) :- B(n)\n\
             input X(n: Int)\nrel Y(n: Int)\nY(n) :- X(n)\n";
  let prog = load(&[("chains.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for i in 0..10 {
    runtime.assert(&format!("A({i}, 0)")).unwrap();
    runtime.assert(&format!("X({i})")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 30);
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 0);
  runtime.assert("X(10)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 1); // X(10), for Y
  assert_eq!(runtime.facts("Y").len(), 11);
  runtime.assert("A(3, 1)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 1); // A(3, 1), for B: the same facts, so C doesn't run
  assert!(runtime.retract("A(3, 0)").unwrap());
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 1); // B(3) holds still, by A(3, 1)
  assert!(runtime.retract("A(3, 1)").unwrap());
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 2); // A(3, 1), for B, which loses B(3), for C
  assert_eq!(runtime.facts("C").len(), 9);
  // Retracting a fact that doesn't hold, or asserting one that does,
  // changes nothing.
  assert!(!runtime.retract("X(99)").unwrap());
  runtime.assert("X(10)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 0);
}

#[test]
fn a_component_with_nothing_to_keep_starts_over_when_most_of_what_it_reads_is_new() {
  // Over has no facts while A holds none over 99. A tick that brings more of
  // A than stayed starts it over, running its rule whole as its first tick
  // did, and visits all of A's facts; one that brings fewer follows what
  // came, and visits just that. Once Over has facts to keep, a tick follows
  // however much came.
  let src = "input A(n: Int)\nrel Over(n: Int)\nOver(n) :- A(n), n > 99\n";
  let prog = load(&[("over.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for i in 0..10 {
    runtime.assert(&format!("A({i})")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 10); // the first tick runs whole
  runtime.assert("A(100)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 1); // 1 came, 10 stayed: it follows
  assert!(runtime.retract("A(100)").unwrap());
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 1); // Over(100) to keep: it follows
  for i in 100..120 {
    runtime.assert(&format!("A({i})")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 30); // 20 came, 10 stayed, nothing to keep
  assert_eq!(runtime.facts("Over").len(), 20);
  for i in 200..240 {
    runtime.assert(&format!("A({i})")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 40); // 40 came, 30 stayed, but 20 to keep
  assert_eq!(runtime.facts("Over").len(), 60);
}

#[test]
fn a_change_that_leaves_a_literal_as_it_was_runs_nothing_for_it() {
  // E's rule has a block's test, T(_), as well as T(n); U reads T only in a
  // negated test; and V negates T(9).
  let src = "input T(n: Int)\ninput S(w: Int, n: Int)\n\
             rel E(w: Int)\nE(w) :- T(_), T(n), S(w, n)\n\
             rel U(w: Int)\nU(w) :- S(w, _), not T(_)\n\
             rel V(w: Int)\nV(w) :- S(w, _), not T(9)\n";
  let prog = load(&[("tests.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for i in 0..10 {
    runtime.assert(&format!("T({i})")).unwrap();
    runtime.assert(&format!("S({}, {i})", 100 + i)).unwrap();
  }
  runtime.tick(None).unwrap();
  // T(10) leaves both tests holding as they did, and isn't T(9): E joins
  // T(10) alone, which visits the test's first fact and T(10), and finds no
  // S for 10; U and V run nothing.
  runtime.assert("T(10)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 2);
  // With every T gone, the tests hold no more, and the rules run whole: E's
  // test fails at once; U and V visit S's 10 facts, and find no T for each.
  for i in 0..=10 {
    assert!(runtime.retract(&format!("T({i})")).unwrap());
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 20);
  assert_eq!(runtime.facts("U").len(), 10);
  assert_eq!(runtime.facts("V").len(), 10);
  // T(5) makes the tests hold again: E's rule runs whole, 1 + 1 + 1, and U's,
  // 10 + 10; but T(5) isn't T(9), so V runs nothing.
  runtime.assert("T(5)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 23);
  assert_eq!(runtime.facts("E").len(), 1);
  assert!(runtime.facts("U").is_empty());
  assert_eq!(runtime.facts("V").len(), 10);
}

#[test]
fn a_rule_follows_two_changes_at_once() {
  // J reads A and B, which change in one tick: it joins what came to A,
  // with B as it was before the tick, and what came to B, with A as it is
  // now, so that each new way to derive a fact counts once, and a tick costs
  // what changed rather than what the relations hold.
  let src = "input A(n: Int)\ninput B(n: Int)\nrel J(n: Int)\nJ(n) :- A(n), B(n)\n";
  let prog = load(&[("two.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for i in 0..1000 {
    runtime.assert(&format!("A({i})")).unwrap();
    runtime.assert(&format!("B({i})")).unwrap();
  }
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 2000);
  // A(1000), finding no B(1000) as B was; then B(1000), finding A(1000).
  runtime.assert("A(1000)").unwrap();
  runtime.assert("B(1000)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 3);
  assert_eq!(runtime.facts("J").len(), 1001);
  // A(5) went, finding B(5) as B was; then B(5), finding no A(5) now.
  assert!(runtime.retract("A(5)").unwrap());
  assert!(runtime.retract("B(5)").unwrap());
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 3);
  assert_eq!(runtime.facts("J").len(), 1000);
}

#[test]
fn an_atom_alone_changes_nothing_while_it_matches_as_many_facts() {
  // L's M(m) has a variable that appears nowhere else in the rule: each of
  // A's facts derives L once for each of M's. A tick that replaces M's fact
  // leaves how many facts M has, and so L's counts, as they were, and visits
  // nothing; one that adds a second visits it and each of A's facts.
  let src = "input A(n: Int)\ninput M(m: Int)\nrel L(n: Int)\nL(n) :- A(n), M(m)\n";
  let prog = load(&[("alone.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  for i in 0..1000 {
    runtime.assert(&format!("A({i})")).unwrap();
  }
  runtime.assert("M(1)").unwrap();
  runtime.tick(None).unwrap();
  assert!(runtime.retract("M(1)").unwrap());
  runtime.assert("M(2)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 0);
  assert_eq!(runtime.facts("L").len(), 1000);
  runtime.assert("M(3)").unwrap();
  runtime.tick(None).unwrap();
  assert_eq!(runtime.visits(), 1001);
  assert!(runtime.retract("M(2)").unwrap());
  assert!(runtime.retract("M(3)").unwrap());
  runtime.tick(None).unwrap();
  assert_eq!(runtime.facts("L").len(), 0);
}

#[test]
fn an_atom_whose_key_is_fixed_is_followed() {
  // K(n) reads A(n, _), whose `n` B binds. Keyed on `n`, A matches one fact
  // at most for each `n`, whatever the order of the join, so a change to A
  // is followed: the fact that went, and the one that came in its place,
  // each meet B's fact. Without the key, every change to A runs K's rule
  // whole.
  for (key, visits) in [(" key n", 4), ("", 2000)] {
    let src = format!(
      "input A(n: Int, m: Int){key}\ninput B(n: Int)\nrel K(n: Int)\nK(n) :- B(n), A(n, _)\n"
    );
    let prog = load(&[("keyed.lem", &src)]).unwrap_or_else(|e| panic!("{}", e.text));
    let mut runtime = Runtime::new(prog);
    for i in 0..1000 {
      runtime.assert(&format!("A({i}, 0)")).unwrap();
      runtime.assert(&format!("B({i})")).unwrap();
    }
    runtime.tick(None).unwrap();
    assert!(runtime.retract("A(5, 0)").unwrap());
    runtime.assert("A(5, 1)").unwrap();
    runtime.tick(None).unwrap();
    assert_eq!(runtime.visits(), visits, "{key:?}");
    assert_eq!(runtime.facts("K").len(), 1000);
  }
}

#[test]
fn a_key_that_doesnt_hold_fails_each_tick_until_it_does() {
  // Keys are checked only where a relation has changed since they last
  // held: a relation whose keys didn't hold is checked again, though
  // nothing changed.
  let src = "input A(n: Int)\nrel K(k: Int, n: Int) key k\nK(0, n) :- A(n)\n";
  let prog = load(&[("keys.lem", src)]).unwrap_or_else(|e| panic!("{}", e.text));
  let mut runtime = Runtime::new(prog);
  runtime.assert("A(1)").unwrap();
  runtime.tick(None).unwrap();
  runtime.assert("A(2)").unwrap();
  let clash = "one key, two facts: K(0, 1) (from the rule at keys.lem:3:1) and K(0, 2) \
               (from the rule at keys.lem:3:1)";
  assert_eq!(runtime.tick(None), Err(clash.to_string()));
  assert_eq!(runtime.tick(None), Err(clash.to_string()));
  assert!(runtime.retract("A(1)").unwrap());
  runtime.tick(None).unwrap();
  assert_eq!(runtime.facts("K").len(), 1);
}

#[test]
fn ticks_remembering_what_rules_read_are_ticks_that_forget_it() {
  for name in [
    "jobs",
    "numbers",
    "recursion",
    "joins",
    "rerun",
    "follow",
    "keys",
  ] {
    let src = read(&format!("test/{name}.lem"));
    let file = format!("lem/test/{name}.lem");
    let runtime = || {
      let prog = load(&[(&file, &src)]).unwrap_or_else(|e| panic!("{}", e.text));
      Runtime::new(prog)
    };
    tick_twice(
      &format!("test/{name}_ticks.txt"),
      [runtime(), runtime()],
      |runtime, lines| runtime.script(lines),
      |runtime| runtime,
    );
  }
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
