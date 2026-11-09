// tools/lem_test.js: runs lem's shared tests on lem.js, lem's runtime in
// JavaScript (src/lem/lem.js), and reports them in TAP (tools/tap.js).
//
// The tests are the tick files that Rust runs on the compiled form, which
// src/lem/compile_test.rs writes as LEM_TESTS to target/web/lem_tests.js:
// for each file, its program's compiled form; for each case, the tick as
// JSON steps, what it shows, and the facts that its joins visit, which
// both runtimes must agree on. Each tick runs again on a second runtime,
// which forgets before it what its rules read, and so derives every
// relation anew: it must show the same, and come to the same facts, and
// why each holds. `make js-test` runs lem.js, the tests, tools/tap.js and
// this as one script, in JavaScriptCore through osascript on a Mac, or in
// another engine (JS=qjs); the script fails if a case does.

(function () {
  "use strict";

  // A tick's steps on a runtime: what it shows, or its error.
  function attempt(runtime, steps) {
    try {
      return runtime.run(steps);
    } catch (e) {
      return tap.error(e);
    }
  }

  // Every relation's facts, as lem writes them, and why each holds.
  function everything(runtime) {
    var prog = runtime.prog;
    var out = "";
    prog.rels.forEach(function (rel, r) {
      runtime.relFacts(r).forEach(function (fact) {
        out += prog.factText(r, fact.tuple) + " (" + runtime.why(fact.why) + ")\n";
      });
    });
    return out;
  }

  LEM_TESTS.forEach(function (suite) {
    var runtime = null;
    var forgets = null;
    var broken = null;
    try {
      runtime = new lem.Runtime(new lem.Program(suite.program));
      forgets = new lem.Runtime(new lem.Program(suite.program));
    } catch (e) {
      broken = "the program doesn't load: " + tap.error(e);
    }
    suite.cases.forEach(function (c) {
      var title = suite.name + ": " + c.title;
      if (broken !== null) {
        tap.report(title, broken);
        return;
      }
      var got = attempt(runtime, c.steps);
      forgets.forget();
      var again = attempt(forgets, c.steps);
      var why = null;
      if (got !== c.want) {
        why = "the tick gives\n" + got + "\nnot\n" + c.want;
      } else if (runtime.visits() !== c.visits) {
        why = "its joins visit " + runtime.visits() + " facts, not " + c.visits;
      } else if (again !== got) {
        why = "forgetting what its rules read, the tick gives\n" + again;
      } else if (everything(forgets) !== everything(runtime)) {
        why = "remembering, the facts are\n" + everything(runtime) + "forgetting, they're\n" + everything(forgets);
      }
      tap.report(title, why);
    });
  });

  // The host's changes that change nothing, as the API allows them:
  // asserting a fact that holds, and retracting one that doesn't. The next
  // tick runs no rule, as in Rust (src/lem/eval_test.rs).
  (function () {
    var suite = LEM_TESTS.filter(function (s) {
      return s.name === "rerun_ticks.txt";
    })[0];
    var why = null;
    try {
      var runtime = new lem.Runtime(new lem.Program(suite.program));
      var a = runtime.prog.rel("A");
      var c = runtime.prog.rel("C");
      runtime.assertTuple(a, [1]);
      runtime.tick(null);
      var first = runtime.visits();
      runtime.assertTuple(a, [1]);
      var held = runtime.retractTuple(c, [9]);
      runtime.tick(null);
      if (first === 0 || held !== false || runtime.visits() !== 0) {
        why = "the first tick visits " + first + " facts, the retraction gives " + held + ", and the next tick visits " + runtime.visits();
      }
    } catch (e) {
      why = tap.error(e);
    }
    tap.report("rerun.lem: changes that change nothing run no rule", why);
  })();
  tap.finish();
})();
