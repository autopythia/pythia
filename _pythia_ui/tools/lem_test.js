// tools/lem_test.js: runs lem's shared tests on lem.js, lem's runtime in
// JavaScript (src/lem/lem.js), and reports them in TAP (tools/tap.js).
//
// The tests are the tick files that Rust runs on the compiled form, which
// src/lem/compile_test.rs writes as LEM_TESTS to target/web/lem_tests.js:
// for each file, its program's compiled form; for each case, the tick as
// JSON steps, what it shows, and the facts that its joins visit, which
// both runtimes must agree on. `make js-test` runs lem.js, the tests,
// tools/tap.js and this as one script, in JavaScriptCore through osascript
// on a Mac, or in another engine (JS=qjs); the script fails if a case does.

(function () {
  "use strict";

  LEM_TESTS.forEach(function (suite) {
    var runtime = null;
    var broken = null;
    try {
      runtime = new lem.Runtime(new lem.Program(suite.program));
    } catch (e) {
      broken = "the program doesn't load: " + tap.error(e);
    }
    suite.cases.forEach(function (c) {
      var title = suite.name + ": " + c.title;
      if (broken !== null) {
        tap.report(title, broken);
        return;
      }
      var got;
      try {
        got = runtime.run(c.steps);
      } catch (e) {
        got = tap.error(e);
      }
      var why = null;
      if (got !== c.want) {
        why = "the tick gives\n" + got + "\nnot\n" + c.want;
      } else if (runtime.visits() !== c.visits) {
        why = "its joins visit " + runtime.visits() + " facts, not " + c.visits;
      }
      tap.report(title, why);
    });
  });
  tap.finish();
})();
