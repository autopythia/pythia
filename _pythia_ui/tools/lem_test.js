// tools/lem_test.js: runs lem's shared tests on lem.js, lem's runtime in
// JavaScript (src/lem/lem.js), and reports them in TAP, as the Rust tests
// do: `ok 1 - "title"`, or `not ok 2 - "title"` and why, then `1..N`.
//
// The tests are the tick files that Rust runs on the compiled form, which
// src/lem/compile_test.rs writes as LEM_TESTS to target/web/lem_tests.js:
// for each file, its program's compiled form; for each case, the tick as
// JSON steps, what it shows, and the facts that its joins visit, which
// both runtimes must agree on. `make js-test` runs lem.js, the tests and
// this as one script, in JavaScriptCore through osascript on a Mac, or in
// another engine (JS=qjs); the script fails if a case does.

var lemTest = (function () {
  "use strict";

  // Runs the suites: gives the report, as TAP text, its number of cases,
  // and how many failed.
  function run(suites) {
    var out = [];
    var count = 0;
    var failed = 0;

    function report(title, why) {
      count++;
      var line = count + " - " + JSON.stringify(title);
      if (why === null) {
        out.push("ok " + line);
        return;
      }
      failed++;
      out.push("not ok " + line);
      why.split("\n").forEach(function (text) {
        out.push("# " + text);
      });
    }

    suites.forEach(function (suite) {
      var runtime = null;
      var broken = null;
      try {
        runtime = new lem.Runtime(new lem.Program(suite.program));
      } catch (e) {
        broken = "the program doesn't load: " + e;
      }
      suite.cases.forEach(function (c) {
        var title = suite.name + ": " + c.title;
        if (broken !== null) {
          report(title, broken);
          return;
        }
        var got;
        try {
          got = runtime.run(c.steps);
        } catch (e) {
          got = e instanceof lem.Error ? "error: " + e.message : "an exception: " + e;
        }
        var why = null;
        if (got !== c.want) {
          why = "the tick gives\n" + got + "\nnot\n" + c.want;
        } else if (runtime.visits() !== c.visits) {
          why = "its joins visit " + runtime.visits() + " facts, not " + c.visits;
        }
        report(title, why);
      });
    });
    out.push("1.." + count);
    return { text: out.join("\n"), count: count, failed: failed };
  }

  return { run: run };
})();

// The report: with print, in a shell such as qjs's; else with console.log,
// which osascript writes to stderr. (In a browser, print would print the
// page.) Then an error, which fails the script, if a case failed or none
// ran.
var LEM_RESULT = lemTest.run(LEM_TESTS);
if (typeof print === "function" && typeof window === "undefined") {
  print(LEM_RESULT.text);
} else if (typeof console !== "undefined") {
  console.log(LEM_RESULT.text);
}
if (LEM_RESULT.count === 0 || LEM_RESULT.failed > 0) {
  throw new Error(
    LEM_RESULT.count === 0
      ? "no cases ran"
      : LEM_RESULT.failed + " of " + LEM_RESULT.count + " cases failed"
  );
}
