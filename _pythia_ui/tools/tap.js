// tools/tap.js: reports of the tests in JavaScript, in TAP, as the Rust
// tests write them: `ok 1 - "title"`, or `not ok 2 - "title"` and why, a
// line of `# ` each, then `1..N`. Each runner reports its cases and
// finishes; the next starts afresh, as in a page that runs several
// (web/test.html).
var tap = (function () {
  "use strict";

  var lines = [];
  var count = 0;
  var failed = 0;
  var api = { report: report, error: error, finish: finish, output: null };

  // A case's result: `why` it failed, or null.
  function report(title, why) {
    count++;
    var line = count + " - " + JSON.stringify(title);
    if (why === null) {
      lines.push("ok " + line);
      return;
    }
    failed++;
    lines.push("not ok " + line);
    why.split("\n").forEach(function (text) {
      lines.push("# " + text);
    });
  }

  // What an exception says: lem's errors their message, anything else
  // what it is.
  function error(e) {
    return e instanceof lem.Error ? "error: " + e.message : "an exception: " + e;
  }

  // Gives the report: to tap.output(text, cases, failed), if a page sets
  // it; else with print, in a shell such as qjs's; else with console.log,
  // which osascript writes to stderr (in a browser, print would print the
  // page). Then an error, which fails the script, if a case failed or none
  // ran.
  function finish() {
    lines.push("1.." + count);
    var text = lines.join("\n");
    var cases = count;
    var failures = failed;
    lines = [];
    count = 0;
    failed = 0;
    if (typeof api.output === "function") {
      api.output(text, cases, failures);
    } else if (typeof print === "function" && typeof window === "undefined") {
      print(text);
    } else if (typeof console !== "undefined") {
      console.log(text);
    }
    if (cases === 0 || failures > 0) {
      throw new Error(cases === 0 ? "no cases ran" : failures + " of " + cases + " cases failed");
    }
  }

  return api;
})();
