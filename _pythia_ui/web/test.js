// web/test.js: the tests in JavaScript, in a browser (web/test.html),
// where the scripts after this one run them, as `make js-test` does: each
// runner in tools/ reports its cases in TAP (tools/tap.js), which this
// shows, with a summary once the page has loaded, and any error that
// stopped a runner, or a script that didn't load.
(function () {
  "use strict";

  var reports = document.getElementById("reports");
  var summary = document.getElementById("summary");
  var totals = { runners: 0, cases: 0, failed: 0 };
  var errors = [];

  function show(tag, text, pass) {
    var e = document.createElement(tag);
    e.textContent = text;
    if (pass !== null) {
      e.setAttribute("class", pass ? "test-pass" : "test-fail");
    }
    reports.appendChild(e);
  }

  // A runner's report, as it finishes, from its script.
  tap.output = function (text, cases, failed) {
    var script = document.currentScript;
    var name = script ? script.getAttribute("src").replace(/^.*\//, "") : "a runner";
    totals.runners++;
    totals.cases += cases;
    totals.failed += failed;
    var verdict = failed === 0 ? "all " + cases + " cases pass" : failed + " of " + cases + " cases fail";
    show("h2", name + ": " + verdict, failed === 0);
    show("pre", text, null);
  };

  // An error: a script or stylesheet that didn't load, or one that a
  // runner threw, but for the one that tap.finish throws after its report
  // when a case fails.
  window.addEventListener(
    "error",
    function (event) {
      var target = event.target;
      var message =
        target && target !== window && target.getAttribute
          ? "couldn't load " + (target.getAttribute("src") || target.getAttribute("href"))
          : String(event.message);
      if (/cases failed$|no cases ran$/.test(message)) {
        return;
      }
      errors.push(message);
      show("h2", "an error: " + message, false);
    },
    true
  );

  window.addEventListener("load", function () {
    var runners = document.querySelectorAll('script[src$="_test.js"]').length;
    var pass = errors.length === 0 && totals.failed === 0 && totals.runners === runners;
    var text = totals.runners + " of " + runners + " runners reported: ";
    text += totals.failed === 0 ? "all " + totals.cases + " cases pass" : totals.failed + " of " + totals.cases + " cases fail";
    if (errors.length > 0) {
      text += "; " + errors.length + (errors.length === 1 ? " error" : " errors");
    }
    summary.textContent = text + ".";
    summary.setAttribute("class", pass ? "test-pass" : "test-fail");
    document.title = (pass ? "Pass" : "Fail") + ": glui's tests";
  });
})();
