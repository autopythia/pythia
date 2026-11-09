// tools/contexts_test.js: runs the tests of web/contexts.js, the host's
// part of pythia's contexts in a browser, and reports them in TAP
// (tools/tap.js). It starts the program, lem/contexts.lem, compiled, which
// src/glui/tests.rs writes as CONTEXTS_TESTS to target/web/
// contexts_tests.js, in a stand-in DOM (tools/stand_in_dom.js), with a
// `get` that holds each request until a case answers it: the page's
// facts; a turn's items, which it asks for when the program sends
// LoadTurn; and items' bodies, which it asks for when the program sends
// Load; each once while it comes, and again after it failed. `make js-test` runs
// lem.js, glem.js, glui.js, contexts.js, the program, tools/tap.js,
// tools/stand_in_dom.js and this as one script.

(function () {
  "use strict";

  // A page that starts, as contexts.start would in a browser, and what it
  // asks for and says.
  function page() {
    var doc = new standInDom.Document();
    var it = { asked: [], waiting: new Map(), said: [] };
    it.view = contexts.start({
      root: doc.createElement("div"),
      program: CONTEXTS_TESTS.program,
      facts: "/f/page",
      turn: "/f/turn/{turn}",
      body: "/f/body/{line}",
      get: function (url, done) {
        it.asked.push(url);
        it.waiting.set(url, done);
      },
      status: function (text, wrong) {
        it.said.push((wrong ? "wrong: " : "") + text.replace(/in \d+ ms/, "in N ms"));
      }
    });
    return it;
  }

  // Answers a request that the page made, with a value or why it failed.
  function answer(it, url, why, value) {
    var done = it.waiting.get(url);
    if (done === undefined) {
      throw new Error("the page didn't ask for " + url);
    }
    it.waiting.delete(url);
    done(why, value);
  }

  // A relation's facts, as lem writes them, a line each.
  function facts(it, name) {
    var prog = it.view.runtime.prog.lem;
    var r = prog.rel(name);
    return it.view.runtime.lem.relFacts(r).map(function (fact) {
      return prog.factText(r, fact.tuple);
    });
  }

  function opened(it, line, open) {
    it.view.send(["Toggled", ["Ent", line], open]);
  }

  function turned(it, n, open) {
    it.view.send(["Toggled", ["TurnBox", n], open]);
  }

  var LOG = [
    ["+", "Page", { title: "ui-0", items: "/f/item/", turns: "/f/turn-page/" }],
    ["+", "Turn", { n: 0, label: "The start \u00b7 1 item", title: "" }],
    ["+", "Turn", { n: 1, label: "Turn 1 \u00b7 2 items", title: "Plan the GUI" }]
  ];

  var TURN = [
    ["+", "Loaded", { turn: 1 }],
    ["+", "Item", { line: 3, turn: 1, tone: "Tool", label: "[tool-call] exec_command", gist: "ls", body: true }],
    ["+", "Item", { line: 4, turn: 1, tone: "Tool", label: "[tool-ret] exec_command", gist: "2 lines", body: true }]
  ];

  // A page with its log, and turn 1 open, with its items.
  function shown() {
    var it = page();
    answer(it, "/f/page", null, LOG);
    turned(it, 1, true);
    answer(it, "/f/turn/1", null, TURN);
    return it;
  }

  function body(line, text) {
    return [["+", "Body", { line: line, part: 0, text: text, diff: false }]];
  }

  var cases = [
    [
      "the page asks for its facts, and shows its turns",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        return it.asked.concat(it.said, facts(it, "Entry").length + " entries");
      },
      ["/f/page", "3 facts, shown in N ms", "2 entries"]
    ],
    [
      "opening a turn asks for its items, once while they come",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        turned(it, 1, true);
        turned(it, 1, false);
        turned(it, 1, true);
        var asked = it.asked.slice();
        answer(it, "/f/turn/1", null, TURN);
        return asked.concat(facts(it, "Entry").length + " entries", facts(it, "Opened"), facts(it, "Loaded"));
      },
      ["/f/page", "/f/turn/1", "4 entries", "Opened(1)", "Loaded(1)"]
    ],
    [
      "a turn whose items can't come: it says why; opened again, they come",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        turned(it, 1, true);
        answer(it, "/f/turn/1", "502 Bad Gateway", null);
        var first = facts(it, "TurnFailed");
        turned(it, 1, false);
        turned(it, 1, true);
        var meanwhile = facts(it, "TurnFailed").length + " failed while they come";
        answer(it, "/f/turn/1", null, TURN);
        return first.concat(meanwhile, facts(it, "TurnFailed"), facts(it, "Loaded"), it.asked);
      },
      ['TurnFailed(1, "502 Bad Gateway")', "0 failed while they come", "Loaded(1)", "/f/page", "/f/turn/1", "/f/turn/1"]
    ],
    [
      "a turn's items that aren't a list of steps",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        turned(it, 1, true);
        answer(it, "/f/turn/1", null, { line: 3 });
        return facts(it, "TurnFailed");
      },
      ["TurnFailed(1, \"its items aren't a list of steps\")"]
    ],
    [
      "opening an entry asks for its body, once while it comes",
      function () {
        var it = shown();
        opened(it, 3, true);
        opened(it, 3, false);
        opened(it, 3, true);
        var asked = it.asked.slice();
        answer(it, "/f/body/3", null, body(3, "a\nb"));
        return asked.concat(facts(it, "Code").length + " code", facts(it, "Asked"));
      },
      ["/f/page", "/f/turn/1", "/f/body/3", "1 code", "Asked(3)"]
    ],
    [
      "a body that can't come: the entry says why; opened again, it comes",
      function () {
        var it = shown();
        opened(it, 4, true);
        answer(it, "/f/body/4", "404 Not Found", null);
        var first = facts(it, "Failed");
        opened(it, 4, false);
        opened(it, 4, true);
        var meanwhile = facts(it, "Failed").length + " failed while it comes";
        answer(it, "/f/body/4", null, body(4, "ok"));
        return first.concat(meanwhile, facts(it, "Failed"), facts(it, "Body"), it.asked);
      },
      [
        'Failed(4, "404 Not Found")',
        "0 failed while it comes",
        'Body(4, 0, "ok", False)',
        "/f/page",
        "/f/turn/1",
        "/f/body/4",
        "/f/body/4"
      ]
    ],
    [
      "a body that isn't a list of steps",
      function () {
        var it = shown();
        opened(it, 3, true);
        answer(it, "/f/body/3", null, { text: "a" });
        return facts(it, "Failed");
      },
      ["Failed(3, \"its body isn't a list of steps\")"]
    ],
    [
      "a body with facts that the program refuses: the page and the entry say so",
      function () {
        var it = shown();
        opened(it, 3, true);
        answer(it, "/f/body/3", null, [["+", "Body", { line: 3, part: "one", text: "a", diff: false }]]);
        return it.said.slice(1).concat(facts(it, "Body").length + " parts", facts(it, "Failed"));
      },
      [
        "wrong: error: Body's `part` is an Int, not a string",
        "0 parts",
        'Failed(3, "Body\'s `part` is an Int, not a string")'
      ]
    ],
    [
      "facts that can't come: the page says why",
      function () {
        var it = page();
        answer(it, "/f/page", "500 Internal Server Error", null);
        return it.said;
      },
      ["wrong: couldn't load the page: 500 Internal Server Error"]
    ]
  ];

  cases.forEach(function (c) {
    var got;
    try {
      got = c[1]().join("\n");
    } catch (e) {
      got = tap.error(e);
    }
    var want = c[2].join("\n");
    tap.report(c[0], got === want ? null : "the case gives\n" + got + "\nnot\n" + want);
  });
  tap.finish();
})();
