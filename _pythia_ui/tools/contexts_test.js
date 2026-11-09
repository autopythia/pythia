// tools/contexts_test.js: runs the tests of web/contexts.js, the host's
// part of pythia's contexts in a browser, and reports them in TAP
// (tools/tap.js). It starts the program, lem/contexts.lem, compiled, which
// src/glui/tests.rs writes as CONTEXTS_TESTS to target/web/
// contexts_tests.js, in a stand-in DOM (tools/stand_in_dom.js), with a
// `get` that holds each request until a case answers it: the page's
// facts; the rows before a line, which it asks for when the program sends
// LoadBefore; a run's items, when it sends LoadRun; and items' bodies,
// when it sends Load; each once while it comes, and again after it
// failed. `make js-test` runs
// lem.js, glem.js, glui.js, contexts.js, the program, tools/tap.js,
// tools/stand_in_dom.js and this as one script.

(function () {
  "use strict";

  // A page that starts, as contexts.start would in a browser, and what it
  // asks for and says.
  function page() {
    var doc = new standInDom.Document();
    var it = { asked: [], waiting: new Map(), said: [], kept: 0, shown: 0 };
    it.view = contexts.start({
      root: doc.createElement("div"),
      program: CONTEXTS_TESTS.program,
      facts: "/f/page",
      before: "/f/before/{line}",
      after: "/f/after/{line}",
      run: "/f/run/{run}",
      body: "/f/body/{line}",
      keep: function (update) {
        it.kept++;
        update();
      },
      shown: function () {
        it.shown++;
      },
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

  function ran(it, line, open) {
    it.view.send(["Toggled", ["RunBox", line], open]);
  }

  // A log of six lines: the start's run, then turn 1's message, run and
  // message, of which the last three rows come with the page.
  var LOG = [
    ["+", "Page", { title: "ui-0", items: "/f/item/", turns: "/f/turn-page/", first: 1, last: 6 }],
    ["+", "Turn", { n: 0, label: "The start \u00b7 2 items", title: "", line: 1 }],
    ["+", "Turn", { n: 1, label: "Turn 1 \u00b7 4 items", title: "Plan the GUI", line: 3 }],
    ["+", "Chunk", { first: 3, last: 6 }],
    ["+", "Message", { line: 3, turn: 1, tone: "User", label: "[user]", text: "Plan the GUI" }],
    ["+", "Run", { line: 4, turn: 1, last: 5, text: "2 tool calls: exec_command 2" }],
    ["+", "Message", { line: 6, turn: 1, tone: "Assistant", label: "[assistant]", text: "Done." }]
  ];

  var EARLIER = [
    ["+", "Chunk", { first: 1, last: 2 }],
    ["+", "Run", { line: 1, turn: 0, last: 2, text: "2 other" }]
  ];

  var RUN = [
    ["+", "RunLoaded", { run: 4 }],
    ["+", "Item", { line: 4, run: 4, tone: "Tool", label: "[tool-call] exec_command", gist: "ls", body: true }],
    ["+", "Item", { line: 5, run: 4, tone: "Tool", label: "[tool-ret] exec_command", gist: "2 lines", body: true }]
  ];

  // A page with its log, and run 4 open, with its items.
  function shown() {
    var it = page();
    answer(it, "/f/page", null, LOG);
    ran(it, 4, true);
    answer(it, "/f/run/4", null, RUN);
    return it;
  }

  function body(line, text) {
    return [["+", "Body", { line: line, part: 0, text: text, diff: false }]];
  }

  var cases = [
    [
      "the page asks for its facts, shows its last rows, and says it's shown",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        return it.asked.concat(it.said, facts(it, "Entry").length + " entries", "shown " + it.shown);
      },
      ["/f/page", "7 facts, shown in N ms", "4 entries", "shown 1"]
    ],
    [
      "the earlier rows: asked for once while they come, and shown in place",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        it.view.send(["Earlier", 3]);
        it.view.send(["Earlier", 3]);
        var asked = it.asked.slice();
        answer(it, "/f/before/3", null, EARLIER);
        return asked.concat(facts(it, "Entry").length + " entries", "kept " + it.kept, facts(it, "Near").length + " near");
      },
      ["/f/page", "/f/before/3", "6 entries", "kept 1", "0 near"]
    ],
    [
      "earlier rows that can't come: the end says why, and asks no more",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        var near = facts(it, "Near");
        it.view.send(["Earlier", 3]);
        answer(it, "/f/before/3", "502 Bad Gateway", null);
        return near.concat(facts(it, "EarlierFailed"), facts(it, "Near").length + " near");
      },
      ["Near(Before, Earlier(3))", 'EarlierFailed(3, "502 Bad Gateway")', "0 near"]
    ],
    [
      "opening a run asks for its items, once while they come",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        ran(it, 4, true);
        ran(it, 4, false);
        ran(it, 4, true);
        var asked = it.asked.slice();
        answer(it, "/f/run/4", null, RUN);
        return asked.concat(facts(it, "Entry").length + " entries", facts(it, "Opened"), facts(it, "RunLoaded"));
      },
      ["/f/page", "/f/run/4", "6 entries", "Opened(4)", "RunLoaded(4)"]
    ],
    [
      "a run whose items can't come: it says why; opened again, they come",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        ran(it, 4, true);
        answer(it, "/f/run/4", "502 Bad Gateway", null);
        var first = facts(it, "RunFailed");
        ran(it, 4, false);
        ran(it, 4, true);
        var meanwhile = facts(it, "RunFailed").length + " failed while they come";
        answer(it, "/f/run/4", null, RUN);
        return first.concat(meanwhile, facts(it, "RunFailed"), facts(it, "RunLoaded"), it.asked);
      },
      ['RunFailed(4, "502 Bad Gateway")', "0 failed while they come", "RunLoaded(4)", "/f/page", "/f/run/4", "/f/run/4"]
    ],
    [
      "a run's items that aren't a list of steps",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        ran(it, 4, true);
        answer(it, "/f/run/4", null, { line: 4 });
        return facts(it, "RunFailed");
      },
      ["RunFailed(4, \"its items aren't a list of steps\")"]
    ],
    [
      "opening an entry asks for its body, once while it comes",
      function () {
        var it = shown();
        opened(it, 4, true);
        opened(it, 4, false);
        opened(it, 4, true);
        var asked = it.asked.slice();
        answer(it, "/f/body/4", null, body(4, "a\nb"));
        return asked.concat(facts(it, "Code").length + " code", facts(it, "Asked"));
      },
      ["/f/page", "/f/run/4", "/f/body/4", "3 code", "Asked(4)"]
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
        "/f/run/4",
        "/f/body/4",
        "/f/body/4"
      ]
    ],
    [
      "a body that isn't a list of steps",
      function () {
        var it = shown();
        opened(it, 5, true);
        answer(it, "/f/body/5", null, { text: "a" });
        return facts(it, "Failed");
      },
      ["Failed(5, \"its body isn't a list of steps\")"]
    ],
    [
      "a body with facts that the program refuses: the page and the entry say so",
      function () {
        var it = shown();
        opened(it, 5, true);
        answer(it, "/f/body/5", null, [["+", "Body", { line: 5, part: "one", text: "a", diff: false }]]);
        return it.said.slice(1).concat(facts(it, "Body").length + " parts", facts(it, "Failed"));
      },
      [
        "wrong: error: Body's `part` is an Int, not a string",
        "0 parts",
        'Failed(5, "Body\'s `part` is an Int, not a string")'
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
