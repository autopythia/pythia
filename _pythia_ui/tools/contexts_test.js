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
  // asks for and says; a log's, unless `more` says otherwise, as with
  // {opening: true}, or {grow: null} for the index. it.tick() is what the
  // browser calls every few seconds.
  function page(more) {
    var doc = new standInDom.Document();
    var it = { asked: [], waiting: new Map(), said: [], kept: 0, shown: 0, tick: null };
    var o = {
      root: doc.createElement("div"),
      program: CONTEXTS_TESTS.program,
      grow: "/f/grow/{lines}",
      every: function (tick) {
        it.tick = tick;
      },
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
    };
    Object.keys(more || {}).forEach(function (name) {
      o[name] = more[name];
    });
    it.view = contexts.start(o);
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
    ["+", "Page", { title: "ui-0", items: "/f/item/", turns: "/f/turn-page/", first: 1 }],
    ["+", "Length", { lines: 6 }],
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
      ["/f/page", "8 facts, shown in N ms", "4 entries", "shown 1"]
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
      "every run open from the start: it asks for its items near the screen, once while they come",
      function () {
        var it = page({ opening: true });
        answer(it, "/f/page", null, LOG);
        var asked = it.asked.slice();
        var near = facts(it, "Near");
        it.view.send(["Fill", 4]);
        it.view.send(["Fill", 4]);
        asked = asked.concat(it.asked.slice(asked.length));
        answer(it, "/f/run/4", null, RUN);
        var runs = facts(it, "Entry").filter(function (fact) {
          return fact.indexOf("Entry(RunBox(") === 0;
        });
        return asked.concat(near, runs, facts(it, "Near").length + " near");
      },
      [
        "/f/page",
        "/f/run/4",
        "Near(Before, Earlier(3))",
        "Near(RunBox(4), Fill(4))",
        'Entry(RunBox(4), "2 tool calls: exec_command 2", "", "4", Dim, True, "/f/item/4", True, False, False, Log, 8)',
        "1 near"
      ]
    ],
    [
      "a log that grows: the page asks what it gained, one request at a time, and its later rows come near the screen",
      function () {
        var it = page();
        it.tick();
        var before = it.asked.length;
        answer(it, "/f/page", null, LOG);
        it.tick();
        it.tick();
        answer(it, "/f/grow/6", null, [
          ["-", "Length", { lines: 6 }],
          ["+", "Length", { lines: 9 }],
          ["+", "Turn", { n: 2, label: "Turn 2 \u00b7 2 items", title: "Go on", line: 7 }]
        ]);
        var near = facts(it, "Near");
        it.tick();
        answer(it, "/f/grow/9", null, []);
        return [before + " asked before its facts"].concat(it.asked, near, facts(it, "Length"), facts(it, "Turn").length + " turns");
      },
      ["1 asked before its facts", "/f/page", "/f/grow/6", "/f/grow/9", "Near(After, Later(6))", "Near(Before, Earlier(3))", "Length(9)", "3 turns"]
    ],
    [
      "what a log gained can't come, or the program refuses it: the page says why, and asks no more",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        it.tick();
        answer(it, "/f/grow/6", "409 Conflict", null);
        it.tick();
        var other = page();
        answer(other, "/f/page", null, LOG);
        other.tick();
        answer(other, "/f/grow/6", null, [["+", "Length", { lines: "many" }]]);
        other.tick();
        return it.asked.concat(it.said.slice(1), other.asked, other.said.slice(1));
      },
      [
        "/f/page",
        "/f/grow/6",
        "wrong: live updates stopped: 409 Conflict",
        "/f/page",
        "/f/grow/6",
        "wrong: error: Length's `lines` is an Int, not a string",
        "wrong: live updates stopped: Length's `lines` is an Int, not a string"
      ]
    ],
    [
      "an index that follows its contexts: what changed, and what went",
      function () {
        var it = page({ grow: null });
        var index = [
          ["+", "Page", { title: "Contexts" }],
          ["+", "Context", { id: "a", title: "One", model: "m", turns: 1, lines: 10, size: "1 KB", changed: "now", rank: 0, to: "/a" }],
          ["+", "Context", { id: "b", title: "Two", model: "m", turns: 2, lines: 20, size: "2 KB", changed: "1 min ago", rank: 1, to: "/b" }]
        ];
        it.tick();
        answer(it, "/f/page", null, index);
        it.tick();
        var later = [
          index[0],
          ["+", "Context", { id: "a", title: "One", model: "m", turns: 2, lines: 30, size: "3 KB", changed: "now", rank: 0, to: "/a" }]
        ];
        answer(it, "/f/page", null, later);
        it.tick();
        answer(it, "/f/page", null, later);
        return it.asked.concat(facts(it, "Context"), facts(it, "Page"), it.said.slice(1));
      },
      ["/f/page", "/f/page", "/f/page", 'Context("a", "One", "m", 2, 30, "3 KB", "now", 0, "/a")', 'Page("Contexts", "", "", 0)']
    ],
    [
      "a star: set at once, and kept; one that can't be kept is taken back",
      function () {
        var posted = [];
        var answers = [];
        var it = page({
          stars: "/f/stars",
          post: function (url, value, done) {
            posted.push(url + " " + JSON.stringify(value));
            answers.push(done);
          }
        });
        answer(it, "/f/page", null, LOG.concat([["+", "Stars", { line: 6 }]]));
        it.view.send(["Starred", ["Said", 3], true]);
        var set = facts(it, "Stars");
        answers.shift()(null);
        it.view.send(["Starred", ["Said", 6], false]);
        answers.shift()("403 Forbidden");
        return posted.concat(set, facts(it, "Stars"), it.said.slice(1));
      },
      [
        '/f/stars {"line":3,"on":true}',
        '/f/stars {"line":6,"on":false}',
        "Stars(3)",
        "Stars(6)",
        "Stars(3)",
        "Stars(6)",
        "wrong: couldn't keep the star: 403 Forbidden"
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
