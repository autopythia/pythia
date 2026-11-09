// tools/contexts_test.js: runs the tests of web/contexts.js, the host's
// part of pythia's contexts in a browser, and reports them in TAP
// (tools/tap.js). It starts the program, lem/contexts.lem, compiled, which
// src/glui/tests.rs writes as CONTEXTS_TESTS to target/web/
// contexts_tests.js, in a stand-in DOM (tools/stand_in_dom.js), with a
// `get` that holds each request until a case answers it: the page's
// facts; the rows before a line, which it asks for when the program sends
// LoadBefore; a run's items, when it sends LoadRun; items' bodies, when
// it sends Load; each once while it comes, and again after it failed; and
// the panel's pins, once a star is kept. `make js-test` runs
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
      around: "/f/around/{line}",
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

  // The panel's pins: the user's message, and the answer, line 6.
  var PIN3 = ["+", "Pin", { line: 3, turn: 1, tone: "User", label: "[user]", gist: "Plan the GUI" }];
  var PIN6 = ["+", "Pin", { line: 6, turn: 1, tone: "Assistant", label: "[assistant]", gist: "Done." }];

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

  // A page of tabs, as contexts.tabs shows one in a browser, and what it
  // asks for, says and keeps: the index's tab first, and those that `more`
  // says, as {kept: {tabs, tab, places}} or {open: n}. Where the tab shown
  // is, as a browser would say, is it.at.
  function tabbed(more) {
    var doc = new standInDom.Document();
    var it = { doc: doc, asked: [], waiting: new Map(), said: [], kept: [], titles: [], addresses: [], top: 0, ends: 0, at: null, places: [] };
    var o = {
      holder: doc.createElement("div"),
      strip: doc.createElement("div"),
      program: CONTEXTS_TESTS.program,
      strip_program: CONTEXTS_TESTS.tabs,
      index: "/f/index",
      tab: "/f/tab/{id}",
      log: "/f/c/{id}",
      home: "/f/",
      get: function (url, done) {
        it.asked.push(url);
        it.waiting.set(url, done);
      },
      post: function (url, value, done) {
        done(null);
      },
      status: function (text, wrong) {
        it.said.push((wrong ? "wrong: " : "") + text.replace(/in \d+ ms/, "in N ms"));
      },
      settings: function () {
        return { opening: false, aside: false };
      },
      kept: { tabs: [], tab: 0 },
      open: null,
      keep_tabs: function (numbers, shown) {
        it.kept.push(numbers.join("-") + " shows " + shown);
      },
      at: function () {
        return it.at;
      },
      keep_place: function (n, at) {
        it.places.push(n + " at " + at);
      },
      title: function (text) {
        it.titles.push(text);
      },
      address: function (url) {
        it.addresses.push(url);
      },
      showing: function () {},
      top: function () {
        return it.top;
      },
      go: function (top) {
        it.top = top;
      },
      end: function () {
        it.ends++;
      },
      keep: function (root, update) {
        update();
      }
    };
    Object.keys(more || {}).forEach(function (name) {
      o[name] = more[name];
    });
    it.holder = o.holder;
    it.tabs = contexts.tabs(o);
    return it;
  }

  // What a context's tab names, for tab `n`.
  function tabOf(n) {
    return {
      id: String(n), title: "ctx (" + n + ")", facts: "/f/page/" + n, before: "/f/before/{line}",
      after: "/f/after/{line}", around: "/f/around/{line}", run: "/f/run/{run}", body: "/f/body/{line}",
      grow: "/f/grow/" + n + "/{lines}", stars: null, pins: null, plain: "/p/" + n
    };
  }

  // The strip's facts of a relation, as lem writes them.
  function strip(it, name) {
    var runtime = it.tabs.strip.runtime;
    var prog = runtime.prog.lem;
    var r = prog.rel(name);
    return runtime.lem.relFacts(r).map(function (fact) {
      return prog.factText(r, fact.tuple);
    });
  }

  // The first link in `e` to `href`, or null.
  function linkTo(e, href) {
    if (e.nodeType === 1 && e.localName === "a" && e.getAttribute("href") === href) {
      return e;
    }
    for (var i = 0; i < e.childNodes.length; i++) {
      var found = linkTo(e.childNodes[i], href);
      if (found !== null) {
        return found;
      }
    }
    return null;
  }

  // A log's rows, in the view in `root`: the entries of its first box.
  function rowsOf(root) {
    var box = null;
    var find = function (e) {
      if (box === null && e.nodeType === 1) {
        if ((" " + e.getAttribute("class") + " ").indexOf(" g-box ") >= 0) {
          box = e;
          return;
        }
        e.childNodes.forEach(find);
      }
    };
    find(root);
    return box === null ? [] : box.childNodes.filter(function (e) {
      return (" " + e.getAttribute("class") + " ").indexOf(" g-entry ") >= 0;
    });
  }

  // How many times the page said something went wrong.
  function wrong(it) {
    return it.said.filter(function (text) {
      return text.indexOf("wrong: ") === 0;
    }).length;
  }

  // Which of the holder's tabs show, by order.
  function showing(it) {
    var shown = [];
    it.holder.childNodes.forEach(function (root, at) {
      if (root.getAttribute("hidden") === null) {
        shown.push(at);
      }
    });
    return shown.join(" ");
  }

  var INDEX = [
    ["+", "Page", { title: "Contexts" }],
    ["+", "Context", { id: "c7", title: "Seven", model: "m", turns: 2, lines: 30, size: "3 KB", changed: "now", rank: 0, to: "/f/c/7" }]
  ];

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
      "a log's dot follows what its context is doing, as the page asks what it gained, with lines or without",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG.concat([
          ["-", "Page", LOG[0][2]],
          ["+", "Page", { title: "ui-0", items: "/f/item/", turns: "/f/turn-page/", first: 1, id: "ui-0" }],
          ["+", "Activity", { id: "ui-0", hue: "Green", why: "Running: sampling." }]
        ]));
        var seen = facts(it, "Dot");
        it.tick();
        answer(it, "/f/grow/6", null, [["+", "Activity", { id: "ui-0", hue: "Green", why: "Running: sampling." }]]);
        seen = seen.concat(facts(it, "Dot"));
        it.tick();
        answer(it, "/f/grow/6", null, [["+", "Activity", { id: "ui-0", hue: "Yellow", why: "Waiting: the CLI waits for input." }]]);
        return seen.concat(facts(it, "Dot"), facts(it, "Activity").length + " activity", it.asked);
      },
      [
        'Dot(TitleDot, Green, "Running: sampling.", TitleRow, 0)',
        'Dot(TitleDot, Green, "Running: sampling.", TitleRow, 0)',
        'Dot(TitleDot, Yellow, "Waiting: the CLI waits for input.", TitleRow, 0)',
        "1 activity",
        "/f/page",
        "/f/grow/6",
        "/f/grow/6"
      ]
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
      ["/f/page", "/f/page", "/f/page", 'Context("a", "One", "m", 2, 30, "3 KB", "now", 0, "/a", "")', 'Page("Contexts", "", "", 0, "")']
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
      "the panel shows from the start, as the bar says; a star kept asks for its pins anew, in place of those before",
      function () {
        var answers = [];
        var it = page({
          aside: true,
          stars: "/f/stars",
          pins: "/f/pins",
          post: function (url, value, done) {
            answers.push(done);
          }
        });
        answer(it, "/f/page", null, LOG.concat([["+", "Stars", { line: 3 }], PIN3]));
        var before = facts(it, "Aside").concat(facts(it, "Pin"), facts(it, "Hud"));
        it.view.send(["Starred", ["Said", 6], true]);
        var asked = it.asked.length + " asked";
        answers.shift()(null);
        answer(it, "/f/pins", null, [PIN6]);
        return before.concat(asked, it.asked.slice(1), facts(it, "Pin"), it.said.slice(1));
      },
      [
        "Aside(True)",
        'Pin(3, 1, User, "[user]", "Plan the GUI")',
        "Hud(Side, TopRight)",
        "1 asked",
        "/f/pins",
        'Pin(6, 1, Assistant, "[assistant]", "Done.")'
      ]
    ],
    [
      "stars kept while the pins come: the page asks once more, after; pins that can't come, the page says why",
      function () {
        var it = page({
          stars: "/f/stars",
          pins: "/f/pins",
          post: function (url, value, done) {
            done(null);
          }
        });
        answer(it, "/f/page", null, LOG.concat([["+", "Stars", { line: 3 }], PIN3]));
        it.view.send(["Starred", ["Said", 6], true]);
        it.view.send(["Starred", ["Said", 3], false]);
        it.view.send(["Starred", ["Said", 3], true]);
        var asked = it.asked.slice(1);
        answer(it, "/f/pins", null, [PIN3, PIN6]);
        var again = it.asked.slice(1 + asked.length);
        var pinned = facts(it, "Pin");
        answer(it, "/f/pins", "503 Service Unavailable", null);
        return asked.concat(again, pinned, facts(it, "Aside").length + " asides", it.said.slice(1));
      },
      [
        "/f/pins",
        "/f/pins",
        'Pin(3, 1, User, "[user]", "Plan the GUI")',
        'Pin(6, 1, Assistant, "[assistant]", "Done.")',
        "0 asides",
        "wrong: couldn't load the starred lines: 503 Service Unavailable"
      ]
    ],
    [
      "a jump asks for the rows around its line; ones that can't come, the page says why",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        it.view.send(["Jump", 1]);
        it.view.send(["Jump", 1]);
        var asked = it.asked.slice();
        answer(it, "/f/around/1", null, EARLIER);
        it.view.send(["Jump", 2]);
        answer(it, "/f/around/2", "502 Bad Gateway", null);
        return asked.concat(it.asked.slice(asked.length), facts(it, "Reveal"), "kept " + it.kept, it.said.slice(1));
      },
      [
        "/f/page",
        "/f/around/1",
        "/f/around/2",
        "Reveal(RunBox(1), 3, False)",
        "kept 1",
        "wrong: couldn't load the rows around line 2: 502 Bad Gateway"
      ]
    ],
    [
      "a page of tabs: the index's, then those the session kept, each asking for its URLs; the one kept shows",
      function () {
        var it = tabbed({ kept: { tabs: [7, 3], tab: 3 } });
        var asked = it.asked.slice();
        answer(it, "/f/tab/7", null, tabOf(7));
        answer(it, "/f/tab/3", null, tabOf(3));
        var more = it.asked.slice(asked.length);
        // The hidden tab's window comes last, and the title stays the shown one's.
        answer(it, "/f/page/3", null, LOG);
        answer(it, "/f/page/7", null, [["+", "Page", { title: "seven", items: "/f/item/", turns: "/f/turn-page/", first: 1 }]].concat(LOG.slice(1)));
        return asked.concat(more, strip(it, "Open"), strip(it, "Shown"), showing(it), it.addresses, it.kept, it.titles.slice(-1));
      },
      [
        "/f/index",
        "/f/tab/7",
        "/f/tab/3",
        "/f/page/7",
        "/f/page/3",
        'Open(0, "Root contexts", 0)',
        'Open(3, "ctx (3)", 2)',
        'Open(7, "ctx (7)", 1)',
        "Shown(3)",
        "2",
        "/f/c/3",
        "7-3 shows 3",
        "ui-0"
      ]
    ],
    [
      "a click on a link to a context opens its tab, and shows it, where it was; with a modifier key, the browser does as it does",
      function () {
        var it = tabbed();
        answer(it, "/f/index", null, INDEX);
        var link = linkTo(it.holder, "/f/c/7");
        var modified = it.doc.fire(link, "click", { ctrlKey: true });
        var before = it.asked.length;
        it.top = 120;
        var plain = it.doc.fire(link, "click");
        answer(it, "/f/tab/7", null, tabOf(7));
        it.top = 40;
        it.tabs.show(0);
        var back = it.top;
        it.tabs.show(7);
        return [modified.defaultPrevented + " " + before + " " + plain.defaultPrevented, back, it.top].concat(strip(it, "Shown"), showing(it), it.addresses, it.kept);
      },
      [
        "false 1 true",
        120,
        40,
        "Shown(7)",
        "1",
        "/f/",
        "/f/c/7",
        "/f/",
        "/f/c/7",
        " shows 0",
        "7 shows 7",
        "7 shows 0",
        "7 shows 7"
      ]
    ],
    [
      "closing the tab shown shows the next, else the one before; the index can't be closed; a tab that can't come closes, and the page says why",
      function () {
        var it = tabbed({ kept: { tabs: [7, 3, 5], tab: 3 } });
        answer(it, "/f/tab/7", null, tabOf(7));
        answer(it, "/f/tab/3", null, tabOf(3));
        it.tabs.strip.send(["Dismissed", ["TabOf", 3]]);
        var first = strip(it, "Shown");
        answer(it, "/f/tab/5", "404 Not Found", null);
        var second = strip(it, "Shown");
        it.tabs.close(0);
        return first.concat(second, strip(it, "Open"), it.said, it.kept.slice(-1));
      },
      [
        "Shown(5)",
        "Shown(7)",
        'Open(0, "Root contexts", 0)',
        'Open(7, "ctx (7)", 1)',
        "wrong: couldn't open context 5: 404 Not Found",
        "7 shows 7"
      ]
    ],
    [
      "a tab dragged onto another moves there, and the session keeps their order; the index stays first",
      function () {
        var it = tabbed({ kept: { tabs: [7, 3, 5], tab: 7 } });
        it.tabs.strip.send(["Moved", ["TabOf", 5], ["TabOf", 7]]);
        var left = strip(it, "Open").concat(it.kept.slice(-1));
        it.tabs.strip.send(["Moved", ["TabOf", 5], ["TabOf", 3]]);
        it.tabs.move(3, 0);
        it.tabs.move(0, 3);
        return left.concat(strip(it, "Open"), it.kept.slice(-1));
      },
      [
        'Open(0, "Root contexts", 0)',
        'Open(3, "3", 3)',
        'Open(5, "5", 1)',
        'Open(7, "7", 2)',
        "5-7-3 shows 7",
        'Open(0, "Root contexts", 0)',
        'Open(3, "3", 2)',
        'Open(5, "5", 3)',
        'Open(7, "7", 1)',
        "7-3-5 shows 7"
      ]
    ],
    [
      "only the tab shown asks what changed; a log's page opens its tab, which starts at its end",
      function () {
        var it = tabbed({ open: 7 });
        answer(it, "/f/tab/7", null, tabOf(7));
        answer(it, "/f/page/7", null, LOG);
        it.tabs.tick();
        answer(it, "/f/grow/7/6", null, []);
        var asked = it.asked.slice();
        it.tabs.show(0);
        answer(it, "/f/index", null, INDEX);
        it.tabs.tick();
        return asked.concat(it.asked.slice(asked.length), it.ends + " ends", it.titles.slice(-1));
      },
      ["/f/index", "/f/tab/7", "/f/page/7", "/f/grow/7/6", "/f/index", "1 ends", "Contexts"]
    ],
    [
      "a tab that the session left at a line goes back to it once its facts are shown, as the program's Restore, which asks for the rows around it",
      function () {
        var it = tabbed({ kept: { tabs: [3], tab: 3, places: { 3: 6 } } });
        answer(it, "/f/tab/3", null, tabOf(3));
        answer(it, "/f/page/3", null, LOG);
        var shown = { view: it.tabs.shown().view };
        return [it.ends + " ends"].concat(facts(shown, "Jumped"), facts(shown, "Reveal"), it.asked.slice(-1), it.places);
      },
      ["1 ends", "Jumped(6, 1, AtRow)", "Reveal(Said(6), 1, True)", "/f/around/6"]
    ],
    [
      "a tab closed and opened again starts at its end: where the session left it was for the page's start",
      function () {
        var it = tabbed({ kept: { tabs: [3], tab: 3, places: { 3: 6 } } });
        answer(it, "/f/tab/3", null, tabOf(3));
        answer(it, "/f/page/3", null, LOG);
        it.tabs.close(3);
        it.tabs.open(3, true);
        answer(it, "/f/tab/3", null, tabOf(3));
        answer(it, "/f/page/3", null, LOG);
        var shown = { view: it.tabs.shown().view };
        return [it.ends + " ends", facts(shown, "Jumped").length + " jumps"].concat(it.places);
      },
      ["2 ends", "0 jumps", "3 at null"]
    ],
    [
      "a hidden tab goes back to where it was left when it first shows, a run's row opening nothing; one left at its end starts there",
      function () {
        var it = tabbed({ kept: { tabs: [7, 3], tab: 3, places: { 7: 4, 3: "end" } } });
        answer(it, "/f/tab/7", null, tabOf(7));
        answer(it, "/f/tab/3", null, tabOf(3));
        answer(it, "/f/page/3", null, LOG);
        answer(it, "/f/page/7", null, LOG);
        var seven = null;
        it.tabs.each(function (tab) {
          if (tab.n === 7) {
            seven = { view: tab.view };
          }
        });
        var before = [it.ends + " ends", facts(seven, "Jumped").length + " jumps"];
        it.tabs.show(7);
        return before.concat(it.ends + " ends", facts(seven, "Jumped"), facts(seven, "Reveal"), facts(seven, "RunOpen"), wrong(it) + " wrong");
      },
      ["1 ends", "0 jumps", "2 ends", "Jumped(4, 1, AtRow)", "Reveal(RunBox(4), 1, True)", "RunOpen(4, False)", "0 wrong"]
    ],
    [
      "where the tab shown is, the session keeps as a scroll settles, once its facts are shown, and as another tab shows; not the index's; and it forgets a tab's that closes",
      function () {
        var it = tabbed({ kept: { tabs: [7, 3], tab: 3 } });
        answer(it, "/f/index", null, INDEX);
        answer(it, "/f/tab/7", null, tabOf(7));
        answer(it, "/f/tab/3", null, tabOf(3));
        it.at = 2;
        it.tabs.settle();
        answer(it, "/f/page/3", null, LOG);
        answer(it, "/f/page/7", null, LOG);
        it.at = 5;
        it.tabs.settle();
        it.tabs.settle();
        it.at = "end";
        it.tabs.show(7);
        it.at = 3;
        it.tabs.settle();
        it.tabs.show(0);
        it.at = 1;
        it.tabs.settle();
        it.tabs.close(7);
        return it.places;
      },
      ["3 at 5", "3 at end", "7 at 3", "7 at null"]
    ],
    [
      "a place is the line of the first row below the screen's top that a message or a run starts; past a turn's mark, the row after it",
      function () {
        var it = page();
        answer(it, "/f/page", null, LOG);
        var rows = rowsOf(it.view.root);
        var from = function (at) {
          return String(
            contexts.placeOf(it.view, rows, function (row) {
              return rows.indexOf(row) >= at;
            })
          );
        };
        return [rows.length + " rows", from(0), from(2), from(3), from(4)];
      },
      ["4 rows", "3", "4", "6", "null"]
    ],
    [
      "an assistant's messages as Markdown writes them, as the bar says, from the start",
      function () {
        var it = page({ formatting: true });
        answer(it, "/f/page", null, LOG);
        return facts(it, "Formatting").concat(facts(it, "Prose"), facts(it, "Code").length + " code");
      },
      ["Formatting(True)", 'Prose(SaidText(6), "Done.", Said(6), 1)', "1 code"]
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
