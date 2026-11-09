// tools/contexts_bench.js: times pythia's contexts (lem/contexts.lem) in
// JavaScript at the size of a long log, as tools/lem_bench.py times lem in
// Rust. For each size, up to 13,394 items, the most that one of pythia's
// contexts had when it was first written: the first tick, from the log's
// turns and its last rows to their widgets' elements, through glui's
// recipes, in a stand-in DOM (tools/stand_in_dom.js); the rows before
// those, as when the page's top comes near the screen; opening a run, with
// the two ticks that it takes, one that asks for its items and one that
// shows them; and opening an entry in it. Then, with every row there, as
// when the page was scrolled to its start, opening an entry again. It
// checks that each shows what it should, and fails if one doesn't.
// `make js-bench` runs it, in JavaScriptCore through osascript on a Mac, or
// in another engine, as JS=qjs, on the program that `make js-test` uses,
// CONTEXTS_TESTS.program.
(function () {
  "use strict";

  var SIZES = [1000, 4000, 13394];
  var ITEMS_PER_TURN = 50;
  var TONES = ["User", "Dim", "Tool", "Tool", "Assistant", "Dim"];

  function out(text) {
    if (typeof print === "function" && typeof window === "undefined") {
      print(text);
    } else {
      console.log(text);
    }
  }

  // As many rows as the node server sends at once.
  var CHUNK = 200;

  // A log's rows, as the node server sends them, in turns of ITEMS_PER_TURN
  // items: the user's message, then runs of six tool items, each followed
  // by the assistant's message; with each run's items, a third of them with
  // a body; in chunks of CHUNK rows, the last first.
  function log(items) {
    var turns = Math.max(1, Math.floor(items / ITEMS_PER_TURN));
    var rows = [];
    var runs = {};
    var marks = [];
    for (var line = 1; line <= items; line++) {
      var turn = Math.min(Math.floor((line - 1) / ITEMS_PER_TURN), turns - 1);
      var at = line - 1 - turn * ITEMS_PER_TURN;
      if (at === 0) {
        marks.push(["+", "Turn", { n: turn, label: "Turn " + turn, title: "Turn " + turn, line: line }]);
      }
      if (at === 0 || at % 7 === 0) {
        var tone = at === 0 ? "User" : "Assistant";
        rows.push({ step: ["+", "Message", { line: line, turn: turn, tone: tone, label: "[" + tone.toLowerCase() + "]", text: "said " + line }], line: line, last: line });
        continue;
      }
      var last = rows[rows.length - 1];
      if (last.step[1] !== "Run" || last.step[2].turn !== turn) {
        last = { step: ["+", "Run", { line: line, turn: turn, last: line, text: "a run" }], line: line, last: line };
        rows.push(last);
        runs[line] = [["+", "RunLoaded", { run: line }]];
      }
      last.last = line;
      last.step[2].last = line;
      var item = { line: line, run: last.line, tone: TONES[line % 6], label: "[tool-call] exec_command" };
      item.gist = "item " + line;
      item.body = line % 3 === 0;
      runs[last.line].push(["+", "Item", item]);
    }
    var chunks = [];
    for (var end = rows.length; end > 0; end -= CHUNK) {
      var some = rows.slice(Math.max(0, end - CHUNK), end);
      var first = end - CHUNK <= 0 ? 1 : some[0].line;
      chunks.push([["+", "Chunk", { first: first, last: some[some.length - 1].last }]].concat(
        some.map(function (row) {
          return row.step;
        })
      ));
    }
    var page = [["+", "Page", { title: "log", items: "/c/log?line=", turns: "/c/log?turn=", first: 1, last: items }]];
    return { page: page.concat(marks, chunks[0]), chunks: chunks, runs: runs, rows: rows };
  }

  function ms(f) {
    var began = Date.now();
    f();
    return Date.now() - began;
  }

  function pad(text, width) {
    text = String(text);
    while (text.length < width) {
      text = " " + text;
    }
    return text;
  }

  // A whole number with commas, as 13,394, in every engine.
  function commas(n) {
    return String(n).replace(/\B(?=(\d{3})+$)/g, ",");
  }

  function count(node, name) {
    var n = (node.getAttribute && (node.getAttribute("class") || "").split(" ").indexOf(name) >= 0) ? 1 : 0;
    for (var i = 0; i < node.childNodes.length; i++) {
      n += count(node.childNodes[i], name);
    }
    return n;
  }

  function check(what, ok) {
    if (!ok) {
      throw new Error("contexts_bench: " + what);
    }
  }

  out(pad("items", 6) + pad("first tick", 13) + pad("earlier rows", 15) + pad("a run", 9) +
      pad("an entry", 11) + pad("every row there: an entry", 28));
  SIZES.forEach(function (items) {
    var made = log(items);
    var effects = [];
    var doc = new standInDom.Document();
    var root = doc.createElement("div");
    var runtime = new glem.Runtime(new glem.Program(CONTEXTS_TESTS.program));
    var view = glui.view(runtime, root, {
      effects: function (list) {
        effects = effects.concat(list);
      },
      error: function (message) {
        throw new Error(message);
      }
    });
    var first = ms(function () {
      view.update(made.page);
    });
    var shown = made.chunks[0].length - 1;
    check("the first tick shows its rows", count(root, "g-entry") >= shown);
    var earlier = 0;
    if (made.chunks.length > 1) {
      earlier = ms(function () {
        view.update(made.chunks[1]);
      });
      check("the earlier rows show", count(root, "g-entry") >= shown + made.chunks[1].length - 1);
    }
    var lastRun = made.chunks[0].filter(function (step) {
      return step[1] === "Run";
    })[0][2].line;
    var opening = ms(function () {
      view.send(["Toggled", ["RunBox", lastRun], true]);
      view.update(made.runs[lastRun]);
    });
    check("opening the run asks for its items", effects.length === 1 && effects[0].name === "LoadRun");
    var line = made.runs[lastRun].filter(function (step) {
      return step[1] === "Item" && step[2].body;
    })[0][2].line;
    var entry = ms(function () {
      view.send(["Toggled", ["Ent", line], true]);
    });
    check("the entry asks for its body", effects.length === 2 && effects[1].name === "Load");
    made.chunks.slice(2).forEach(function (chunk) {
      view.update(chunk);
    });
    check("every row shows", count(root, "g-entry") >= made.rows.length);
    var firstRun = made.rows.filter(function (row) {
      return row.step[1] === "Run";
    })[0].line;
    view.send(["Toggled", ["RunBox", firstRun], true]);
    view.update(made.runs[firstRun]);
    var other = made.runs[firstRun].filter(function (step) {
      return step[1] === "Item" && step[2].body;
    })[0][2].line;
    var all = ms(function () {
      view.send(["Toggled", ["Ent", other], true]);
    });
    check("that entry asks for its body", effects[effects.length - 1].name === "Load");
    out(pad(commas(items), 6) + pad(first + " ms", 13) + pad(earlier + " ms", 15) + pad(opening + " ms", 9) +
        pad(entry + " ms", 11) + pad(all + " ms", 28));
  });
})();
