// tools/contexts_bench.js: times pythia's contexts (lem/contexts.lem) in
// JavaScript at the size of a long log, as tools/lem_bench.py times lem in
// Rust. For each size, up to 13,394 items, the most that one of pythia's
// contexts has had: the first tick, from the log's facts to its widgets'
// elements, through glui's recipes, in a stand-in DOM
// (tools/stand_in_dom.js); then a tick that changes an item, and one that
// opens an entry. It checks that each shows what it should, and fails if
// one doesn't. `make js-bench` runs it, in JavaScriptCore through
// osascript on a Mac, or in another engine, as JS=qjs, on the program that
// `make js-test` uses, CONTEXTS_TESTS.program.
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

  // A log's facts, as JSON steps: its turns, and its items, a third of
  // them with a body.
  function log(items) {
    var turns = Math.max(1, Math.floor(items / ITEMS_PER_TURN));
    var steps = [["+", "Page", { title: "log", items: "/c/log?line=" }]];
    for (var t = 0; t < turns; t++) {
      steps.push(["+", "Turn", { n: t, title: "Turn " + (t + 1) }]);
    }
    for (var line = 1; line <= items; line++) {
      var item = { line: line, turn: (line - 1) % turns, tone: TONES[line % 6], label: "[tool-call] exec_command" };
      item.gist = "item " + line;
      item.body = line % 3 === 0;
      steps.push(["+", "Item", item]);
    }
    return steps;
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

  out(pad("items", 6) + pad("first tick", 13) + pad("a change", 11) + pad("an opening", 13));
  SIZES.forEach(function (items) {
    var steps = log(items);
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
      view.update(steps);
    });
    check("the first tick shows " + count(root, "g-entry") + " entries", count(root, "g-entry") === items);
    var last = steps[steps.length - 1][2];
    var changed = { line: last.line, turn: last.turn, tone: last.tone, label: last.label, gist: "changed", body: last.body };
    var change = ms(function () {
      view.update([["+", "Item", changed]]);
    });
    check("the change shows", runtime.lem.relFacts(runtime.prog.lem.rel("Item")).length === items);
    var opening = ms(function () {
      view.send(["Toggled", ["Ent", 3], true]);
    });
    check("the opening asks for the body", effects.length === 1 && effects[0].name === "Load");
    out(pad(commas(items), 6) + pad(first + " ms", 13) + pad(change + " ms", 11) + pad(opening + " ms", 13));
  });
})();
