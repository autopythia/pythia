// tools/contexts_bench.js: times pythia's contexts (lem/contexts.lem) in
// JavaScript at the size of a long log, as tools/lem_bench.py times lem in
// Rust. For each size, up to 13,394 items, the most that one of pythia's
// contexts has had: the first tick, from the log's turns, folded, to their
// widgets' elements, through glui's recipes, in a stand-in DOM
// (tools/stand_in_dom.js); opening a turn, with the two ticks that it
// takes, one that asks for its items and one that shows them; and opening
// an entry in it. Then, with every turn's items there, as when each was
// opened, opening an entry again. It checks that each shows what it
// should, and fails if one doesn't. `make js-bench` runs it, in JavaScriptCore through
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

  // A log's facts, as JSON steps: its turns, folded; and each turn's items,
  // as the host sends them when it's opened, a third of them with a body.
  function log(items) {
    var turns = Math.max(1, Math.floor(items / ITEMS_PER_TURN));
    var steps = [["+", "Page", { title: "log", items: "/c/log?line=", turns: "/c/log?turn=" }]];
    var byTurn = [];
    for (var t = 0; t < turns; t++) {
      steps.push(["+", "Turn", { n: t, label: "Turn " + t, title: "Turn " + (t + 1) }]);
      byTurn.push([["+", "Loaded", { turn: t }]]);
    }
    for (var line = 1; line <= items; line++) {
      var item = { line: line, turn: (line - 1) % turns, tone: TONES[line % 6], label: "[tool-call] exec_command" };
      item.gist = "item " + line;
      item.body = line % 3 === 0;
      byTurn[item.turn].push(["+", "Item", item]);
    }
    return { steps: steps, byTurn: byTurn };
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

  out(pad("items", 6) + pad("first tick", 13) + pad("a turn", 10) + pad("an entry", 11) +
      pad("every turn open: an entry", 28));
  SIZES.forEach(function (items) {
    var made = log(items);
    var turns = made.byTurn.length;
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
      view.update(made.steps);
    });
    check("the first tick shows " + count(root, "g-entry") + " entries", count(root, "g-entry") === turns);
    var last = turns - 1;
    var turn = made.byTurn[last];
    var opening = ms(function () {
      view.send(["Toggled", ["TurnBox", last], true]);
      view.update(turn);
    });
    check("opening the turn asks for its items", effects.length === 1 && effects[0].name === "LoadTurn");
    check("the turn shows its items", count(root, "g-entry") === turns + turn.length - 1);
    var line = turn.filter(function (step) {
      return step[1] === "Item" && step[2].body;
    })[0][2].line;
    var entry = ms(function () {
      view.send(["Toggled", ["Ent", line], true]);
    });
    check("the entry asks for its body", effects.length === 2 && effects[1].name === "Load");
    view.update([].concat.apply([], made.byTurn.slice(0, last)));
    check("every turn shows its items", count(root, "g-entry") === turns + items);
    var all = ms(function () {
      view.send(["Toggled", ["Ent", 3], true]);
    });
    check("that entry asks for its body", effects.length === 3 && effects[2].name === "Load");
    out(pad(commas(items), 6) + pad(first + " ms", 13) + pad(opening + " ms", 10) + pad(entry + " ms", 11) +
        pad(all + " ms", 28));
  });
})();
