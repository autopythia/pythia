// tools/glem_test.js: runs glem's shared tests on glem.js, glem's runtime
// in JavaScript (src/glem/glem.js), and reports them in TAP (tools/tap.js).
//
// The tests are what src/glem/compile_test.rs writes as GLEM_TESTS to
// target/web/glem_tests.js: the task list's compiled form
// (lem/test/tasks.lem), and its ticks (lem/test/tasks_ticks.txt) as JSON
// steps, with what each shows and the facts that its joins visit, which
// both runtimes must agree on; and the cases of lem/test/tasks_dom.txt,
// which only JavaScript runs. In a stand-in DOM (tools/stand_in_dom.js),
// with toy recipes that make an element per widget, with its fields as
// attributes, the view must show what the runtime does, after each tick of
// both files, keep the elements of the widgets that stay, and send what
// clicks, focus and keys say. `make js-test` runs lem.js, glem.js, the
// tests, tools/tap.js, tools/stand_in_dom.js and this as one script.

(function () {
  "use strict";

  var tests = GLEM_TESTS;

  // The tick file, on glem.js's runtime.
  function ticks() {
    var runtime = new glem.Runtime(new glem.Program(tests.program));
    tests.ticks.forEach(function (c) {
      var got;
      try {
        got = runtime.run(c.steps, null);
      } catch (e) {
        got = tap.error(e);
      }
      var why = null;
      if (got !== c.want) {
        why = "the tick gives\n" + got + "\nnot\n" + c.want;
      } else if (runtime.lem.visits() !== c.visits) {
        why = "its joins visit " + runtime.lem.visits() + " facts, not " + c.visits;
      }
      tap.report("tasks_ticks.txt: " + c.title, why);
    });
  }

  // Toy recipes, for the program's kinds: an element named by the kind,
  // with a `key` attribute, its id, and one for each field, as lem writes
  // them, holding its children; an element with an `on` field sends it
  // when clicked. The kinds in `leaves` hold no children.
  function toys(prog, leaves) {
    var recipes = {};
    prog.kinds.forEach(function (kind) {
      var name = prog.lem.rels[kind.rel].name;
      recipes[name] = function (w, view) {
        var element = view.doc.createElement(name);
        element.addEventListener("click", function () {
          if (w.names.indexOf("on") >= 0) {
            view.send(w.fields.on);
          }
        });
        return {
          element: element,
          children: leaves.indexOf(name) >= 0 ? null : element,
          update: function () {
            element.setAttribute("key", w.key);
            w.names.forEach(function (field) {
              element.setAttribute(field, w.text(field));
            });
          }
        };
      };
    });
    return recipes;
  }

  // The page under `root`, as glem's outline writes widgets: an element a
  // line, its tag and its `key`, then its other attributes.
  function page(root) {
    var lines = [];
    (function walk(node, depth) {
      node.childNodes.forEach(function (child) {
        var line = new Array(depth + 1).join("  ") + child.tagName + "(" + child.getAttribute("key");
        child.attributes.forEach(function (attribute) {
          if (attribute[0] !== "key") {
            line += ", " + attribute[0] + "=" + attribute[1];
          }
        });
        lines.push(line + ")");
        walk(child, depth + 1);
      });
    })(root, 0);
    return lines.join("\n");
  }

  // The elements under `root`, by their `key`.
  function elements(root) {
    var all = new Map();
    (function walk(node) {
      node.childNodes.forEach(function (child) {
        all.set(child.getAttribute("key"), child);
        walk(child);
      });
    })(root);
    return all;
  }

  // A view of the task list, from no facts, in a new stand-in DOM, with
  // toy recipes; and `heard`, what its host hears, as lines.
  function fixture(leaves) {
    var doc = new standInDom.Document();
    var root = doc.createElement("root");
    var prog = new glem.Program(tests.program);
    var runtime = new glem.Runtime(prog);
    var it = { doc: doc, root: root, runtime: runtime, heard: [] };
    it.view = new glem.View(runtime, root, toys(prog, leaves), {
      effects: function (effects) {
        effects.forEach(function (effect) {
          it.heard.push("effect " + effect.name + " " + JSON.stringify(effect.fields));
        });
      },
      error: function (message) {
        it.heard.push("error: " + message);
      }
    });
    return it;
  }

  // The tick file, through the view: after each tick, the page shows what
  // the runtime's outline does; or, after a tick that fails, what it showed,
  // and the host hears the error. The elements of widgets that stay are the
  // same ones, and only new widgets get elements; and the host hears each
  // effect.
  function rendered() {
    var it = fixture([]);
    tests.ticks.forEach(function (c) {
      var before = elements(it.root);
      var shown = page(it.root);
      var made = it.doc.made;
      it.heard = [];
      it.view.update(c.steps);
      var why = null;
      var effects = c.want.split("\n").filter(function (line) {
        return line.indexOf("effect ") === 0;
      });
      if (c.want.indexOf("error: ") === 0) {
        if (it.heard.join("\n") !== c.want || page(it.root) !== shown) {
          why = "the host hears\n" + it.heard.join("\n") + "\nand the page is\n" + page(it.root);
        }
      } else if (page(it.root) !== it.runtime.outline(null).join("\n")) {
        why = "the page is\n" + page(it.root) + "\nnot\n" + it.runtime.outline(null).join("\n");
      } else if (it.heard.length !== effects.length) {
        why = "the host hears\n" + it.heard.join("\n") + "\nnot " + effects.length + " effects";
      }
      tap.report("tasks_ticks.txt, in the DOM: " + c.title, why || kept(it, before, made));
    });
  }

  // Whether the elements of the widgets that stayed, since `before`, are the
  // same ones, and only new widgets got elements, of which there were
  // `made` before: null, or why not.
  function kept(it, before, made) {
    var fresh = 0;
    var why = null;
    elements(it.root).forEach(function (element, key) {
      if (!before.has(key)) {
        fresh++;
      } else if (before.get(key) !== element && why === null) {
        why = key + " has a new element";
      }
    });
    if (why === null && it.doc.made - made !== fresh) {
      why = it.doc.made - made + " elements made, for " + fresh + " new widgets";
    }
    return why;
  }

  // The cases of tasks_dom.txt: the host's steps, a tick each run of them;
  // and what a user does. Each shows the page, the Focused fact, and what
  // the host heard.
  function dom() {
    var it = fixture([]);
    var r = it.runtime.prog.lem.rel("Focused");
    tests.dom.forEach(function (c) {
      it.heard = [];
      var before = elements(it.root);
      var made = it.doc.made;
      var got;
      try {
        var steps = [];
        c.steps.concat([["end"]]).forEach(function (step) {
          if (step[0] === "+" || step[0] === "-" || step[0] === "!") {
            steps.push(step);
            return;
          }
          if (steps.length > 0) {
            it.view.update(steps);
            steps = [];
          }
          act(it, step);
        });
        var focused = it.runtime.lem.relFacts(r).map(function (fact) {
          return it.runtime.prog.lem.factText(r, fact.tuple);
        });
        got = [page(it.root) || "(no widgets)"].concat(focused, it.heard).join("\n");
      } catch (e) {
        got = tap.error(e);
      }
      var why = got === c.want ? null : "the case gives\n" + got + "\nnot\n" + c.want;
      tap.report("tasks_dom.txt: " + c.title, why || kept(it, before, made));
    });
  }

  // What a user does, as tasks_dom.txt writes it.
  function act(it, step) {
    var byKey = elements(it.root);
    var target = function (key) {
      if (!byKey.has(key)) {
        throw new Error("no element shows " + key);
      }
      return byKey.get(key);
    };
    switch (step[0]) {
      case "click":
        it.doc.fire(target(step[1]), "click");
        return;
      case "focus":
        it.doc.focus(target(step[1]));
        return;
      case "blur":
        it.doc.focus(null);
        return;
      case "key":
        var at = it.doc.activeElement !== null && it.root.contains(it.doc.activeElement);
        var init = { key: step[1] };
        Object.keys(step[2] || {}).forEach(function (name) {
          init[name] = step[2][name];
        });
        it.doc.fire(at ? it.doc.activeElement : it.root, "keydown", init);
        return;
      case "end":
        return;
    }
    throw new Error("there's no step " + JSON.stringify(step));
  }

  // What the view refuses: recipes that don't match the program's kinds;
  // and a widget in one whose recipe holds no children, which leaves the
  // page as it was.
  function refused() {
    var prog = new glem.Program(tests.program);
    var made = function (recipes) {
      try {
        new glem.View(new glem.Runtime(prog), new standInDom.Document().createElement("root"), recipes, {});
        return "it's made";
      } catch (e) {
        return tap.error(e);
      }
    };
    var some = toys(prog, []);
    delete some.Group;
    var more = toys(prog, []);
    more.Todo = more.Item;
    var cases = [
      ["recipes without one for a kind", made(some), "error: there's no recipe for Group"],
      [
        "a recipe for a relation that isn't a widget",
        made(more),
        "error: there's a recipe for Todo, which isn't a widget or a surface"
      ]
    ];
    var it = fixture(["Item"]);
    it.view.update([["+", "Todo", { id: "a", title: "first", at: 10 }]]);
    it.view.update([["!", ["Type", "a", "draft"]]]);
    var shown = page(it.root);
    it.heard = [];
    it.view.update([["!", ["Open", "a"]]]);
    var whole = it.heard.concat(page(it.root) === shown ? [] : ["and the page changed"]).join("\n");
    var want = "error: Item's recipe holds no children, but Item(Row(\"a\")) has some";
    cases.push(["a widget in one whose recipe holds no children", whole, want]);
    cases.forEach(function (c) {
      tap.report(c[0], c[1] === c[2] ? null : c[1] + "\nnot\n" + c[2]);
    });
  }

  ticks();
  rendered();
  dom();
  refused();
  tap.finish();
})();
