// tools/glem_test.js: runs glem's shared tests on glem.js, glem's runtime
// in JavaScript (src/glem/glem.js), and reports them in TAP (tools/tap.js).
//
// The tests are what src/glem/compile_test.rs writes as GLEM_TESTS to
// target/web/glem_tests.js: the task list's compiled form
// (lem/test/tasks.lem), and its ticks (lem/test/tasks_ticks.txt) as JSON
// steps, with what each shows and the facts that its joins visit, which
// both runtimes must agree on, and which a runtime that forgets what its
// rules read before each tick must show too, with the same facts and why
// each holds; the tree that glem.js keeps from tick to tick must be the
// one that tree() builds anew; and the cases of lem/test/tasks_dom.txt,
// which only JavaScript runs. In a stand-in DOM (tools/stand_in_dom.js),
// with toy recipes that make an element per widget, with its fields as
// attributes, the view must show what the runtime does, after each tick of
// both files, keep the elements of the widgets that stay, and send what
// clicks, focus and keys say. `make js-test` runs lem.js, glem.js, the
// tests, tools/tap.js, tools/stand_in_dom.js and this as one script.

(function () {
  "use strict";

  var tests = GLEM_TESTS;

  // A tick's steps on a runtime: what it shows, or its error.
  function attempt(runtime, steps) {
    try {
      return runtime.run(steps, null);
    } catch (e) {
      return tap.error(e);
    }
  }

  // Every relation's facts, as lem writes them, and why each holds.
  function everything(runtime) {
    var prog = runtime.prog;
    var out = "";
    prog.rels.forEach(function (rel, r) {
      runtime.relFacts(r).forEach(function (fact) {
        out += prog.factText(r, fact.tuple) + " (" + runtime.why(fact.why) + ")\n";
      });
    });
    return out;
  }

  // A tree's nodes, a line each: its depth, and its fact as lem writes it.
  function treeText(runtime, nodes) {
    var prog = runtime.prog.lem;
    return nodes
      .map(function (node) {
        return node.depth + " " + prog.factText(node.rel, node.tuple);
      })
      .join("\n");
  }

  // The tick file, on glem.js's runtime; and again on one that forgets,
  // before each tick, what its rules read. After a tick that doesn't fail,
  // the tree that the runtime keeps is the one that tree() builds anew.
  function ticks() {
    var runtime = new glem.Runtime(new glem.Program(tests.program));
    var forgets = new glem.Runtime(new glem.Program(tests.program));
    tests.ticks.forEach(function (c) {
      var got = attempt(runtime, c.steps);
      forgets.lem.forget();
      var again = attempt(forgets, c.steps);
      var why = null;
      if (got !== c.want) {
        why = "the tick gives\n" + got + "\nnot\n" + c.want;
      } else if (runtime.lem.visits() !== c.visits) {
        why = "its joins visit " + runtime.lem.visits() + " facts, not " + c.visits;
      } else if (again !== got) {
        why = "forgetting what its rules read, the tick gives\n" + again;
      } else if (everything(forgets.lem) !== everything(runtime.lem)) {
        why = "remembering, the facts are\n" + everything(runtime.lem) + "forgetting, they're\n" + everything(forgets.lem);
      } else if (got.indexOf("error: ") !== 0) {
        var kept = treeText(runtime, runtime.nodes());
        var anew = treeText(runtime, runtime.tree());
        if (kept !== anew) {
          why = "the tree that it keeps is\n" + kept + "\nnot\n" + anew;
        }
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
          if (attribute.name !== "key") {
            line += ", " + attribute.name + "=" + attribute.value;
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
  // effect. A tick that follows one that didn't fail, and doesn't fail
  // itself, changes the tree and the page where they changed: it neither
  // builds the tree anew nor shows it whole.
  function rendered() {
    var it = fixture([]);
    var anew = 0;
    var whole = 0;
    var build = it.runtime.build;
    it.runtime.build = function () {
      anew++;
      return build.apply(this, arguments);
    };
    var renderAll = it.view.renderAll;
    it.view.renderAll = function () {
      whole++;
      return renderAll.apply(this, arguments);
    };
    var steady = false;
    tests.ticks.forEach(function (c) {
      var before = elements(it.root);
      var shown = page(it.root);
      var made = it.doc.made;
      it.heard = [];
      anew = 0;
      whole = 0;
      var thrown = null;
      try {
        it.view.update(c.steps);
      } catch (e) {
        thrown = tap.error(e);
      }
      var why = null;
      var effects = c.want.split("\n").filter(function (line) {
        return line.indexOf("effect ") === 0;
      });
      var failed = c.want.indexOf("error: ") === 0;
      if (thrown !== null) {
        why = "the view throws " + thrown;
      } else if (failed) {
        if (it.heard.join("\n") !== c.want || page(it.root) !== shown) {
          why = "the host hears\n" + it.heard.join("\n") + "\nand the page is\n" + page(it.root);
        }
      } else if (page(it.root) !== it.runtime.outline(null).join("\n")) {
        why = "the page is\n" + page(it.root) + "\nnot\n" + it.runtime.outline(null).join("\n");
      } else if (it.heard.length !== effects.length) {
        why = "the host hears\n" + it.heard.join("\n") + "\nnot " + effects.length + " effects";
      } else if (steady && (anew > 0 || whole > 0)) {
        why = "the tick built the tree anew " + anew + " times, and showed it whole " + whole + " times";
      }
      steady = !failed;
      tap.report("tasks_ticks.txt, in the DOM: " + c.title, why || kept(it, before, made));
    });
  }

  // Whether the elements of the widgets that stayed, since `before`, are the
  // same ones, and only new widgets got elements, of which there were
  // `made` before: null, or why not. A widget whose kind changed is new: its
  // kind's recipe makes it an element, whose tag here is the kind.
  function kept(it, before, made) {
    var fresh = 0;
    var why = null;
    elements(it.root).forEach(function (element, key) {
      if (!before.has(key) || before.get(key).tagName !== element.tagName) {
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

  // A tick that changes widgets changes the tree by what lem's tick logged
  // (glem.Runtime's grow): with 200 rows, a tick that adds one walks no
  // widget kind's facts and builds nothing anew, and the tree that it keeps
  // is the one that tree() builds.
  function logged() {
    var runtime = new glem.Runtime(new glem.Program(tests.program));
    var kinds = runtime.prog.kinds;
    var steps = [];
    for (var i = 0; i < 200; i++) {
      steps.push(["+", "Todo", { id: "t" + i, title: "row " + i, at: i }]);
    }
    runtime.run(steps, null);
    var walked = 0;
    var anew = 0;
    var grow = runtime.grow;
    var build = runtime.build;
    runtime.build = function () {
      anew++;
      return build.apply(this, arguments);
    };
    runtime.grow = function () {
      var spied = kinds.map(function (kind) {
        return runtime.lem.tables[kind.rel].facts;
      });
      spied.forEach(function (facts) {
        facts.forEach = function () {
          walked++;
          return Map.prototype.forEach.apply(this, arguments);
        };
      });
      try {
        return grow.apply(this, arguments);
      } finally {
        spied.forEach(function (facts) {
          delete facts.forEach;
        });
      }
    };
    runtime.run([["+", "Todo", { id: "t200", title: "row 200", at: 200 }]], null);
    var why = null;
    if (walked > 0 || anew > 0) {
      why = "the tick walked " + walked + " kinds' facts, and built the tree anew " + anew + " times";
    } else if (treeText(runtime, runtime.nodes()) !== treeText(runtime, runtime.tree())) {
      why = "the tree that it keeps isn't the one that tree() builds";
    } else if (runtime.nodes().length !== 203) {
      why = "the tree has " + runtime.nodes().length + " widgets, not 203";
    }
    tap.report("a tick that adds a row to 200 walks no kind's facts", why);
  }

  ticks();
  rendered();
  logged();
  dom();
  refused();
  tap.finish();
})();
