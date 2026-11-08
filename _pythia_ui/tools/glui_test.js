// tools/glui_test.js: runs the tests of glui.js, glui's web look
// (src/glui/glui.js), and reports them in TAP (tools/tap.js).
//
// The tests are what src/glui/tests.rs writes as GLUI_TESTS to
// target/web/glui_tests.js: programs of glui's widgets, compiled, each with
// the cases of its file in lem/test/, which run through glui's recipes in a
// stand-in DOM (tools/stand_in_dom.js): the approval HUD
// (approval_dom.txt), a page of the kinds for documents (page_dom.txt) and
// a form of glui's controls (form_dom.txt); and glui's themes, compiled,
// with their values as CSS variables (themes_css.txt). In a browser
// (web/test.html), each program's first cases run in the page's own DOM
// too, up to the first that needs what that can't do at once (AT_ONCE),
// and must give the same. `make js-test` runs lem.js, glem.js, glui.js,
// the tests, tools/tap.js, tools/stand_in_dom.js and this as one script.

(function () {
  "use strict";

  var tests = GLUI_TESTS;

  // An element as a line: its tag and its classes, as CSS writes them; its
  // other attributes, in order, each with its value unless that's empty; an
  // input's `checked` and its text, which aren't attributes, as `:checked`
  // and `:value`; and its text, if text is all it holds. For the elements of
  // either DOM, the stand-in's or a browser's.
  function describe(e) {
    var line = e.localName;
    var classes = e.getAttribute("class");
    if (classes) {
      line += "." + classes.split(" ").join(".");
    }
    for (var i = 0; i < e.attributes.length; i++) {
      var a = e.attributes[i];
      if (a.name !== "class") {
        line += a.value === "" ? " " + a.name : " " + a.name + "=" + JSON.stringify(a.value);
      }
    }
    if (e.localName === "input" && e.checked === true) {
      line += " :checked";
    }
    if (e.localName === "input" && e.getAttribute("type") === "text" && e.value) {
      line += " :value=" + JSON.stringify(e.value);
    }
    var text = textAlone(e);
    return text === null ? line : line + " " + JSON.stringify(text);
  }

  // An element's text, if it holds text and nothing else; else null.
  function textAlone(e) {
    if (e.childNodes.length === 0) {
      return null;
    }
    var text = "";
    for (var i = 0; i < e.childNodes.length; i++) {
      if (e.childNodes[i].nodeType !== 3) {
        return null;
      }
      text += e.childNodes[i].data;
    }
    return text;
  }

  // The page under `root`, as lines: an element a line, as describe()
  // writes it, with what it holds under it, indented, unless that's text
  // alone; and text among elements, quoted, as a line of its own.
  function page(root) {
    var lines = [];
    (function walk(node, depth) {
      for (var i = 0; i < node.childNodes.length; i++) {
        var child = node.childNodes[i];
        var pad = new Array(depth + 1).join("  ");
        if (child.nodeType === 3) {
          lines.push(pad + JSON.stringify(child.data));
        } else {
          lines.push(pad + describe(child));
          if (textAlone(child) === null) {
            walk(child, depth + 1);
          }
        }
      }
    })(root, 0);
    return lines.join("\n");
  }

  // The widgets' elements, by the widgets' ids, as lem writes them.
  function elements(view) {
    var all = new Map();
    view.widgets.forEach(function (w) {
      all.set(w.key, w.made.element);
    });
    return all;
  }

  // The first element of one of `tags` in `e`, or `e` itself, else null.
  function find(e, tags) {
    if (e.nodeType === 1 && tags.indexOf(e.localName) >= 0) {
      return e;
    }
    for (var i = 0; i < e.childNodes.length; i++) {
      var found = find(e.childNodes[i], tags);
      if (found !== null) {
        return found;
      }
    }
    return null;
  }

  // A view of a compiled program in `root`, of `doc`, through glui's
  // recipes, from no facts; and `heard`, what its host hears, as lines.
  // `real` says whether `doc` is a browser's.
  function fixture(program, doc, root) {
    var runtime = new glem.Runtime(new glem.Program(program));
    var real = typeof document !== "undefined" && doc === document;
    var it = { doc: doc, root: root, runtime: runtime, heard: [], composing: false, real: real };
    it.view = glui.view(runtime, root, {
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

  // What a case shows: the page's title, if it has one; the page, or "(no
  // widgets)"; the Focused fact, if any; and what the host heard.
  function shown(it) {
    var r = it.runtime.prog.lem.rel("Focused");
    var focused = it.runtime.lem.relFacts(r).map(function (fact) {
      return it.runtime.prog.lem.factText(r, fact.tuple);
    });
    var title = it.doc.title ? ["page title: " + JSON.stringify(it.doc.title)] : [];
    return title.concat([page(it.root) || "(no widgets)"], focused, it.heard).join("\n");
  }

  // Whether a step is the host's, which the view runs with the others
  // next to it as one tick.
  function host(step) {
    return step[0] === "+" || step[0] === "-" || step[0] === "!";
  }

  // Runs a case's steps: the host's, a tick each run of them, and what a
  // user does, each in turn.
  function run(it, steps) {
    var facts = [];
    steps.concat([["end"]]).forEach(function (step) {
      if (host(step)) {
        facts.push(step);
        return;
      }
      if (facts.length > 0) {
        it.view.update(facts);
        facts = [];
      }
      act(it, step);
    });
  }

  // What a browser's DOM can do as a user would, at once, in a page's
  // hidden part: the rest needs the keyboard's focus, an input method, or a
  // details element's toggle, which comes later.
  var AT_ONCE = ["click", "check", "type"];

  // What a user does, as lem/test/*_dom.txt writes it, to a widget, by its
  // id as lem writes it: "click", on its element; "check", its checkbox;
  // "type", text into its field; "compose", text with an input method, and
  // "composed", the composition's end; "open" and "close", its entry;
  // "focus", on its control, if it has one, or its element; and, as for
  // glem's tests, "blur", and "key" with its name and modifiers. A disabled
  // control takes no click, change or text. In a browser, a click and a
  // check are the element's own click(), which does nothing on a disabled
  // control.
  function act(it, step) {
    if (step[0] === "end") {
      return;
    }
    if (step[0] === "blur") {
      it.doc.focus(null);
      return;
    }
    if (step[0] === "key") {
      var init = { key: step[1] };
      Object.keys(step[2] || {}).forEach(function (name) {
        init[name] = step[2][name];
      });
      var at = it.doc.activeElement !== null && it.root.contains(it.doc.activeElement);
      it.doc.fire(at ? it.doc.activeElement : it.root, "keydown", init);
      return;
    }
    var byId = elements(it.view);
    if (!byId.has(step[1])) {
      throw new Error("no widget " + step[1] + " is shown");
    }
    var e = byId.get(step[1]);
    var input = find(e, ["input"]);
    var off = input !== null && input.getAttribute("disabled") !== null;
    switch (step[0]) {
      case "click":
        if (it.real) {
          e.click();
        } else {
          it.doc.fire(e, "click");
        }
        return;
      case "check":
        if (it.real) {
          input.click();
        } else if (!off) {
          input.checked = !input.checked;
          it.doc.fire(input, "input");
          it.doc.fire(input, "change");
        }
        return;
      case "type":
        if (off) {
          return;
        }
        input.value = step[2];
        if (it.real) {
          input.dispatchEvent(new Event("input", { bubbles: true }));
        } else {
          it.doc.fire(input, "input", { isComposing: false });
        }
        return;
      case "compose":
        if (!it.composing) {
          it.composing = true;
          it.doc.fire(input, "compositionstart");
        }
        input.value = step[2];
        it.doc.fire(input, "input", { isComposing: true });
        return;
      case "composed":
        it.composing = false;
        it.doc.fire(input, "compositionend");
        return;
      case "open":
      case "close":
        var details = find(e, ["details"]);
        details.open = step[0] === "open";
        it.doc.fire(details, "toggle");
        return;
      case "focus":
        it.doc.focus(find(e, ["button", "input", "a", "summary"]) || e);
        return;
    }
    throw new Error("there's no step " + JSON.stringify(step));
  }

  // Whether the widgets that stayed, since `before`, kept their elements:
  // null, or why not.
  function kept(it, before) {
    var why = null;
    elements(it.view).forEach(function (element, id) {
      if (why === null && before.has(id) && before.get(id) !== element) {
        why = id + " has a new element";
      }
    });
    return why;
  }

  // A program's cases, in a stand-in DOM, each on what the cases before it
  // left.
  function suite(s) {
    var doc = new standInDom.Document();
    var it = fixture(s.program, doc, doc.createElement("div"));
    s.cases.forEach(function (c) {
      var before = elements(it.view);
      var got;
      it.heard = [];
      try {
        run(it, c.steps);
        got = shown(it);
      } catch (e) {
        got = tap.error(e);
      }
      var why = got === c.want ? null : "the case gives\n" + got + "\nnot\n" + c.want;
      tap.report(s.name + ": " + c.title, why || kept(it, before));
    });
  }

  // Whether a browser's DOM can take a case's steps at once (AT_ONCE).
  function atOnce(step) {
    return host(step) || AT_ONCE.indexOf(step[0]) >= 0;
  }

  // In a browser, a program's first cases, up to the first that it can't
  // take at once, in the page's own DOM, hidden, with its title for the
  // page's while they run: they must give what they give in the stand-in,
  // and keep the elements of the widgets that stay.
  function inThePage(s) {
    var title = document.title;
    var root = document.createElement("div");
    root.setAttribute("hidden", "");
    document.body.appendChild(root);
    document.title = "";
    var it = fixture(s.program, document, root);
    for (var i = 0; i < s.cases.length && s.cases[i].steps.every(atOnce); i++) {
      var c = s.cases[i];
      var before = elements(it.view);
      var got;
      it.heard = [];
      try {
        run(it, c.steps);
        got = shown(it);
      } catch (e) {
        got = tap.error(e);
      }
      var why = got === c.want ? null : "the page's DOM gives\n" + got + "\nnot\n" + c.want;
      tap.report("in the page's DOM: " + s.name + ": " + c.title, why || kept(it, before));
    }
    document.body.removeChild(root);
    document.title = title;
  }

  // Each theme's values, as the looks' stylesheets read them.
  function themes() {
    tests.themes.forEach(function (t) {
      var got;
      try {
        var runtime = new glem.Runtime(new glem.Program(t.program));
        runtime.tick(null);
        got = glui
          .theme(runtime.lem)
          .map(function (value) {
            return value[0] + ": " + value[1];
          })
          .join("\n");
      } catch (e) {
        got = tap.error(e);
      }
      var why = got === t.want ? null : "the theme gives\n" + got + "\nnot\n" + t.want;
      tap.report("themes_css.txt: " + t.name, why);
    });
  }

  // Where a link may go: only to one of the host's own paths.
  function paths() {
    [
      ["/c/ui-0", true],
      ["/", true],
      ["/web/demo.html?program=page#top", true],
      ["//example.com/", false],
      ["/\\example.com", false],
      ["https://example.com/", false],
      ["javascript:alert(1)", false],
      ["c/ui-0", false],
      ["", false],
      ["/a b", false],
      ["/a\nb", false]
    ].forEach(function (c) {
      var title = "a link to " + JSON.stringify(c[0]) + (c[1] ? " goes there" : " is refused");
      tap.report(title, glui.own(c[0]) === c[1] ? null : "glui.own gives " + !c[1]);
    });
  }

  tests.suites.forEach(suite);
  if (typeof document !== "undefined" && document.body) {
    tests.suites.forEach(inThePage);
  }
  themes();
  paths();
  tap.finish();
})();
