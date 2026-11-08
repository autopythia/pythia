// glem.js: glem's runtime in JavaScript, on lem.js (src/lem/lem.js). It
// runs a glem program's compiled form (src/glem/compile.rs) as
// src/glem/mod.rs runs a program: lem's tick, then the widget tree's
// checks, then dropping the state whose widget is gone, with the same
// outline and errors, which the shared tests check (tools/glem_test.js,
// `make js-test`). A view (View) keeps a program's widgets in the DOM, each
// made by a recipe from the layer above, and sends messages from the
// recipes' events and from keys. It's a plain script that defines one
// global, `glem`, after lem.js's `lem`; it writes no HTML, and runs no code
// that it's given.

var glem = (function () {
  "use strict";

  function fail(message) {
    throw new lem.Error(message);
  }

  function indent(depth) {
    var out = "";
    for (var i = 0; i < depth; i++) {
      out += "  ";
    }
    return out;
  }

  // A glem program: lem's (lem.Program), and glem's member of its compiled
  // form, as src/glem/compile.rs reads it: its widget kinds and surfaces,
  // as {rel, widget}, by relation; and its `keep` relations.
  function Program(form) {
    this.lem = new lem.Program(form);
    var member = form.glem;
    if (member === null || typeof member !== "object") {
      fail("this isn't the compiled form of a glem program: it has no `glem`");
    }
    var rels = this.lem.rels;
    var kinds = items(member, "kinds");
    this.kinds = [];
    this.widget = new Map();
    for (var k = 0; k < kinds.length; k++) {
      var kind = kinds[k];
      if (!Array.isArray(kind) || kind.length !== 2 || typeof kind[1] !== "string") {
        fail("a kind is [relation, \"widget\" or \"surface\"]");
      }
      var r = relation(kind[0], rels.length);
      var fields = rels[r].fields;
      var n = fields.length;
      var widget = kind[1] === "widget";
      var fits = widget
        ? n >= 3 && fields[0][1] === "Id" && fields[n - 2][1] === "Id" && fields[n - 1][1] === "Any"
        : kind[1] === "surface" && n >= 1 && fields[0][1] === "Id";
      if (!fits || rels[r].kind !== "rel" || this.widget.has(r)) {
        fail(rels[r].name + " isn't a " + kind[1] + "'s relation");
      }
      this.kinds.push({ rel: r, widget: widget });
      this.widget.set(r, widget);
    }
    this.kinds.sort(function (a, b) {
      return a.rel - b.rel;
    });
    this.keep = items(member, "keep").map(function (j) {
      var r = relation(j, rels.length);
      if (rels[r].kind !== "state") {
        fail(rels[r].name + " is `keep`, but isn't a `state` relation");
      }
      return r;
    });
  }

  // The array that's the member `name` of glem's member.
  function items(member, name) {
    if (!Array.isArray(member[name])) {
      fail("glem's `" + name + "` is an array");
    }
    return member[name];
  }

  // A relation, by its number below `len`.
  function relation(j, len) {
    if (typeof j !== "number" || Math.floor(j) !== j || j < 0 || j >= len) {
      fail("a relation is a number below " + len);
    }
    return j;
  }

  // A glem program's runtime: lem's (lem.Runtime), and the widget tree.
  function Runtime(prog) {
    this.prog = prog;
    this.lem = new lem.Runtime(prog.lem);
  }

  // One tick: lem's, then the tree's checks, then dropping the state whose
  // widget is gone. Gives the effects that the `on` blocks sent, each as
  // {rel, tuple}.
  Runtime.prototype.tick = function (msg) {
    var effects = this.lem.tick(msg);
    this.tree();
    this.dropOrphans();
    return effects;
  };

  // Runs a tick given as JSON steps (src/lem/host.rs), and gives the
  // outline, or "(no widgets)", then the facts asked for and the effects
  // sent, as src/glem/mod.rs's Runtime::run does. `notes`, from the layer
  // above, gives a widget's notes (outline), or is null.
  Runtime.prototype.run = function (steps, notes) {
    var script = this.lem.readSteps(steps);
    var effects = this.tick(script.msg);
    var lines = this.outline(notes);
    var out = lines.length > 0 ? lines.join("\n") : "(no widgets)";
    var shown = this.lem.show(script, effects);
    return shown === "" ? out : out + "\n" + shown;
  };

  // The widget tree: each surface, and each widget under its parent, by
  // `order` and then id, each as {depth, rel, tuple, why}. An id names one
  // widget, every parent is a widget, and every widget is under a surface;
  // the errors name the first that isn't so, in Rust's order.
  Runtime.prototype.tree = function () {
    var prog = this.prog.lem;
    var facts = this.lem;
    var nodes = new Map();
    var kids = new Map();
    var roots = [];
    var i, k;
    for (k = 0; k < this.prog.kinds.length; k++) {
      var kind = this.prog.kinds[k];
      var all = facts.relFacts(kind.rel);
      for (i = 0; i < all.length; i++) {
        var tuple = all[i].tuple;
        var id = tuple[0];
        var first = nodes.get(lem.key(id));
        if (first !== undefined) {
          var a = prog.rels[first.rel].name + " (" + facts.why(first.why) + ")";
          var b = prog.rels[kind.rel].name + " (" + facts.why(all[i].why) + ")";
          fail("the id " + prog.text(id, "Id") + " names two widgets: " + a + " and " + b);
        }
        nodes.set(lem.key(id), { rel: kind.rel, tuple: tuple, why: all[i].why });
        if (kind.widget) {
          // A widget's last fields are its `parent` and `order`.
          var parent = tuple[tuple.length - 2];
          var under = kids.get(lem.key(parent));
          if (under === undefined) {
            under = { parent: parent, kids: [] };
            kids.set(lem.key(parent), under);
          }
          under.kids.push({ order: tuple[tuple.length - 1], id: id });
        } else {
          roots.push(id);
        }
      }
    }
    var parents = [];
    kids.forEach(function (under) {
      parents.push(under);
    });
    parents.sort(function (a, b) {
      return lem.compare(a.parent, b.parent);
    });
    for (i = 0; i < parents.length; i++) {
      if (!nodes.has(lem.key(parents[i].parent))) {
        var child = parents[i].kids[0].id;
        var why = facts.why(nodes.get(lem.key(child)).why);
        var whose = prog.text(child, "Id") + "'s parent, " + prog.text(parents[i].parent, "Id");
        fail(whose + ", isn't a widget (" + why + ")");
      }
    }
    roots.sort(lem.compare);
    var out = [];
    var stack = [];
    for (i = roots.length - 1; i >= 0; i--) {
      stack.push({ id: roots[i], depth: 0 });
    }
    while (stack.length > 0) {
      var top = stack.pop();
      var node = nodes.get(lem.key(top.id));
      out.push({ depth: top.depth, rel: node.rel, tuple: node.tuple, why: node.why });
      var below = kids.get(lem.key(top.id));
      if (below !== undefined) {
        var sorted = below.kids.slice().sort(function (a, b) {
          return lem.compare(a.order, b.order) || lem.compare(a.id, b.id);
        });
        for (i = sorted.length - 1; i >= 0; i--) {
          stack.push({ id: sorted[i].id, depth: top.depth + 1 });
        }
      }
    }
    if (out.length < nodes.size) {
      var placed = new Set();
      for (i = 0; i < out.length; i++) {
        placed.add(lem.key(out[i].tuple[0]));
      }
      var lost = [];
      nodes.forEach(function (node, name) {
        if (!placed.has(name)) {
          lost.push(node.tuple[0]);
        }
      });
      lost.sort(lem.compare);
      var names = lost.map(function (id) {
        return prog.text(id, "Id");
      });
      fail("these widgets' parents form a cycle: " + names.join(", "));
    }
    return out;
  };

  // The widget tree as lines of text: a widget a line, indented under its
  // parent, with its own fields by name, then its notes, if any, which
  // notes(lemRuntime, id), from the layer above, gives as a list.
  Runtime.prototype.outline = function (notes) {
    var prog = this.prog;
    var runtime = this.lem;
    return this.tree().map(function (node) {
      var rel = prog.lem.rels[node.rel];
      var fields = rel.fields;
      var end = prog.widget.get(node.rel) ? fields.length - 2 : fields.length;
      var line = indent(node.depth) + rel.name + "(" + prog.lem.text(node.tuple[0], fields[0][1]);
      for (var i = 1; i < end; i++) {
        line += ", " + fields[i][0] + "=" + prog.lem.text(node.tuple[i], fields[i][1]);
      }
      line += ")";
      var more = notes ? notes(runtime, node.tuple[0]) : [];
      return more.length > 0 ? line + " -- " + more.join("; ") : line;
    });
  };

  // Drops the `state` facts that name a widget that's gone, unless their
  // relation says `keep`: those with a value of type Id that isn't a
  // widget's or a surface's.
  Runtime.prototype.dropOrphans = function () {
    var runtime = this.lem;
    var live = new Set();
    this.prog.kinds.forEach(function (kind) {
      runtime.relFacts(kind.rel).forEach(function (fact) {
        live.add(lem.key(fact.tuple[0]));
      });
    });
    var keep = this.prog.keep;
    this.prog.lem.rels.forEach(function (rel, r) {
      if (rel.kind !== "state" || keep.indexOf(r) >= 0) {
        return;
      }
      var cols = [];
      rel.fields.forEach(function (field, i) {
        if (field[1] === "Id") {
          cols.push(i);
        }
      });
      if (cols.length > 0) {
        runtime.dropFacts(r, function (tuple) {
          return cols.some(function (i) {
            return !live.has(lem.key(tuple[i]));
          });
        });
      }
    });
  };

  // The view: a glem runtime's widgets in the DOM.

  // Escape, Return and Tab, glem's Chord, by the names that the DOM gives
  // their keys.
  var CHORDS = { Escape: "Escape", Enter: "Return", Tab: "Tab" };

  // A widget in a view: its relation, kind and id; `key`, its id as lem
  // writes it; and its own fields: by name, in `fields`; and in order, in
  // `names`, `types` and `values`. `made` is what its recipe made.
  function Widget(prog, rel, id) {
    var fields = prog.lem.rels[rel].fields;
    var end = prog.widget.get(rel) ? fields.length - 2 : fields.length;
    this.prog = prog.lem;
    this.rel = rel;
    this.kind = prog.lem.rels[rel].name;
    this.id = id;
    this.key = prog.lem.text(id, "Id");
    this.names = [];
    this.types = [];
    for (var i = 1; i < end; i++) {
      this.names.push(fields[i][0]);
      this.types.push(fields[i][1]);
    }
    this.values = [];
    this.fields = {};
    this.made = null;
  }

  // Sets the widget's own fields, in order.
  Widget.prototype.set = function (values) {
    this.values = values;
    this.fields = {};
    for (var i = 0; i < values.length; i++) {
      this.fields[this.names[i]] = values[i];
    }
  };

  // A field's value as lem writes it.
  Widget.prototype.text = function (name) {
    var i = this.names.indexOf(name);
    if (i < 0) {
      fail(this.kind + " has no field `" + name + "`");
    }
    return this.prog.text(this.values[i], this.types[i]);
  };

  // A view: a glem runtime's widgets in the DOM, in `root`, which holds the
  // surfaces' elements and nothing else. Each widget's element comes from
  // its kind's recipe, from the layer above, and stays from tick to tick
  // while its id names a widget of that kind, so that it keeps its focus,
  // its caret and its scroll: only new widgets get elements, only changed
  // ones are updated, and elements move only to take their places.
  //
  // A recipe is a function (w, view), for a Widget, that gives {element,
  // children, update}: the widget's element; the element that holds its
  // children, and nothing else, or null for a kind that has none; and
  // update(), if any, which the view calls when the widget is new, and
  // each time its fields change, after setting them. A recipe sends a
  // message with view.send(msg), reading the widget's fields when its event
  // comes, not when it's made.
  //
  // The view fills in glem's Focused, the widget whose element holds the
  // keyboard's focus; and on Escape, Return or Tab, without a modifier, and
  // unless a recipe took the key, it sends the messages that glem's Key
  // facts give for it, a tick each. `host` hears what happens:
  // host.effects(list) after a tick that sends effects, each as {name,
  // fields}, as JSON (src/lem/host.rs); and host.error(message) after one
  // that fails, which leaves the page as it was.
  function View(runtime, root, recipes, host) {
    var prog = runtime.prog;
    var view = this;
    prog.kinds.forEach(function (kind) {
      var name = prog.lem.rels[kind.rel].name;
      if (typeof recipes[name] !== "function") {
        fail("there's no recipe for " + name);
      }
    });
    Object.keys(recipes).forEach(function (name) {
      if (!prog.widget.has(prog.lem.rel(name))) {
        fail("there's a recipe for " + name + ", which isn't a widget or a surface");
      }
    });
    this.runtime = runtime;
    this.root = root;
    this.doc = root.ownerDocument;
    this.recipes = recipes;
    this.host = host || {};
    // The widgets, by their ids, as lem.key gives them; and by element.
    this.widgets = new Map();
    this.elements = new Map();
    // What waits for a tick: functions that change facts, and give the
    // tick's message, or null. One runs at a time (drain).
    this.queue = [];
    this.busy = false;
    // The id that Focused names, and the one that it will.
    this.focused = null;
    this.focusing = null;
    root.addEventListener("focusin", function (e) {
      view.focus(view.widgetOf(e.target));
    });
    root.addEventListener("focusout", function (e) {
      if (!e.relatedTarget || !root.contains(e.relatedTarget)) {
        view.focus(null);
      }
    });
    root.addEventListener("keydown", function (e) {
      view.key(e);
    });
  }

  // Changes the host's facts, by JSON steps (src/lem/host.rs), and runs a
  // tick: at once, or after the ticks before it, if one is running.
  View.prototype.update = function (steps) {
    var runtime = this.runtime.lem;
    this.queue.push(function () {
      return runtime.readSteps(steps).msg;
    });
    this.drain();
  };

  // Sends a message, a value as lem.js holds it, in a tick of its own.
  View.prototype.send = function (msg) {
    this.update([["!", msg]]);
  };

  // Runs what waits, a tick each, unless a tick is running already: what
  // happens during one, such as a host that answers an effect at once,
  // waits for it.
  View.prototype.drain = function () {
    if (this.busy) {
      return;
    }
    this.busy = true;
    try {
      while (this.queue.length > 0) {
        this.run(this.queue.shift());
      }
    } finally {
      this.busy = false;
    }
  };

  // Changes facts as `change` does, runs a tick, tells the host its
  // effects, and shows it; or tells the host why it failed.
  View.prototype.run = function (change) {
    var prog = this.runtime.prog.lem;
    var effects;
    try {
      effects = this.runtime.tick(change());
    } catch (e) {
      this.failed(e);
      return;
    }
    if (effects.length > 0) {
      this.tell(
        "effects",
        effects.map(function (effect) {
          var fields = prog.factJson(effect.rel, effect.tuple);
          return { name: prog.rels[effect.rel].name, fields: fields };
        })
      );
    }
    try {
      this.render();
    } catch (e) {
      this.failed(e);
    }
  };

  View.prototype.failed = function (e) {
    if (!(e instanceof lem.Error)) {
      throw e;
    }
    this.tell("error", e.message);
  };

  // Tells the host: host.effects or host.error. Without host.error, an
  // error is thrown, to whatever runs the view.
  View.prototype.tell = function (what, value) {
    if (typeof this.host[what] === "function") {
      this.host[what](value);
    } else if (what === "error") {
      throw new lem.Error(value);
    }
  };

  // The widget whose element holds a node: the innermost, or null.
  View.prototype.widgetOf = function (node) {
    for (; node && node !== this.root; node = node.parentNode) {
      var w = this.elements.get(node);
      if (w !== undefined) {
        return w;
      }
    }
    return null;
  };

  // The keyboard's focus moves to a widget, or away (null): Focused names
  // it, from a tick.
  View.prototype.focus = function (w) {
    var id = w === null ? null : w.id;
    if (lem.key(id) === lem.key(this.focusing)) {
      return;
    }
    this.focusing = id;
    var view = this;
    var runtime = this.runtime.lem;
    var r = this.runtime.prog.lem.rel("Focused");
    this.queue.push(function () {
      if (view.focused !== null) {
        runtime.retractTuple(r, [view.focused]);
      }
      if (id !== null) {
        runtime.assertTuple(r, [id]);
      }
      view.focused = id;
      return null;
    });
    this.drain();
  };

  // A key that's pressed: what glem's Key facts give for it, as the last
  // tick left them.
  View.prototype.key = function (e) {
    var chord = Object.prototype.hasOwnProperty.call(CHORDS, e.key) ? CHORDS[e.key] : null;
    var plain = !(e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.isComposing);
    if (chord === null || !plain || e.defaultPrevented) {
      return;
    }
    var r = this.runtime.prog.lem.rel("Key");
    var msgs = [];
    this.runtime.lem.relFacts(r).forEach(function (fact) {
      if (fact.tuple[0][0] === chord) {
        msgs.push(fact.tuple[1]);
      }
    });
    if (msgs.length > 0) {
      e.preventDefault();
      for (var i = 0; i < msgs.length; i++) {
        this.send(msgs[i]);
      }
    }
  };

  // Shows the runtime's widget tree: makes the new widgets' elements,
  // updates the changed ones, removes those that are gone, and puts each in
  // its place in its parent's, in order. It makes the new elements first,
  // and changes nothing in the page if a widget's parent has no place for
  // it.
  View.prototype.render = function () {
    var prog = this.runtime.prog;
    var nodes = this.runtime.tree();
    var next = new Map();
    var made = [];
    var changed = [];
    // Each widget's children, and the root's surfaces, by lem.key of the
    // widget's id, or "".
    var under = new Map();
    var path = [];
    var i, w;
    for (i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      var name = lem.key(node.tuple[0]);
      var values = node.tuple.slice(1, prog.widget.get(node.rel) ? node.tuple.length - 2 : node.tuple.length);
      w = this.widgets.get(name);
      if (w !== undefined && w.rel !== node.rel) {
        w = undefined;
      }
      if (w === undefined) {
        w = new Widget(prog, node.rel, node.tuple[0]);
        w.set(values);
        made.push(w);
      } else if (lem.compare(w.values, values) !== 0) {
        changed.push({ widget: w, values: values });
      }
      next.set(name, w);
      path[node.depth] = name;
      var parent = node.depth === 0 ? "" : path[node.depth - 1];
      if (!under.has(parent)) {
        under.set(parent, []);
      }
      under.get(parent).push(w);
    }
    for (i = 0; i < made.length; i++) {
      made[i].made = this.recipes[made[i].kind](made[i], this);
    }
    under.forEach(function (kids, parent) {
      var p = next.get(parent);
      if (parent !== "" && !p.made.children) {
        var whose = p.kind + "(" + p.key + ")";
        fail(p.kind + "'s recipe holds no children, but " + whose + " has some");
      }
    });
    // Nothing in the page has changed yet; now it does.
    changed.forEach(function (c) {
      c.widget.set(c.values);
      update(c.widget);
    });
    made.forEach(update);
    var view = this;
    this.widgets.forEach(function (old, name) {
      if (next.get(name) !== old) {
        var element = old.made.element;
        if (element.parentNode) {
          element.parentNode.removeChild(element);
        }
        view.elements.delete(element);
      }
    });
    made.forEach(function (w) {
      view.elements.set(w.made.element, w);
    });
    this.widgets = next;
    place(this.root, under.get("") || []);
    next.forEach(function (w, name) {
      if (w.made.children) {
        place(w.made.children, under.get(name) || []);
      }
    });
    // A widget that had the focus, and is gone, has it no more.
    if (this.focused !== null && !next.has(lem.key(this.focused))) {
      this.focus(null);
    }
  };

  function update(w) {
    if (typeof w.made.update === "function") {
      w.made.update();
    }
  }

  // Puts widgets' elements in a container, in order, moving only those out
  // of place. One that's in another container moves here; and one that's
  // here, but belongs elsewhere, moves there when its own container's turn
  // comes, so that each holds just its own in the end.
  function place(container, widgets) {
    var nodes = container.childNodes;
    for (var i = 0; i < widgets.length; i++) {
      var element = widgets[i].made.element;
      if (nodes[i] !== element) {
        container.insertBefore(element, nodes[i] || null);
      }
    }
  }

  return {
    Program: Program,
    Runtime: Runtime,
    View: View,
    Widget: Widget
  };
})();
