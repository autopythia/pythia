// glem.js: glem's runtime in JavaScript, on lem.js (src/lem/lem.js). It
// runs a glem program's compiled form (src/glem/compile.rs) as
// src/glem/mod.rs runs a program: lem's tick, then the widget tree's
// checks, then dropping the state whose widget is gone, with the same
// outline and errors, which the shared tests check (tools/glem_test.js,
// `make js-test`). Where Rust builds the tree anew each tick, glem.js keeps
// it, and changes it where a widget's relation changed. A view (View)
// keeps a program's widgets in the DOM, each made by a recipe from the
// layer above, and sends messages from the recipes' events and from keys;
// it shows what changed in the tree since it last showed it. It's a plain
// script that defines one global, `glem`, after lem.js's `lem`; it writes
// no HTML, and runs no code that it's given.

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

  // A glem program's runtime: lem's (lem.Runtime), and the widget tree, as
  // the last tick left it. The tree stays from tick to tick, and a tick
  // changes it where a kind's relation has a new version, by the facts that
  // it gained and lost (grow). It's built anew from tree() the first time,
  // after a tick whose tree failed, and when a change might break its rules,
  // so that tree() gives the error, as Rust's does. Its nodes, each {kind,
  // rel, tuple, id, key, fact, parent, order}, are by their ids, as
  // lem.key gives them (byId); under each parent's key, or "" for the
  // surfaces, in order (children); and, for each kind, by their facts'
  // keys, with the version of the kind's table that they are (byFact,
  // versions). `touched` is what changed since the view last showed the
  // tree (untouch): the widgets, and the parents whose children changed, by
  // key; or all of it. A runtime has one view at most.
  function Runtime(prog) {
    this.prog = prog;
    this.lem = new lem.Runtime(prog.lem);
    this.byId = null;
    this.children = null;
    this.byFact = null;
    this.versions = null;
    this.touched = { all: true, ids: new Set(), parents: new Set() };
  }

  // One tick: lem's, then the tree's checks, then dropping the state whose
  // widget is gone. Gives the effects that the `on` blocks sent, each as
  // {rel, tuple}.
  Runtime.prototype.tick = function (msg) {
    var effects = this.lem.tick(msg);
    this.grow();
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

  // The order of a node among its siblings: a widget by its `order`, then
  // its id; a surface by its id, as tree() sorts them.
  function before(a, b) {
    return (a.order === null ? 0 : lem.compare(a.order, b.order)) || lem.compare(a.id, b.id);
  }

  // `list`, in order, without the nodes that went, and with `more`.
  function merged(list, more) {
    more.sort(before);
    var out = [];
    var i = 0;
    var j = 0;
    while (i < list.length || j < more.length) {
      if (i < list.length && list[i].gone) {
        i++;
      } else if (j >= more.length || (i < list.length && before(list[i], more[j]) <= 0)) {
        out.push(list[i++]);
      } else {
        out.push(more[j++]);
      }
    }
    return out;
  }

  // A node for a fact of kind `k`, whose key in its table is `fact`.
  Runtime.prototype.node = function (k, tuple, fact) {
    var kind = this.prog.kinds[k];
    var n = tuple.length;
    return {
      kind: k,
      rel: kind.rel,
      tuple: tuple,
      id: tuple[0],
      key: lem.key(tuple[0]),
      fact: fact,
      parent: kind.widget ? lem.key(tuple[n - 2]) : "",
      order: kind.widget ? tuple[n - 1] : null
    };
  };

  // Builds the tree anew, from tree(), which gives its errors; meanwhile,
  // and after an error, there's none.
  Runtime.prototype.build = function () {
    this.byId = null;
    this.touched.all = true;
    var built = this.tree();
    var kindOf = new Map();
    this.prog.kinds.forEach(function (kind, k) {
      kindOf.set(kind.rel, k);
    });
    var byId = new Map();
    var children = new Map();
    var byFact = this.prog.kinds.map(function () {
      return new Map();
    });
    for (var i = 0; i < built.length; i++) {
      var k = kindOf.get(built[i].rel);
      var node = this.node(k, built[i].tuple, lem.key(built[i].tuple));
      byId.set(node.key, node);
      byFact[k].set(node.fact, node);
      var list = children.get(node.parent);
      if (list === undefined) {
        list = [];
        children.set(node.parent, list);
      }
      list.push(node);
    }
    var tables = this.lem.tables;
    this.children = children;
    this.byFact = byFact;
    this.versions = this.prog.kinds.map(function (kind) {
      return tables[kind.rel].version;
    });
    this.byId = byId;
  };

  // Brings the tree up to lem's facts: for each kind whose table has a new
  // version, the nodes of the facts that it lost go, and those of the facts
  // that it gained come; or anew (build).
  Runtime.prototype.grow = function () {
    if (this.byId === null) {
      this.build();
      return;
    }
    var kinds = this.prog.kinds;
    var tables = this.lem.tables;
    var gone = [];
    var come = [];
    for (var k = 0; k < kinds.length; k++) {
      var table = tables[kinds[k].rel];
      if (table.version === this.versions[k]) {
        continue;
      }
      var had = this.byFact[k];
      table.facts.forEach(function (entry, fact) {
        if (!had.has(fact)) {
          come.push({ kind: k, tuple: entry.tuple, fact: fact });
        }
      });
      had.forEach(function (node, fact) {
        if (!table.facts.has(fact)) {
          gone.push(node);
        }
      });
    }
    if (!this.change(gone, come)) {
      this.build();
      return;
    }
    this.versions = kinds.map(function (kind) {
      return tables[kind.rel].version;
    });
  };

  // Takes the nodes `gone` from the tree, and adds one for each fact that
  // `come`s, keeping each parent's children in order: whether the tree
  // still keeps its rules, as far as the change can tell; if not, it's
  // half changed. An id names one node; a widget that went leaves no
  // children; and each new widget is under a surface, not in a cycle.
  Runtime.prototype.change = function (gone, come) {
    var touched = this.touched;
    var more = new Map();
    var i, node;
    for (i = 0; i < gone.length; i++) {
      node = gone[i];
      node.gone = true;
      this.byId.delete(node.key);
      this.byFact[node.kind].delete(node.fact);
      if (!more.has(node.parent)) {
        more.set(node.parent, []);
      }
      touched.ids.add(node.key);
    }
    var added = [];
    for (i = 0; i < come.length; i++) {
      node = this.node(come[i].kind, come[i].tuple, come[i].fact);
      if (this.byId.has(node.key)) {
        return false;
      }
      this.byId.set(node.key, node);
      this.byFact[node.kind].set(node.fact, node);
      if (!more.has(node.parent)) {
        more.set(node.parent, []);
      }
      more.get(node.parent).push(node);
      touched.ids.add(node.key);
      added.push(node);
    }
    var children = this.children;
    more.forEach(function (list, parent) {
      var now = merged(children.get(parent) || [], list);
      if (now.length > 0) {
        children.set(parent, now);
      } else {
        children.delete(parent);
      }
      touched.parents.add(parent);
    });
    for (i = 0; i < gone.length; i++) {
      if (!this.byId.has(gone[i].key) && children.has(gone[i].key)) {
        return false;
      }
    }
    for (i = 0; i < added.length; i++) {
      if (!this.rooted(added[i])) {
        return false;
      }
    }
    return true;
  };

  // Whether a node's parents lead to a surface: not to a node that isn't
  // in the tree, nor back to it.
  Runtime.prototype.rooted = function (node) {
    var key = node.parent;
    for (var steps = 0; key !== "" && steps <= this.byId.size; steps++) {
      var up = this.byId.get(key);
      if (key === node.key || up === undefined) {
        return false;
      }
      key = up.parent;
    }
    return key === "";
  };

  // The tree's nodes in its order, each as {depth, rel, tuple}, as tree()
  // gives them.
  Runtime.prototype.nodes = function () {
    var children = this.children;
    var out = [];
    (function walk(list, depth) {
      for (var i = 0; i < list.length; i++) {
        out.push({ depth: depth, rel: list[i].rel, tuple: list[i].tuple });
        var below = children.get(list[i].key);
        if (below !== undefined) {
          walk(below, depth + 1);
        }
      }
    })(children.get("") || [], 0);
    return out;
  };

  // The view has shown what changed: there's nothing new to show.
  Runtime.prototype.untouch = function () {
    this.touched = { all: false, ids: new Set(), parents: new Set() };
  };

  // The widget tree, as the last tick left it, as lines of text: a widget a
  // line, indented under its parent, with its own fields by name, then its
  // notes, if any, which notes(lemRuntime, id), from the layer above, gives
  // as a list.
  Runtime.prototype.outline = function (notes) {
    var prog = this.prog;
    var runtime = this.lem;
    return this.nodes().map(function (node) {
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
  // widget's or a surface's, which the tree has by id.
  Runtime.prototype.dropOrphans = function () {
    var runtime = this.lem;
    var keep = this.prog.keep;
    var doomed = [];
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
      if (cols.length > 0 && runtime.tables[r].facts.size > 0) {
        doomed.push({ rel: r, cols: cols });
      }
    });
    if (doomed.length === 0) {
      return;
    }
    var live = this.byId;
    doomed.forEach(function (d) {
      runtime.dropFacts(d.rel, function (tuple) {
        return d.cols.some(function (i) {
          return !live.has(lem.key(tuple[i]));
        });
      });
    });
  };

  // The view: a glem runtime's widgets in the DOM.

  // Escape, Return and Tab, glem's Chord, by the names that the DOM gives
  // their keys.
  var CHORDS = { Escape: "Escape", Enter: "Return", Tab: "Tab" };

  // A widget in a view: its relation, kind and id; `key`, its id as lem
  // writes it; and its own fields: by name, in `fields`; and in order, in
  // `names`, `types` and `values`. `state` is what the layer above shows of
  // it besides its fields (View), and `made` is what its recipe made.
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
    this.state = null;
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
  // `state`, if given, is from the layer above too: state(lemRuntime),
  // after each tick, gives a function of a widget's id, which gives what
  // the layer above shows of the widget besides its fields, such as why
  // it's disabled: a value that JSON can write, or null. The view keeps it
  // in the widget's `state`, and a widget whose state changes is updated,
  // as one whose fields change is. `state.rels` names the relations whose
  // facts it reads, each naming a widget's id first: the view asks again
  // for the state of the widgets that their changed facts name. Without
  // it, the view asks for every widget's, after every tick.
  //
  // The view fills in glem's Focused, the widget whose element holds the
  // keyboard's focus; and on Escape, Return or Tab, without a modifier, and
  // unless a recipe took the key, it sends the messages that glem's Key
  // facts give for it, a tick each. `host` hears what happens:
  // host.effects(list) after a tick that sends effects, each as {name,
  // fields}, as JSON (src/lem/host.rs); and host.error(message) after one
  // that fails, which leaves the page as it was.
  function View(runtime, root, recipes, host, state) {
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
    this.state = state || null;
    // The relations that the state reads, by number, or null; and their
    // facts as the view last showed them (seen), or null, before it first
    // shows the tree, and when it can't tell what the state reads.
    this.stateRels = null;
    if (this.state !== null && Array.isArray(this.state.rels)) {
      this.stateRels = this.state.rels.map(function (name) {
        var r = prog.lem.rel(name);
        if (r < 0) {
          fail("there's no relation " + name + ", which the view's state reads");
        }
        return r;
      });
    }
    this.seen = null;
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

  // Shows the tree as the last tick left it: all of it, the first time,
  // after the tree was built anew, and when the view can't tell what the
  // state reads (renderAll); else what changed since the view last showed
  // it, Runtime's `touched`, and the widgets whose state's facts changed
  // (renderSome).
  View.prototype.render = function () {
    if (this.runtime.touched.all || this.seen === null) {
      this.renderAll();
    } else {
      this.renderSome(this.runtime.touched);
    }
    this.runtime.untouch();
  };

  // The facts of the relations that the state reads, as the view sees them
  // now: for each, its table's version, and the key of the widget that each
  // fact names, by the fact's key.
  View.prototype.stateFacts = function () {
    var tables = this.runtime.lem.tables;
    return this.stateRels.map(function (r) {
      var facts = new Map();
      tables[r].facts.forEach(function (entry, fact) {
        facts.set(fact, lem.key(entry.tuple[0]));
      });
      return { version: tables[r].version, facts: facts };
    });
  };

  // The keys of the widgets whose state's facts changed since the view
  // last showed the tree; and the facts as they are now (next).
  View.prototype.stateChanged = function () {
    var tables = this.runtime.lem.tables;
    var keys = new Set();
    var seen = this.seen;
    var next = this.stateRels.map(function (r, i) {
      var was = seen[i];
      var table = tables[r];
      if (table.version === was.version) {
        return was;
      }
      var facts = new Map();
      table.facts.forEach(function (entry, fact) {
        var key = was.facts.has(fact) ? was.facts.get(fact) : lem.key(entry.tuple[0]);
        facts.set(fact, key);
        if (!was.facts.has(fact)) {
          keys.add(key);
        }
      });
      was.facts.forEach(function (key, fact) {
        if (!facts.has(fact)) {
          keys.add(key);
        }
      });
      return { version: table.version, facts: facts };
    });
    return { keys: keys, next: next };
  };

  // Shows what changed: the widgets `touched` and those whose state's facts
  // changed, each made, updated or taken away; and, of each parent whose
  // children changed, which a new widget's parent has, or whose element is
  // new, the children, placed in order. As renderAll does, it changes
  // nothing in the page until each new widget's recipe has made its
  // element.
  View.prototype.renderSome = function (touched) {
    var view = this;
    var runtime = this.runtime;
    var prog = runtime.prog;
    var stateOf = this.state === null ? null : this.state(runtime.lem);
    var states = this.state === null ? null : this.stateChanged();
    var keys = new Set(touched.ids);
    if (states !== null) {
      states.keys.forEach(function (key) {
        keys.add(key);
      });
    }
    var parents = new Set(touched.parents);
    var made = [];
    var changed = [];
    // The widgets whose elements go, each as [its key, the widget]; and the
    // new ones, by key.
    var gone = [];
    var fresh = new Map();
    keys.forEach(function (key) {
      var node = runtime.byId.get(key);
      var w = view.widgets.get(key);
      if (w !== undefined && (node === undefined || w.rel !== node.rel)) {
        gone.push([key, w]);
        w = undefined;
      }
      if (node === undefined) {
        return;
      }
      var values = node.tuple.slice(1, prog.widget.get(node.rel) ? node.tuple.length - 2 : node.tuple.length);
      var state = stateOf === null ? null : stateOf(node.id);
      if (w === undefined) {
        w = new Widget(prog, node.rel, node.id);
        w.set(values);
        w.state = state;
        made.push(w);
        fresh.set(key, w);
        if (runtime.children.has(key)) {
          parents.add(key);
        }
      } else if (lem.compare(w.values, values) !== 0 || JSON.stringify(w.state) !== JSON.stringify(state)) {
        changed.push({ widget: w, values: values, state: state });
      }
    });
    function widget(key) {
      return fresh.has(key) ? fresh.get(key) : view.widgets.get(key);
    }
    for (var i = 0; i < made.length; i++) {
      made[i].made = this.recipes[made[i].kind](made[i], this);
    }
    parents.forEach(function (key) {
      var p = key === "" || !runtime.children.has(key) ? undefined : widget(key);
      if (p !== undefined && !p.made.children) {
        var whose = p.kind + "(" + p.key + ")";
        fail(p.kind + "'s recipe holds no children, but " + whose + " has some");
      }
    });
    // Nothing in the page has changed yet; now it does.
    changed.forEach(function (c) {
      c.widget.set(c.values);
      c.widget.state = c.state;
      update(c.widget);
    });
    made.forEach(update);
    gone.forEach(function (pair) {
      var element = pair[1].made.element;
      if (element.parentNode) {
        element.parentNode.removeChild(element);
      }
      view.elements.delete(element);
      view.widgets.delete(pair[0]);
    });
    fresh.forEach(function (w, key) {
      view.elements.set(w.made.element, w);
      view.widgets.set(key, w);
    });
    parents.forEach(function (key) {
      var p = key === "" ? null : widget(key);
      var container = key === "" ? view.root : p !== undefined ? p.made.children : null;
      if (container) {
        var below = runtime.children.get(key) || [];
        place(
          container,
          below.map(function (node) {
            return view.widgets.get(node.key);
          })
        );
      }
    });
    if (states !== null) {
      this.seen = states.next;
    }
    // A widget that had the focus, and is gone, has it no more.
    if (this.focused !== null && !runtime.byId.has(lem.key(this.focused))) {
      this.focus(null);
    }
  };

  // Shows the whole tree, as the last tick left it: makes the new widgets'
  // elements, updates the changed ones, removes those that are gone, and
  // puts each in its place in its parent's, in order, keeping the elements
  // of the widgets that stay. It makes the new elements first, and changes
  // nothing in the page if a widget's parent has no place for it.
  View.prototype.renderAll = function () {
    var prog = this.runtime.prog;
    var nodes = this.runtime.nodes();
    var next = new Map();
    var made = [];
    var changed = [];
    // Each widget's children, and the root's surfaces, by lem.key of the
    // widget's id, or "".
    var under = new Map();
    var path = [];
    var stateOf = this.state === null ? null : this.state(this.runtime.lem);
    var i, w;
    for (i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      var name = lem.key(node.tuple[0]);
      var values = node.tuple.slice(1, prog.widget.get(node.rel) ? node.tuple.length - 2 : node.tuple.length);
      var state = stateOf === null ? null : stateOf(node.tuple[0]);
      w = this.widgets.get(name);
      if (w !== undefined && w.rel !== node.rel) {
        w = undefined;
      }
      if (w === undefined) {
        w = new Widget(prog, node.rel, node.tuple[0]);
        w.set(values);
        w.state = state;
        made.push(w);
      } else if (lem.compare(w.values, values) !== 0 || JSON.stringify(w.state) !== JSON.stringify(state)) {
        changed.push({ widget: w, values: values, state: state });
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
      c.widget.state = c.state;
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
    // What the state reads, as shown: nothing, without a state; or null,
    // when the view can't tell what it reads, so that it shows all again.
    if (this.state === null) {
      this.seen = [];
    } else {
      this.seen = this.stateRels === null ? null : this.stateFacts();
    }
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
