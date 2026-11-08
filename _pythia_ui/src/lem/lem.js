// lem.js: lem's runtime in JavaScript. It runs a program's compiled form
// (src/lem/compile.rs) as src/lem/eval.rs runs a checked program, with the
// same ticks, facts, effects and errors, which the shared tests check
// (tools/lem_test.js, `make js-test`). It's a plain script that defines
// one global, `lem`, so that the same file runs in a browser and in
// macOS's JavaScriptCore. It writes no HTML, and runs no code that it's
// given.
//
// Values carry no types: Bool and the numbers are JavaScript's own, a Text
// is a string, and a constructor an array of its name and arguments, as
// ["Row", "c"]. The compiled form gives each column and each operator its
// type, which says how to compute and print. The host's facts come as
// JSON, by relation, with fields by name (src/lem/host.rs).

var lem = (function () {
  "use strict";

  // The compiled form's version, as src/lem/compile.rs writes it.
  var VERSION = 1;
  // The largest whole number, 2^53 - 1, as in src/lem/value.rs.
  var MAX_INT = 9007199254740991;
  var BASE = ["Bool", "Int", "Num", "Text", "Time", "Any"];
  var HOST = { host: true };
  var NONE = [];

  // An error that lem gives, with Rust's message.
  function LemError(message) {
    this.message = message;
  }
  LemError.prototype.toString = function () {
    return this.message;
  };

  function fail(message) {
    throw new LemError(message);
  }

  function has(object, name) {
    return Object.prototype.hasOwnProperty.call(object, name);
  }

  // Values.

  // A UTF-16 unit, moved so that units compare as their code points do:
  // the surrogates, which make the code points above U+FFFF, after the
  // rest.
  function unit(c) {
    return c >= 0xe000 ? c - 0x800 : c >= 0xd800 ? c + 0x2000 : c;
  }

  // Texts in the order of their code points, as Rust's are.
  function compareText(a, b) {
    var n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) {
      var x = a.charCodeAt(i);
      var y = b.charCodeAt(i);
      if (x !== y) {
        return unit(x) < unit(y) ? -1 : 1;
      }
    }
    return a.length < b.length ? -1 : a.length > b.length ? 1 : 0;
  }

  // Values of different types order by type, Bools, then numbers, Texts
  // and constructors, as Rust's do; numbers are never mixed, since a
  // column of type Any takes no Num.
  function rank(v) {
    var t = typeof v;
    return t === "boolean" ? 0 : t === "number" ? 1 : t === "string" ? 3 : 4;
  }

  function compare(a, b) {
    if (a === b) {
      return 0;
    }
    var ra = rank(a);
    var rb = rank(b);
    if (ra !== rb) {
      return ra < rb ? -1 : 1;
    }
    if (ra === 0) {
      return a ? 1 : -1;
    }
    if (ra === 1) {
      return a < b ? -1 : 1;
    }
    if (ra === 3) {
      return compareText(a, b);
    }
    return compareLists(a, b);
  }

  // Lists of values, as a fact's or a constructor's: by their first
  // values that differ, else the shorter first.
  function compareLists(a, b) {
    var n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) {
      var c = compare(a[i], b[i]);
      if (c !== 0) {
        return c;
      }
    }
    return a.length < b.length ? -1 : a.length > b.length ? 1 : 0;
  }

  function equal(a, b) {
    return a === b || (typeof a === "object" && typeof b === "object" && compareLists(a, b) === 0);
  }

  // Values as a key in a Map: the same values give the same text, since a
  // number is never -0.
  function key(values) {
    return JSON.stringify(values);
  }

  function project(tuple, cols) {
    var values = [];
    for (var i = 0; i < cols.length; i++) {
      values.push(tuple[cols[i]]);
    }
    return values;
  }

  // -0 becomes 0.
  function zero(x) {
    return x + 0;
  }

  // Printing, as src/lem/value.rs and src/lem/print.rs print.

  // A Text, quoted as lem writes it.
  function quote(s) {
    var out = "\"";
    for (var i = 0; i < s.length; i++) {
      var c = s.charCodeAt(i);
      if (c === 0x22) {
        out += "\\\"";
      } else if (c === 0x5c) {
        out += "\\\\";
      } else if (c === 0x0a) {
        out += "\\n";
      } else if (c === 0x0d) {
        out += "\\r";
      } else if (c === 0x09) {
        out += "\\t";
      } else if (c < 0x20 || c === 0x7f) {
        out += "\\u" + ("000" + c.toString(16)).slice(-4);
      } else {
        out += s.charAt(i);
      }
    }
    return out + "\"";
  }

  function zeros(n) {
    var out = "";
    for (var i = 0; i < n; i++) {
      out += "0";
    }
    return out;
  }

  // A positive number's shortest digits, as String gives them, without
  // zeros at either end, and the power of ten of the first.
  function digitsOf(x) {
    var s = String(x);
    var power = 0;
    var e = s.indexOf("e");
    if (e >= 0) {
      power = parseInt(s.slice(e + 1), 10);
      s = s.slice(0, e);
    }
    var point = s.indexOf(".");
    var whole = point < 0 ? s : s.slice(0, point);
    var digits = whole + (point < 0 ? "" : s.slice(point + 1));
    power += whole.length - 1;
    var first = 0;
    while (first < digits.length - 1 && digits.charAt(first) === "0") {
      first++;
      power--;
    }
    var last = digits.length;
    while (last > first + 1 && digits.charAt(last - 1) === "0") {
      last--;
    }
    return { digits: digits.slice(first, last), power: power };
  }

  // A Num as Rust's {:?} writes it: with a point, and with an exponent
  // below 1e-4 and from 1e16, as 2.0, 0.0001, 1e-5 and 1.5e16.
  function numText(x) {
    if (x === 0) {
      return "0.0";
    }
    var sign = x < 0 ? "-" : "";
    var a = Math.abs(x);
    var d = digitsOf(a);
    var digits = d.digits;
    var power = d.power;
    if (a < 1e-4 || a >= 1e16) {
      var rest = digits.length > 1 ? "." + digits.slice(1) : "";
      return sign + digits.charAt(0) + rest + "e" + power;
    }
    if (power < 0) {
      return sign + "0." + zeros(-power - 1) + digits;
    }
    if (digits.length <= power + 1) {
      return sign + digits + zeros(power + 1 - digits.length) + ".0";
    }
    return sign + digits.slice(0, power + 1) + "." + digits.slice(power + 1);
  }

  // A type after "a" or "an", as Rust's errors name it.
  function article(ty) {
    if (ty === "Any") {
      return "any value";
    }
    return ("AEIOU".indexOf(ty.charAt(0)) >= 0 ? "an " : "a ") + ty;
  }

  // What a JSON value is, for errors, as src/json.rs names it.
  function jsonKind(j) {
    if (j === null) {
      return "null";
    }
    switch (typeof j) {
      case "boolean":
        return "a boolean";
      case "number":
        return "a number";
      case "string":
        return "a string";
    }
    return Array.isArray(j) ? "an array" : "an object";
  }

  function whole(j) {
    return typeof j === "number" && Math.floor(j) === j && Math.abs(j) <= MAX_INT;
  }

  // "Job's `due`", or "Ring's field 1"; a field is [name, type] or [name,
  // type, default], and a constructor's name may be null.
  function fieldWhat(owner, field, i) {
    return field[0] !== null ? owner + "'s `" + field[0] + "`" : owner + "'s field " + (i + 1);
  }

  function fieldName(field, i) {
    return field[0] !== null ? "`" + field[0] + "`" : "field " + (i + 1);
  }

  // A program: its compiled form, read, with its parts by name. A body's
  // parts become objects: {tag, rel, pats, key, needs} for an atom and a
  // negation, {tag, v, e} for a binding, {tag, op, a, b} for a test.
  function Program(form) {
    if (form === null || typeof form !== "object" || form.lem !== VERSION) {
      fail("this isn't the compiled form of a lem program, version " + VERSION);
    }
    this.types = form.types;
    this.rels = form.rels;
    this.ctors = form.ctors;
    this.rules = form.rules.map(function (rule) {
      return {
        at: rule.at,
        head: rule.head,
        args: rule.args,
        body: rule.body.map(part),
        vars: rule.vars
      };
    });
    this.components = form.components;
    this.actions = form.actions.map(function (action) {
      return {
        at: action.at,
        msg: action.msg,
        body: action.body.map(part),
        rel: action.rel,
        insert: has(action, "insert") ? action.insert : null,
        remove: has(action, "delete") ? action["delete"] : null,
        vars: action.vars
      };
    });
    this.relByName = new Map();
    this.ctorByName = new Map();
    var self = this;
    this.rels.forEach(function (rel, i) {
      self.relByName.set(rel.name, i);
    });
    this.ctors.forEach(function (ctor) {
      self.ctorByName.set(ctor.name, ctor);
    });
  }

  function part(lit) {
    switch (lit[0]) {
      case "atom":
      case "not":
        return { tag: lit[0], rel: lit[1], pats: lit[2], key: lit[3], needs: lit[4] };
      case "let":
        return { tag: "let", v: lit[1], e: lit[2] };
      case "test":
      case "nottest":
        return { tag: lit[0], op: lit[1], a: lit[2], b: lit[3] };
    }
    return fail("there's no part `" + lit[0] + "` of a body");
  }

  // A relation's number, or -1.
  Program.prototype.rel = function (name) {
    var r = this.relByName.get(name);
    return r === undefined ? -1 : r;
  };

  // A constructor, as {name, type, fields}, or undefined.
  Program.prototype.ctor = function (name) {
    return this.ctorByName.get(name);
  };

  // The type that a type made with `type` is made from, through any chain;
  // any other type itself.
  Program.prototype.base = function (ty) {
    for (var n = 0; n < 64; n++) {
      if (BASE.indexOf(ty) >= 0 || !has(this.types, ty) || typeof this.types[ty] !== "string") {
        return ty;
      }
      ty = this.types[ty];
    }
    return ty;
  };

  // A value of type `ty` as lem writes it.
  Program.prototype.text = function (v, ty) {
    switch (typeof v) {
      case "boolean":
        return v ? "True" : "False";
      case "number":
        return this.base(ty) === "Num" ? numText(v) : String(v);
      case "string":
        return quote(v);
    }
    var ctor = this.ctor(v[0]);
    return this.call(v[0], v.slice(1), ctor.fields);
  };

  // `name(a, b)`, or `name` alone, its values typed by `fields`.
  Program.prototype.call = function (name, values, fields) {
    if (values.length === 0) {
      return name;
    }
    var parts = [];
    for (var i = 0; i < values.length; i++) {
      parts.push(this.text(values[i], fields[i][1]));
    }
    return name + "(" + parts.join(", ") + ")";
  };

  // A relation's fact as lem writes it.
  Program.prototype.factText = function (r, tuple) {
    return this.call(this.rels[r].name, tuple, this.rels[r].fields);
  };

  // A value of type `ty` from the host's JSON; `what` names it, for
  // errors (src/lem/host.rs).
  Program.prototype.value = function (ty, j, what) {
    var base = this.base(ty);
    switch (base) {
      case "Bool":
        if (typeof j === "boolean") {
          return j;
        }
        break;
      case "Int":
      case "Time":
        if (typeof j === "number") {
          if (whole(j)) {
            return zero(j);
          }
          fail(what + " is " + article(ty) + ", a whole number within \u00b1(2^53 \u2212 1)");
        }
        break;
      case "Num":
        if (typeof j === "number") {
          if (isFinite(j)) {
            return zero(j);
          }
          fail(what + " is " + article(ty) + ", a finite number");
        }
        break;
      case "Text":
        if (typeof j === "string") {
          return j;
        }
        break;
      case "Any":
        if (typeof j === "boolean" || typeof j === "string") {
          return j;
        }
        if (typeof j === "number") {
          if (whole(j)) {
            return zero(j);
          }
          fail(what + " takes any value but a Num");
        }
        if (Array.isArray(j)) {
          return this.ctorValue(null, j, what);
        }
        break;
      default:
        if (typeof j === "string") {
          return this.ctorValue(base, [j], what);
        }
        if (Array.isArray(j)) {
          return this.ctorValue(base, j, what);
        }
    }
    return fail(what + " is " + article(ty) + ", not " + jsonKind(j));
  };

  // A constructor from its array: its name, then its fields by position;
  // of the type `type`, unless that's null.
  Program.prototype.ctorValue = function (type, items, what) {
    var name = items.length > 0 ? items[0] : null;
    if (typeof name !== "string") {
      fail(what + " is a constructor: its name, or an array of its name and its fields");
    }
    var ctor = this.ctor(name);
    if (ctor === undefined) {
      fail("there's no constructor `" + name + "`");
    }
    if (type !== null && ctor.type !== type) {
      fail(what + " is " + article(type) + ", but " + name + " makes " + article(ctor.type));
    }
    var fields = ctor.fields;
    var given = items.length - 1;
    if (given > fields.length) {
      var n = fields.length === 1 ? "1 value" : fields.length + " values";
      fail(name + " takes " + n + ", not " + given);
    }
    var value = [ctor.name];
    for (var i = 0; i < fields.length; i++) {
      var field = fields[i];
      if (i < given) {
        value.push(this.value(field[1], items[i + 1], fieldWhat(name, field, i)));
      } else if (field.length > 2) {
        value.push(field[2]);
      } else {
        fail(name + " needs " + fieldName(field, i));
      }
    }
    return value;
  };

  // A fact of the relation named `name`, from its fields by name, as
  // {rel, tuple}; fields with defaults may be left out.
  Program.prototype.fact = function (name, j) {
    var r = this.rel(name);
    if (r < 0) {
      fail("there's no relation `" + name + "`");
    }
    if (j === null || typeof j !== "object" || Array.isArray(j)) {
      fail("a fact of " + name + " is an object of its fields, not " + jsonKind(j));
    }
    var fields = this.rels[r].fields;
    Object.keys(j).forEach(function (member) {
      var known = fields.some(function (field) {
        return field[0] === member;
      });
      if (!known) {
        fail(name + " has no field `" + member + "`");
      }
    });
    var tuple = [];
    for (var i = 0; i < fields.length; i++) {
      var field = fields[i];
      if (has(j, field[0])) {
        tuple.push(this.value(field[1], j[field[0]], fieldWhat(name, field, i)));
      } else if (field.length > 2) {
        tuple.push(field[2]);
      } else {
        fail(name + " needs `" + field[0] + "`");
      }
    }
    return { rel: r, tuple: tuple };
  };

  // The runtime, as src/lem/eval.rs's: the host's facts in, the rules run
  // to a fixed point, the derived facts out; and messages, which `on`
  // blocks turn into state changes and effects. It takes the same steps
  // in the same order as Rust's, down to the order in which a join visits
  // facts, so that the two give the same errors too; the shared tests
  // compare the facts that their joins visit.

  // Why a fact holds: the host asserted it (HOST), an `on` block's
  // sentence, an action, added it, or rules derived it: the
  // lowest-numbered of them.
  function byAction(a) {
    return { action: a };
  }

  function byRule(r) {
    return { rule: r };
  }

  // A body's variables, none of them bound.
  function vars(n) {
    var env = [];
    for (var i = 0; i < n; i++) {
      env.push(undefined);
    }
    return env;
  }

  function unbind(env, trail) {
    for (var i = 0; i < trail.length; i++) {
      env[trail[i]] = undefined;
    }
  }

  // Whether two facts differ in any of the columns `cols`.
  function differs(a, b, cols) {
    for (var i = 0; i < cols.length; i++) {
      if (!equal(a[cols[i]], b[cols[i]])) {
        return true;
      }
    }
    return false;
  }

  function columns(n) {
    var cols = [];
    for (var i = 0; i < n; i++) {
      cols.push(i);
    }
    return cols;
  }

  // An index maps the values of some columns to the facts that have them,
  // in the order that they came: those there when it was built, in order,
  // then those that rules derived since.
  function addTo(map, cols, tuple) {
    var k = key(project(tuple, cols));
    var bucket = map.get(k);
    if (bucket === undefined) {
      map.set(k, [tuple]);
    } else {
      bucket.push(tuple);
    }
  }

  // The index on `cols` among `indexes`, built from `tuples()` if it isn't
  // there.
  function indexOn(indexes, cols, tuples) {
    var name = cols.join(",");
    var index = indexes.get(name);
    if (index === undefined) {
      var map = new Map();
      var all = tuples();
      for (var i = 0; i < all.length; i++) {
        addTo(map, cols, all[i]);
      }
      index = { cols: cols, map: map };
      indexes.set(name, index);
    }
    return index.map;
  }

  // A relation's facts, and why each holds, by their values as text; the
  // facts in order, once asked for; and indexes on them, which only
  // derive() keeps as facts change: every other change drops them.
  function Table() {
    this.facts = new Map();
    this.sorted = null;
    this.indexes = new Map();
  }

  // The facts, in order.
  Table.prototype.tuples = function () {
    if (this.sorted === null) {
      var tuples = [];
      this.facts.forEach(function (entry) {
        tuples.push(entry.tuple);
      });
      this.sorted = tuples.sort(compareLists);
    }
    return this.sorted;
  };

  // The index on `cols`, which mustn't be empty.
  Table.prototype.index = function (cols) {
    var table = this;
    return indexOn(this.indexes, cols, function () {
      return table.tuples();
    });
  };

  // A fact that rule `r` derives, into the indexes too: whether it's new.
  // One that holds already keeps the lowest-numbered of its rules.
  Table.prototype.derive = function (tuple, r) {
    var k = key(tuple);
    var entry = this.facts.get(k);
    if (entry !== undefined) {
      if (has(entry.why, "rule") && r < entry.why.rule) {
        entry.why = byRule(r);
      }
      return false;
    }
    this.facts.set(k, { tuple: tuple, why: byRule(r) });
    this.sorted = null;
    this.indexes.forEach(function (index) {
      addTo(index.map, index.cols, tuple);
    });
    return true;
  };

  // The facts themselves, to change otherwise: so the indexes go.
  Table.prototype.change = function () {
    this.indexes = new Map();
    this.sorted = null;
    return this.facts;
  };

  // A fact, and why it holds, in place of one with its values.
  Table.prototype.put = function (tuple, why) {
    this.change().set(key(tuple), { tuple: tuple, why: why });
  };

  // Removes a fact: whether it held.
  Table.prototype.remove = function (tuple) {
    return this.change().delete(key(tuple));
  };

  // Drops the facts that `drop` picks.
  Table.prototype.drop = function (drop) {
    var facts = this.change();
    var gone = [];
    facts.forEach(function (entry, k) {
      if (drop(entry.tuple)) {
        gone.push(k);
      }
    });
    for (var i = 0; i < gone.length; i++) {
      facts.delete(gone[i]);
    }
  };

  // Drops the facts that agree with `tuple` in the columns `cols`.
  Table.prototype.dropKey = function (tuple, cols) {
    this.drop(function (t) {
      return !differs(t, tuple, cols);
    });
  };

  // A runtime for a program (Program), as src/lem/eval.rs's Runtime.
  function Runtime(prog) {
    this.prog = prog;
    this.tables = prog.rels.map(function () {
      return new Table();
    });
    this.visited = 0;
  }

  // The facts that the last tick's joins visited: a measure of its work,
  // which the shared tests compare with Rust's.
  Runtime.prototype.visits = function () {
    return this.visited;
  };

  // Whether the host may change a relation: only an `input` one.
  Runtime.prototype.input = function (r) {
    var rel = this.prog.rels[r];
    if (rel.kind !== "input") {
      fail(rel.name + " isn't an `input` relation, so the host can't change it");
    }
  };

  // The host's fact, in an `input` relation. In a relation with a key, it
  // replaces the fact with its key.
  Runtime.prototype.assertTuple = function (r, tuple) {
    this.input(r);
    var cols = this.prog.rels[r].key;
    if (cols.length > 0) {
      this.tables[r].dropKey(tuple, cols);
    }
    this.tables[r].put(tuple, HOST);
  };

  // Retracts the host's fact: whether it held.
  Runtime.prototype.retractTuple = function (r, tuple) {
    this.input(r);
    return this.tables[r].remove(tuple);
  };

  // A message, from JSON.
  Runtime.prototype.msg = function (j) {
    return this.prog.value("Msg", j, "a message");
  };

  // Takes a step of a tick, from JSON (src/lem/host.rs): asserts or
  // retracts the host's fact at once, or notes in `script`, {msg, show},
  // the tick's message, or a relation to show after it.
  Runtime.prototype.step = function (j, script) {
    var prog = this.prog;
    var items = Array.isArray(j) ? j : NONE;
    var tag = items.length > 0 ? items[0] : null;
    var named = items.length > 1 && typeof items[1] === "string";
    if (items.length === 3 && named && (tag === "+" || tag === "-")) {
      var fact = prog.fact(items[1], items[2]);
      if (tag === "+") {
        this.assertTuple(fact.rel, fact.tuple);
      } else if (!this.retractTuple(fact.rel, fact.tuple)) {
        fail(prog.factText(fact.rel, fact.tuple) + " doesn't hold, so it can't be retracted");
      }
    } else if (items.length === 2 && tag === "!") {
      var msg = this.msg(items[1]);
      if (script.msg !== null) {
        fail("a tick has one message at most");
      }
      script.msg = msg;
    } else if (items.length === 2 && named && tag === "?") {
      if (prog.rel(items[1]) < 0) {
        fail("there's no relation `" + items[1] + "`");
      }
      script.show.push(items[1]);
    } else if (items.length === 2 && named && tag === "error") {
      fail(items[1]);
    } else {
      fail(
        "a step is [\"+\", relation, fields], [\"-\", relation, fields], " +
          "[\"!\", message] or [\"?\", relation]"
      );
    }
  };

  // Runs a tick given as JSON steps, taking each as it's read, and gives
  // what it shows: the facts that it asked for, then its effects, a line
  // each; or "(nothing to show)".
  Runtime.prototype.run = function (steps) {
    if (!Array.isArray(steps)) {
      fail("a tick is an array of steps, not " + jsonKind(steps));
    }
    var script = { msg: null, show: [] };
    for (var i = 0; i < steps.length; i++) {
      this.step(steps[i], script);
    }
    var out = this.show(script, this.tick(script.msg));
    return out === "" ? "(nothing to show)" : out;
  };

  // The facts of the relations that a script asked for, then the effects,
  // as `effect Send(...)`, a line each.
  Runtime.prototype.show = function (script, effects) {
    var prog = this.prog;
    var lines = [];
    for (var i = 0; i < script.show.length; i++) {
      var r = prog.rel(script.show[i]);
      var tuples = this.tables[r].tuples();
      for (var j = 0; j < tuples.length; j++) {
        lines.push(prog.factText(r, tuples[j]));
      }
    }
    for (var k = 0; k < effects.length; k++) {
      lines.push("effect " + prog.factText(effects[k].rel, effects[k].tuple));
    }
    return lines.join("\n");
  };

  // One tick: the message's `on` blocks, if there's a message, then the
  // rules. Gives the effects that the `on` blocks sent, in order, each as
  // {rel, tuple}.
  Runtime.prototype.tick = function (msg) {
    this.visited = 0;
    var effects = [];
    if (msg !== null && msg !== undefined) {
      var changes = this.update(msg);
      this.apply(changes.inserts, changes.deletes, effects);
    }
    this.derive();
    this.checkKeys();
    return effects;
  };

  // What the `on` blocks of a message would add and delete, each as {rel,
  // tuple, n}, where `n` is its action's number.
  Runtime.prototype.update = function (msg) {
    var self = this;
    var inserts = [];
    var deletes = [];
    this.prog.actions.forEach(function (action, a) {
      var env = vars(action.vars.length);
      if (!self.matchPat(action.msg, msg, env, [])) {
        return;
      }
      self.solve(action.body, env, null, function (env) {
        if (action.insert !== null) {
          var tuple = action.insert.map(function (e) {
            return self.compute(e, env);
          });
          inserts.push({ rel: action.rel, tuple: tuple, n: a });
          return;
        }
        var tuples = self.tables[action.rel].tuples();
        for (var i = 0; i < tuples.length; i++) {
          if (self.matchAll(action.remove, tuples[i], env.slice(), [])) {
            deletes.push({ rel: action.rel, tuple: tuples[i], n: a });
          }
        }
      });
    });
    return { inserts: inserts, deletes: deletes };
  };

  // Applies what the `on` blocks add and delete: no fact both, deletes
  // first, then inserts, each in place of the facts with its key, of which
  // a tick adds one; effects go to `effects`.
  Runtime.prototype.apply = function (inserts, deletes, effects) {
    var prog = this.prog;
    var i, j, ins, del;
    for (i = 0; i < inserts.length; i++) {
      ins = inserts[i];
      for (j = 0; j < deletes.length; j++) {
        del = deletes[j];
        if (del.rel === ins.rel && compareLists(del.tuple, ins.tuple) === 0) {
          var at = this.why(byAction(ins.n));
          var other = this.why(byAction(del.n));
          var both = prog.factText(ins.rel, ins.tuple) + " is both added (" + at + ")";
          fail(both + " and deleted (" + other + ")");
        }
      }
    }
    for (j = 0; j < deletes.length; j++) {
      this.tables[deletes[j].rel].remove(deletes[j].tuple);
    }
    var added = new Map();
    for (i = 0; i < inserts.length; i++) {
      ins = inserts[i];
      var rel = prog.rels[ins.rel];
      if (rel.kind === "effect") {
        effects.push({ rel: ins.rel, tuple: ins.tuple });
        continue;
      }
      var cols = rel.key.length > 0 ? rel.key : columns(rel.fields.length);
      var k = ins.rel + " " + key(project(ins.tuple, cols));
      var before = added.get(k);
      if (before !== undefined && compareLists(before.tuple, ins.tuple) !== 0) {
        this.keyClash(ins.rel, before, byAction(before.n), ins.tuple, byAction(ins.n));
      }
      this.tables[ins.rel].dropKey(ins.tuple, cols);
      this.tables[ins.rel].put(ins.tuple, byAction(ins.n));
      added.set(k, ins);
    }
  };

  // The error for two facts with one key: {tuple} first, `why` it holds,
  // then the other.
  Runtime.prototype.keyClash = function (r, first, why, tuple, other) {
    var prog = this.prog;
    var x = prog.factText(r, first.tuple) + " (" + this.why(why) + ")";
    var y = prog.factText(r, tuple) + " (" + this.why(other) + ")";
    fail("one key, two facts: " + x + " and " + y);
  };

  // Runs the rules again from the input and state facts, component by
  // component, in the checker's order.
  Runtime.prototype.derive = function () {
    var self = this;
    this.prog.rels.forEach(function (rel, r) {
      if (rel.kind === "rel") {
        self.tables[r].change().clear();
      }
    });
    for (var c = 0; c < this.prog.components.length; c++) {
      this.component(c);
    }
  };

  // Runs a component's rules: once on every fact; then, if it's recursive,
  // again for each of their atoms on its relations, joining just what the
  // last round added there, until a round adds nothing.
  Runtime.prototype.component = function (c) {
    var prog = this.prog;
    var component = prog.components[c];
    var found = [];
    var i;
    for (i = 0; i < component.rules.length; i++) {
      this.solutions(component.rules[i], null, found);
    }
    var fresh = this.insert(found);
    while (component.recursive && fresh.size > 0) {
      var added = new Map();
      fresh.forEach(function (tuples, rel) {
        added.set(rel, { tuples: tuples, indexes: new Map() });
      });
      found = [];
      for (i = 0; i < component.rules.length; i++) {
        var r = component.rules[i];
        var body = prog.rules[r].body;
        for (var at = 0; at < body.length; at++) {
          if (body[at].tag === "atom" && added.has(body[at].rel)) {
            this.solutions(r, { at: at, fresh: added.get(body[at].rel) }, found);
          }
        }
      }
      fresh = this.insert(found);
    }
  };

  // Adds rule `r`'s solutions to `found`, as {rel, tuple, n}. `delta` is
  // null, or a round's new facts, {tuples, indexes}, for the atom at a
  // place in the body: {at, fresh}.
  Runtime.prototype.solutions = function (r, delta, found) {
    var self = this;
    var rule = this.prog.rules[r];
    this.solve(rule.body, vars(rule.vars.length), delta, function (env) {
      var tuple;
      try {
        tuple = rule.args.map(function (e) {
          return self.compute(e, env);
        });
      } catch (e) {
        if (e instanceof LemError) {
          fail(e.message + ", in the rule at " + rule.at);
        }
        throw e;
      }
      found.push({ rel: rule.head, tuple: tuple, n: r });
    });
  };

  // Adds rules' new facts: by relation, those that didn't hold before.
  Runtime.prototype.insert = function (found) {
    var fresh = new Map();
    for (var i = 0; i < found.length; i++) {
      var f = found[i];
      if (this.tables[f.rel].derive(f.tuple, f.n)) {
        var tuples = fresh.get(f.rel);
        if (tuples === undefined) {
          fresh.set(f.rel, [f.tuple]);
        } else {
          tuples.push(f.tuple);
        }
      }
    }
    return fresh;
  };

  // Calls `out` with each solution of a body, as the variables' values:
  // its positive atoms, which come first, joined as join() does; then the
  // rest in order.
  Runtime.prototype.solve = function (body, env, delta, out) {
    var atoms = 0;
    while (atoms < body.length && body[atoms].tag === "atom") {
      atoms++;
    }
    var taken = [];
    for (var i = 0; i < atoms; i++) {
      taken.push(false);
    }
    this.join(body, atoms, taken, atoms, env, delta, out);
  };

  // Joins the first `atoms` parts of a body, the atoms, `left` of which
  // aren't `taken`: of those that are ready, it looks each up by the
  // columns bound so far; fails if one has no facts; and else takes the
  // one with the fewest, the first written on a tie. Then the rest of the
  // body, in order.
  Runtime.prototype.join = function (body, atoms, taken, left, env, delta, out) {
    if (left === 0) {
      this.rest(body, atoms, env, out);
      return;
    }
    var best = -1;
    var bucket = NONE;
    for (var at = 0; at < atoms; at++) {
      var atom = body[at];
      if (taken[at] || !bound(atom.needs, env)) {
        continue;
      }
      var cols = [];
      for (var c = 0; c < atom.pats.length; c++) {
        if (known(atom.pats[c], env)) {
          cols.push(c);
        }
      }
      var fresh = delta !== null && delta.at === at ? delta.fresh : null;
      var tuples = this.bucket(atom, cols, env, fresh);
      if (left === 1) {
        best = at;
        bucket = tuples;
        break;
      }
      if (tuples.length === 0) {
        return;
      }
      if (best < 0 || tuples.length < bucket.length) {
        best = at;
        bucket = tuples;
      }
    }
    if (best < 0) {
      fail("no atom is ready to join (a bug in lem's checker)");
    }
    var pats = body[best].pats;
    taken[best] = true;
    for (var i = 0; i < bucket.length; i++) {
      this.visited++;
      var trail = [];
      if (this.matchAll(pats, bucket[i], env, trail)) {
        this.join(body, atoms, taken, left - 1, env, delta, out);
      }
      unbind(env, trail);
    }
    taken[best] = false;
  };

  // Whether the variables `needs` are bound.
  function bound(needs, env) {
    for (var i = 0; i < needs.length; i++) {
      if (env[needs[i]] === undefined) {
        return false;
      }
    }
    return true;
  }

  // Whether a pattern's value is known, from the variables bound so far:
  // in an atom that's ready, an expression's is.
  function known(pat, env) {
    switch (pat[0]) {
      case "any":
        return false;
      case "var":
        return env[pat[1]] !== undefined;
      case "ctor":
        return pat[2].every(function (p) {
          return known(p, env);
        });
    }
    return true;
  }

  // The facts that might match an atom, by the columns `cols` whose values
  // are known: from what a round added, if `fresh` is this atom's, else
  // from its relation.
  Runtime.prototype.bucket = function (atom, cols, env, fresh) {
    if (cols.length === 0) {
      return fresh !== null ? fresh.tuples : this.tables[atom.rel].tuples();
    }
    var values = [];
    for (var i = 0; i < cols.length; i++) {
      values.push(this.keyValue(atom.pats[cols[i]], env));
    }
    var map;
    if (fresh !== null) {
      map = indexOn(fresh.indexes, cols, function () {
        return fresh.tuples;
      });
    } else {
      map = this.tables[atom.rel].index(cols);
    }
    var bucket = map.get(key(values));
    return bucket === undefined ? NONE : bucket;
  };

  // The rest of a body, after its atoms, in order: bindings, tests and
  // negations, each negation looked up by its key.
  Runtime.prototype.rest = function (body, at, env, out) {
    if (at === body.length) {
      out(env);
      return;
    }
    var lit = body[at];
    switch (lit.tag) {
      case "let":
        env[lit.v] = this.compute(lit.e, env);
        this.rest(body, at + 1, env, out);
        env[lit.v] = undefined;
        return;
      case "test":
        if (this.test(lit, env)) {
          this.rest(body, at + 1, env, out);
        }
        return;
      case "nottest":
        if (!this.test(lit, env)) {
          this.rest(body, at + 1, env, out);
        }
        return;
      case "not":
        var tuples = this.bucket(lit, lit.key, env, null);
        for (var i = 0; i < tuples.length; i++) {
          this.visited++;
          var trail = [];
          var found = this.matchAll(lit.pats, tuples[i], env, trail);
          unbind(env, trail);
          if (found) {
            return;
          }
        }
        this.rest(body, at + 1, env, out);
        return;
    }
    fail("a body's atoms come first (a bug in lem's checker)");
  };

  // A pattern's value, when it's known: a key column's.
  Runtime.prototype.keyValue = function (pat, env) {
    switch (pat[0]) {
      case "const":
        return pat[1];
      case "var":
        if (env[pat[1]] === undefined) {
          fail("a key's variable isn't bound (a bug in lem's checker)");
        }
        return env[pat[1]];
      case "expr":
        return this.compute(pat[1], env);
      case "ctor":
        var value = [pat[1]];
        for (var i = 0; i < pat[2].length; i++) {
          value.push(this.keyValue(pat[2][i], env));
        }
        return value;
    }
    return fail("`_` can't be in a key (a bug in lem's checker)");
  };

  Runtime.prototype.matchAll = function (pats, tuple, env, trail) {
    for (var i = 0; i < pats.length; i++) {
      if (!this.matchPat(pats[i], tuple[i], env, trail)) {
        return false;
      }
    }
    return true;
  };

  // Whether a value matches; binds the pattern's unbound variables, and
  // records them in `trail`, so that the caller can unbind them.
  Runtime.prototype.matchPat = function (pat, value, env, trail) {
    switch (pat[0]) {
      case "any":
        return true;
      case "var":
        var v = env[pat[1]];
        if (v !== undefined) {
          return equal(v, value);
        }
        env[pat[1]] = value;
        trail.push(pat[1]);
        return true;
      case "const":
        return equal(pat[1], value);
      case "ctor":
        var pats = pat[2];
        if (!Array.isArray(value) || value[0] !== pat[1] || value.length !== pats.length + 1) {
          return false;
        }
        for (var i = 0; i < pats.length; i++) {
          if (!this.matchPat(pats[i], value[i + 1], env, trail)) {
            return false;
          }
        }
        return true;
      case "expr":
        return equal(this.compute(pat[1], env), value);
    }
    return fail("there's no pattern `" + pat[0] + "`");
  };

  // An expression's value, once the body has bound its variables.
  Runtime.prototype.compute = function (e, env) {
    switch (e[0]) {
      case "var":
        if (env[e[1]] === undefined) {
          fail("a variable isn't bound (a bug in lem's checker)");
        }
        return env[e[1]];
      case "const":
        return e[1];
      case "ctor":
        var value = [e[1]];
        for (var i = 0; i < e[2].length; i++) {
          value.push(this.compute(e[2][i], env));
        }
        return value;
      case "neg":
        var x = this.compute(e[1], env);
        if (typeof x !== "number") {
          fail("`-` negates numbers (a bug in lem's checker)");
        }
        // Within +/-(2^53 - 1), and finite, a negation is too.
        return zero(-x);
      case "arith":
        return arith(e[1], e[2], this.compute(e[3], env), this.compute(e[4], env));
      case "num":
        return this.compute(e[1], env);
      case "text":
        var t = this.compute(e[2], env);
        return typeof t === "string" ? t : this.prog.text(t, e[1]);
    }
    return fail("there's no expression `" + e[0] + "`");
  };

  // Arithmetic, by its result's type, `ty`: Int and Time are whole
  // numbers, exact within +/-(2^53 - 1), as JavaScript's numbers are; Num is
  // finite; and Text joins.
  function arith(op, ty, a, b) {
    if (ty === "Text") {
      return a + b;
    }
    var x;
    if (op === "/" || ty === "Num") {
      if (op === "/" && b === 0) {
        fail("a division by zero");
      }
      x = op === "+" ? a + b : op === "-" ? a - b : op === "*" ? a * b : a / b;
      if (!isFinite(x)) {
        fail("`" + op + "` overflowed: a Num stays finite");
      }
      return zero(x);
    }
    x = op === "+" ? a + b : op === "-" ? a - b : a * b;
    if (Math.abs(x) > MAX_INT) {
      var within = "a whole number stays within \u00b1" + MAX_INT + " (2^53 \u2212 1)";
      fail("`" + op + "` overflowed: " + within);
    }
    return zero(x);
  }

  Runtime.prototype.test = function (lit, env) {
    var a = this.compute(lit.a, env);
    var b = this.compute(lit.b, env);
    switch (lit.op) {
      case "=":
        return equal(a, b);
      case "/=":
        return !equal(a, b);
      case "<":
        return compare(a, b) < 0;
      case "<=":
        return compare(a, b) <= 0;
      case ">":
        return compare(a, b) > 0;
      case ">=":
        return compare(a, b) >= 0;
    }
    return fail("there's no test `" + lit.op + "`");
  };

  // Facts that agree on their relation's key agree on the rest. Of a
  // relation whose don't, the error names the first two that clash, in
  // order, as Rust's does.
  Runtime.prototype.checkKeys = function () {
    var rels = this.prog.rels;
    for (var r = 0; r < rels.length; r++) {
      if (rels[r].key.length > 0 && !this.keysHold(r, rels[r].key)) {
        this.keyError(r, rels[r].key);
      }
    }
  };

  // Whether no two of a relation's facts share a key, by `cols`.
  Runtime.prototype.keysHold = function (r, cols) {
    var seen = new Set();
    var holds = true;
    this.tables[r].facts.forEach(function (entry) {
      var k = key(project(entry.tuple, cols));
      holds = holds && !seen.has(k);
      seen.add(k);
    });
    return holds;
  };

  Runtime.prototype.keyError = function (r, cols) {
    var table = this.tables[r];
    var seen = new Map();
    var tuples = table.tuples();
    for (var i = 0; i < tuples.length; i++) {
      var k = key(project(tuples[i], cols));
      var why = table.facts.get(key(tuples[i])).why;
      var first = seen.get(k);
      if (first !== undefined) {
        this.keyClash(r, first, first.why, tuples[i], why);
      }
      seen.set(k, { tuple: tuples[i], why: why });
    }
  };

  // Where a fact comes from, for errors.
  Runtime.prototype.why = function (why) {
    if (has(why, "action")) {
      return "from the `on` block at " + this.prog.actions[why.action].at;
    }
    if (has(why, "rule")) {
      return "from the rule at " + this.prog.rules[why.rule].at;
    }
    return "from the host";
  };

  return {
    VERSION: VERSION,
    MAX_INT: MAX_INT,
    Error: LemError,
    Program: Program,
    Runtime: Runtime,
    compare: compare,
    numText: numText,
    quote: quote
  };
})();
