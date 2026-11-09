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

  // A value of type `ty` as the host's JSON (src/lem/host.rs): as lem.js
  // holds it, but a constructor without fields by its name, unless the
  // type is Any.
  Program.prototype.valueJson = function (v, ty) {
    if (!Array.isArray(v)) {
      return v;
    }
    if (v.length === 1 && this.base(ty) !== "Any") {
      return v[0];
    }
    var fields = this.ctor(v[0]).fields;
    var out = [v[0]];
    for (var i = 1; i < v.length; i++) {
      out.push(this.valueJson(v[i], fields[i - 1][1]));
    }
    return out;
  };

  // A relation's fact as the host's JSON: its fields by name.
  Program.prototype.factJson = function (r, tuple) {
    var fields = this.rels[r].fields;
    var out = {};
    for (var i = 0; i < fields.length; i++) {
      out[fields[i][0]] = this.valueJson(tuple[i], fields[i][1]);
    }
    return out;
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
  // then those added since, each in its place until it goes.
  function addTo(map, cols, tuple) {
    var k = key(project(tuple, cols));
    var bucket = map.get(k);
    if (bucket === undefined) {
      map.set(k, [tuple]);
    } else {
      bucket.push(tuple);
    }
  }

  // Takes a fact, as its table holds it, out of an index.
  function takeFrom(map, cols, tuple) {
    var k = key(project(tuple, cols));
    var bucket = map.get(k) || NONE;
    var at = bucket.indexOf(tuple);
    if (at < 0) {
      fail("an index lost a fact of its table (a bug in lem.js)");
    }
    bucket.splice(at, 1);
    if (bucket.length === 0) {
      map.delete(k);
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

  // What came to a relation, and what went from it, since the last tick
  // that ran the rules (Runtime.settle), and none before the first, when
  // nothing follows what changed yet: facts by their values as text, in
  // the order that they came or went. Rust's runtime keeps the facts that
  // came in the same order, which sets the order of what follows them in
  // the indexes (Runtime.follow): a whole run notes those that it adds in
  // the order that its rules first derived them, which a new table's facts
  // keep (Runtime.rerun). The order of those that went matters to nothing,
  // since an index keeps the rest in their order.
  function Log() {
    this.came = new Map();
    this.went = new Map();
  }

  // Notes that a fact came, or went: one that went and comes back, or came
  // and goes again, is no change.
  Log.prototype.note = function (k, tuple, came) {
    var to = came ? this.came : this.went;
    var from = came ? this.went : this.came;
    if (from.has(k)) {
      from.delete(k);
    } else {
      to.set(k, tuple);
    }
  };

  // Whether anything came or went.
  Log.prototype.changed = function () {
    return this.came.size > 0 || this.went.size > 0;
  };

  // The facts that came, or went, in the order that they did.
  function listed(facts) {
    var tuples = [];
    facts.forEach(function (tuple) {
      tuples.push(tuple);
    });
    return tuples;
  }

  // What a relation had before the tick's changes, from its log (Log), for
  // the joins that follow several changes at once (Runtime.follow): its
  // facts now, but those that came, and with those that went, in order of
  // their values, so that both runtimes take them alike; with indexes on
  // those that came and on those that went, as joins ask for them.
  function Old(log) {
    this.came = log.came;
    this.went = listed(log.went).sort(compareLists);
    this.cameIndexes = new Map();
    this.wentIndexes = new Map();
  }

  // What a relation had before the tick's changes (Old), whole or at a
  // key's values: its facts now, `now`, a table or a bucket, but those that
  // came, then those that went, `went`; and how many (length), which a
  // join compares before it takes them (tuples).
  function OldBucket(now, old, went, length) {
    this.now = now;
    this.old = old;
    this.went = went;
    this.length = length;
  }

  OldBucket.prototype.tuples = function () {
    var now = this.now instanceof Table ? this.now.tuples() : this.now;
    var came = this.old.came;
    var out = [];
    for (var i = 0; i < now.length; i++) {
      if (!came.has(key(now[i]))) {
        out.push(now[i]);
      }
    }
    return out.concat(this.went);
  };

  // A relation's facts, and why each holds, by their values as text; the
  // facts in order, once asked for; and indexes on them, which it keeps as
  // facts come and go. Its version moves on with each change to its facts
  // but a rule's (bump): a table that rules fill is new, a version on from
  // the one that it replaces (Runtime.rerun). Its log (Log) notes what
  // comes and goes; a table that rules fill has none until it replaces one.
  function Table(log) {
    this.version = 0;
    this.facts = new Map();
    this.sorted = null;
    this.indexes = new Map();
    // For the host's facts in a relation with a key, the facts by their
    // key's values (putKeyed), which only the host's changes keep.
    this.keyed = null;
    // For a relation with a key, how many facts have each of its values, and
    // how many of those values more than one fact has (keyCount), which it
    // keeps as facts come and go, once asked for.
    this.keys = null;
    this.log = log;
  }

  // How many facts it has: a join's bucket can be a whole table, whose
  // size counts before it's walked, which sorts it.
  Object.defineProperty(Table.prototype, "length", {
    get: function () {
      return this.facts.size;
    }
  });

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

  // The index on `cols`, which mustn't be empty. Its buckets hold their
  // facts in order, as Rust's runtime builds them from its sorted facts:
  // from the facts in order, if the table has them so; else each bucket is
  // sorted, which costs least where a bucket holds one fact.
  Table.prototype.index = function (cols) {
    var name = cols.join(",");
    var index = this.indexes.get(name);
    if (index === undefined) {
      var map = new Map();
      if (this.sorted !== null) {
        this.sorted.forEach(function (tuple) {
          addTo(map, cols, tuple);
        });
      } else {
        this.facts.forEach(function (entry) {
          addTo(map, cols, entry.tuple);
        });
        map.forEach(function (bucket) {
          if (bucket.length > 1) {
            bucket.sort(compareLists);
          }
        });
      }
      index = { cols: cols, map: map };
      this.indexes.set(name, index);
    }
    return index.map;
  };

  // A fact that rule `r` derives, into the indexes too: whether it's new.
  // One that holds already keeps the lowest-numbered of its rules. `known`
  // is the fact's key, if it's known already.
  Table.prototype.derive = function (tuple, r, known) {
    var k = known === undefined ? key(tuple) : known;
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
    if (this.keys !== null) {
      countKey(this.keys, tuple, true);
    }
    return true;
  };

  // Adds a fact that doesn't hold, and why it holds: to the indexes too,
  // and to the log.
  Table.prototype.add = function (tuple, why) {
    var k = key(tuple);
    this.facts.set(k, { tuple: tuple, why: why });
    this.sorted = null;
    this.indexes.forEach(function (index) {
      addTo(index.map, index.cols, tuple);
    });
    if (this.keys !== null) {
      countKey(this.keys, tuple, true);
    }
    if (this.log !== null) {
      this.log.note(k, tuple, true);
    }
  };

  // Takes away the fact whose values, as text, are `k`, which holds: from
  // the indexes too, and notes it in the log.
  Table.prototype.cut = function (k) {
    var tuple = this.facts.get(k).tuple;
    this.facts.delete(k);
    this.sorted = null;
    this.indexes.forEach(function (index) {
      takeFrom(index.map, index.cols, tuple);
    });
    if (this.keys !== null) {
      countKey(this.keys, tuple, false);
    }
    if (this.log !== null) {
      this.log.note(k, tuple, false);
    }
  };

  // The facts changed otherwise than by the host's facts with a key
  // (putKeyed), which no longer finds them by key; and the version moves
  // on.
  Table.prototype.changed = function () {
    this.keyed = null;
    this.version += 1;
  };

  // The host's fact, in place of the one with its values in the columns
  // `cols`, the relation's key. The facts by key, made once, find that
  // one, so that the host's facts take time linear in their number.
  Table.prototype.putKeyed = function (tuple, cols) {
    var keyed = this.keyed;
    if (keyed === null) {
      keyed = new Map();
      this.facts.forEach(function (entry) {
        keyed.set(key(project(entry.tuple, cols)), entry.tuple);
      });
    }
    var k = key(project(tuple, cols));
    var old = keyed.get(k);
    if (old !== undefined) {
      this.cut(key(old));
    }
    this.add(tuple, HOST);
    keyed.set(k, tuple);
    this.keyed = keyed;
    this.version += 1;
  };

  // Removes the host's fact, which holds, from a relation whose key is
  // `cols`, and from the facts by key.
  Table.prototype.removeKeyed = function (tuple, cols) {
    this.cut(key(tuple));
    if (this.keyed !== null) {
      this.keyed.delete(key(project(tuple, cols)));
    }
    this.version += 1;
  };

  // A fact that doesn't hold, and why it holds.
  Table.prototype.put = function (tuple, why) {
    this.add(tuple, why);
    this.changed();
  };

  // Removes a fact: whether it held. One that doesn't hold changes
  // nothing.
  Table.prototype.remove = function (tuple) {
    var k = key(tuple);
    if (!this.facts.has(k)) {
      return false;
    }
    this.cut(k);
    this.changed();
    return true;
  };

  // Drops the facts that `drop` picks. If it picks none, nothing changes,
  // so the table keeps its version.
  Table.prototype.drop = function (drop) {
    var gone = [];
    this.facts.forEach(function (entry, k) {
      if (drop(entry.tuple)) {
        gone.push(k);
      }
    });
    if (gone.length === 0) {
      return;
    }
    for (var i = 0; i < gone.length; i++) {
      this.cut(gone[i]);
    }
    this.changed();
  };

  // Drops the facts that agree with `tuple` in the columns `cols`.
  Table.prototype.dropKey = function (tuple, cols) {
    this.drop(function (t) {
      return !differs(t, tuple, cols);
    });
  };

  // Takes away every fact, if it has any, noting each in the log; so its
  // indexes go.
  Table.prototype.clear = function () {
    if (this.facts.size === 0) {
      return;
    }
    var log = this.log;
    if (log !== null) {
      this.facts.forEach(function (entry, k) {
        log.note(k, entry.tuple, false);
      });
    }
    this.facts = new Map();
    this.sorted = null;
    this.indexes = new Map();
    this.keys = null;
    this.changed();
  };

  // The facts by their values in the columns `cols`, the relation's key:
  // {cols, counts, clashes}, how many facts have each of the key's values,
  // and how many values more than one fact has. Made once, without sorting,
  // and kept as facts come and go.
  Table.prototype.keyCount = function (cols) {
    if (this.keys === null) {
      var keys = { cols: cols, counts: new Map(), clashes: 0 };
      this.facts.forEach(function (entry) {
        countKey(keys, entry.tuple, true);
      });
      this.keys = keys;
    }
    return this.keys;
  };

  // Counts a fact's key's values once more, or once less.
  function countKey(keys, tuple, more) {
    var k = key(project(tuple, keys.cols));
    var n = keys.counts.get(k) || 0;
    if (more) {
      keys.clashes += n === 1 ? 1 : 0;
      keys.counts.set(k, n + 1);
    } else {
      keys.clashes -= n === 2 ? 1 : 0;
      if (n === 1) {
        keys.counts.delete(k);
      } else {
        keys.counts.set(k, n - 1);
      }
    }
  }

  // A runtime for a program (Program), as src/lem/eval.rs's Runtime.
  function Runtime(prog) {
    this.prog = prog;
    this.tables = prog.rels.map(function () {
      return new Table(null);
    });
    this.visited = 0;
    // The components that a tick skips, unless the host asks for one of
    // their relations, which the next tick derives: `asked`.
    this.skip = skipped(prog, []);
    this.asked = [];
    // For each component, the relations that its rules read (reads); and
    // the versions that they had when it last ran, or null: it hasn't run
    // since the runtime began, or since the rules failed, or a tick skipped
    // it.
    this.reads = reads(prog);
    this.ran = prog.components.map(function () {
      return null;
    });
    // For each relation, the version at which its keys last held, or null.
    this.held = prog.rels.map(function () {
      return null;
    });
    // For each rule, of a component that isn't recursive, how many ways it
    // derives each fact, as its component last ran, by the fact's key, each
    // as {tuple, n} (follow); and for each of its atoms, whether a tick
    // follows a change to that atom's relation alone (followed).
    this.counts = prog.rules.map(function () {
      return null;
    });
    this.followed = prog.rules.map(followed);
    // Whether the next tick derives every relation anew (forget).
    this.anew = false;
    // What the last tick changed, by relation, and how many ticks have
    // ended (settle).
    this.changes = null;
    // What relations had before the tick's changes, by relation (Old),
    // while follow joins.
    this.olds = null;
    this.settled = 0;
  }

  // The variables that a pattern binds, into `out`.
  function patVars(pat, out) {
    if (pat[0] === "var") {
      out.push(pat[1]);
    } else if (pat[0] === "ctor") {
      pat[2].forEach(function (p) {
        patVars(p, out);
      });
    }
  }

  // Whether a pattern has `_`.
  function loose(pat) {
    return pat[0] === "any" || (pat[0] === "ctor" && pat[2].some(loose));
  }

  // Whether a pattern has no variable and no expression: an atom of such
  // patterns alone binds nothing, a test, which holds or doesn't for every
  // solution alike.
  function fixed(pat) {
    switch (pat[0]) {
      case "var":
      case "expr":
        return false;
      case "ctor":
        return pat[2].every(fixed);
    }
    return true;
  }

  // Whether a fact could match an atom's patterns, whatever its variables
  // and expressions are: a variable matches anything, but the same value
  // wherever it recurs; an expression, anything.
  function couldMatch(pats, tuple) {
    var env = new Map();
    function one(pat, value) {
      switch (pat[0]) {
        case "var":
          if (env.has(pat[1])) {
            return equal(env.get(pat[1]), value);
          }
          env.set(pat[1], value);
          return true;
        case "const":
          return equal(pat[1], value);
        case "ctor":
          var pats = pat[2];
          if (!Array.isArray(value) || value[0] !== pat[1] || value.length !== pats.length + 1) {
            return false;
          }
          for (var i = 0; i < pats.length; i++) {
            if (!one(pats[i], value[i + 1])) {
              return false;
            }
          }
          return true;
      }
      return true;
    }
    for (var i = 0; i < pats.length; i++) {
      if (!one(pats[i], tuple[i])) {
        return false;
      }
    }
    return true;
  }

  // Whether one of the facts that came, or went, in a log (Log), could
  // match an atom's patterns (couldMatch).
  function anyMatch(pats, facts) {
    var it = facts.values();
    for (var step = it.next(); !step.done; step = it.next()) {
      if (couldMatch(pats, step.value)) {
        return true;
      }
    }
    return false;
  }

  // For each of a rule's atoms, whether a tick follows a change to its
  // relation alone by joining there just the facts that came and went
  // (Runtime.follow). A rule's solutions must count the same whatever order
  // its join takes: a solution counts each fact of an atom with `_` that
  // binds a variable that no other atom binds, and once an atom without
  // `_`, which matches one fact at most; an atom with `_` and no variable
  // is a test, which counts once, so a change there isn't followed. But an
  // atom with `_` whose variables other atoms bind counts one fact or each,
  // by the order of the join: then no change to the rule is followed.
  function followed(rule) {
    var atoms = [];
    for (var i = 0; i < rule.body.length && rule.body[i].tag === "atom"; i++) {
      atoms.push(rule.body[i]);
    }
    var vars = atoms.map(function (atom) {
      var out = [];
      atom.pats.forEach(function (pat) {
        patVars(pat, out);
      });
      return out;
    });
    function own(i) {
      return vars[i].some(function (v) {
        return vars.every(function (others, j) {
          return j === i || others.indexOf(v) < 0;
        });
      });
    }
    var ok = atoms.map(function (atom, i) {
      return !atom.pats.some(loose) || own(i);
    });
    var counts = ok.every(function (yes, i) {
      return yes || vars[i].length === 0;
    });
    return ok.map(function (yes) {
      return counts && yes;
    });
  }

  // What each component's rules read, as atoms or negations: the relations
  // of other components, the host's and the `on` blocks', in order.
  function reads(prog) {
    return prog.components.map(function (c) {
      var read = [];
      c.rules.forEach(function (r) {
        prog.rules[r].body.forEach(function (lit) {
          var rel = lit.rel;
          if ((lit.tag === "atom" || lit.tag === "not") && c.rels.indexOf(rel) < 0 && read.indexOf(rel) < 0) {
            read.push(rel);
          }
        });
      });
      return read.sort(function (a, b) {
        return a - b;
      });
    });
  }

  // The components that a tick skips: those whose relations are all lazy,
  // and that nothing derived reads: no rule of a relation that's derived,
  // no `on` block, and not the host, which asks for the relations `asked`.
  // Their relations have no facts.
  function skipped(prog, asked) {
    var read = prog.rels.map(function (rel) {
      return rel.lazy !== true;
    });
    asked.forEach(function (r) {
      read[r] = true;
    });
    var todo = [];
    read.forEach(function (yes, r) {
      if (yes) {
        todo.push(r);
      }
    });
    function reads(body) {
      body.forEach(function (lit) {
        if ((lit.tag === "atom" || lit.tag === "not") && !read[lit.rel]) {
          read[lit.rel] = true;
          todo.push(lit.rel);
        }
      });
    }
    prog.actions.forEach(function (action) {
      reads(action.body);
    });
    while (todo.length > 0) {
      var r = todo.pop();
      prog.rules.forEach(function (rule) {
        if (rule.head === r) {
          reads(rule.body);
        }
      });
    }
    return prog.components.map(function (c) {
      return c.rels.every(function (rel) {
        return !read[rel];
      });
    });
  }

  // The facts that the last tick's joins visited: a measure of its work,
  // which the shared tests compare with Rust's.
  Runtime.prototype.visits = function () {
    return this.visited;
  };

  // Forgets what the components read when they last ran, and which keys
  // held: the next tick, after its `on` blocks, clears every derived
  // relation, runs every rule and checks every key, as every tick did
  // before ticks kept what they derived. For tests, which compare the two.
  Runtime.prototype.forget = function () {
    this.ran = this.ran.map(function () {
      return null;
    });
    this.held = this.held.map(function () {
      return null;
    });
    this.anew = true;
  };

  // A relation's facts, by its number, in order, each as {tuple, why}: for
  // a layer above lem.
  Runtime.prototype.relFacts = function (r) {
    var table = this.tables[r];
    return table.tuples().map(function (tuple) {
      return { tuple: tuple, why: table.facts.get(key(tuple)).why };
    });
  };

  // Drops a relation's facts that `drop` picks: for a layer above lem that
  // cleans up between ticks.
  Runtime.prototype.dropFacts = function (r, drop) {
    this.tables[r].drop(drop);
  };

  // Whether the host may change a relation: only an `input` one.
  Runtime.prototype.input = function (r) {
    var rel = this.prog.rels[r];
    if (rel.kind !== "input") {
      fail(rel.name + " isn't an `input` relation, so the host can't change it");
    }
  };

  // The host's fact, in an `input` relation. In a relation with a key, it
  // replaces the fact with its key. A fact that holds already changes
  // nothing.
  Runtime.prototype.assertTuple = function (r, tuple) {
    this.input(r);
    if (this.tables[r].facts.has(key(tuple))) {
      return;
    }
    var cols = this.prog.rels[r].key;
    if (cols.length > 0) {
      this.tables[r].putKeyed(tuple, cols);
    } else {
      this.tables[r].put(tuple, HOST);
    }
  };

  // Retracts the host's fact: whether it held. One that doesn't hold
  // changes nothing.
  Runtime.prototype.retractTuple = function (r, tuple) {
    this.input(r);
    if (!this.tables[r].facts.has(key(tuple))) {
      return false;
    }
    var cols = this.prog.rels[r].key;
    if (cols.length > 0) {
      this.tables[r].removeKeyed(tuple, cols);
      return true;
    }
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
      this.asked.push(prog.rel(items[1]));
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

  // Reads a tick given as JSON steps, and takes each as it's read: gives
  // the script, {msg, show}.
  Runtime.prototype.readSteps = function (steps) {
    if (!Array.isArray(steps)) {
      fail("a tick is an array of steps, not " + jsonKind(steps));
    }
    var script = { msg: null, show: [] };
    for (var i = 0; i < steps.length; i++) {
      this.step(steps[i], script);
    }
    return script;
  };

  // Runs a tick given as JSON steps (readSteps), and gives what it shows:
  // the facts that it asked for, then its effects, a line each; or
  // "(nothing to show)".
  Runtime.prototype.run = function (steps) {
    var script = this.readSteps(steps);
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
    try {
      this.derive();
      this.checkKeys();
    } finally {
      this.settle();
    }
    return effects;
  };

  // The end of a tick that ran the rules, whether or not they failed: each
  // relation's log begins again. The logs that end are what the tick
  // changed, by relation (changes), for a layer above lem, such as glem's
  // tree: null for each relation before the first tick, which logs
  // nothing. `settled` counts the ticks that ended so, so that a layer
  // above can tell whether it saw every one.
  Runtime.prototype.settle = function () {
    var changes = [];
    for (var r = 0; r < this.tables.length; r++) {
      changes.push(this.tables[r].log);
      this.tables[r].log = new Log();
    }
    this.changes = changes;
    this.settled += 1;
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

  // Runs the rules from the input and state facts, component by component,
  // in the checker's order: each only if what it reads has changed. A
  // component that isn't recursive, and has run since the rules last
  // failed, follows what changed (follow), unless it starts over
  // (startsOver); one that fails to follow runs again whole, as any other
  // does (rerun). A component that the tick skips has no facts. If a
  // component's rules fail, those after it have no facts, as when every
  // tick cleared them first, and the next tick runs every rule.
  Runtime.prototype.derive = function () {
    var prog = this.prog;
    var tables = this.tables;
    if (this.anew) {
      this.anew = false;
      prog.rels.forEach(function (rel, r) {
        if (rel.kind === "rel") {
          tables[r].clear();
        }
      });
    }
    var asked = this.asked;
    this.asked = [];
    var lazy = asked.some(function (r) {
      return prog.rels[r].lazy === true;
    });
    var skip = lazy ? skipped(prog, asked) : this.skip;
    for (var c = 0; c < prog.components.length; c++) {
      try {
        if (skip[c]) {
          this.ran[c] = null;
          this.clear(c);
        } else if (prog.components[c].recursive || this.ran[c] === null || this.startsOver(c)) {
          this.rerun(c);
        } else {
          try {
            this.follow(c);
          } catch (e) {
            if (!(e instanceof LemError)) {
              throw e;
            }
            this.ran[c] = null;
            this.rerun(c);
          }
        }
      } catch (e) {
        this.ran = this.ran.map(function () {
          return null;
        });
        for (var later = c + 1; later < prog.components.length; later++) {
          this.clear(later);
        }
        throw e;
      }
    }
  };

  // Clears component `c`'s relations: those that have facts, so that one
  // without keeps its version.
  Runtime.prototype.clear = function (c) {
    var rels = this.prog.components[c].rels;
    for (var i = 0; i < rels.length; i++) {
      this.tables[rels[i]].clear();
    }
  };

  // Runs component `c`'s rules, unless the relations that they read have
  // the versions that they had when it last ran. Each of its relations gets
  // a new table, a version on from the old one; but a relation whose facts
  // come out the same keeps the old table, with its version, indexes and
  // order, and why each fact holds now. A new table takes the old one's
  // log, and notes there what came and went.
  Runtime.prototype.rerun = function (c) {
    var tables = this.tables;
    var read = this.reads[c].map(function (r) {
      return tables[r].version;
    });
    if (same(read, this.ran[c])) {
      return;
    }
    var rels = this.prog.components[c].rels;
    var olds = rels.map(function (r) {
      var old = tables[r];
      tables[r] = new Table(null);
      tables[r].version = old.version + 1;
      return old;
    });
    this.component(c);
    rels.forEach(function (r, i) {
      var now = tables[r];
      var old = olds[i];
      if (sameFacts(now, old)) {
        now.facts.forEach(function (entry, k) {
          old.facts.get(k).why = entry.why;
        });
        tables[r] = old;
        return;
      }
      var log = old.log;
      now.log = log;
      if (log === null) {
        return;
      }
      old.facts.forEach(function (entry, k) {
        if (!now.facts.has(k)) {
          log.note(k, entry.tuple, false);
        }
      });
      now.facts.forEach(function (entry, k) {
        if (!old.facts.has(k)) {
          log.note(k, entry.tuple, true);
        }
      });
    });
    this.ran[c] = read;
  };

  // Whether component `c` starts over this tick, running whole as in its
  // first tick, rather than following what changed (follow): when its
  // relations have no facts to keep, and more of what it reads came or
  // went since the last tick than stayed. Following such a change would
  // join about as much as a whole run does, and keep more for each fact,
  // where a whole run fills new tables with nothing to compare: it's
  // semi-naive evaluation's first round, where every fact is new, which is
  // naive. Both runtimes decide alike, from the logs and the sizes of the
  // tables, so that their joins visit the same facts.
  Runtime.prototype.startsOver = function (c) {
    var tables = this.tables;
    var empty = this.prog.components[c].rels.every(function (r) {
      return tables[r].facts.size === 0;
    });
    if (!empty) {
      return false;
    }
    var changed = 0;
    var stayed = 0;
    this.reads[c].forEach(function (r) {
      var log = tables[r].log;
      changed += log.came.size + log.went.size;
      stayed += tables[r].facts.size - log.came.size;
    });
    return changed > stayed;
  };

  // Whether the versions that a component read are those that it read when
  // it last ran, if it has.
  function same(read, ran) {
    return (
      ran !== null &&
      read.every(function (version, i) {
        return version === ran[i];
      })
    );
  }

  // Whether two tables have the same facts.
  function sameFacts(a, b) {
    if (a.facts.size !== b.facts.size) {
      return false;
    }
    var same = true;
    a.facts.forEach(function (entry, k) {
      same = same && b.facts.has(k);
    });
    return same;
  }

  // Runs a component's rules: once on every fact; then, if it's recursive,
  // again for each of their atoms on its relations, joining just what the
  // last round added there, until a round adds nothing. Of one that isn't,
  // it counts how many ways each rule derives each fact (follow).
  Runtime.prototype.component = function (c) {
    var prog = this.prog;
    var component = prog.components[c];
    var found = [];
    var i;
    for (i = 0; i < component.rules.length; i++) {
      this.solutions(component.rules[i], null, found);
    }
    if (!component.recursive) {
      var counts = this.counts;
      component.rules.forEach(function (r) {
        counts[r] = new Map();
      });
      for (i = 0; i < found.length; i++) {
        found[i].k = key(found[i].tuple);
        tally(counts[found[i].n], found[i].k, found[i].tuple, 1);
      }
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

  // Follows what changed, since component `c` last ran, in what its rules read,
  // unless that has the versions that it had then: in place, by how many ways
  // each rule derives each fact (counts). A rule heeds only the atoms and
  // negations that what changed could make match otherwise (plan, unchanged). A
  // rule that follows each such atom (followed) joins, at each in turn, just
  // the facts that came there, in the order that they came, and those that
  // went, in order of their values, with its atoms before that one as they are
  // now and those after it as they were before the tick (Old), so that each way
  // to derive a fact that comes or goes counts once; one with such a negation,
  // or with such an atom that it doesn't follow, runs whole; one with none does
  // nothing. Every rule joins before any count changes. A fact whose counts
  // change holds if a rule derives it, by the lowest-numbered of them (place);
  // the facts that it adds come in the order that the rules' solutions first
  // gave them, so that both runtimes' indexes keep them in one order.
  Runtime.prototype.follow = function (c) {
    var self = this;
    var prog = this.prog;
    var tables = this.tables;
    var read = this.reads[c].map(function (r) {
      return tables[r].version;
    });
    if (same(read, this.ran[c])) {
      return;
    }
    var rules = prog.components[c].rules;
    var plans = rules.map(function (r) {
      return self.plan(r);
    });
    // What the relations at the places after a rule's first that changed
    // had before the tick, for the terms of the places before them.
    var olds = new Map();
    plans.forEach(function (plan, i) {
      if (plan !== null && plan !== -1) {
        plan.slice(1).forEach(function (at) {
          var rel = prog.rules[rules[i]].body[at].rel;
          if (!olds.has(rel)) {
            olds.set(rel, new Old(tables[rel].log));
          }
        });
      }
    });
    // Each rule's change: its counts anew ({whole: true}), or by how many
    // ways more, or fewer, it derives each fact; or null, for none.
    var changes;
    this.olds = olds;
    try {
      changes = rules.map(function (r, i) {
        var plan = plans[i];
        var found = [];
        if (plan === null) {
          return null;
        }
        if (plan === -1) {
          self.solutions(r, null, found);
          return { whole: true, counts: tallied(found, 1, new Map()) };
        }
        var by = new Map();
        plan.forEach(function (at, m) {
          var old = [];
          plan.slice(m + 1).forEach(function (place) {
            old[place] = true;
          });
          var log = tables[prog.rules[r].body[at].rel].log;
          [
            [listed(log.came), 1],
            [listed(log.went).sort(compareLists), -1]
          ].forEach(function (pair) {
            if (pair[0].length > 0) {
              found = [];
              self.solutions(r, { at: at, fresh: { tuples: pair[0], indexes: new Map() }, old: old }, found);
              tallied(found, pair[1], by);
            }
          });
        });
        return { whole: false, counts: by };
      });
    } finally {
      this.olds = null;
    }
    // The facts whose counts change, by relation; and the counts.
    var touched = new Map();
    rules.forEach(function (r, i) {
      var change = changes[i];
      if (change === null) {
        return;
      }
      var head = prog.rules[r].head;
      if (!touched.has(head)) {
        touched.set(head, new Map());
      }
      var heads = touched.get(head);
      var counts = self.counts[r];
      if (change.whole) {
        counts.forEach(function (entry, k) {
          heads.set(k, entry.tuple);
        });
        change.counts.forEach(function (entry, k) {
          heads.set(k, entry.tuple);
        });
        self.counts[r] = change.counts;
        return;
      }
      change.counts.forEach(function (entry, k) {
        heads.set(k, entry.tuple);
        tally(counts, k, entry.tuple, entry.n);
        if (counts.has(k) && counts.get(k).n < 0) {
          fail("a rule derives a fact fewer than no ways (a bug in lem.js)");
        }
      });
    });
    prog.components[c].rels.forEach(function (rel) {
      if (touched.has(rel)) {
        self.place(c, rel, touched.get(rel));
      }
    });
    this.ran[c] = read;
  };

  // What a tick does for rule `r`, by the atoms and negations in its body
  // that what changed in their relations changes (unchanged): nothing (null);
  // join just what came and went at each such atom, which it follows (an
  // array of their places); or run whole (-1).
  Runtime.prototype.plan = function (r) {
    var body = this.prog.rules[r].body;
    var places = [];
    for (var i = 0; i < body.length; i++) {
      var lit = body[i];
      var reads = lit.tag === "atom" || lit.tag === "not";
      if (reads && this.tables[lit.rel].log.changed() && !this.unchanged(lit)) {
        if (lit.tag !== "atom" || !this.followed[r][i]) {
          return -1;
        }
        places.push(i);
      }
    }
    return places.length === 0 ? null : places;
  };

  // Whether what changed in the relation of a rule's atom or negation, since
  // the last tick, leaves what it matches as it was, whatever its variables
  // are, and so changes none of the rule's solutions: when no fact that came
  // or went could match it (couldMatch); or, for a test, which binds no
  // variable (fixed), when it holds as it did. A test of `_` alone, such as
  // a block's condition `Turn()` gives, holds while its relation has facts;
  // any other holds before and after when facts that it matches both came
  // and went. Else it counts as changed.
  Runtime.prototype.unchanged = function (lit) {
    var table = this.tables[lit.rel];
    var log = table.log;
    var came = anyMatch(lit.pats, log.came);
    var went = anyMatch(lit.pats, log.went);
    if (!came && !went) {
      return true;
    }
    if (!lit.pats.every(fixed)) {
      return false;
    }
    var any = lit.pats.every(function (pat) {
      return pat[0] === "any";
    });
    if (any) {
      var now = table.facts.size;
      return (now > 0) === (now + log.went.size > log.came.size);
    }
    return came && went;
  };

  // Brings relation `rel`, of component `c`, up to its rules' counts, at the
  // facts `heads`, by their keys, in their order: a fact holds if a rule
  // derives it, by the lowest-numbered of them.
  Runtime.prototype.place = function (c, rel, heads) {
    var prog = this.prog;
    var counts = this.counts;
    var rules = prog.components[c].rules.filter(function (r) {
      return prog.rules[r].head === rel;
    });
    var table = this.tables[rel];
    var changed = false;
    heads.forEach(function (tuple, k) {
      var by = -1;
      for (var i = 0; i < rules.length && by < 0; i++) {
        by = counts[rules[i]].has(k) ? rules[i] : -1;
      }
      var entry = table.facts.get(k);
      if (by < 0 && entry !== undefined) {
        table.cut(k);
        changed = true;
      } else if (by >= 0 && entry === undefined) {
        table.add(tuple, byRule(by));
        changed = true;
      } else if (by >= 0) {
        entry.why = byRule(by);
      }
    });
    if (changed) {
      table.changed();
    }
  };

  // Counts a fact, whose key is `k`, `by` more ways, or fewer, in `counts`,
  // each as {tuple, n}: one that's counted no ways goes.
  function tally(counts, k, tuple, by) {
    var entry = counts.get(k);
    if (entry === undefined) {
      counts.set(k, { tuple: tuple, n: by });
    } else if (entry.n + by === 0) {
      counts.delete(k);
    } else {
      entry.n += by;
    }
  }

  // Counts the facts of rules' solutions, `found`, `by` more ways each, or
  // fewer, in `counts`.
  function tallied(found, by, counts) {
    for (var i = 0; i < found.length; i++) {
      tally(counts, key(found[i].tuple), found[i].tuple, by);
    }
    return counts;
  }

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
      if (this.tables[f.rel].derive(f.tuple, f.n, f.k)) {
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
  // one with the fewest, the first written on a tie. An atom that binds no
  // variable is a test, which it takes as soon as it's ready: the join
  // goes on once if a fact matches it, rather than once for each fact.
  // Then the rest of the body, in order.
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
      var old = delta !== null && delta.old !== undefined && delta.old[at] === true ? this.olds.get(atom.rel) : null;
      var tuples = this.bucket(atom, cols, env, fresh, old);
      if (!bindsAny(atom.pats, env)) {
        if (this.matchesOne(atom.pats, tuples, env)) {
          taken[at] = true;
          this.join(body, atoms, taken, left - 1, env, delta, out);
          taken[at] = false;
        }
        return;
      }
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
    if (bucket instanceof Table || bucket instanceof OldBucket) {
      bucket = bucket.tuples();
    }
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

  // Whether a pattern binds a variable: one that isn't bound yet.
  function binds(pat, env) {
    switch (pat[0]) {
      case "var":
        return env[pat[1]] === undefined;
      case "ctor":
        return bindsAny(pat[2], env);
    }
    return false;
  }

  function bindsAny(pats, env) {
    for (var i = 0; i < pats.length; i++) {
      if (binds(pats[i], env)) {
        return true;
      }
    }
    return false;
  }

  // Whether one of `tuples` matches an atom's patterns, which bind no
  // variable, visiting them in order, up to the first that does: of a
  // whole table's, which patterns of `_` alone all match, the first.
  Runtime.prototype.matchesOne = function (pats, tuples, env) {
    if (tuples instanceof Table || tuples instanceof OldBucket) {
      var any = pats.every(function (pat) {
        return pat[0] === "any";
      });
      if (any && tuples.length > 0) {
        this.visited++;
        return true;
      }
      tuples = tuples.tuples();
    }
    for (var i = 0; i < tuples.length; i++) {
      this.visited++;
      var trail = [];
      var matched = this.matchAll(pats, tuples[i], env, trail);
      unbind(env, trail);
      if (matched) {
        return true;
      }
    }
    return false;
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
  Runtime.prototype.bucket = function (atom, cols, env, fresh, old) {
    var table = this.tables[atom.rel];
    old = old === undefined ? null : old;
    if (cols.length === 0) {
      if (fresh !== null) {
        return fresh.tuples;
      }
      if (old !== null) {
        return new OldBucket(table, old, old.went, table.length - old.came.size + old.went.length);
      }
      return table;
    }
    var values = [];
    for (var i = 0; i < cols.length; i++) {
      values.push(this.keyValue(atom.pats[cols[i]], env));
    }
    var k = key(values);
    var map;
    if (fresh !== null) {
      map = indexOn(fresh.indexes, cols, function () {
        return fresh.tuples;
      });
    } else {
      map = table.index(cols);
    }
    var bucket = map.get(k);
    bucket = bucket === undefined ? NONE : bucket;
    if (fresh !== null || old === null) {
      return bucket;
    }
    var came = indexOn(old.cameIndexes, cols, function () {
      return listed(old.came);
    }).get(k);
    var went = indexOn(old.wentIndexes, cols, function () {
      return old.went;
    }).get(k);
    came = came === undefined ? NONE : came;
    went = went === undefined ? NONE : went;
    return new OldBucket(bucket, old, went, bucket.length - came.length + went.length);
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
        if (!this.matchesOne(lit.pats, this.bucket(lit, lit.key, env, null), env)) {
          this.rest(body, at + 1, env, out);
        }
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

  // Facts that agree on their relation's key agree on the rest: checked
  // where the relation's version has moved since its keys last held, by
  // how many of its key's values more than one fact has (Table.keyCount).
  // Of a relation whose don't, the error names the first two that clash, in
  // order, as Rust's does.
  Runtime.prototype.checkKeys = function () {
    var rels = this.prog.rels;
    for (var r = 0; r < rels.length; r++) {
      var version = this.tables[r].version;
      if (rels[r].key.length === 0 || this.held[r] === version) {
        continue;
      }
      if (this.tables[r].keyCount(rels[r].key).clashes > 0) {
        this.keyError(r, rels[r].key);
      }
      this.held[r] = version;
    }
  };

  // The error for a relation whose key `cols` doesn't hold: its first two
  // facts, in order, that clash; or, if none do, the count of its key's
  // values that said so is wrong.
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
    fail("the facts of " + this.prog.rels[r].name + " by their key were miscounted (a bug in lem.js)");
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
    key: key,
    numText: numText,
    quote: quote
  };
})();
