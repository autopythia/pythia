// glui.js: glui's web look, on glem.js (src/glem/glem.js). It gives a
// recipe for each of glui's widget kinds (widgets.lem), which makes the
// widget a native element (button, a, details, table, pre, progress) with
// classes, `g-` and its kind's name, which the looks' stylesheets draw
// (web/glui.css, then web/native.css or web/log.css); the messages that
// widgets send (Edited, Toggled); what Disabled and Tooltip say of each
// widget; and the theme's values, as CSS variables. Text goes into the page
// as text, never as HTML, and a link goes only to one of the host's own
// paths. Like lem.js and glem.js, it's a plain script that defines one
// global, `glui`, and runs in a browser or in the tests' stand-in DOM
// (tools/glui_test.js, `make js-test`).
var glui = (function () {
  "use strict";

  // A name in CamelCase as CSS writes names: PanelBack, panel-back.
  function kebab(name) {
    return name.replace(/[A-Z]/g, function (c, at) {
      return (at > 0 ? "-" : "") + c.toLowerCase();
    });
  }

  // An element of the view's document, with a class, if one is given.
  function element(view, tag, name) {
    var e = view.doc.createElement(tag);
    if (name) {
      e.setAttribute("class", name);
    }
    return e;
  }

  // Sets an attribute, or removes it, for null.
  function attr(e, name, value) {
    if (value === null) {
      e.removeAttribute(name);
    } else if (e.getAttribute(name) !== value) {
      e.setAttribute(name, value);
    }
  }

  // Sets an element's text, unless it's that already.
  function text(e, value) {
    if (e.textContent !== value) {
      e.textContent = value;
    }
  }

  // An element that acts on Return itself, as a browser's button, link or
  // summary does: the view doesn't hear the key, so it sends nothing that
  // glem's Key facts give for it.
  function takesReturn(e) {
    e.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        event.stopPropagation();
      }
    });
  }

  // A tone's class (widgets.lem's Tone): none for Normal.
  function tone(value) {
    return value[0] === "Normal" ? [] : ["g-tone-" + kebab(value[0])];
  }

  // What a recipe gives the view (glem.View): the widget's element; the
  // element that holds its children, or null; and an update, which calls
  // `show`, which shows the widget's fields and gives its classes, then
  // shows what Disabled and Tooltip say of it (state): it's disabled, by a
  // class, and on `control`, the element that takes clicks and keys, if
  // it has one, by the attribute; and its title gives its tooltip, then
  // why it's disabled, a line each.
  function made(w, e, children, control, show) {
    return {
      element: e,
      children: children,
      update: function () {
        var classes = show();
        var why = w.state === null ? [] : w.state.disabled;
        var tip = w.state === null ? [] : [w.state.tooltip];
        if (why.length > 0) {
          classes.push("g-disabled");
        }
        attr(e, "class", classes.join(" "));
        var title = tip.concat(why).filter(function (line) {
          return line !== "";
        });
        attr(e, "title", title.length > 0 ? title.join("\n") : null);
        if (control !== null) {
          attr(control, "disabled", why.length > 0 ? "" : null);
        }
      }
    };
  }

  // Containers, and their flags.

  // A window, which is the page: its title is the document's.
  function window_(w, view) {
    var e = element(view, "main");
    return made(w, e, e, null, function () {
      view.doc.title = w.fields.title;
      return ["g-window"];
    });
  }

  // A HUD, which floats at its place, such as g-top-right.
  function hud(w, view) {
    var e = element(view, "aside");
    return made(w, e, e, null, function () {
      return ["g-hud", "g-" + kebab(w.fields.place[0])];
    });
  }

  function panel(w, view) {
    var e = element(view, "section");
    var title = element(view, "div", "g-panel-title");
    var body = element(view, "div", "g-panel-body");
    e.appendChild(title);
    e.appendChild(body);
    return made(w, e, body, null, function () {
      text(title, w.fields.title);
      return ["g-panel"];
    });
  }

  // A row, a column or a box, which holds its widgets: as one group, with
  // g-aligned, if it's aligned.
  function group(name) {
    return function (w, view) {
      var e = element(view, "div");
      return made(w, e, e, null, function () {
        return w.fields.aligned ? [name, "g-aligned"] : [name];
      });
    };
  }

  // A table, whose children are its rows.
  function table(w, view) {
    var e = element(view, "table");
    var body = view.doc.createElement("tbody");
    e.appendChild(body);
    return made(w, e, body, null, function () {
      return ["g-table"];
    });
  }

  // A table's row, whose children are its cells, a widget each, which the
  // looks show as cells.
  function tableRow(w, view) {
    var e = element(view, "tr");
    return made(w, e, e, null, function () {
      return w.fields.header ? ["g-tablerow", "g-header"] : ["g-tablerow"];
    });
  }

  // Text.

  function label(w, view) {
    var e = element(view, "span");
    return made(w, e, null, null, function () {
      text(e, w.fields.text);
      return ["g-label"].concat(tone(w.fields.tone));
    });
  }

  // A heading, of level 1 to 6, as ARIA says, so that its level can change.
  function heading(w, view) {
    var e = element(view, "div");
    e.setAttribute("role", "heading");
    return made(w, e, null, null, function () {
      var level = Math.min(Math.max(w.fields.level, 1), 6);
      attr(e, "aria-level", String(level));
      text(e, w.fields.text);
      return ["g-heading", "g-level-" + level];
    });
  }

  // Whether `to` is one of the host's own paths: from the root of its
  // origin, as `/c/ui-0`; not `//` or `/\`, which a browser takes to be
  // another host; and without spaces or controls.
  function own(to) {
    return /^\/(?![\/\\])[^\x00-\x20\x7f\\]*$/.test(to);
  }

  // A link, to one of the host's own paths: one to anywhere else has no
  // `href`, and the class g-refused. A disabled link goes nowhere.
  function link(w, view) {
    var e = element(view, "a");
    takesReturn(e);
    e.addEventListener("click", function (event) {
      if (w.state !== null && w.state.disabled.length > 0) {
        event.preventDefault();
      }
    });
    return made(w, e, null, null, function () {
      text(e, w.fields.text);
      var off = w.state !== null && w.state.disabled.length > 0;
      attr(e, "aria-disabled", off ? "true" : null);
      if (own(w.fields.to)) {
        attr(e, "href", w.fields.to);
        return ["g-link"];
      }
      attr(e, "href", null);
      return ["g-link", "g-refused"];
    });
  }

  // A diff's line's class, by how it starts, as wui's pages show diffs; or
  // "" for one that's neither added, deleted, a hunk nor a file's name.
  function diffClass(line) {
    if (/^(\+\+\+|---|\*\*\* |diff --git)/.test(line)) {
      return "g-diff-file";
    }
    if (line.indexOf("@@") === 0) {
      return "g-diff-hunk";
    }
    if (line.charAt(0) === "+") {
      return "g-diff-add";
    }
    return line.charAt(0) === "-" ? "g-diff-del" : "";
  }

  // Code, kept as written; with `diff`, a diff, whose lines that are added,
  // deleted, hunks or files' names each have a class: such a line is an
  // element, and the lines between them text.
  function code(w, view) {
    var e = element(view, "pre");
    var shown = null;
    return made(w, e, null, null, function () {
      var now = [w.fields.text, w.fields.diff];
      if (shown === null || shown[0] !== now[0] || shown[1] !== now[1]) {
        e.textContent = w.fields.diff ? "" : w.fields.text;
        if (w.fields.diff) {
          diff(view, e, w.fields.text);
        }
        shown = now;
      }
      return w.fields.diff ? ["g-code", "g-diff"] : ["g-code"];
    });
  }

  function diff(view, e, value) {
    var lines = value.split("\n");
    var plain = "";
    for (var i = 0; i < lines.length; i++) {
      var line = i < lines.length - 1 ? lines[i] + "\n" : lines[i];
      var name = diffClass(lines[i]);
      if (name === "") {
        plain += line;
        continue;
      }
      if (plain !== "") {
        e.appendChild(view.doc.createTextNode(plain));
        plain = "";
      }
      var span = element(view, "span", name);
      span.textContent = line;
      e.appendChild(span);
    }
    if (plain !== "") {
      e.appendChild(view.doc.createTextNode(plain));
    }
  }

  // A log's entry: its mark, label and text, then its children. Its mark
  // links to `to`, if that's one of the host's own paths, as a link does;
  // and to nothing for "". With `fold`, it's a details element, whose
  // summary shows the first three, and which sends Toggled when it opens or
  // closes; its children show when it's open. It opens or closes as `open`
  // says when it's made, and whenever `open` changes, but not otherwise,
  // so that it stays as it was opened or closed in between; and the toggle
  // event that follows, which a browser fires later, sends nothing, since
  // the program knows.
  function entry(w, view) {
    var e = element(view, "div");
    var mark = element(view, "a");
    takesReturn(mark);
    var label_ = element(view, "span", "g-entry-label");
    var gist = element(view, "span", "g-entry-text");
    var body = element(view, "div", "g-entry-body");
    var folds = null;
    // The details element, with fold; the `open` last shown in it, or null
    // for none yet; and whether it's open as `open` made it, until its
    // toggle event comes, or null.
    var details = null;
    var shown = null;
    var echo = null;
    // Makes the entry's parts anew, for `fold`, with the same mark, label,
    // text and body, the body holding the children still.
    function build(fold) {
      e.textContent = "";
      var head = element(view, fold ? "summary" : "div", "g-entry-head");
      if (fold) {
        takesReturn(head);
      }
      head.appendChild(mark);
      head.appendChild(label_);
      head.appendChild(gist);
      var holder = e;
      details = null;
      shown = null;
      echo = null;
      if (fold) {
        var it = view.doc.createElement("details");
        it.addEventListener("toggle", function () {
          var mine = echo !== null && it.open === echo;
          echo = null;
          if (!mine) {
            view.send(["Toggled", w.id, it.open]);
          }
        });
        e.appendChild(it);
        holder = details = it;
      }
      holder.appendChild(head);
      holder.appendChild(body);
      folds = fold;
    }
    return made(w, e, body, null, function () {
      if (folds !== w.fields.fold) {
        build(w.fields.fold);
      }
      if (details !== null && shown !== w.fields.open) {
        shown = w.fields.open;
        if (details.open !== shown) {
          echo = shown;
          details.open = shown;
        }
      }
      var to = w.fields.to;
      text(mark, w.fields.mark);
      attr(mark, "class", to === "" || own(to) ? "g-entry-mark" : "g-entry-mark g-refused");
      attr(mark, "href", to !== "" && own(to) ? to : null);
      text(label_, w.fields.label);
      text(gist, w.fields.text);
      return ["g-entry"].concat(w.fields.fold ? ["g-fold"] : [], tone(w.fields.tone));
    });
  }

  // Controls.

  // A button, or a list's row, which sends its message when clicked.
  function sends(name) {
    return function (w, view) {
      var e = element(view, "button");
      e.setAttribute("type", "button");
      takesReturn(e);
      e.addEventListener("click", function () {
        view.send(w.fields.on);
      });
      return made(w, e, null, e, function () {
        text(e, w.fields.text);
        return [name];
      });
    };
  }

  // A checkbox, which sends Toggled. It shows the value that its widget
  // gives: after a click, what the tick that it sends gives.
  function toggle(w, view) {
    var e = element(view, "label");
    var box = view.doc.createElement("input");
    var span = view.doc.createElement("span");
    box.setAttribute("type", "checkbox");
    e.appendChild(box);
    e.appendChild(span);
    box.addEventListener("change", function () {
      var value = box.checked;
      box.checked = w.fields.value;
      view.send(["Toggled", w.id, value]);
    });
    return made(w, e, null, box, function () {
      box.checked = w.fields.value;
      text(span, w.fields.text);
      return ["g-toggle"];
    });
  }

  // A text field, which sends Edited as it's edited. Its text changes only
  // when its widget's does, and differs from what it holds, so that typing
  // keeps the caret. While an input method composes text, it sends
  // nothing, and keeps what it holds: it sends the text when the
  // composition ends.
  function field(w, view) {
    var e = element(view, "input");
    var composing = false;
    e.setAttribute("type", "text");
    function edited() {
      view.send(["Edited", w.id, e.value]);
    }
    e.addEventListener("compositionstart", function () {
      composing = true;
    });
    e.addEventListener("compositionend", function () {
      composing = false;
      edited();
    });
    e.addEventListener("input", function (event) {
      if (!composing && !event.isComposing) {
        edited();
      }
    });
    return made(w, e, null, e, function () {
      if (!composing && e.value !== w.fields.text) {
        e.value = w.fields.text;
      }
      return ["g-field"];
    });
  }

  // A progress bar, from 0 to 1.
  function progress(w, view) {
    var e = element(view, "progress");
    e.setAttribute("max", "1");
    return made(w, e, null, null, function () {
      attr(e, "value", String(Math.min(Math.max(w.fields.frac, 0), 1)));
      return ["g-progress"];
    });
  }

  // The recipes, by kind: every kind and surface of widgets.lem's.
  var RECIPES = {
    Window: window_,
    Hud: hud,
    Panel: panel,
    Row: group("g-row"),
    Column: group("g-column"),
    Box: group("g-box"),
    Table: table,
    TableRow: tableRow,
    Label: label,
    Heading: heading,
    Button: sends("g-button"),
    Toggle: toggle,
    Field: field,
    ListRow: sends("g-listrow"),
    Progress: progress,
    Link: link,
    Entry: entry,
    Code: code
  };

  // glui's recipes, for glem.View: a new object, which a page may add to.
  function recipes() {
    var all = {};
    Object.keys(RECIPES).forEach(function (kind) {
      all[kind] = RECIPES[kind];
    });
    return all;
  }

  // What glui shows of a widget besides its fields, for glem.View: after a
  // tick, a function of a widget's id that gives {disabled, tooltip}: why
  // it's disabled, the reasons in order, and its tooltip, "" for none; or
  // null, for neither. It reads the relations that `state.rels` names, so
  // that the view asks again only for the widgets whose facts there change.
  function state(runtime) {
    var prog = runtime.prog;
    var disabled = new Map();
    var tooltips = new Map();
    runtime.relFacts(prog.rel("Disabled")).forEach(function (fact) {
      var id = lem.key(fact.tuple[0]);
      if (!disabled.has(id)) {
        disabled.set(id, []);
      }
      disabled.get(id).push(fact.tuple[1]);
    });
    runtime.relFacts(prog.rel("Tooltip")).forEach(function (fact) {
      tooltips.set(lem.key(fact.tuple[0]), fact.tuple[1]);
    });
    return function (id) {
      var key = lem.key(id);
      var why = disabled.has(key) ? disabled.get(key) : [];
      var tip = tooltips.has(key) ? tooltips.get(key) : "";
      return why.length === 0 && tip === "" ? null : { disabled: why, tooltip: tip };
    };
  }
  state.rels = ["Disabled", "Tooltip"];

  // The metrics that are times, in milliseconds (theme.lem); the rest are
  // sizes, in points, which CSS calls px.
  var TIMES = ["TooltipDelay", "Collapse"];

  // Whether a color, as 0xRRGGBBAA, is dark: its luminance below half.
  function dark(rgba) {
    var r = (rgba >>> 24) & 255;
    var g = (rgba >>> 16) & 255;
    var b = (rgba >>> 8) & 255;
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 128;
  }

  // The theme as CSS, each [property, value]: first `color-scheme`, dark
  // or light as the stage is, by which a browser draws its own controls
  // and scroll bars; then a variable for each value: `--glui-`, then the
  // paint's or the metric's name, as CSS writes names: a color as
  // #rrggbbaa, a size in px, a time in ms, and a page's width of 0 as
  // none; then one for each tone (TonePaint), `--glui-`, the tone's name
  // and `-text`, its paint's color. They come from the theme's facts,
  // after a tick of lem's runtime.
  function theme(runtime) {
    var prog = runtime.prog;
    var values = [];
    var colors = new Map();
    runtime.relFacts(prog.rel("Color")).forEach(function (fact) {
      var hex = "#" + ("0000000" + fact.tuple[1].toString(16)).slice(-8);
      colors.set(fact.tuple[0][0], hex);
      values.push(["--glui-" + kebab(fact.tuple[0][0]), hex]);
      if (fact.tuple[0][0] === "Stage") {
        values.unshift(["color-scheme", dark(fact.tuple[1]) ? "dark" : "light"]);
      }
    });
    runtime.relFacts(prog.rel("Size")).forEach(function (fact) {
      var metric = fact.tuple[0][0];
      var size = fact.tuple[1];
      var value = TIMES.indexOf(metric) >= 0 ? size + "ms" : size + "px";
      if (metric === "PageWidth" && size === 0) {
        value = "none";
      }
      values.push(["--glui-" + kebab(metric), value]);
    });
    runtime.relFacts(prog.rel("TonePaint")).forEach(function (fact) {
      var paint = fact.tuple[1][0];
      if (!colors.has(paint)) {
        throw new lem.Error("the theme gives no Color(" + paint + ")");
      }
      values.push(["--glui-" + kebab(fact.tuple[0][0]) + "-text", colors.get(paint)]);
    });
    return values;
  }

  // Paints an element, such as the view's root, with the theme, which the
  // looks' stylesheets read.
  function paint(e, runtime) {
    theme(runtime).forEach(function (value) {
      e.style.setProperty(value[0], value[1]);
    });
  }

  // A view of a glui program's widgets in `root`: glem.js's, with glui's
  // recipes and state, for `host` (glem.View).
  function view(runtime, root, host) {
    return new glem.View(runtime, root, recipes(), host, state);
  }

  return {
    recipes: recipes,
    state: state,
    theme: theme,
    paint: paint,
    view: view,
    own: own
  };
})();
