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

  // Prose: text as Markdown writes it, as elements, never as HTML, which
  // shows as written. Its blocks: paragraphs; headings, by #; rules;
  // quotes, by >; lists, by -, * or +, or by 1. or 1); fenced code, by ```
  // or ~~~, a diff's lines as a diff's; and tables, as GitHub writes them.
  // Its inlines: code, emphasis, strong, struck, links, autolinks and bare
  // URLs, and backslash's escapes; a line that ends in two spaces or a
  // backslash breaks. A link goes to one of the host's own paths, or to
  // http, https or mailto, in a new tab, without a referrer; any other
  // shows as text.
  function prose(w, view) {
    var e = element(view, "div");
    var shown = null;
    return made(w, e, null, null, function () {
      if (shown !== w.fields.text) {
        e.textContent = "";
        var lines = w.fields.text.replace(/\r\n?/g, "\n").split("\n").map(function (line) {
          return line.replace(/^\t+/, function (tabs) {
            return "    ".repeat(tabs.length);
          });
        });
        mdBlocks(view, e, mdParse(lines), false);
        shown = w.fields.text;
      }
      return ["g-prose"];
    });
  }

  var FENCE = /^( {0,3})(`{3,}|~{3,})[ \t]*([^`\s]*)[^`]*$/;
  var HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
  var RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  var QUOTE = /^ {0,3}> ?(.*)$/;
  var ITEM = /^( {0,3})([-*+]|\d{1,9}[.)])(?:([ \t]+)(.*))?$/;
  var BLANK = /^[ \t]*$/;
  var DELIMITER = /^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;
  // How deep emphasis and links nest at most, past which their text is
  // plain.
  var MD_DEPTH = 32;
  // What matches at a place in a text, from its lastIndex: never in a copy
  // of the text's rest, which an engine may flatten whole for each match,
  // as QuickJS does, so that a text of many takes no longer than another.
  var TICKS = /`+/y;
  var AUTOLINK = /<((?:https?:\/\/|mailto:)[^\s<>]*)>/iy;
  var BARE = /https?:\/\/[^\s<>`]+/iy;
  var LINK_TAIL = /[ \t]*(?:<([^<>\n]*)>|((?:[^\s()\\]|\\.|\((?:[^\s()\\]|\\.)*\))*))(?:[ \t]+(?:"[^"]*"|'[^']*'|\([^)]*\)))?[ \t]*\)/y;

  // Whether a line starts a block that ends a paragraph.
  function mdStarts(line) {
    return FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || ITEM.test(line);
  }

  // Markdown's blocks, from its lines: each {type, ...}.
  function mdParse(lines) {
    var out = [];
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];
      var m;
      if (BLANK.test(line)) {
        i++;
      } else if ((m = FENCE.exec(line)) !== null) {
        var close = new RegExp("^ {0,3}" + (m[2][0] === "`" ? "`" : "~") + "{" + m[2].length + ",}[ \\t]*$");
        var indent = new RegExp("^ {0," + m[1].length + "}");
        var body = [];
        i++;
        while (i < lines.length && !close.test(lines[i])) {
          body.push(lines[i].replace(indent, ""));
          i++;
        }
        i++;
        out.push({ type: "code", info: m[3].toLowerCase(), text: body.join("\n") });
      } else if ((m = HEADING.exec(line)) !== null) {
        out.push({ type: "heading", level: m[1].length, text: m[2] || "" });
        i++;
      } else if (RULE.test(line)) {
        out.push({ type: "rule" });
        i++;
      } else if (QUOTE.test(line)) {
        var quoted = [];
        while (i < lines.length && (m = QUOTE.exec(lines[i])) !== null) {
          quoted.push(m[1]);
          i++;
        }
        out.push({ type: "quote", blocks: mdParse(quoted) });
      } else if (ITEM.test(line)) {
        i = mdList(lines, i, out);
      } else if (line.indexOf("|") >= 0 && i + 1 < lines.length && lines[i + 1].indexOf("-") >= 0 && DELIMITER.test(lines[i + 1])) {
        i = mdTable(lines, i, out);
      } else {
        var para = [line];
        i++;
        while (i < lines.length && !BLANK.test(lines[i]) && !mdStarts(lines[i])) {
          para.push(lines[i]);
          i++;
        }
        out.push({ type: "paragraph", lines: para });
      }
    }
    return out;
  }

  // A list, from line i: its items, each its blocks, and whether it's
  // loose, its items apart, which shows each paragraph as one. Gives the
  // line after it.
  function mdList(lines, i, out) {
    var first = ITEM.exec(lines[i]);
    var ordered = /\d/.test(first[2]);
    var it = { type: "list", ordered: ordered, start: ordered ? parseInt(first[2], 10) : 1, items: [], loose: false };
    while (i < lines.length) {
      var m = ITEM.exec(lines[i]);
      if (m === null || /\d/.test(m[2]) !== ordered) {
        break;
      }
      var gap = m[3] === undefined ? 1 : m[3].length > 4 ? 1 : m[3].length;
      var width = m[1].length + m[2].length + gap;
      var body = [m[3] === undefined ? "" : m[3].slice(gap) + m[4]];
      var blank = false;
      i++;
      while (i < lines.length) {
        var next = lines[i];
        if (BLANK.test(next)) {
          body.push("");
          blank = true;
        } else if (/^ */.exec(next)[0].length >= width) {
          body.push(next.slice(width));
          blank = false;
        } else if (!blank && !mdStarts(next)) {
          body.push(next.replace(/^ +/, ""));
        } else {
          break;
        }
        i++;
      }
      while (body.length > 0 && body[body.length - 1] === "") {
        body.pop();
        it.loose = it.loose || (i < lines.length && ITEM.test(lines[i]));
      }
      it.loose = it.loose || body.indexOf("") >= 0 && mdParse(body).length > 1;
      it.items.push(mdParse(body));
    }
    out.push(it);
    return i;
  }

  // A table, from line i: its head, its columns' alignments, and its rows.
  // Gives the line after it.
  function mdTable(lines, i, out) {
    var head = mdCells(lines[i]);
    var aligns = mdCells(lines[i + 1]).map(function (cell) {
      var left = cell[0] === ":";
      var right = cell[cell.length - 1] === ":";
      return left && right ? "center" : right ? "right" : left ? "left" : "";
    });
    var rows = [];
    i += 2;
    while (i < lines.length && !BLANK.test(lines[i]) && lines[i].indexOf("|") >= 0) {
      rows.push(mdCells(lines[i]));
      i++;
    }
    out.push({ type: "table", head: head, aligns: aligns, rows: rows });
    return i;
  }

  // A table row's cells: its text between unescaped pipes.
  function mdCells(line) {
    var row = line.trim();
    if (row[0] === "|") {
      row = row.slice(1);
    }
    if (row[row.length - 1] === "|" && row[row.length - 2] !== "\\") {
      row = row.slice(0, -1);
    }
    var found = [];
    var cell = "";
    for (var k = 0; k < row.length; k++) {
      if (row[k] === "\\" && row[k + 1] === "|") {
        cell += "|";
        k++;
      } else if (row[k] === "|") {
        found.push(cell.trim());
        cell = "";
      } else {
        cell += row[k];
      }
    }
    found.push(cell.trim());
    return found;
  }

  // Blocks, as elements in `parent`; with `tight`, a paragraph is its
  // inlines alone, as in a list's item.
  function mdBlocks(view, parent, found, tight) {
    found.forEach(function (b) {
      var e;
      if (b.type === "paragraph") {
        e = tight ? parent : element(view, "p");
        var marked = b.lines.map(function (line, at) {
          var last = at === b.lines.length - 1;
          return !last && /( {2,}|\\)$/.test(line) ? line.replace(/( {2,}|\\)$/, "") + "\u0000" : line;
        });
        mdInline(view, e, marked.join("\n").replace(/\u0000\n/g, "\u0000"));
      } else if (b.type === "heading") {
        e = element(view, "div", "g-prose-heading g-prose-h" + b.level);
        e.setAttribute("role", "heading");
        e.setAttribute("aria-level", String(Math.min(b.level + 2, 6)));
        mdInline(view, e, b.text);
      } else if (b.type === "code") {
        e = element(view, "pre", b.info === "diff" ? "g-code g-diff" : "g-code");
        if (b.info === "diff") {
          diff(view, e, b.text);
        } else {
          e.textContent = b.text;
        }
      } else if (b.type === "rule") {
        e = element(view, "hr");
      } else if (b.type === "quote") {
        e = element(view, "blockquote");
        mdBlocks(view, e, b.blocks, false);
      } else if (b.type === "list") {
        e = element(view, b.ordered ? "ol" : "ul");
        if (b.ordered && b.start !== 1) {
          e.setAttribute("start", String(b.start));
        }
        b.items.forEach(function (item) {
          var li = element(view, "li");
          mdBlocks(view, li, item, !b.loose);
          e.appendChild(li);
        });
      } else {
        e = element(view, "table");
        var row = function (cellsOf, tag, into) {
          var tr = element(view, "tr");
          b.head.forEach(function (_, k) {
            var cell = element(view, tag, b.aligns[k] ? "g-align-" + b.aligns[k] : null);
            mdInline(view, cell, k < cellsOf.length ? cellsOf[k] : "");
            tr.appendChild(cell);
          });
          into.appendChild(tr);
        };
        var thead = element(view, "thead");
        var tbody = element(view, "tbody");
        row(b.head, "th", thead);
        b.rows.forEach(function (cellsOf) {
          row(cellsOf, "td", tbody);
        });
        e.appendChild(thead);
        if (b.rows.length > 0) {
          e.appendChild(tbody);
        }
      }
      if (e !== parent) {
        parent.appendChild(e);
      }
    });
  }

  // Inline Markdown, as nodes in `parent`: "\u0000" is a line's break.
  function mdInline(view, parent, source, depth) {
    depth = depth || 0;
    if (depth > MD_DEPTH) {
      parent.appendChild(view.doc.createTextNode(source));
      return;
    }
    // What this text's links and emphasis have found (mdLinkAt, mdEmphasis).
    var memo = { match: null, failed: {} };
    var plain = "";
    var flush = function () {
      if (plain !== "") {
        parent.appendChild(view.doc.createTextNode(plain));
        plain = "";
      }
    };
    var i = 0;
    while (i < source.length) {
      var c = source[i];
      var found = null;
      if (c === "\\" && /[!-\/:-@\[-`{-~]/.test(source[i + 1] || "")) {
        plain += source[i + 1];
        i += 2;
        continue;
      }
      if (c === "\u0000") {
        flush();
        parent.appendChild(element(view, "br"));
        i++;
        continue;
      }
      if (c === "`") {
        var run = mdRun(source, i);
        var end = mdTicks(source, i + run, run);
        if (end < 0) {
          plain += source.slice(i, i + run);
          i += run;
          continue;
        }
        var inner = source.slice(i + run, end).replace(/\n/g, " ");
        if (/^ [\s\S]* $/.test(inner) && inner.trim() !== "") {
          inner = inner.slice(1, -1);
        }
        flush();
        var code = element(view, "code", "g-prose-code");
        code.textContent = inner;
        parent.appendChild(code);
        i = end + run;
        continue;
      }
      if (c === "[" || (c === "!" && source[i + 1] === "[")) {
        found = mdLinkAt(source, c === "!" ? i + 1 : i, memo);
        if (found !== null) {
          flush();
          mdLink(view, parent, found.url, found.text, false, depth);
          i = found.end;
          continue;
        }
      }
      if (c === "<") {
        AUTOLINK.lastIndex = i;
        found = AUTOLINK.exec(source);
        if (found !== null) {
          flush();
          mdLink(view, parent, found[1], found[1], true, depth);
          i += found[0].length;
          continue;
        }
      }
      if ((c === "h" || c === "H") && /^https?:\/\/\S/i.test(source.slice(i, i + 9)) && (i === 0 || /[\s(\[<*_~"']/.test(source[i - 1]))) {
        var url = mdBare(source, i);
        flush();
        mdLink(view, parent, url, url, true, depth);
        i += url.length;
        continue;
      }
      if (c === "*" || c === "_" || c === "~") {
        found = mdEmphasis(source, i, memo);
        if (found !== null) {
          flush();
          var em = element(view, found.tag);
          mdInline(view, em, found.inner, depth + 1);
          parent.appendChild(em);
          i = found.end;
          continue;
        }
      }
      plain += c;
      i++;
    }
    flush();
  }

  // How many backticks run from `at`, where one is.
  function mdRun(source, at) {
    TICKS.lastIndex = at;
    return TICKS.exec(source)[0].length;
  }

  // Where the run of exactly `n` backticks after `from` starts, or -1.
  function mdTicks(source, from, n) {
    var k = from;
    while (k < source.length) {
      var at = source.indexOf("`", k);
      if (at < 0) {
        return -1;
      }
      var run = mdRun(source, at);
      if (run === n) {
        return at;
      }
      k = at + run;
    }
    return -1;
  }

  // A link at `i`, at its `[`: {text, url, end}, or null. Its `]` is the
  // one that `memo` matches it with (mdBrackets), found once for the text.
  function mdLinkAt(source, i, memo) {
    if (memo.match === null) {
      memo.match = mdBrackets(source);
    }
    var k = memo.match[i];
    if (k < 0 || source[k + 1] !== "(") {
      return null;
    }
    LINK_TAIL.lastIndex = k + 2;
    var m = LINK_TAIL.exec(source);
    if (m === null) {
      return null;
    }
    var url = m[1] !== undefined ? m[1] : m[2];
    return { text: source.slice(i + 1, k), url: url.replace(/\\([!-\/:-@\[-`{-~])/g, "$1"), end: k + 2 + m[0].length };
  }

  // Each `[` of `source` with the `]` that closes it, as far as they nest,
  // past escapes and code spans: its index, or -1; in one pass, so that a
  // text of many `[` that nothing closes takes no longer than another.
  function mdBrackets(source) {
    var match = new Int32Array(source.length).fill(-1);
    var open = [];
    for (var k = 0; k < source.length; k++) {
      var c = source[k];
      if (c === "\\") {
        k++;
      } else if (c === "`") {
        var run = mdRun(source, k);
        var end = mdTicks(source, k + run, run);
        k = end < 0 ? k + run - 1 : end + run - 1;
      } else if (c === "[") {
        open.push(k);
      } else if (c === "]" && open.length > 0) {
        match[open.pop()] = k;
      }
    }
    return match;
  }

  // The bare URL at `at` in `source`, without the punctuation that ends a
  // sentence, or a parenthesis that it didn't open.
  function mdBare(source, at) {
    BARE.lastIndex = at;
    var url = BARE.exec(source)[0];
    var opened = url.split("(").length - 1;
    var closed = url.split(")").length - 1;
    var end = url.length;
    while (end > 0) {
      var last = url[end - 1];
      if (/[.,;:!?'"*_~]/.test(last)) {
        end--;
      } else if (last === ")" && opened < closed) {
        end--;
        closed--;
      } else {
        break;
      }
    }
    return url.slice(0, end);
  }

  // Emphasis at `i`: *, _, **, __ or ~~, as far as its match: {tag, inner,
  // end}, or null. An _ within a word is none. Where the search for a
  // delimiter's match goes on depends on where it is alone, so `memo` marks
  // the places that a search that found none went through, and another
  // search for that delimiter stops at one: a text of many delimiters that
  // nothing matches takes no longer than another.
  function mdEmphasis(source, i, memo) {
    var c = source[i];
    var double = source[i + 1] === c;
    if (c === "~" && !double) {
      return null;
    }
    var delim = double ? c + c : c;
    var from = i + delim.length;
    if (from >= source.length || /\s/.test(source[from]) || (c === "_" && i > 0 && /[A-Za-z0-9]/.test(source[i - 1]))) {
      return null;
    }
    var marks = memo.failed[delim];
    if (marks === undefined) {
      marks = memo.failed[delim] = new Uint8Array(source.length);
    }
    var seen = [];
    for (var k = from + 1; k < source.length && marks[k] === 0; k++) {
      seen.push(k);
      var ch = source[k];
      if (ch === "\\") {
        k++;
      } else if (ch === "`") {
        var run = mdRun(source, k);
        var end = mdTicks(source, k + run, run);
        k = end < 0 ? k + run - 1 : end + run - 1;
      } else if (source.slice(k, k + delim.length) === delim && !/\s/.test(source[k - 1])) {
        if (!double && source[k + 1] === c) {
          k++;
        } else if (!(c === "_" && /[A-Za-z0-9]/.test(source[k + delim.length] || ""))) {
          return { tag: c === "~" ? "del" : double ? "strong" : "em", inner: source.slice(from, k), end: k + delim.length };
        }
      }
    }
    seen.forEach(function (at) {
      marks[at] = 1;
    });
    return null;
  }

  // A link to `url`, of `label`, which is inlines, but with `raw`, as a
  // URL's own text: to one of the host's own paths, or to http, https or
  // mailto, in a new tab, without a referrer; else text, with its URL.
  function mdLink(view, parent, url, label, raw, depth) {
    var safe = own(url) || /^(https?:\/\/|mailto:)\S+$/i.test(url);
    if (!safe) {
      mdInline(view, parent, label, depth + 1);
      parent.appendChild(view.doc.createTextNode(" (" + url + ")"));
      return;
    }
    var a = element(view, "a", "g-prose-link");
    a.setAttribute("href", url);
    if (!own(url)) {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    }
    if (raw) {
      a.textContent = label;
    } else {
      mdInline(view, a, label, depth + 1);
    }
    parent.appendChild(a);
  }

  // A strip of tabs: a tab list, to a screen reader.
  function tabs(w, view) {
    var e = element(view, "div");
    e.setAttribute("role", "tablist");
    return made(w, e, e, null, function () {
      return ["g-tabs"];
    });
  }

  // The tab that a drag carries, {view, id}, while it lasts, or null.
  var dragging = null;

  // A tab of a strip: its text, which is the tab, selected or not, to a
  // screen reader, and takes the keyboard's focus; and, if it closes, a
  // button that sends Dismissed. A click on the tab, but on that button, or
  // Return or space on its text, sends Chosen. One that moves is dragged,
  // by the browser's drag and drop, which a button within would stop; and
  // dropped on another that moves, of its strip, it sends Moved, and
  // data-drop marks that one while it's over it. g-active marks the active
  // one.
  function tabItem(w, view) {
    var e = element(view, "span");
    var label = element(view, "span", "g-tab-label");
    var close = element(view, "button", "g-tab-close");
    label.setAttribute("role", "tab");
    label.setAttribute("tabindex", "0");
    close.setAttribute("type", "button");
    takesReturn(label);
    takesReturn(close);
    text(close, "\u00d7");
    e.appendChild(label);
    e.addEventListener("click", function () {
      view.send(["Chosen", w.id]);
    });
    label.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        view.send(["Chosen", w.id]);
      }
    });
    close.addEventListener("click", function (event) {
      event.stopPropagation();
      view.send(["Dismissed", w.id]);
    });
    var onto = function () {
      return dragging !== null && dragging.view === view && w.fields.moves && lem.key(dragging.id) !== lem.key(w.id);
    };
    e.addEventListener("dragstart", function (event) {
      if (!w.fields.moves) {
        event.preventDefault();
        return;
      }
      dragging = { view: view, id: w.id };
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", w.fields.text);
      }
    });
    e.addEventListener("dragover", function (event) {
      if (onto()) {
        event.preventDefault();
        attr(e, "data-drop", "");
      }
    });
    e.addEventListener("dragleave", function () {
      attr(e, "data-drop", null);
    });
    e.addEventListener("drop", function (event) {
      attr(e, "data-drop", null);
      if (onto()) {
        event.preventDefault();
        var from = dragging.id;
        dragging = null;
        view.send(["Moved", from, w.id]);
      }
    });
    e.addEventListener("dragend", function () {
      dragging = null;
    });
    return made(w, e, null, null, function () {
      text(label, w.fields.text);
      attr(label, "aria-selected", w.fields.active ? "true" : "false");
      attr(close, "aria-label", "close " + w.fields.text);
      attr(e, "draggable", w.fields.moves ? "true" : null);
      if (w.fields.closes && close.parentNode !== e) {
        e.appendChild(close);
      } else if (!w.fields.closes && close.parentNode === e) {
        e.removeChild(close);
      }
      return w.fields.active ? ["g-tab", "g-active"] : ["g-tab"];
    });
  }

  // A dot, as a light of its hue (glui.css), which says its text as an
  // image's label, and in its tooltip, after any other.
  function dot(w, view) {
    var e = element(view, "span");
    e.setAttribute("role", "img");
    var it = made(w, e, null, null, function () {
      attr(e, "aria-label", w.fields.text);
      return ["g-dot", "g-hue-" + kebab(w.fields.hue[0])];
    });
    var update = it.update;
    it.update = function () {
      update();
      var tip = e.getAttribute("title");
      attr(e, "title", tip === null ? w.fields.text : w.fields.text + "\n" + tip);
    };
    return it;
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
  // the program knows. With `star`, a star comes first, a button that sends
  // Starred, as set or cleared, and that opens or folds nothing.
  function entry(w, view) {
    var e = element(view, "div");
    var star = element(view, "button", "g-entry-star");
    star.setAttribute("type", "button");
    takesReturn(star);
    star.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      view.send(["Starred", w.id, !w.fields.starred]);
    });
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
      if (star.parentNode !== null) {
        star.parentNode.removeChild(star);
      }
      e.textContent = "";
      var head = element(view, fold ? "summary" : "div", "g-entry-head");
      if (fold) {
        takesReturn(head);
      }
      if (w.fields.star) {
        head.appendChild(star);
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
      if (folds !== w.fields.fold || (star.parentNode !== null) !== w.fields.star) {
        build(w.fields.fold);
      }
      text(star, w.fields.starred ? "\u2605" : "\u2606");
      attr(star, "aria-pressed", w.fields.starred ? "true" : "false");
      attr(star, "aria-label", w.fields.starred ? "starred" : "star");
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
    Prose: prose,
    Dot: dot,
    Tabs: tabs,
    TabItem: tabItem,
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
