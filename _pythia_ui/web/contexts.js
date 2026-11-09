// web/contexts.js: the host's part of pythia's contexts (lem/contexts.lem) in
// a browser, which the glui view of pythia's node server runs on its pages,
// for the root contexts and for a context's log
// (pythia/interaction/node/view.py). A page gives, on its <html> element,
// `data-facts`, the URL of its facts, JSON steps (src/lem/host.rs); and, for
// a log, `data-before`, `data-after` and `data-around`, the URLs of the rows
// before a line, after one and around one, `data-run`, the URL of a run's
// items, and `data-body`, the URL of an item's body, each with `{line}` or
// `{run}` in place of its line; `data-grow`, the URL of what the log gained
// since it had `{lines}` lines; `data-stars`, the URL that keeps its stars;
// and `data-pins`, the URL of its panel's pins; and, on each page,
// `data-store`, the URL of the settings that the server keeps for the view
// (pythia/interaction/node/store.py). The compiled program and glui's themes
// are beside this script, in compiled/. A page of the node server's also
// gives what its tabs need (tabs()): `data-index`, the URL of the index's
// facts; `data-tab`, of a context's tab, with `{id}`; `data-log` and
// `data-home`, the addresses of a context's page and the index's; and, for a
// log, `data-open`, its number. It shows the page in glui's widgets
// (src/glui/glui.js), with a tab for the index and for each context open, the
// strip of them at the top (lem/tabs.lem), where the page names them, in the
// look and the theme that its bar chooses, which the server keeps, and the
// browser too; a log from its end, or, in a tab, from where the session left
// it (the program's Restore), with every run folded or every run open, as the
// bar's Runs says (Opening), with its panel beside it, or not, as the bar's
// Panel says (Aside), and with an assistant's messages as Markdown writes
// them, or as written, as the bar's Text says (Formatting). It answers the
// program's LoadBefore(line) and LoadAfter(line) with rows, or with
// EarlierFailed(line, why) or LaterFailed(line, why); its LoadAround(line)
// with rows, or a status that says why not; its LoadRun(run) with the run's
// items, or with RunFailed(run, why); and its Load(line) with the item's
// body, or with Failed(line, why). Every few seconds, while the page shows,
// it asks what a log gained, or an index's facts anew, and tells the program
// what changed. Like lem.js, it's a plain script that defines one global,
// `contexts`, and it starts itself on a page that gives `data-facts`.
var contexts = (function () {
  "use strict";

  // The looks and themes, each [name, what the bar calls it], the first the
  // default.
  var LOOKS = [
    ["native", "Native"],
    ["log", "Log"]
  ];
  var THEMES = [
    ["dark", "dark"],
    ["light", "light"]
  ];
  // How often a page asks what changed, while it shows.
  var EVERY_MS = 4000;
  // Whether a log's runs fold, the default, or open, all at once.
  var RUNS = [
    ["folded", "folded"],
    ["open", "open"]
  ];
  // Whether a log's panel, of its starred lines and its turns, hides, the
  // default, or shows beside it.
  var PANEL = [
    ["hidden", "hidden"],
    ["shown", "shown"]
  ];
  // Whether an assistant's messages show as Markdown writes them, the
  // default, or as written.
  var TEXT = [
    ["markdown", "markdown"],
    ["plain", "plain"]
  ];
  // The tabs that the session keeps, as words of the store's (store.py):
  // the open contexts' numbers, oldest first, and the number of the one
  // shown, 0 for the index.
  var TABS = /^([1-9][0-9]{0,8}(-[1-9][0-9]{0,8})*)?$/;
  var TAB = /^(0|[1-9][0-9]{0,8})$/;
  // The panel's width, in pixels, as its edge was dragged, or "" for as
  // contexts.css says; and the least it may be.
  var WIDTH = /^([1-9][0-9]{1,3})?$/;
  var NARROWEST = 180;
  // Where a log's tab was left, which the session keeps for each tab open,
  // as `at-N`, a word of the store's: the line of its first row on the
  // screen, or its end; for PLACES tabs at most, as a scroll settles, once
  // it has been still for SETTLE_MS.
  var PLACE = /^(end|[1-9][0-9]{0,8})$/;
  var PLACES = 16;
  var SETTLE_MS = 500;

  // Shows a page of a compiled program, contexts.lem's, in a view in
  // `o.root`, and is its host:
  //
  // - o.get(url, done) fetches JSON, and calls done(null, value), or
  //   done(why, null) if it can't, later rather than at once;
  // - o.facts is the URL of the page's facts; o.before, o.after and
  //   o.around, the URLs of the rows before a line, after one and around
  //   one, with {line}; o.run, the
  //   URL of a run's items, with {run}; and o.body, the URL of an item's
  //   body, with {line}; each null for the index;
  // - o.status(text, wrong), if given, hears how it went: how long the
  //   page took to show, and what went wrong;
  // - o.keep(update), if given, runs update(), which shows what came, rows
  //   or a run's items or a body, keeping what's on the screen in place; and
  //   o.shown(), if given, hears that the page's facts are shown;
  // - o.opening, for a log, says whether every run opens, as an Opening
  //   fact from the start, o.aside whether its panel shows, as an Aside
  //   fact, and o.formatting whether an assistant's messages show as
  //   Markdown writes them, as a Formatting fact, each of which the bar
  //   changes later (boot);
  // - o.grow, for a log, is the URL of what it gained since it had {lines}
  //   lines; and o.every(tick), if given, calls tick() every few seconds
  //   while the page shows, when a log asks what it gained, and an index its
  //   facts anew, one request at a time. What a log gained is steps; what
  //   an index's facts gained and lost, the host works out. Should either
  //   fail, the page says why, and asks no more;
  // - o.stars, for a log, is the URL that keeps a star, set or cleared, and
  //   o.post(url, value, done) sends it JSON, and calls done(null), or
  //   done(why): the program's Star(line, on) changes Stars on the page at
  //   once, and then in the server's store; one that can't be kept is
  //   taken back, and the page says why; and, for the index, the program's
  //   KeepRoot(to, label, tags) sends a context's label and tags to `to`,
  //   the tags as words, in lowercase, after which the page asks for its
  //   facts anew, at once;
  // - o.recipes, if given, are the recipes that the view draws with, in
  //   place of glui's (glui.recipes);
  // - o.pins, for a log, is the URL of its panel's pins, which the page
  //   asks for anew once a star is kept, one request at a time: the Pin
  //   facts that come take the place of those before. Should they fail, the
  //   page says why.
  //
  // It asks for rows, a run's items, or a body, once while they're coming,
  // and again when the program asks again after they failed; meanwhile the
  // run or the entry no longer says why. What can't come, or what the
  // program refuses, fails, which the run, the entry or the end of the rows
  // says. Gives the view.
  function start(o) {
    var status = o.status || function () {};
    var keep =
      o.keep ||
      function (update) {
        update();
      };
    var runtime = new glem.Runtime(new glem.Program(o.program));
    // What the program asks for, by its effect: where it comes from; the
    // field that names it, in the effect and in the fact that says it
    // failed; that fact; and why steps that aren't a list are wrong.
    var asks = {
      LoadBefore: { url: o.before, field: "line", failed: "EarlierFailed", wrong: "its rows aren't a list of steps" },
      LoadAfter: { url: o.after, field: "line", failed: "LaterFailed", wrong: "its rows aren't a list of steps" },
      LoadAround: { url: o.around, field: "line", failed: null, wrong: "its rows aren't a list of steps" },
      LoadRun: { url: o.run, field: "run", failed: "RunFailed", wrong: "its items aren't a list of steps" },
      Load: { url: o.body, field: "line", failed: "Failed", wrong: "its body isn't a list of steps" }
    };
    // What's coming, and what failed, with its fact's fields, each by its
    // fact's name and the value that names it.
    var loading = new Set();
    var failed = new Map();
    // Why the program refused the host's last change, or null.
    var refused = null;
    // Whether the page's facts are shown; how many lines a log has, as the
    // host last said (Length), or null; an index's facts, as they last came,
    // by their text; whether a request for what changed is out; and whether
    // the page still asks.
    var log = typeof o.grow === "string";
    var ready = false;
    var length = null;
    var facts = null;
    var asking = false;
    var live = true;
    // Whether to ask what changed once more, as soon as the request that's
    // out is back.
    var soon = false;
    // The lines starred, as the page shows them (Stars); the panel's pins,
    // as they last came, by their text; whether a request for them is out;
    // and whether to ask again once it's back, as a star was kept meanwhile.
    var starred = new Set();
    var pins = new Map();
    var pinning = false;
    var again = false;
    var host = {
      effects: function (effects) {
        effects.forEach(function (effect) {
          if (effect.name === "Star") {
            star(effect.fields.line, effect.fields.on);
            return;
          }
          if (effect.name === "KeepRoot") {
            keepRoot(effect.fields.to, effect.fields.label, effect.fields.tags);
            return;
          }
          var ask = asks[effect.name];
          if (ask !== undefined) {
            load(ask, effect.fields[ask.field]);
          }
        });
      },
      error: function (message) {
        refused = message;
        status("error: " + message, true);
      }
    };
    var view = o.recipes ? new glem.View(runtime, o.root, o.recipes, host, glui.state) : glui.view(runtime, o.root, host);
    var settings = [];
    if (typeof o.opening === "boolean") {
      settings.push(["+", "Opening", { open: o.opening }]);
    }
    if (typeof o.aside === "boolean") {
      settings.push(["+", "Aside", { shown: o.aside }]);
    }
    if (typeof o.formatting === "boolean") {
      settings.push(["+", "Formatting", { markdown: o.formatting }]);
    }
    if (settings.length > 0) {
      view.update(settings);
    }

    // Sets or clears the star of line `line`: on the page at once, and then
    // in the server's store, if there's one. One that can't be kept is
    // taken back, and the page says why.
    function star(line, on) {
      if (starred.has(line) === on) {
        return;
      }
      var change = function (set) {
        if (set) {
          starred.add(line);
        } else {
          starred.delete(line);
        }
        view.update([[set ? "+" : "-", "Stars", { line: line }]]);
      };
      change(on);
      if (typeof o.stars === "string" && o.post) {
        o.post(o.stars, { line: line, on: on }, function (why) {
          if (why !== null) {
            if (starred.has(line) === on) {
              change(!on);
            }
            status("couldn't keep the star: " + why, true);
            return;
          }
          pin();
        });
      }
    }

    // Keeps a context's label and tags at `url`, as KeepRoot gives them: the
    // tags apart by spaces or commas, in lowercase, once each; then asks for
    // the index's facts anew, at once. What can't be kept, the page says why.
    function keepRoot(url, label, tags) {
      if (!o.post) {
        return;
      }
      var words = [];
      tags
        .toLowerCase()
        .split(/[\s,]+/)
        .forEach(function (word) {
          if (word !== "" && words.indexOf(word) < 0) {
            words.push(word);
          }
        });
      o.post(url, { label: label, tags: words }, function (why) {
        if (why !== null) {
          status("couldn't keep the label and tags: " + why, true);
          return;
        }
        if (asking) {
          soon = true;
        } else {
          tick();
        }
      });
    }

    // Asks for the panel's pins anew, as a star was kept: one request at a
    // time, and once more if another star was kept meanwhile.
    function pin() {
      if (typeof o.pins !== "string") {
        return;
      }
      if (pinning) {
        again = true;
        return;
      }
      pinning = true;
      o.get(o.pins, function (why, steps) {
        pinning = false;
        if (why === null && !Array.isArray(steps)) {
          why = "they aren't a list of steps";
        }
        if (why === null) {
          var now = byText(steps);
          var change = changed(pins, now);
          refused = null;
          if (change.length > 0) {
            view.update(change);
          }
          why = refused;
          if (why === null) {
            pins = now;
          }
        }
        if (why !== null) {
          status("couldn't load the starred lines: " + why, true);
        }
        if (again) {
          again = false;
          pin();
        }
      });
    }

    // What was asked for, `value`, fails: the turn or the entry says why;
    // or, for what has no fact to say so, the page.
    function fail(ask, value, why) {
      if (ask.failed === null) {
        status("couldn't load the rows around line " + value + ": " + why, true);
        return;
      }
      var fields = { why: why };
      fields[ask.field] = value;
      failed.set(ask.failed + " " + value, fields);
      view.update([["+", ask.failed, fields]]);
    }

    function load(ask, value) {
      var name = ask.url + " " + value;
      if (ask.url === null || ask.url === undefined || loading.has(name)) {
        return;
      }
      loading.add(name);
      var gone = ask.failed + " " + value;
      if (failed.has(gone)) {
        view.update([["-", ask.failed, failed.get(gone)]]);
        failed.delete(gone);
      }
      o.get(ask.url.split("{" + ask.field + "}").join(String(value)), function (why, steps) {
        loading.delete(name);
        if (why === null && !Array.isArray(steps)) {
          why = ask.wrong;
        }
        if (why !== null) {
          fail(ask, value, why);
          return;
        }
        refused = null;
        keep(function () {
          view.update(steps);
        });
        if (refused !== null) {
          fail(ask, value, refused);
        }
      });
    }

    o.get(o.facts, function (why, steps) {
      if (why === null && !Array.isArray(steps)) {
        why = "its facts aren't a list of steps";
      }
      if (why !== null) {
        status("couldn't load the page: " + why, true);
        return;
      }
      refused = null;
      var began = Date.now();
      view.update(steps);
      if (refused === null) {
        status(steps.length + " facts, shown in " + (Date.now() - began) + " ms", false);
        ready = true;
        steps.forEach(function (step) {
          if (step[0] === "+" && step[1] === "Stars") {
            starred.add(step[2].line);
          } else if (step[0] === "+" && step[1] === "Pin") {
            pins.set(JSON.stringify([step[1], step[2]]), step);
          }
        });
        length = lengthOf(steps, length);
        facts = log ? null : byText(steps);
        if (o.shown) {
          o.shown();
        }
      }
    });

    // Asks what changed, unless a request is out, or the page isn't shown
    // yet, or asks no more.
    function tick() {
      if (asking || !live || !ready || (log && length === null)) {
        return;
      }
      var url = log ? o.grow.split("{lines}").join(String(length)) : o.facts;
      asking = true;
      o.get(url, function (why, steps) {
        asking = false;
        if (why === null && !Array.isArray(steps)) {
          why = "what changed isn't a list of steps";
        }
        var change = why === null && !log ? changed(facts, byText(steps)) : steps;
        if (why === null && change.length > 0) {
          refused = null;
          view.update(change);
          why = refused;
        }
        if (why !== null) {
          live = false;
          status("live updates stopped: " + why, true);
          return;
        }
        if (log) {
          length = lengthOf(steps, length);
        } else {
          facts = byText(steps);
        }
        if (soon) {
          soon = false;
          tick();
        }
      });
    }
    if (o.every) {
      o.every(tick);
    }
    return view;
  }

  // A page of tabs (V3): the index's tab, and a tab for each root context
  // that's open, each a view of the program (start) in an element of its
  // own, in o.holder, which only the tab shown shows; and their strip, a
  // view of the strip's program (lem/tabs.lem) in o.strip. It takes:
  //
  // - o.program and o.strip_program, the compiled programs;
  // - o.index, the URL of the index's facts; o.tab, the URL of a context's
  //   tab, with {id}, which gives its title and its log's URLs as JSON
  //   (pythia/interaction/node/view.py); o.log, the address of a
  //   context's page, with {id}, and o.home, the index's, which a click on
  //   a link to either, without a modifier key, shows in its tab instead;
  // - o.get, o.post and o.status, as start() takes them;
  // - o.settings(), what a log's tab starts with: {opening, aside,
  //   formatting};
  // - o.kept, the tabs that the session kept, {tabs: numbers, tab:
  //   number, places: {number: line or "end"}}, and o.open, the number of
  //   the tab to show at first, if a page names one; o.keep_tabs(numbers,
  //   shown) keeps them anew as they change;
  // - o.at(tab), where the tab shown is: the line of its first row on the
  //   screen (placeOf), "end", or null if it can't say; and
  //   o.keep_place(n, at), which keeps that for tab n, or forgets it, with
  //   null. A tab that the session kept a line for goes back to it, as the
  //   program's Restore does, once its facts are shown, when it first shows;
  //   the rest start at their end;
  // - o.title(text), o.address(url) and o.showing(tab) follow the tab
  //   shown, its window's title, its page's address, and the tab itself;
  //   o.top() and o.go(top), its scroll, which each tab keeps, and o.end(),
  //   a log's end, where it starts; o.keep(root, update), as start()'s
  //   keep, for the tab shown.
  //
  // Gives {open(n, show), show(n), close(n), move(n, m), shown(), each(f),
  // tick(), settle()}: tick() asks the tab shown what changed, as only it
  // asks; settle(), which the host calls as a scroll settles, keeps where
  // the tab shown is, if that changed.
  function tabs(o) {
    var doc = o.holder.ownerDocument;
    var places = o.kept.places || {};
    var all = new Map();
    var order = [];
    var rank = 0;
    var current = null;
    var ready = false;
    var strip = glui.view(new glem.Runtime(new glem.Program(o.strip_program)), o.strip, {
      effects: function (effects) {
        effects.forEach(function (effect) {
          if (effect.name === "Show") {
            show(effect.fields.n);
          } else if (effect.name === "Close") {
            close(effect.fields.n);
          } else if (effect.name === "Move") {
            move(effect.fields.n, effect.fields.to);
          }
        });
      },
      error: function (message) {
        o.status("error: " + message, true);
      }
    });

    function fact(tab) {
      return { n: tab.n, title: tab.title, rank: tab.rank };
    }

    // glui's recipes, with a window that keeps the document's title the
    // tab shown's.
    function recipes(tab) {
      var all = glui.recipes();
      var window_ = all.Window;
      all.Window = function (w, view) {
        var made = window_(w, view);
        var update = made.update;
        made.update = function () {
          update();
          tab.heading = w.fields.title;
          o.title(current === null ? tab.heading : current.heading || current.title);
        };
        return made;
      };
      return all;
    }

    // The tab's view, of the URLs that `urls` gives.
    function begin(tab, urls) {
      var settings = o.settings();
      var log = tab.n !== 0;
      tab.view = start({
        root: tab.root,
        program: o.program,
        recipes: recipes(tab),
        opening: log ? settings.opening : undefined,
        aside: log ? settings.aside : undefined,
        formatting: log ? settings.formatting : undefined,
        facts: urls.facts,
        before: urls.before || null,
        after: urls.after || null,
        around: urls.around || null,
        run: urls.run || null,
        body: urls.body || null,
        grow: urls.grow || null,
        stars: urls.stars || null,
        pins: urls.pins || null,
        post: o.post,
        get: o.get,
        status: o.status,
        every: function (tick) {
          tab.tick = tick;
        },
        keep: function (update) {
          if (tab === current) {
            o.keep(tab.root, update);
          } else {
            update();
          }
        },
        shown: function () {
          tab.ready = true;
          if (!log) {
            return;
          }
          if (tab === current) {
            o.end();
            restore(tab);
          } else {
            tab.place = "end";
          }
        }
      });
    }

    // Sends the tab the program's Restore, to the line that the session
    // kept for it, once.
    function restore(tab) {
      if (tab.restore !== null) {
        var line = tab.restore;
        tab.restore = null;
        tab.view.send(["Restore", line]);
      }
    }

    // Keeps where the tab shown is, a log's, as the host says, if that
    // changed, once its facts are shown, and it has gone back to where it
    // was, as it then has.
    function settle() {
      if (current === null || current.n === 0 || !current.ready) {
        return;
      }
      var at = o.at(current);
      if (at !== null && at !== current.at) {
        current.at = at;
        o.keep_place(current.n, at);
      }
    }

    // Opens tab `n`, the index's if it's 0, if it isn't open; and shows it,
    // with `andShow`. A context's tab asks for its URLs first; one that
    // can't come closes, and the page says why.
    function open(n, andShow) {
      var tab = all.get(n);
      if (tab === undefined) {
        tab = { n: n, rank: rank++, title: n === 0 ? "Root contexts" : String(n), heading: null, root: doc.createElement("div"), view: null, tick: null, place: null, ready: false, restore: null, at: null };
        // Where the session left it, only as the page starts.
        if (n !== 0 && Object.prototype.hasOwnProperty.call(places, n)) {
          tab.at = places[n];
          tab.restore = typeof places[n] === "number" ? places[n] : null;
          delete places[n];
        }
        tab.root.setAttribute("hidden", "");
        o.holder.appendChild(tab.root);
        all.set(n, tab);
        order.push(n);
        strip.update([["+", "Open", fact(tab)]]);
        if (n === 0) {
          begin(tab, { facts: o.index });
        } else {
          o.get(o.tab.split("{id}").join(String(n)), function (why, urls) {
            if (all.get(n) !== tab) {
              return;
            }
            if (why === null && (urls === null || typeof urls !== "object" || typeof urls.facts !== "string")) {
              why = "its tab isn't one";
            }
            if (why !== null) {
              o.status("couldn't open context " + n + ": " + why, true);
              close(n);
              return;
            }
            if (typeof urls.title === "string" && urls.title !== tab.title) {
              var was = fact(tab);
              tab.title = urls.title;
              strip.update([
                ["-", "Open", was],
                ["+", "Open", fact(tab)]
              ]);
            }
            begin(tab, urls);
          });
        }
        if (!andShow) {
          keep();
        }
      }
      if (andShow) {
        show(n);
      }
      return tab;
    }

    // Shows tab `n`, if it's open: where it was, as the tab shown before
    // keeps its place.
    function show(n) {
      var tab = all.get(n);
      if (tab === undefined || tab === current) {
        return;
      }
      if (current !== null) {
        settle();
        current.place = o.top();
        current.root.setAttribute("hidden", "");
      }
      current = tab;
      tab.root.removeAttribute("hidden");
      if (tab.restore !== null && tab.ready) {
        o.end();
        restore(tab);
      } else if (tab.place === "end") {
        o.end();
      } else {
        o.go(tab.place === null ? 0 : tab.place);
      }
      strip.update([["+", "Shown", { n: n }]]);
      o.title(tab.heading || tab.title);
      o.address(n === 0 ? o.home : o.log.split("{id}").join(String(n)));
      o.showing(tab);
      if (tab.tick !== null) {
        tab.tick();
      }
      keep();
    }

    // Closes tab `n`, but the index's: then the next shows, if it showed,
    // or the one before.
    function close(n) {
      var tab = all.get(n);
      if (n === 0 || tab === undefined) {
        return;
      }
      var at = order.indexOf(n);
      all.delete(n);
      order.splice(at, 1);
      if (tab.root.parentNode !== null) {
        tab.root.parentNode.removeChild(tab.root);
      }
      strip.update([["-", "Open", fact(tab)]]);
      o.keep_place(n, null);
      if (tab === current) {
        current = null;
        show(order[Math.min(at, order.length - 1)]);
      }
      keep();
    }

    // Moves tab `n` to where tab `m` is: after it, if it was before it, or
    // before it, if it was after; the index stays first. The tabs' ranks
    // follow their order.
    function move(n, m) {
      var from = order.indexOf(n);
      var to = order.indexOf(m);
      if (n === 0 || m === 0 || from < 0 || to < 0 || from === to) {
        return;
      }
      order.splice(from, 1);
      order.splice(to, 0, n);
      var steps = [];
      order.forEach(function (k, at) {
        var tab = all.get(k);
        if (tab.rank !== at) {
          steps.push(["-", "Open", fact(tab)]);
          tab.rank = at;
          steps.push(["+", "Open", fact(tab)]);
        }
      });
      strip.update(steps);
      keep();
    }

    function keep() {
      if (ready) {
        o.keep_tabs(order.filter(function (n) {
          return n !== 0;
        }), current === null ? 0 : current.n);
      }
    }

    // The number of a context whose page `href` is, 0 for the index's, or
    // null.
    function numberOf(href) {
      if (href === null) {
        return null;
      }
      if (href === o.home) {
        return 0;
      }
      var parts = o.log.split("{id}");
      if (href.length <= parts[0].length + parts[1].length || href.indexOf(parts[0]) !== 0 || href.slice(href.length - parts[1].length) !== parts[1]) {
        return null;
      }
      var id = href.slice(parts[0].length, href.length - parts[1].length);
      return /^[1-9][0-9]{0,8}$/.test(id) ? Number(id) : null;
    }

    o.holder.addEventListener("click", function (event) {
      if (event.defaultPrevented || event.button > 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }
      for (var e = event.target; e && e !== o.holder; e = e.parentNode) {
        if (e.nodeType === 1 && e.localName === "a") {
          var n = numberOf(e.getAttribute("href"));
          if (n !== null) {
            event.preventDefault();
            open(n, true);
          }
          return;
        }
      }
    });

    open(0, false);
    (o.kept.tabs || []).forEach(function (n) {
      open(n, false);
    });
    var first = typeof o.open === "number" ? o.open : all.has(o.kept.tab) ? o.kept.tab : 0;
    open(first, false);
    ready = true;
    show(first);
    return {
      open: open,
      show: show,
      close: close,
      move: move,
      shown: function () {
        return current;
      },
      each: function (f) {
        all.forEach(f);
      },
      tick: function () {
        if (current !== null && current.tick !== null) {
          current.tick();
        }
      },
      settle: settle,
      strip: strip
    };
  }

  // The line of the first of a log's rows, `rows`, in order, that
  // `below(row)` says reaches below the screen's top, as far as rows say:
  // a message's or a run's, which the program's Restore goes back to; past
  // a turn's mark, the row after it. Or null, if none does.
  function placeOf(view, rows, below) {
    var low = firstBelow(rows, below);
    for (var i = low; i < rows.length; i++) {
      var w = view.elements.get(rows[i]);
      if (w !== undefined && Array.isArray(w.id) && (w.id[0] === "Said" || w.id[0] === "RunBox")) {
        return w.id[1];
      }
    }
    return null;
  }

  // The index of the first of `rows`, in order, that `below(row)` says
  // reaches below the screen's top, by halves; or rows.length.
  function firstBelow(rows, below) {
    var low = 0;
    var high = rows.length;
    while (low < high) {
      var mid = (low + high) >> 1;
      if (below(rows[mid])) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return low;
  }

  // The length that steps say a log has, by their last Length fact, else
  // `was`.
  function lengthOf(steps, was) {
    steps.forEach(function (step) {
      if (step[0] === "+" && step[1] === "Length") {
        was = step[2].lines;
      }
    });
    return was;
  }

  // Steps that add facts, by their text.
  function byText(steps) {
    var all = new Map();
    steps.forEach(function (step) {
      all.set(JSON.stringify([step[1], step[2]]), step);
    });
    return all;
  }

  // What takes facts `was` to facts `now`, each by its text: those that went,
  // then those that came.
  function changed(was, now) {
    var steps = [];
    was.forEach(function (step, text) {
      if (!now.has(text)) {
        steps.push(["-", step[1], step[2]]);
      }
    });
    now.forEach(function (step, text) {
      if (!was.has(text)) {
        steps.push(step);
      }
    });
    return steps;
  }

  // In a browser.

  // This script's URL, which only its first run knows.
  var here = typeof document !== "undefined" && document.currentScript ? document.currentScript.src : null;

  // JSON from a URL, for start(): what fetch gives, or why it couldn't.
  function get(url, done) {
    fetch(url, { cache: "no-store", credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error(response.status + " " + response.statusText);
        }
        return response.json();
      })
      .then(
        function (value) {
          done(null, value);
        },
        function (error) {
          done(error instanceof Error ? error.message : String(error), null);
        }
      );
  }

  // Shows what came, by update(), keeping what's on the screen in place:
  // the first of the log's rows in `root` that shows stays where it is,
  // though rows come above it, or a run above it opens. The page's own
  // scroll anchoring is off (web/contexts.css), so that only this moves it.
  function keepRows(root, update) {
    var page = document.scrollingElement;
    var rows = page ? root.querySelectorAll(".g-box > .g-entry") : [];
    // The first row whose bottom is below the screen's top.
    var low = firstBelow(rows, function (row) {
      return row.getBoundingClientRect().bottom > 0;
    });
    var anchor = low < rows.length ? rows[low] : null;
    var top = anchor !== null ? anchor.getBoundingClientRect().top : 0;
    update();
    if (anchor !== null && anchor.isConnected) {
      page.scrollTop += anchor.getBoundingClientRect().top - top;
    }
  }

  // Sends JSON to the server's store, with the header that only the view's
  // scripts send (pythia/interaction/node/server.py): then done(null), or
  // done(why) if it couldn't.
  function keepIn(url, value, done) {
    fetch(url, {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Pythia-View": "1" },
      body: JSON.stringify(value)
    }).then(
      function (response) {
        done(response.ok ? null : response.status + " " + response.statusText);
      },
      function (error) {
        done(error instanceof Error ? error.message : String(error));
      }
    );
  }

  // A choice: `kept`, as the server's store keeps it, if it's one of
  // `choices`; else as the browser remembers it, if it lets the page; else
  // the first.
  function remembered(name, choices, kept) {
    var value = kept;
    var known = function () {
      return choices.some(function (choice) {
        return choice[0] === value;
      });
    };
    if (!known()) {
      try {
        value = window.localStorage.getItem("glui." + name);
      } catch (e) {
        value = null;
      }
    }
    return known() ? value : choices[0][0];
  }

  // The edge of a log's panel, which a drag with the pointer moves, to make
  // the panel wider or narrower, and a double click puts back, as the page
  // keeps it: in --contexts-panel, which contexts.css reads, from `width`,
  // pixels or "", and `keep(width)` as the drag ends.
  function edged(html, width, keep) {
    var edge = document.createElement("div");
    edge.setAttribute("class", "contexts-edge");
    edge.setAttribute("aria-hidden", "true");
    document.body.appendChild(edge);
    var set = function (px) {
      if (px === "") {
        html.style.removeProperty("--contexts-panel");
      } else {
        html.style.setProperty("--contexts-panel", px + "px");
      }
    };
    set(width);
    edge.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      var panel = document.querySelector(".g-hud.g-top-right");
      var right = panel ? parseFloat(window.getComputedStyle(panel).right) || 0 : 0;
      var widest = Math.max(NARROWEST, Math.round(window.innerWidth * 0.7));
      var px = width;
      var move = function (e) {
        px = String(Math.max(NARROWEST, Math.min(widest, Math.round(window.innerWidth - right - e.clientX))));
        set(px);
      };
      var up = function () {
        edge.removeEventListener("pointermove", move);
        edge.removeEventListener("pointerup", up);
        edge.removeEventListener("pointercancel", up);
        edge.classList.remove("contexts-dragging");
        if (px !== width) {
          width = px;
          keep(px);
        }
      };
      edge.setPointerCapture(event.pointerId);
      edge.classList.add("contexts-dragging");
      edge.addEventListener("pointermove", move);
      edge.addEventListener("pointerup", up);
      edge.addEventListener("pointercancel", up);
    });
    edge.addEventListener("dblclick", function () {
      width = "";
      set("");
      keep("");
    });
  }

  // A word that the server's store keeps (`kept`), if it's one that
  // `pattern` takes; else as the browser remembers it, if it lets the page;
  // else "".
  function rememberedWord(name, pattern, kept) {
    var value = typeof kept === "string" ? kept : null;
    if (value === null || !pattern.test(value)) {
      try {
        value = window.localStorage.getItem("glui." + name);
      } catch (e) {
        value = null;
      }
    }
    return value !== null && pattern.test(value) ? value : "";
  }

  function remember(name, value) {
    try {
      window.localStorage.setItem("glui." + name, value);
    } catch (e) {
      // The page keeps the choice until it's left.
    }
  }

  // A menu of choices, in the bar, which calls `chosen` with the value
  // that's chosen.
  function menu(bar, label, choices, value, chosen) {
    var wrap = document.createElement("label");
    var select = document.createElement("select");
    choices.forEach(function (choice) {
      var option = document.createElement("option");
      option.value = choice[0];
      option.textContent = choice[1];
      select.appendChild(option);
    });
    select.value = value;
    select.addEventListener("change", function () {
      chosen(select.value);
    });
    wrap.appendChild(document.createTextNode(label + " "));
    wrap.appendChild(select);
    bar.appendChild(wrap);
  }

  // The page: its settings, as the server's store keeps them, if it does;
  // then its bar, with the look's and the theme's menus and a line of
  // status; and the view.
  function boot() {
    var store = document.documentElement.getAttribute("data-store");
    if (store === null) {
      show({});
      return;
    }
    get(store, function (why, kept) {
      var plain = why === null && kept !== null && typeof kept === "object" && !Array.isArray(kept);
      show(plain ? kept : {}, store);
    });
  }

  // The page, with the settings that the store keeps (`kept`), and the URL
  // that keeps them anew, if there's one.
  function show(kept, store) {
    var html = document.documentElement;
    // A page with `data-tab` shows tabs (tabs()), with the strip and the bar
    // on a line of their own at the top; else one page, the bar over it.
    var tabbing = html.getAttribute("data-tab") !== null;
    var bar = document.createElement("nav");
    var line = document.createElement("span");
    var root = document.createElement("div");
    var stripRoot = document.createElement("div");
    var top = null;
    bar.setAttribute("class", "contexts-bar");
    line.setAttribute("class", "contexts-status");
    if (tabbing) {
      top = document.createElement("div");
      top.setAttribute("class", "contexts-strip");
      top.appendChild(stripRoot);
      top.appendChild(bar);
      document.body.appendChild(top);
      html.classList.add("contexts-tabbed");
      // How tall the strip is, which a row that goes to the top of the
      // screen goes below (contexts.css).
      var tall = function () {
        html.style.setProperty("--contexts-strip", Math.ceil(top.getBoundingClientRect().height) + "px");
      };
      if (typeof ResizeObserver === "function") {
        new ResizeObserver(tall).observe(top);
      } else {
        window.addEventListener("resize", tall);
        window.setTimeout(tall, 0);
      }
    } else {
      document.body.appendChild(bar);
    }
    document.body.appendChild(root);
    var status = function (text, wrong) {
      line.textContent = text;
      line.setAttribute("class", wrong ? "contexts-status contexts-wrong" : "contexts-status");
    };
    var look = function (name) {
      LOOKS.forEach(function (one) {
        html.classList.toggle("g-" + one[0], one[0] === name);
      });
    };
    var theme = function (name) {
      get(new URL("compiled/theme-" + name + ".json", here).href, function (why, form) {
        if (why !== null) {
          status("couldn't load the " + name + " theme: " + why, true);
          return;
        }
        var it = new glem.Runtime(new glem.Program(form));
        it.tick(null);
        glui.paint(html, it.lem);
      });
    };
    var chosen = {
      look: remembered("look", LOOKS, kept.look),
      theme: remembered("theme", THEMES, kept.theme),
      runs: remembered("runs", RUNS, kept.runs),
      panel: remembered("panel", PANEL, kept.panel),
      text: remembered("text", TEXT, kept.text),
      width: rememberedWord("width", WIDTH, kept.width),
      tabs: rememberedWord("tabs", TABS, kept.tabs),
      tab: rememberedWord("tab", TAB, kept.tab)
    };
    // Where each tab open was left, as the store keeps it, if it does.
    var places = {};
    var open_ = chosen.tabs === "" ? [] : chosen.tabs.split("-");
    open_.forEach(function (n) {
      var at = kept["at-" + n];
      if (typeof at === "string" && PLACE.test(at)) {
        chosen["at-" + n] = at;
        places[n] = at === "end" ? at : Number(at);
      }
    });
    // Settings that changed: the browser remembers them, and the server's
    // store keeps them all (keep), with where each tab was left.
    var keep = function () {
      if (store !== undefined) {
        keepIn(store, chosen, function (why) {
          if (why !== null) {
            status("couldn't keep the settings: " + why, true);
          }
        });
      }
    };
    var save = function (name, value) {
      chosen[name] = value;
      remember(name, value);
      keep();
    };
    // The view, or the tabs, once the program has come; and whether the
    // page shows logs.
    var shown = null;
    var log = tabbing || html.getAttribute("data-run") !== null;
    // A log's panel shows beside it, as contexts.css makes room for it.
    var aside = function (name) {
      var now = tabbing ? shown !== null && shown.shown() !== null && shown.shown().n !== 0 : log;
      html.classList.toggle("contexts-aside", now && name === "shown");
    };
    // Each view of a log, for a setting that changes.
    var logs = function (f) {
      if (shown === null) {
        return;
      }
      if (!tabbing) {
        f(shown);
        return;
      }
      shown.each(function (tab) {
        if (tab.n !== 0 && tab.view !== null) {
          f(tab.view);
        }
      });
    };
    menu(bar, "Look", LOOKS, chosen.look, function (value) {
      save("look", value);
      look(value);
    });
    menu(bar, "Theme", THEMES, chosen.theme, function (value) {
      save("theme", value);
      theme(value);
    });
    if (log) {
      menu(bar, "Runs", RUNS, chosen.runs, function (value) {
        var was = chosen.runs;
        logs(function (view) {
          view.update([
            ["-", "Opening", { open: was === "open" }],
            ["+", "Opening", { open: value === "open" }]
          ]);
        });
        save("runs", value);
      });
      menu(bar, "Panel", PANEL, chosen.panel, function (value) {
        var was = chosen.panel;
        logs(function (view) {
          view.update([
            ["-", "Aside", { shown: was === "shown" }],
            ["+", "Aside", { shown: value === "shown" }]
          ]);
        });
        save("panel", value);
        aside(value);
      });
      menu(bar, "Text", TEXT, chosen.text, function (value) {
        var was = chosen.text;
        logs(function (view) {
          view.update([
            ["-", "Formatting", { markdown: was === "markdown" }],
            ["+", "Formatting", { markdown: value === "markdown" }]
          ]);
        });
        save("text", value);
      });
    }
    bar.appendChild(line);
    if (log) {
      edged(html, chosen.width, function (px) {
        save("width", px);
      });
    }
    look(chosen.look);
    theme(chosen.theme);
    aside(chosen.panel);
    var page = document.scrollingElement;
    var every = function (tick) {
      var showing = function () {
        if (document.visibilityState !== "hidden") {
          tick();
        }
      };
      window.setInterval(showing, EVERY_MS);
      document.addEventListener("visibilitychange", showing);
    };
    get(new URL("compiled/contexts.json", here).href, function (why, program) {
      if (why !== null) {
        status("couldn't load the program: " + why, true);
        return;
      }
      if (tabbing) {
        get(new URL("compiled/tabs.json", here).href, function (why, strip) {
          if (why !== null) {
            status("couldn't load the tabs' program: " + why, true);
            return;
          }
          var open = html.getAttribute("data-open");
          shown = tabs({
            holder: root,
            strip: stripRoot,
            program: program,
            strip_program: strip,
            index: html.getAttribute("data-index"),
            tab: html.getAttribute("data-tab"),
            log: html.getAttribute("data-log"),
            home: html.getAttribute("data-home"),
            open: open !== null && TAB.test(open) ? Number(open) : null,
            kept: {
              tabs: open_.map(Number),
              tab: Number(chosen.tab),
              places: places
            },
            keep_tabs: function (numbers, n) {
              var words = numbers.map(String);
              while (words.join("-").length > 64) {
                words.shift();
              }
              chosen.tabs = words.join("-");
              remember("tabs", chosen.tabs);
              Object.keys(chosen).forEach(function (name) {
                if (/^at-/.test(name) && words.indexOf(name.slice(3)) < 0) {
                  delete chosen[name];
                }
              });
              save("tab", String(n));
            },
            at: function (tab) {
              if (!page || tab.view === null) {
                return null;
              }
              if (page.scrollTop + page.clientHeight >= page.scrollHeight - 8) {
                return "end";
              }
              var below = top.getBoundingClientRect().bottom;
              return placeOf(tab.view, tab.root.querySelectorAll(".g-box > .g-entry"), function (row) {
                return row.getBoundingClientRect().bottom > below;
              });
            },
            keep_place: function (n, at) {
              var name = "at-" + n;
              if (at === null) {
                if (Object.prototype.hasOwnProperty.call(chosen, name)) {
                  delete chosen[name];
                  keep();
                }
                return;
              }
              var others = Object.keys(chosen).filter(function (one) {
                return /^at-/.test(one) && one !== name;
              });
              if (others.length >= PLACES) {
                delete chosen[others[0]];
              }
              chosen[name] = String(at);
              keep();
            },
            get: get,
            post: keepIn,
            status: status,
            settings: function () {
              return {
                opening: chosen.runs === "open",
                aside: chosen.panel === "shown",
                formatting: chosen.text === "markdown"
              };
            },
            title: function (text) {
              document.title = text;
            },
            address: function (url) {
              if (window.location.pathname !== url) {
                window.history.replaceState(null, "", url);
              }
            },
            showing: function () {
              aside(chosen.panel);
            },
            top: function () {
              return page ? page.scrollTop : 0;
            },
            go: function (top) {
              if (page) {
                page.scrollTop = top;
              }
            },
            end: function () {
              if (page) {
                page.scrollTop = page.scrollHeight;
              }
            },
            keep: keepRows
          });
          aside(chosen.panel);
          every(shown.tick);
          var settling = null;
          window.addEventListener(
            "scroll",
            function () {
              if (settling !== null) {
                window.clearTimeout(settling);
              }
              settling = window.setTimeout(function () {
                settling = null;
                shown.settle();
              }, SETTLE_MS);
            },
            { passive: true }
          );
        });
        return;
      }
      shown = start({
        root: root,
        program: program,
        opening: log ? chosen.runs === "open" : undefined,
        aside: log ? chosen.panel === "shown" : undefined,
        formatting: log ? chosen.text === "markdown" : undefined,
        grow: html.getAttribute("data-grow"),
        stars: html.getAttribute("data-stars"),
        pins: html.getAttribute("data-pins"),
        post: keepIn,
        every: every,
        facts: html.getAttribute("data-facts"),
        before: html.getAttribute("data-before"),
        after: html.getAttribute("data-after"),
        around: html.getAttribute("data-around"),
        run: html.getAttribute("data-run"),
        body: html.getAttribute("data-body"),
        get: get,
        status: status,
        keep: function (update) {
          keepRows(document, update);
        },
        shown: function () {
          if (page && html.getAttribute("data-before") !== null) {
            page.scrollTop = page.scrollHeight;
          }
        }
      });
    });
  }

  if (here !== null && document.documentElement.getAttribute("data-facts") !== null) {
    if (document.body) {
      boot();
    } else {
      document.addEventListener("DOMContentLoaded", boot);
    }
  }

  return { start: start, tabs: tabs, placeOf: placeOf, LOOKS: LOOKS, THEMES: THEMES, RUNS: RUNS, PANEL: PANEL, TEXT: TEXT };
})();
