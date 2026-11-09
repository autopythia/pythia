// web/contexts.js: the host's part of pythia's contexts (lem/contexts.lem)
// in a browser, which the glui view of pythia's node server runs on its
// pages, for the root contexts and for a context's log
// (pythia/interaction/node/view.py). A page gives, on its <html> element,
// `data-facts`, the URL of its facts, JSON steps (src/lem/host.rs); and,
// for a log, `data-before` and `data-after`, the URLs of the rows before a
// line and after one, `data-run`, the URL of a run's items, and
// `data-body`, the URL of an item's body, each with `{line}` or `{run}` in
// place of its line; and `data-grow`, the URL of what the log gained since it
// had `{lines}` lines, and `data-stars`, the URL that keeps its stars; and,
// on each page, `data-store`, the URL of the settings that the server keeps
// for the view (pythia/interaction/node/store.py). The compiled program and
// glui's themes are beside
// this script, in compiled/. It shows the page in glui's widgets
// (src/glui/glui.js), in the look and the theme that its bar chooses,
// which the server keeps, and the browser too, from the log's end, with
// every run folded or every run open, as the bar's Runs says (Opening); and
// it answers the
// program's LoadBefore(line) and LoadAfter(line) with rows, or with
// EarlierFailed(line, why) or LaterFailed(line, why); its LoadRun(run)
// with the run's items, or with RunFailed(run, why); and its Load(line)
// with the item's body, or with Failed(line, why). Every few seconds, while
// the page shows, it asks what a log gained, or an index's facts anew, and
// tells the program what changed. Like lem.js, it's a plain script that
// defines one global, `contexts`, and it starts itself on a page that gives
// `data-facts`.
var contexts = (function () {
  "use strict";

  // The looks and themes, each [name, what the bar calls it], the first the
  // default, as the node server's plain pages are in Log dark.
  var LOOKS = [
    ["log", "Log"],
    ["native", "Native"]
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

  // Shows a page of a compiled program, contexts.lem's, in a view in
  // `o.root`, and is its host:
  //
  // - o.get(url, done) fetches JSON, and calls done(null, value), or
  //   done(why, null) if it can't, later rather than at once;
  // - o.facts is the URL of the page's facts; o.before and o.after, the
  //   URLs of the rows before a line and after one, with {line}; o.run, the
  //   URL of a run's items, with {run}; and o.body, the URL of an item's
  //   body, with {line}; each null for the index;
  // - o.status(text, wrong), if given, hears how it went: how long the
  //   page took to show, and what went wrong;
  // - o.keep(update), if given, runs update(), which shows rows that came
  //   before those on the page, keeping what's on the screen in place; and
  //   o.shown(), if given, hears that the page's facts are shown;
  // - o.opening, for a log, says whether every run opens, as an Opening
  //   fact from the start, which the bar changes later (boot);
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
  //   taken back, and the page says why.
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
    // The lines starred, as the page shows them (Stars).
    var starred = new Set();
    var view = glui.view(runtime, o.root, {
      effects: function (effects) {
        effects.forEach(function (effect) {
          if (effect.name === "Star") {
            star(effect.fields.line, effect.fields.on);
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
    });
    if (typeof o.opening === "boolean") {
      view.update([["+", "Opening", { open: o.opening }]]);
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
          }
        });
      }
    }

    // What was asked for, `value`, fails: the turn or the entry says why.
    function fail(ask, value, why) {
      var fields = { why: why };
      fields[ask.field] = value;
      failed.set(ask.failed + " " + value, fields);
      view.update([["+", ask.failed, fields]]);
    }

    function load(ask, value) {
      var name = ask.failed + " " + value;
      if (ask.url === null || ask.url === undefined || loading.has(name)) {
        return;
      }
      loading.add(name);
      if (failed.has(name)) {
        view.update([["-", ask.failed, failed.get(name)]]);
        failed.delete(name);
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
        if (ask.failed === "EarlierFailed") {
          keep(function () {
            view.update(steps);
          });
        } else {
          view.update(steps);
        }
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
      });
    }
    if (o.every) {
      o.every(tick);
    }
    return view;
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

  // Shows rows that came before those on the page, by update(), keeping
  // what's on the screen in place: they're all above it, so the page
  // scrolls down as far as they reach. The page's own scroll anchoring is
  // off (web/contexts.css), so that only this moves it.
  function keep(update) {
    var page = document.scrollingElement;
    var height = page ? page.scrollHeight : 0;
    update();
    if (page) {
      page.scrollTop += page.scrollHeight - height;
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
    var bar = document.createElement("nav");
    var line = document.createElement("span");
    var root = document.createElement("div");
    bar.setAttribute("class", "contexts-bar");
    line.setAttribute("class", "contexts-status");
    document.body.appendChild(bar);
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
      runs: remembered("runs", RUNS, kept.runs)
    };
    // A setting that changed: the browser remembers it, and the server's
    // store keeps them all.
    var save = function (name, value) {
      chosen[name] = value;
      remember(name, value);
      if (store !== undefined) {
        keepIn(store, chosen, function (why) {
          if (why !== null) {
            status("couldn't keep the settings: " + why, true);
          }
        });
      }
    };
    // The view, once the program has come; and whether the page is a log.
    var shown = null;
    var log = html.getAttribute("data-run") !== null;
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
        if (shown !== null) {
          shown.update([
            ["-", "Opening", { open: chosen.runs === "open" }],
            ["+", "Opening", { open: value === "open" }]
          ]);
        }
        save("runs", value);
      });
    }
    bar.appendChild(line);
    look(chosen.look);
    theme(chosen.theme);
    get(new URL("compiled/contexts.json", here).href, function (why, program) {
      if (why !== null) {
        status("couldn't load the program: " + why, true);
        return;
      }
      shown = start({
        root: root,
        program: program,
        opening: log ? chosen.runs === "open" : undefined,
        grow: html.getAttribute("data-grow"),
        stars: html.getAttribute("data-stars"),
        post: keepIn,
        every: function (tick) {
          var showing = function () {
            if (document.visibilityState !== "hidden") {
              tick();
            }
          };
          window.setInterval(showing, EVERY_MS);
          document.addEventListener("visibilitychange", showing);
        },
        facts: html.getAttribute("data-facts"),
        before: html.getAttribute("data-before"),
        after: html.getAttribute("data-after"),
        run: html.getAttribute("data-run"),
        body: html.getAttribute("data-body"),
        get: get,
        status: status,
        keep: keep,
        shown: function () {
          var page = document.scrollingElement;
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

  return { start: start, LOOKS: LOOKS, THEMES: THEMES, RUNS: RUNS };
})();
