// web/contexts.js: the host's part of pythia's contexts (lem/contexts.lem)
// in a browser, which the pages that wui serves for its index of contexts,
// and for a context's log, run. A page gives, on its <html> element,
// `data-facts`, the URL of its facts, JSON steps (src/lem/host.rs); and,
// for a log, `data-body`, the URL of an item's body, steps of Body facts,
// with `{line}` in place of the item's line. The compiled program and
// glui's themes are beside this script, in compiled/. It shows the page
// in glui's widgets (src/glui/glui.js), in the look and the theme that its
// bar chooses, which the browser remembers; and it answers the program's
// Load(line) with the item's body, or with Failed(line, why). Like lem.js,
// it's a plain script that defines one global, `contexts`, and it starts
// itself on a page that gives `data-facts`.
var contexts = (function () {
  "use strict";

  // The looks and themes, each [name, what the bar calls it], the first the
  // default: wui's own default is Log dark.
  var LOOKS = [
    ["log", "Log"],
    ["native", "Native"]
  ];
  var THEMES = [
    ["dark", "dark"],
    ["light", "light"]
  ];

  // Shows a page of a compiled program, contexts.lem's, in a view in
  // `o.root`, and is its host:
  //
  // - o.get(url, done) fetches JSON, and calls done(null, value), or
  //   done(why, null) if it can't, later rather than at once;
  // - o.facts is the URL of the page's facts, and o.body the URL of an
  //   item's body, with {line}, or null;
  // - o.status(text, wrong), if given, hears how it went: how long the
  //   page took to show, and what went wrong.
  //
  // It asks for a body once while it's coming, and again when the program
  // asks again after it failed; meanwhile the entry no longer says why. A
  // body that can't come, or that the program refuses, fails, which the
  // entry says. Gives the view.
  function start(o) {
    var status = o.status || function () {};
    var runtime = new glem.Runtime(new glem.Program(o.program));
    var loading = new Set();
    var failed = new Map();
    // Why the program refused the host's last change, or null.
    var refused = null;
    var view = glui.view(runtime, o.root, {
      effects: function (effects) {
        effects.forEach(function (effect) {
          if (effect.name === "Load") {
            load(effect.fields.line);
          }
        });
      },
      error: function (message) {
        refused = message;
        status("error: " + message, true);
      }
    });

    // An item's body fails: the entry says why.
    function fail(line, why) {
      failed.set(line, why);
      view.update([["+", "Failed", { line: line, why: why }]]);
    }

    function load(line) {
      if (o.body === null || loading.has(line)) {
        return;
      }
      loading.add(line);
      if (failed.has(line)) {
        view.update([["-", "Failed", { line: line, why: failed.get(line) }]]);
        failed.delete(line);
      }
      o.get(o.body.split("{line}").join(String(line)), function (why, steps) {
        loading.delete(line);
        if (why === null && !Array.isArray(steps)) {
          why = "its body isn't a list of steps";
        }
        if (why !== null) {
          fail(line, why);
          return;
        }
        refused = null;
        view.update(steps);
        if (refused !== null) {
          fail(line, refused);
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
      }
    });
    return view;
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

  // A choice that the browser remembers, if it lets the page: one of
  // `choices`, else the first.
  function remembered(name, choices) {
    var value = null;
    try {
      value = window.localStorage.getItem("glui." + name);
    } catch (e) {
      value = null;
    }
    var known = choices.some(function (choice) {
      return choice[0] === value;
    });
    return known ? value : choices[0][0];
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

  // The page: its bar, with the look's and the theme's menus and a line of
  // status; and the view.
  function boot() {
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
    var chosen = { look: remembered("look", LOOKS), theme: remembered("theme", THEMES) };
    menu(bar, "Look", LOOKS, chosen.look, function (value) {
      remember("look", value);
      look(value);
    });
    menu(bar, "Theme", THEMES, chosen.theme, function (value) {
      remember("theme", value);
      theme(value);
    });
    bar.appendChild(line);
    look(chosen.look);
    theme(chosen.theme);
    get(new URL("compiled/contexts.json", here).href, function (why, program) {
      if (why !== null) {
        status("couldn't load the program: " + why, true);
        return;
      }
      start({
        root: root,
        program: program,
        facts: html.getAttribute("data-facts"),
        body: html.getAttribute("data-body"),
        get: get,
        status: status
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

  return { start: start, LOOKS: LOOKS, THEMES: THEMES };
})();
