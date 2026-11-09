// web/demo.js: glui's demo (web/demo.html). It shows one of the programs
// that `make web` compiles to target/web/ in a view, through glui's
// recipes (src/glui/glui.js), in either look, and in any theme that `make
// web` compiles; and it's the program's host, as pythia's node server is:
// its buttons send the program's facts, as JSON steps (src/lem/host.rs),
// and it lists what the program sends back, and answers some of it. The
// page's address keeps its choices, as ?program=approval&look=native&
// theme=dark.
(function () {
  "use strict";

  // Where `make web` writes the compiled programs and themes.
  var WEB = "../target/web/";

  // The looks: the class that each one's stylesheet applies under, and
  // its name in the menu.
  var LOOKS = [
    ["native", "Native"],
    ["log", "Log"]
  ];

  // The programs: each one's name, its title, its compiled form's file,
  // and its host, below.
  var PROGRAMS = [
    ["approval", "The approval HUD", "approval.json", Approval],
    ["page", "A page of a log", "page.json", Page],
    ["form", "A form", "form.json", Form]
  ];

  // A program's host gives the steps that start the program (start), and
  // its buttons, each a label and a function that gives steps; it may
  // answer an effect that the program sends, by steps (answer), and keep
  // time, giving steps four times a second (clock). Either may say what
  // happens with say(text).

  // The approval HUD (lem/approval.lem, with lem/host.lem): calls that
  // wait to run, each with a countdown, which the HUD runs now or
  // withdraws. The host takes a call away once it's answered, or once its
  // countdown ends and it runs, as pythia would.
  var CALLS = [
    ["ui", "screenshot()"],
    ["ui", "click(120, 48)"],
    ["exec_command", "make test"]
  ];

  function Approval() {
    var host = this;
    this.now = Date.now();
    this.calls = new Map();
    this.next = 1;
    this.allowed = false;
    this.buttons = [
      [
        "A call",
        function () {
          return host.call();
        }
      ],
      [
        "Allow input",
        function () {
          return host.allow(true);
        }
      ],
      [
        "Forbid input",
        function () {
          return host.allow(false);
        }
      ]
    ];
  }

  Approval.prototype.start = function () {
    return [["+", "Now", { t: this.now }]].concat(this.call());
  };

  // A call, whose countdown takes ten seconds.
  Approval.prototype.call = function () {
    var n = this.next++;
    var what = CALLS[(n - 1) % CALLS.length];
    var id = "toolu_" + n;
    var pending = { call: id, tool: what[0], summary: what[1], since: this.now };
    var countdown = { call: id, ends: this.now + 10000, total: 10000 };
    this.calls.set(id, [pending, countdown]);
    return [
      ["+", "Pending", pending],
      ["+", "Countdown", countdown]
    ];
  };

  Approval.prototype.allow = function (allowed) {
    if (allowed === this.allowed) {
      return [];
    }
    this.allowed = allowed;
    return [[allowed ? "+" : "-", "Allowed", { what: "Input" }]];
  };

  // The steps that take a call away, if it's still there.
  Approval.prototype.drop = function (id) {
    var facts = this.calls.get(id);
    if (facts === undefined) {
      return [];
    }
    this.calls.delete(id);
    return [
      ["-", "Countdown", facts[1]],
      ["-", "Pending", facts[0]]
    ];
  };

  // Send(Approve(c)), and the call runs; or Send(Deny(c)), and it doesn't.
  Approval.prototype.answer = function (effect, say) {
    var msg = effect.fields.msg;
    say(msg[1] + (msg[0] === "Approve" ? " runs, approved" : " won't run, withdrawn"));
    return this.drop(msg[1]);
  };

  // The time, and the calls whose countdowns have ended, which run.
  Approval.prototype.clock = function (say) {
    var host = this;
    var then = this.now;
    this.now = Date.now();
    var steps = [
      ["-", "Now", { t: then }],
      ["+", "Now", { t: this.now }]
    ];
    this.calls.forEach(function (facts, id) {
      if (facts[1].ends <= host.now) {
        say(id + " runs, its countdown over");
        steps = steps.concat(host.drop(id));
      }
    });
    return steps;
  };

  // A page of a log (lem/test/page.lem): contexts in a table, with links,
  // one of them to elsewhere, which glui refuses; and a log's lines, some
  // of which open to show their text, one of them a diff.
  var LINES = [
    ["[user]", "Let's start stage 8.", "User", null],
    ["[assistant]", "I'll read the plan for the web look.", "Assistant", null],
    [
      "[tool-call] exec_command",
      "git diff web/glui.css",
      "Tool",
      [
        "diff --git a/web/glui.css b/web/glui.css\n@@ -1,2 +1,2 @@\n" +
          "-.g-row{display:flex}\n+.g-row{display:flex;align-items:center}\n .g-aligned{gap:0}",
        true
      ]
    ],
    ["[tool-result] exec_command", "Process exited with code 0", "Normal", ["Chunk ID: 1\nOutput:\nok", false]],
    ["[reasoning]", "Both looks share the recipes.", "Dim", null],
    ["[error]", "The command failed.", "Error", null]
  ];

  var CONTEXTS = [
    ["ui-0", "13.7 MB", "/web/demo.html?program=page"],
    ["dev-1", "48.2 MB", "/web/demo.html?program=page"],
    ["elsewhere", "0 B", "https://example.com/"]
  ];

  function Page() {
    var host = this;
    this.lines = 0;
    this.contexts = [];
    this.buttons = [
      [
        "A line",
        function () {
          return host.line();
        }
      ],
      [
        "A context",
        function () {
          return host.context();
        }
      ],
      [
        "One context fewer",
        function () {
          var fact = host.contexts.pop();
          return fact === undefined ? [] : [["-", "Context", fact]];
        }
      ]
    ];
  }

  Page.prototype.start = function () {
    var steps = [];
    for (var i = 0; i < LINES.length; i++) {
      steps = steps.concat(this.line());
    }
    for (var j = 0; j < CONTEXTS.length; j++) {
      steps = steps.concat(this.context());
    }
    return steps;
  };

  // The log's next line, from the samples in turn.
  Page.prototype.line = function () {
    var n = ++this.lines;
    var sample = LINES[(n - 1) % LINES.length];
    var line = { n: n, label: sample[0], text: sample[1], tone: sample[2], fold: sample[3] !== null };
    var steps = [["+", "Line", line]];
    if (sample[3] !== null) {
      steps.push(["+", "Text_", { n: n, text: sample[3][0], diff: sample[3][1] }]);
    }
    return steps;
  };

  Page.prototype.context = function () {
    var k = this.contexts.length;
    var sample = CONTEXTS[k % CONTEXTS.length];
    var id = k < CONTEXTS.length ? sample[0] : sample[0] + "-" + k;
    var fact = { id: id, name: id, size: sample[1], href: sample[2] };
    this.contexts.push(fact);
    return [["+", "Context", fact]];
  };

  // A form of glui's controls (lem/test/form.lem): its options, a task's
  // progress, and what Save sends.
  function Form() {
    var host = this;
    this.options = 0;
    this.due = 0.25;
    this.buttons = [
      [
        "An option",
        function () {
          return host.option();
        }
      ],
      [
        "Later",
        function () {
          var then = host.due;
          host.due = then >= 1 ? 0 : then + 0.25;
          return [
            ["-", "Due", { frac: then }],
            ["+", "Due", { frac: host.due }]
          ];
        }
      ]
    ];
  }

  Form.prototype.start = function () {
    return this.option().concat(this.option(), this.option(), [["+", "Due", { frac: this.due }]]);
  };

  Form.prototype.option = function () {
    var n = ++this.options;
    return [["+", "Option", { n: n, text: "Option " + n }]];
  };

  // Saved(text): the host keeps it.
  Form.prototype.answer = function (effect, say) {
    say("saved " + JSON.stringify(effect.fields.text));
    return [];
  };

  // The page.

  var html = document.documentElement;
  var heard = document.getElementById("heard");
  var actions = document.getElementById("actions");
  var menus = {
    program: document.getElementById("program"),
    look: document.getElementById("look"),
    theme: document.getElementById("theme")
  };
  var asked = new URLSearchParams(location.search);

  // A line of what the host heard, or of what went wrong; the last 50.
  function hear(text, wrong) {
    var item = document.createElement("li");
    item.textContent = text;
    if (wrong) {
      item.setAttribute("class", "demo-error");
    }
    heard.appendChild(item);
    while (heard.childNodes.length > 50) {
      heard.removeChild(heard.firstChild);
    }
  }

  function failed(e) {
    hear(e instanceof Error ? e.message : String(e), true);
  }

  // A file that `make web` writes, as JSON.
  function load(file) {
    return fetch(WEB + file, { cache: "no-store" }).then(function (response) {
      if (!response.ok) {
        throw new Error(WEB + file + ": " + response.status + " " + response.statusText + "; make web writes it");
      }
      return response.json();
    });
  }

  // Fills a menu with choices, [value, label], and chooses `value`, else
  // the first; gives what it chose.
  function fill(select, choices, value) {
    select.textContent = "";
    choices.forEach(function (choice) {
      var option = document.createElement("option");
      option.value = choice[0];
      option.textContent = choice[1];
      select.appendChild(option);
    });
    var known = choices.some(function (choice) {
      return choice[0] === value;
    });
    select.value = known ? value : choices[0][0];
    return select.value;
  }

  // Keeps the menus' choices in the page's address.
  function remember() {
    var query = new URLSearchParams();
    Object.keys(menus).forEach(function (name) {
      if (menus[name].value) {
        query.set(name, menus[name].value);
      }
    });
    history.replaceState(null, "", "?" + query.toString());
  }

  function look(name) {
    LOOKS.forEach(function (one) {
      html.classList.toggle("g-" + one[0], one[0] === name);
    });
  }

  // The themes, by name, each a promise of its runtime, after a tick.
  var themes = new Map();

  // Paints the page with a theme, unless another is chosen by then.
  function theme(name) {
    if (!themes.has(name)) {
      var runtime = load("theme-" + name + ".json").then(function (form) {
        var it = new glem.Runtime(new glem.Program(form));
        it.tick(null);
        return it.lem;
      });
      themes.set(name, runtime);
      runtime.catch(function () {
        themes.delete(name);
      });
    }
    themes
      .get(name)
      .then(function (runtime) {
        if (menus.theme.value === name) {
          glui.paint(html, runtime);
        }
      })
      .catch(failed);
  }

  // The program on show: its clock, if it keeps time. `showing` counts the
  // programs asked for, so that only the last one asked for shows.
  var clock = null;
  var showing = 0;

  // Shows a program, by name, in a new view, with a new host, from no
  // facts.
  function program(name) {
    var chosen = PROGRAMS.filter(function (p) {
      return p[0] === name;
    })[0];
    var turn = ++showing;
    if (clock !== null) {
      clearInterval(clock);
      clock = null;
    }
    actions.textContent = "";
    var old = document.getElementById("view");
    var root = document.createElement("div");
    root.id = "view";
    root.setAttribute("class", "demo-view");
    old.parentNode.replaceChild(root, old);
    document.title = chosen[1];
    load(chosen[2])
      .then(function (form) {
        if (turn !== showing) {
          return;
        }
        var host = new chosen[3]();
        var view = glui.view(new glem.Runtime(new glem.Program(form)), root, {
          effects: function (effects) {
            effects.forEach(function (effect) {
              hear("effect " + effect.name + " " + JSON.stringify(effect.fields));
              if (typeof host.answer === "function") {
                send(host.answer(effect, hear));
              }
            });
          },
          error: function (message) {
            hear("error: " + message, true);
          }
        });
        var send = function (steps) {
          if (steps.length > 0) {
            view.update(steps);
          }
        };
        host.buttons.forEach(function (one) {
          var button = document.createElement("button");
          button.type = "button";
          button.textContent = one[0];
          button.addEventListener("click", function () {
            send(one[1]());
          });
          actions.appendChild(button);
        });
        if (typeof host.clock === "function") {
          clock = setInterval(function () {
            send(host.clock(hear));
          }, 250);
        }
        send(host.start());
      })
      .catch(failed);
  }

  var programs = PROGRAMS.map(function (p) {
    return [p[0], p[1]];
  });
  fill(menus.program, programs, asked.get("program"));
  look(fill(menus.look, LOOKS, asked.get("look")));
  program(menus.program.value);
  menus.program.addEventListener("change", function () {
    remember();
    program(menus.program.value);
  });
  menus.look.addEventListener("change", function () {
    remember();
    look(menus.look.value);
  });
  menus.theme.addEventListener("change", function () {
    remember();
    theme(menus.theme.value);
  });
  load("themes.json")
    .then(function (names) {
      var choices = names.map(function (name) {
        return [name, name];
      });
      theme(fill(menus.theme, choices, asked.get("theme") || "dark"));
      remember();
    })
    .catch(failed);
})();
