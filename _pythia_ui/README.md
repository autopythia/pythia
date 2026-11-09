# pythia_ui

Screen info and screenshots for pythia's `ui` tool, on Apple Silicon Macs
with macOS 11 or later. This is phase 1 of `doc/interaction-ui-tool-plan.md`:
the ops `info` and `screenshot`, and no input yet. It has no crates.io
dependencies, and it builds on Linux, or on a Mac without Xcode or Apple's
SDK, as research-gui does.

One dispatcher (`src/lib.rs`) takes a JSON request and returns a JSON reply.
Two layers expose it, with the same requests and replies:

- **The plugin**, `libpythia_ui.dylib`, which pythia loads into its own
  process with ctypes (`--enable-experimental-ui-plugin`, from
  `~/.pythia/plugin/`, where `python3 install.py -e
  --enable-experimental-ui-plugin` builds and installs it). Its C interface
  is three functions (below).
- **The program**, `pythia-ui`: one request per line on stdin, one reply per
  line on stdout. It's for trying the crate by hand now, and for pythia on
  another machine later.

## Build

You need a rustup toolchain with Rust 1.78 or later, which includes the
`rust-lld` linker (Homebrew's Rust doesn't), with the macOS target, and
`python3` for `tools/`:

```sh
rustup target add aarch64-apple-darwin   # on Linux; a Mac has it already
make               # target/aarch64-apple-darwin/debug/: pythia-ui, libpythia_ui.dylib
make release       # the same, in release/
make test          # the unit tests, on this machine
make check         # rustfmt; clippy for macOS and for this machine; and
                   # every macOS binary, tests too, linked against the stubs
make bench         # times lem at the size of a long log, and on a hard join
make js-test       # lem.js, glem.js and glui.js, the runtimes in JavaScript,
                   # on the tests, in JavaScriptCore; elsewhere with
                   # another engine, JS=qjs
make js-bench      # times pythia's contexts in JavaScript, at log size
make web           # what the web look's pages fetch, in target/web/: the
                   # programs and themes, compiled, the tests' cases, and
                   # a sample of pythia's contexts
make web-serve     # then serves the crate on 127.0.0.1:8000, for a browser
make compiled      # web/compiled/: the contexts and the themes, compiled
make msrv          # check and test with Rust 1.78.0, the oldest supported
make inspect       # from any host: will the binaries load on macOS?
make plugin-check  # loads this machine's build of the plugin with ctypes
```

Plain `cargo build` works too: `.cargo/config.toml` picks the target, the
linker and the stub SDK. These commands work the same on Linux and on a Mac.

## Try it on the Mac

In a terminal, in the pythia checkout:

```sh
cd _pythia_ui
make && make test && make inspect && make plugin-check
target/aarch64-apple-darwin/debug/pythia-ui info
target/aarch64-apple-darwin/debug/pythia-ui screenshot shot.png
```

That first screenshot fails with a `permission` error, and macOS asks, once,
whether the terminal may record the screen. Allow it in System Settings >
Privacy & Security > Screen Recording (Screen & System Audio Recording on
macOS 15 and later), quit and reopen the terminal, and then:

```sh
target/aarch64-apple-darwin/debug/pythia-ui screenshot shot.png
target/aarch64-apple-darwin/debug/pythia-ui screenshot --capture coregraphics cg.png
make plugin-check SCREENSHOT=plugin.png   # with the Python that runs pythia
```

What should happen:
- `make test` passes on the Mac too. That also proves the stubs: dyld refuses
  to load a binary that imports a symbol macOS's libraries don't export.
- `info` lists each display, with its bounds in points and its size in
  pixels; `capture.paths` (`screencapturekit` and `coregraphics` on macOS 14
  and later, `coregraphics` alone before); and whether capture and input
  events are allowed.
- `shot.png` has the display's own pixels, such as 3024×1964 on a 14-inch
  MacBook Pro at its default resolution, with the cursor; `cg.png` the same,
  by the older path, without the cursor.
- Without the permission, a screenshot is a `permission` error, never a
  blank image.
- `make plugin-check` passes, which shows that this Python loads the plugin.

## The protocol

A request is a JSON object with an `op`; its `id`, if any, comes back first
in the reply. A reply has `"ok": true` and the op's fields, or `"ok": false`,
a `kind` and an `error`, which is text for the model to read. The kinds:
- `request`: the request is malformed: not JSON, without an op, or with a
  value of the wrong type;
- `unsupported`: this build or this macOS can't do it: an op or a field that
  it doesn't know, or a capture path that this macOS lacks;
- `permission`: macOS doesn't allow it yet;
- `failed`: it was tried, and it failed.

A reply is compact JSON with no raw newline, so the program writes each on one
line. A request takes at most 10 seconds of waiting for macOS.

`info` takes no fields. Its reply, wrapped here:

```json
{"id":1,"ok":true,"name":"pythia_ui","version":"0.1.0","abi":1,"ops":["info","screenshot"],
 "system":{"os":"macos","version":"14.5","arch":"aarch64"},
 "displays":[{"id":1,"main":true,"builtin":true,"bounds":{"x":0,"y":0,"width":1512,"height":982},
              "pixels":{"width":3024,"height":1964},"scale":2}],
 "capture":{"paths":["screencapturekit","coregraphics"],"default":"screencapturekit","allowed":true},
 "input":{"allowed":false}}
```

`screenshot` takes `display`, an ID from `info` (default: the main display);
`cursor` (default `true`); and `capture`: `auto` (the default),
`screencapturekit` or `coregraphics`. Its reply gives the PNG in base64, at the
display's own pixel size, and the display's bounds in points, which map the
image's pixels to the screen. The bounds are in macOS's global display
coordinates: the origin is the main display's top-left corner, y grows
downward, and input events use the same space.

```json
{"id":2,"ok":true,"display":1,"capture":"screencapturekit","width":3024,"height":1964,
 "bounds":{"x":0,"y":0,"width":1512,"height":982},"cursor":true,"media_type":"image/png",
 "bytes":4123456,"png":"iVBORw0KGgo..."}
```

ScreenCaptureKit (`SCScreenshotManager`) arrived in macOS 14, and is the
default there. CoreGraphics (`CGDisplayCreateImage`) is on every macOS, but
deprecated from 14; it's the default before, and never draws the cursor.
Which paths a Mac has is checked at run time, not by its version number.

## The plugin interface

```c
uint32_t pythia_plugin_abi(void);  /* 1 */
int32_t pythia_plugin_call(const uint8_t *request, size_t request_len,
                           uint8_t **reply, size_t *reply_len);
void pythia_plugin_free(uint8_t *reply, size_t reply_len);
```

- `pythia_plugin_call` answers one request of the protocol: it sets `*reply`
  and `*reply_len` and returns 0, even when the reply reports a failure. It
  returns 1, setting nothing, only when `reply` or `reply_len` is null.
- A reply belongs to the plugin until `pythia_plugin_free`, which takes the
  same length.
- A Rust panic becomes a `failed` reply and prints nothing. An Objective-C
  exception, or a crash in unsafe code, ends the host process.
- A call blocks for up to 10 seconds; ctypes releases the GIL meanwhile. The
  plugin waits for ScreenCaptureKit's completion handlers itself, with that
  deadline, and a handler that answers late finds live memory.
- The ABI version changes only with these three functions' signatures or
  contracts; `info` gives the plugin's version and ops.
- The plugin is never unloaded. Replace its file by renaming a new one over it
  (as `rsync`, `mv` and `install.py` do), never by writing into it: macOS can
  kill a process whose loaded library changes underneath it.

## lem

lem is a prototype of the rule language for pythia's GUI that
`doc/pythia-ui-gui-design.md` designs: a typed, Datalog-like language.
Rules derive facts from facts, with negation and recursion, and `on`
blocks turn messages into state changes and effects, as Elm's update does.
Its frontend follows the conventions of zinterp's Pratt parser
(`_pythia/src/zinterp/parse.rs`). It runs on any system: `make test`
checks it with the programs and cases in `lem/test/`, and the program
runs ticks given on stdin, a tick at each blank line:

```sh
target/aarch64-apple-darwin/debug/pythia-ui lem lem/test/jobs.lem   # on the Mac
cargo --offline run --target x86_64-unknown-linux-gnu -- lem lem/test/jobs.lem
```

```
+ Job("a", "write", 60000)
+ Now(120000)
? Late
```

`+` asserts a fact for the host, `-` retracts one, `!` sends a message,
and `?` names a relation whose facts to show after the tick. Each tick
prints those facts, then the effects that it sends.

A relation's `key` names columns: facts that agree on them must agree on the
rest. The host's new fact, and an `on` block's, takes the place of the one
with its key; two facts that a tick derives with one key fail it. A relation
declared `one` has one fact at most, as a key of no columns would give: a
page's title, a log's length, a setting. Before a tick runs, the checker warns
of a rule that may give one key two facts, or one id two widgets, where keys,
constructors, `one`, negations and tests that must fail don't show that it
can't (`src/lem/clash.rs`); `pythia-ui lem` prints the warnings. A `rel`
declared `lazy` is derived only when something reads it: a rule of a relation
that's derived, an `on` block, or the host, with `?`. glem's `Within` is lazy,
since few programs read it.

A tick runs a relation's rules only when a relation that they read has changed
since they last ran; if its facts come out the same, what reads it doesn't run
either. Once its rules have run, a relation that isn't recursive follows what
changed: each rule keeps how many ways it derives each fact, and joins just
the facts that came to, or went from, each relation that it reads, where that
counts exactly, with the others as they were before the tick where they must
be; an atom with `_` counts exactly where it binds a variable of its own, or
has `_` only outside its relation's key, so that it matches one fact at most,
and a key that doesn't hold makes the next tick run every rule whole; a rule
that reads such a relation in a negation runs whole. A change counts for an
atom or a negation only if a fact that came or went could match it, and, for a
test that binds nothing, such as the `Turn()` that a block's condition gives
each rule inside it, only if the test's truth changed; and, for an atom whose
variables appear nowhere else in its rule, which counts each solution once for
each fact that it matches, only if how many facts it matches changed. A
relation that has no facts to keep, and most of whose inputs are new, starts
over: it runs whole, as on its first tick. The facts, effects and errors are
those of a tick that derives everything anew, which the tests run too, and
compare (`lem/test/rerun.lem`, `lem/test/follow.lem`); and both runtimes'
joins visit the same facts.

For the web, a checked program can be written as data, its compiled form
(`src/lem/compile.rs`): JSON, which lem's runtime in JavaScript,
`src/lem/lem.js`, runs as `src/lem/eval.rs` runs a program. The host's
facts, its messages and the steps of a tick are JSON too
(`src/lem/host.rs`). `make test` runs every tick test on the compiled form,
read back in Rust, with each tick as JSON, and writes the same cases to
`target/web/lem_tests.js`; `make js-test` runs them on `lem.js`, with
`tools/lem_test.js`. The two runtimes must agree on what each tick shows,
and on how many facts its joins visit:

```sh
make js-test                   # on the Mac: JavaScriptCore, through osascript
make js-test JS=/path/to/qjs   # elsewhere, with QuickJS, for example
```

### glem

glem (`src/glem/`) is lem's GUI layer. It adds `widget` and `surface`
declarations, blocks that give the widgets inside them their parent and
order, and the widget tree, which it checks after each tick; state that
names a widget goes with the widget, unless its relation says `keep`.
glem rewrites a program as plain lem, which lem checks and runs. With
glem, each tick prints the widget tree first.

On the web, a glem program's compiled form has glem's part too
(`src/glem/compile.rs`); `pythia-ui lem --compile FILE.lem...` writes it.
glem's runtime in JavaScript, `src/glem/glem.js`, runs it as `src/glem/mod.rs`
runs a program, but keeps the widget tree from tick to tick, and changes it by
the widgets that lem's logs say each tick added and removed. Its view keeps
the widgets in the DOM, and shows only what changed: each widget's element
comes from a recipe for its kind, from the layer above, and stays from tick to
tick while the widget does, so that it keeps its focus and caret. Recipes send
messages from their events; the view fills in `Focused`, sends what `Key`
facts give for Escape, Return and Tab, sends what `Near` facts give for a
widget whose element comes near the screen, and scrolls into view the widget
that a `Reveal` fact names, once for each of its `at`, as soon as it shows: to
the middle of the screen, marked for a look to show, or, if the fact says
`top`, to the top, unmarked. `make js-test` runs glem's ticks on `glem.js`, as
Rust runs them, checks the tree that it keeps against one built anew, and runs
the view in a stand-in DOM (`tools/stand_in_dom.js`), with the clicks, focus
and keys of `lem/test/tasks_dom.txt`.

### glui

glui (`src/glui/`) is the design language of pythia's GUI, on glem: a library
in lem, with no syntax or meaning of its own. `widgets.lem` has its widget
kinds, among them those that documents need (headings, links, tables, a log's
entries and code) a dot of a hue, a light that says a state, tabs, a strip of
them (`Tabs`, `TabItem`), and prose, text as Markdown writes it (`Prose`), and
`theme.lem` says what a theme gives: the colors, sizes and times that a front
end draws with. `themes/dark.lem`, the default, and `light.lem` give them,
after wui's log styles. `pythia-ui lem` loads glui's files and a theme first:
`--theme light`, or `--theme FILE.lem` for another, which it refuses if a
value is missing. The example is the countdown after approval,
`lem/approval.lem`, with what pythia sends and accepts, `lem/host.lem`:

```sh
cargo --offline run --target x86_64-unknown-linux-gnu -- lem lem/host.lem lem/approval.lem
```

```
+ Pending("toolu_1", "ui", "screenshot()", 1000)
+ Countdown("toolu_1", 5000, 4000)
+ Now(3000)

! Withdraw("toolu_1")
```

On the web, `src/glui/glui.js` draws them, with a recipe for each kind for
glem.js's view. It makes each widget a native element (`button`, `a`,
`details`, `table`, `pre`, `progress`) with classes: `g-` and its kind's name,
and its flags. Its widgets send `Edited` and `Toggled`, a tab `Chosen`, its
close button `Dismissed`, and one dragged onto another `Moved`, it shows
`Disabled` and `Tooltip`, a link goes only to one of the host's own paths, a
dot says its text as an image's label and in its tooltip, and prose shows
Markdown as elements, never as HTML, its links to the host's own paths, or to
http, https or mailto in a new tab, without a referrer, in colors of its own
that show on light stages and dark ones. It sets a theme's values on the page
as CSS variables, `--glui-` and a value's name, and a look's stylesheet draws
with them, over `web/glui.css`: `web/native.css`, the design doc's native
look, or `web/log.css`, after wui's log. `make js-test` runs its cases in a
stand-in DOM (`lem/test/*_dom.txt`), with a form of its controls,
`lem/test/form.lem`.

In a browser, `make web-serve` serves the web look's two pages.
`web/demo.html` shows the approval HUD, a page of a log, or the form, in
either look and any theme, with the page as their host. `web/test.html` runs
the tests in JavaScript, and some of glui's cases in the page's own DOM too.
Both pages carry the CSP that pythia's node server sends the pages of its
glui view, but `frame-ancestors`, which only a header can give, and run
only the crate's files:

```sh
make web-serve     # then http://127.0.0.1:8000/web/demo.html, and test.html
```

### The contexts

`lem/contexts.lem` shows pythia's contexts in glui's widgets: the index of the
root contexts, and one context's log, a page each. Its host is the glui view
of pythia's node server (`pythia/interaction/node/view.py` in pythia), at
`/TOKEN/lem/`. It sends a page's facts: `Context` for each context, with the
label that it's given, if any, which the index shows in a column of its own,
and a `Tagged` for each of its tags; with them, its tags as one text, and the
URL that keeps them, by which its row has a ✎, which opens a row under it,
with a field for its label and one for its tags, which Save, or Return, sends
to the host (`KeepRoot(to, label, tags)`), and Cancel, or Escape, closes; and,
for each remote node whose roots the index shows after this node's, a `Node`,
which says how many it has, or why it can't be reached, and dims its row,
before its roots, by ids of their own, as `NAME/ID`; or, for a log, its
length, `Length`, a `Turn` for each turn, with the line where it starts, and
the log's last rows: a `Chunk`, the lines that they cover, and a row for each
message, with its text, `Message`, and for each run of other items between two
messages within a turn, with what's in it, `Run`. Before each context, in the
index, and before a log's title, a dot (glui's `Dot`) says what the context is
doing, as the host says (`Activity`): gray, no CLI runs on it; green, a CLI
works on a turn; yellow, a CLI waits, for input, a login or icron; orange, it
retries a model request that failed; red, its last turn failed, or its CLI
stopped in the middle of one. An assistant's messages show as Markdown writes
them, as the host says (`Formatting`), and a user's as written. A log's page
shows its rows in order, each turn marked where it starts, and each run
folded. As either end of the rows comes near the screen, the program sends
`LoadBefore(line)` or `LoadAfter(line)` (glem's `Near`), and the host answers
with another chunk, or with `EarlierFailed` or `LaterFailed`; so a page holds
what's been scrolled through. As a log grows, the host says its new `Length`,
in place of the last, and the turns that changed; the program then asks for
the later rows as it does for others, and the run that the log ended with
comes whole, with its new items, if it went on. A run's items come when it's
first opened: the program sends `LoadRun(run)`, and the host answers with
`RunLoaded(run)` and an `Item` for each, or with `RunFailed(run, why)`. The
host's `Opening(open)` opens every run, or folds them all, through glui's
`Entry`'s `open`, and each still opens and folds alone; a run that's open asks
for its items as it comes near the screen. A jump to a line (`Jump(line)`)
asks for the rows around it (`LoadAround(line)`), opens the run that holds it,
and brings its row into view (glem's `Reveal`); where the stretches of rows
that came don't meet, a marker at either side asks for the rows between as it
comes near the screen. A message, and an item, has a star (glui's `Entry`'s
`star`), faint until it's set: setting or clearing it sends `Star(line, on)`,
and the host sets or clears `Stars(line)`. A log's panel shows beside it as
the host says (`Aside(shown)`), in a HUD: the lines starred, under their
turns, as the host sends them (`Pin`), a starred answer with the user's
messages that it answers; and the outline of the log's turns. A click on a pin
jumps to its line, and one on a turn visits it (`Visit(turn)`): a jump to the
line where it first shows that brings its mark into view, and opens nothing.
Restoring a place (`Restore(line)`), as a page does where it was left, is a
jump to the row that starts at a line, which opens nothing either, and brings
the row to the top of the screen, unmarked. An item's body comes the same way,
when its entry is first opened: the program sends `Load(line)`, and the host
answers with `Body` facts, or `Failed(line, why)`.
`lem/test/contexts_ticks.txt` and `contexts_dom.txt` test it.

In a browser, `web/contexts.js` is the host's part. The node server's pages
show tabs: the index's, and one for each context that's open, each a runtime
and a view of its own, of which only the tab shown shows, and asks what
changed; a click on a link to a context, without a modifier key, opens its
tab, a tab dragged onto another moves there, and the address follows the tab
shown, which the server's store keeps, with the tabs open and where each was
left: the line of its first row on the screen, or its end, to which it goes
back after a reload. `lem/tabs.lem` is their strip, in a HUD at the top: the
host says which are open (`Open(n, title, rank)`, the index as 0) and which
shows (`Shown(n)`), and hears a pick (`Show(n)`), a close (`Close(n)`) and a
move (`Move(n, to)`); the index can't be closed. A page names its facts on its
`html` element, `data-facts`, and, for a log, where the rows before a line and
after one are, `data-before`, `data-after` and `data-around`, where a run's
items are, `data-run`, where an item's body is, `data-body`, each with
`{line}` or `{run}`, where what the log gained since it had `{lines}` lines
is, `data-grow`, where its stars are kept, `data-stars`, by the root's key,
and where its panel's pins are, `data-pins`, which it asks for anew once a
star is kept; it keeps a context's label and tags as `KeepRoot` says, the tags
as words, in lowercase, and then asks for the index's facts anew; the script
fetches the program and the themes, compiled, from `web/compiled/`, which
`make compiled` writes, and a test checks against the files they're compiled
from. A log's page starts at its end, and keeps the first row on the screen in
place as rows come, above it or below. Every few seconds while it shows, a
page asks what changed: a log, what it gained; the index, its facts anew,
whose difference from the last it applies. Its bar picks the look and the
theme, and, for a log, whether its runs are folded or open, whether its panel
shows, and how wide, as a drag on its edge makes it, and whether an
assistant's messages show as Markdown or as written, which the node server
keeps for the view, at `data-store`, and the browser too. `make web` writes a
sample, an index and a log of 13,394 items, the most that one of pythia's
contexts had when it was first written, in turns of 50, for two pages that
show it as the node server's do:

```sh
make web-serve     # then http://127.0.0.1:8000/web/contexts-index.html
make js-bench      # the same log in JavaScriptCore, here as JS=qjs
```

## Layout

```
src/lib.rs         the dispatcher, the protocol, and the plugin's C functions
src/main.rs        the program
src/json.rs        a small JSON reader and writer
src/base64.rs      base64, for the PNG
src/pending.rs     a result that a completion handler hands over, with a deadline
src/elsewhere.rs   other systems: info, and screenshots unsupported
src/lem/           lem (below): token.rs, parse.rs, print.rs, value.rs,
                   check.rs, eval.rs, its prelude.lem; compile.rs and
                   host.rs (its JSON), lem.js (its runtime in JavaScript);
                   and the tests
src/glem/          glem, lem's GUI layer: lower.rs (the rewrite into plain
                   lem), mod.rs (the tree), its prelude.lem; compile.rs
                   (its part of the compiled form), glem.js (its runtime and
                   view in JavaScript); and the tests
src/glui/          glui, the design language: widgets.lem, theme.lem,
                   themes/, mod.rs (loading them), glui.js (its web look's
                   recipes), and the tests
lem/               the approval example (host.lem, approval.lem), the
                   contexts (contexts.lem), and test/, the tests' programs
                   and cases
web/               the web look's stylesheets, glui.css, native.css and
                   log.css; its pages, demo.html and test.html; the
                   contexts' script and stylesheet, contexts.js and
                   contexts.css, their sample pages, and compiled/
src/macos/         objc.rs (the Objective-C runtime), block.rs (completion
                   blocks), cf.rs, cg.rs (displays, permissions), png.rs
                   (ImageIO), capture.rs (the two capture paths)
build.rs           the plugin's install name, @rpath/libpythia_ui.dylib
stub-sdk/          link-time symbol lists (below)
tools/             macho_info.py (make inspect), gen_tbd.py (make tbd),
                   plugin_check.py (make plugin-check), lem_bench.py
                   (make bench); lem_test.js, glem_test.js, glui_test.js,
                   contexts_test.js, tap.js and stand_in_dom.js (make
                   js-test); contexts_bench.js (make js-bench); and
                   contexts_sample.py (make web)
```

## Stub SDK

The linker must know which library exports each symbol that a binary uses.
Instead of Apple's SDK, `stub-sdk/` holds text stubs (`.tbd`) that list only
names; at run time, macOS's own libraries supply the code.

- `usr/lib/libSystem.tbd` is generated by `make tbd` from what the program,
  the plugin and the unit tests import. It only grows: each run keeps what
  the stub lists, and adds what that Rust version's std imports, so one stub
  serves several versions. Rust 1.78's std imports `pthread_cond_timedwait`
  and `getentropy`, where 1.93's imports `pthread_cond_timedwait_relative_np`
  and `CCRandomGenerateBytes`. `libc.tbd` and `libm.tbd` are symlinks to it,
  as in Apple's SDK.
- `usr/lib/libobjc.tbd`, and the CoreFoundation, CoreGraphics and ImageIO
  stubs, are hand-written: the functions that `src/macos/` declares.
- Foundation's stub lists nothing: classes are looked up by name, so linking
  only makes dyld load it.
- ScreenCaptureKit has no stub: it's loaded with `dlopen`, since macOS 11
  lacks it, and its classes are looked up by name. `CGDisplayCreateImage` is
  looked up with `dlsym`, so that a macOS without it still loads the plugin.
- Version numbers are roughly the macOS 11 SDK's, to match the deployment
  target.

If a link fails with `undefined symbol`:
- For a libc or std function (new std usage, or another Rust version), run
  `make tbd` with the Rust whose link failed.
- For an `objc_`/`CF`/`CG` function, add it to that library's `.tbd` by hand;
  `make tbd` refuses these.

A symbol listed in the wrong library links, but shows up at launch as
`dyld: Symbol not found`; `python3 tools/macho_info.py -v BINARY` lists the
symbols per library. On a Mac with Xcode or its command-line tools,
`SDKROOT="$(xcrun --show-sdk-path)" cargo build` links against the real SDK
instead, as a cross-check.

## From research-gui

Copied from `../research-gui/_glomery` at commit a5099c9, which stays as it
was:
- `src/macos/objc.rs`, from `src/objc.rs`, trimmed and changed for a library
  that runs inside another program (its header lists the changes);
- `tools/gen_tbd.py`, which now keeps what the stub lists (`--replace`
  does as before); `tools/macho_info.py`, which learned dylibs;
- the build settings (`.cargo/config.toml`, `rustfmt.toml`), the Makefile's
  structure, and the stub format (libobjc's stub lists other functions).
