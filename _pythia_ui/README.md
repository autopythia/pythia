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
make js-test       # lem.js, lem's runtime in JavaScript, on lem's tests, in
                   # JavaScriptCore; elsewhere with another engine, JS=qjs
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

### glui

glui (`src/glui/`) is the design language of pythia's GUI, on glem: a
library in lem, with no syntax or meaning of its own. `widgets.lem` has
its widget kinds, among them those that documents need (headings, links,
tables, a log's entries and code), and `theme.lem` says what a theme
gives: the colors, sizes and times that a front end draws with.
`themes/dark.lem`, the default, and `light.lem` give them, after wui's
log styles. Nothing draws them yet. `pythia-ui lem` loads glui's files
and a theme first: `--theme light`, or `--theme FILE.lem` for another,
which it refuses if a value is missing. The example is the countdown
after approval, `lem/approval.lem`, with what pythia sends and accepts,
`lem/host.lem`:

```sh
cargo --offline run --target x86_64-unknown-linux-gnu -- lem lem/host.lem lem/approval.lem
```

```
+ Pending("toolu_1", "ui", "screenshot()", 1000)
+ Countdown("toolu_1", 5000, 4000)
+ Now(3000)

! Withdraw("toolu_1")
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
                   lem), mod.rs (the tree), its prelude.lem, and the tests
src/glui/          glui, the design language: widgets.lem, theme.lem,
                   themes/, mod.rs (loading them), and the tests
lem/               the approval example (host.lem, approval.lem), and test/,
                   the tests' programs and cases
src/macos/         objc.rs (the Objective-C runtime), block.rs (completion
                   blocks), cf.rs, cg.rs (displays, permissions), png.rs
                   (ImageIO), capture.rs (the two capture paths)
build.rs           the plugin's install name, @rpath/libpythia_ui.dylib
stub-sdk/          link-time symbol lists (below)
tools/             macho_info.py (make inspect), gen_tbd.py (make tbd),
                   plugin_check.py (make plugin-check), lem_bench.py
                   (make bench), lem_test.js (make js-test)
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
