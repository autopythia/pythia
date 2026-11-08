# Builds pythia_ui for Apple Silicon macOS, the same way on Linux and on a Mac
# (README.md): the plugin, libpythia_ui.dylib, and the program, pythia-ui, in
# target/aarch64-apple-darwin/. From research-gui's (_glomery/Makefile, commit
# a5099c9), with test and plugin-check, which run on this machine, and the
# plugin in tbd and inspect.
CARGO := cargo --offline
PYTHON := python3

TARGET := aarch64-apple-darwin
BIN := pythia-ui
LIB := libpythia_ui.dylib
OUT := target/$(TARGET)
# This machine's own target: x86_64-unknown-linux-gnu here, and TARGET on a Mac.
HOST := $(shell rustc -vV | sed -n 's/^host: //p')
HOST_LIB := target/$(HOST)/release/libpythia_ui.$(if $(findstring apple-darwin,$(HOST)),dylib,so)

.PHONY: all build dev debug rel release test check bench js-test js-bench web web-serve compiled msrv \
  plugin-check tbd inspect clean

all: build

build dev debug:
	$(CARGO) build

rel release:
	$(CARGO) build --release

# The unit tests, on this machine: the plain parts (JSON, base64, the
# protocol, the plugin interface) anywhere, and on a Mac the macOS code too.
test:
	$(CARGO) test --target $(HOST)

# rustfmt; clippy for macOS and for this machine; and every macOS binary,
# the unit tests too, linked against the stubs.
check:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets -- -D warnings
ifneq ($(HOST),$(TARGET))
	$(CARGO) clippy --all-targets --target $(HOST) -- -D warnings
endif
	$(CARGO) build --all-targets

# Times lem, the GUI's rule language, on a log's shape, up to the most items
# that one of pythia's contexts has had, and on triangles in a star, a join
# that needs a good order (tools/lem_bench.py), with pythia-ui built in
# release for this machine.
bench:
	$(CARGO) build --release --bin $(BIN) --target $(HOST)
	$(PYTHON) tools/lem_bench.py target/$(HOST)/release/$(BIN)

# The runtimes in JavaScript, src/lem/lem.js, src/glem/glem.js and
# src/glui/glui.js, and the contexts' page script, web/contexts.js, on the
# cases that the Rust tests write to target/web/: tools/lem_test.js,
# glem_test.js, glui_test.js and contexts_test.js run them, each as one
# script with what it needs, in JavaScriptCore through osascript on a Mac,
# or in another engine, as JS=qjs. They report in TAP (tools/tap.js),
# which osascript writes to stderr, and fail if a case does.
JS := osascript -l JavaScript
WEB := target/web
js-test:
	$(CARGO) test --target $(HOST) --lib -- lem::compile_test glem::compile_test glui::tests
	cat src/lem/lem.js $(WEB)/lem_tests.js tools/tap.js tools/lem_test.js > $(WEB)/lem_test_run.js
	$(JS) $(WEB)/lem_test_run.js
	cat src/lem/lem.js src/glem/glem.js $(WEB)/glem_tests.js tools/tap.js tools/stand_in_dom.js \
	  tools/glem_test.js > $(WEB)/glem_test_run.js
	$(JS) $(WEB)/glem_test_run.js
	cat src/lem/lem.js src/glem/glem.js src/glui/glui.js $(WEB)/glui_tests.js tools/tap.js \
	  tools/stand_in_dom.js tools/glui_test.js > $(WEB)/glui_test_run.js
	$(JS) $(WEB)/glui_test_run.js
	cat src/lem/lem.js src/glem/glem.js src/glui/glui.js web/contexts.js $(WEB)/contexts_tests.js \
	  tools/tap.js tools/stand_in_dom.js tools/contexts_test.js > $(WEB)/contexts_test_run.js
	$(JS) $(WEB)/contexts_test_run.js

# What the web look's pages, web/demo.html and web/test.html, fetch, in
# target/web/: the approval HUD, and lem/test's page and form, compiled by
# pythia-ui lem --compile; each theme in src/glui/themes/, alone, as
# theme-NAME.json, with their names in themes.json; from the tests, their
# cases for JavaScript; and, for the contexts' sample pages,
# web/contexts-index.html and web/contexts-log.html, a sample of their
# facts (tools/contexts_sample.py). web-serve serves the crate to this
# machine alone, at 127.0.0.1:PORT, for a browser.
WEB_BIN := target/$(HOST)/debug/$(BIN)
WEB_THEMES := $(basename $(notdir $(sort $(wildcard src/glui/themes/*.lem))))
comma := ,
web:
	$(CARGO) build --bin $(BIN) --target $(HOST)
	$(CARGO) test --target $(HOST) --lib -- lem::compile_test glem::compile_test glui::tests
	mkdir -p $(WEB)
	$(WEB_BIN) lem --compile lem/host.lem lem/approval.lem > $(WEB)/approval.json
	$(WEB_BIN) lem --compile lem/test/page.lem > $(WEB)/page.json
	$(WEB_BIN) lem --compile lem/test/form.lem > $(WEB)/form.json
	$(foreach t,$(WEB_THEMES),$(WEB_BIN) lem --theme src/glui/themes/$(t).lem --compile \
	  > $(WEB)/theme-$(t).json &&) true
	echo '[$(subst " ","$(comma) ",$(patsubst %,"%",$(WEB_THEMES)))]' > $(WEB)/themes.json
	$(PYTHON) tools/contexts_sample.py $(WEB)/sample

# The compiled programs that the contexts' pages fetch, which wui will
# serve: lem/contexts.lem, and glui's themes, each alone, in web/compiled/.
# A test compiles them again and compares (src/glui/tests.rs): after a
# change to what they're compiled from, this writes them anew.
compiled:
	$(CARGO) build --bin $(BIN) --target $(HOST)
	mkdir -p web/compiled
	$(WEB_BIN) lem --compile lem/contexts.lem > web/compiled/contexts.json
	$(foreach t,dark light,$(WEB_BIN) lem --theme $(t) --compile > web/compiled/theme-$(t).json &&) true

PORT := 8000
web-serve: web
	@echo "glui's demo: http://127.0.0.1:$(PORT)/web/demo.html; its tests: /web/test.html;"
	@echo "pythia's contexts, a sample: /web/contexts-index.html"
	$(PYTHON) -m http.server $(PORT) --bind 127.0.0.1

# Times pythia's contexts in JavaScript at the size of a long log
# (tools/contexts_bench.js), in JavaScriptCore through osascript on a Mac,
# or in another engine, as JS=qjs.
js-bench:
	$(CARGO) test --target $(HOST) --lib -- glui::tests::the_compiled_files_are_up_to_date
	cat src/lem/lem.js src/glem/glem.js src/glui/glui.js $(WEB)/contexts_tests.js tools/stand_in_dom.js \
	  tools/contexts_bench.js > $(WEB)/contexts_bench_run.js
	$(JS) $(WEB)/contexts_bench_run.js

# check and test with the oldest Rust that Cargo.toml's rust-version names,
# which must be installed:
#   rustup toolchain install 1.78.0 --target aarch64-apple-darwin
MSRV := $(shell sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml)
msrv:
	RUSTUP_TOOLCHAIN=$(MSRV) $(MAKE) check test

# Loads this machine's build of the plugin with ctypes, as pythia will
# (tools/plugin_check.py). On a Mac, SCREENSHOT=shot.png saves one too.
plugin-check:
	$(CARGO) build --release --lib --target $(HOST)
	$(PYTHON) tools/plugin_check.py $(if $(SCREENSHOT),--screenshot $(SCREENSHOT)) $(HOST_LIB)

# Adds to stub-sdk/usr/lib/libSystem.tbd what the program, the plugin and the
# unit tests import, and keeps what it lists: another Rust version's std
# imports other functions. Run it, with the Rust whose link failed, when a
# link fails with "undefined symbol". Every link here allows unresolved
# symbols; RUSTFLAGS replaces the configs' flags, so it repeats the CPU from
# .cargo/config.toml.
TBD_OUT := target/tbd/$(TARGET)
TBD_FLAGS := -C target-cpu=apple-m1 -C link-arg=-undefined -C link-arg=dynamic_lookup
TBD_CARGO := RUSTFLAGS="$(TBD_FLAGS)" $(CARGO)
tbd:
	$(TBD_CARGO) build --target-dir target/tbd
	$(TBD_CARGO) build --target-dir target/tbd --release
	$(TBD_CARGO) test --no-run --target-dir target/tbd
	$(PYTHON) tools/gen_tbd.py stub-sdk/usr/lib/libSystem.tbd \
	  $(TBD_OUT)/debug/$(BIN) $(TBD_OUT)/release/$(BIN) \
	  $(TBD_OUT)/debug/$(LIB) $(TBD_OUT)/release/$(LIB) \
	  $$(find $(TBD_OUT)/debug/deps -maxdepth 1 -type f -name 'pythia_ui-*' ! -name '*.*')

# Checks, from any host, that the binaries can load on macOS: platform,
# code signature, imports per library, and the plugin's name and exports.
inspect: build release
	$(PYTHON) tools/macho_info.py $(OUT)/debug/$(BIN) $(OUT)/release/$(BIN) \
	  $(OUT)/debug/$(LIB) $(OUT)/release/$(LIB)

clean:
	$(CARGO) clean
