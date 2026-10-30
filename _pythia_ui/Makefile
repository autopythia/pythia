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

.PHONY: all build dev debug rel release test check plugin-check tbd inspect clean

all: build

build dev debug:
	$(CARGO) build

rel release:
	$(CARGO) build --release

# The unit tests, on this machine: the plain parts (JSON, base64, the
# protocol, the plugin interface) anywhere, and on a Mac the macOS code too.
test:
	$(CARGO) test --target $(HOST)

check:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets -- -D warnings
ifneq ($(HOST),$(TARGET))
	$(CARGO) clippy --all-targets --target $(HOST) -- -D warnings
endif

# Loads this machine's build of the plugin with ctypes, as pythia will
# (tools/plugin_check.py). On a Mac, SCREENSHOT=shot.png saves one too.
plugin-check:
	$(CARGO) build --release --lib --target $(HOST)
	$(PYTHON) tools/plugin_check.py $(if $(SCREENSHOT),--screenshot $(SCREENSHOT)) $(HOST_LIB)

# Rewrites stub-sdk/usr/lib/libSystem.tbd to list what the program, the
# plugin and the unit tests import. Run it when a link fails with "undefined
# symbol" (new std usage, or another Rust version). Every link here allows
# unresolved symbols; RUSTFLAGS replaces the configs' flags, so it repeats the
# CPU from .cargo/config.toml.
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
