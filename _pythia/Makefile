MAKE := make
CARGO := cargo --offline
RUSTC := rustc

PACKAGE := _pythia

.PHONY: all
all: release

.PHONY: v version
v: version
version:
	$(RUSTC) --version > rustc.version
	$(CARGO) --version > cargo.version

.PHONY: dev debug
dev: debug
debug:
	date
	$(CARGO) build -p $(PACKAGE) --lib --bins
	date

.PHONY: rel release
rel: release
release:
	date
	$(CARGO) build -p $(PACKAGE) --release --lib --bins
	date

.PHONY: clean
clean:
	date
	rm -rf target
	date

.PHONY: pt
pt: release
	./target/release/parse_test

.PHONY: pt1
pt1: release
	./target/release/parse_test 1

.PHONY: t
t: release
	./target/release/mach_test

.PHONY: t1
t1: release
	./target/release/mach_test 1

.PHONY: j
j: release
	./target/release/jit_test
