.PHONY: all
all: build lint test

.PHONY: clean
clean:
	cargo clean

.PHONY: format
format:
	RUST_LOG=error; cargo fmt

.PHONY: lint
lint:
	cargo fmt --all -- --check
	# Sadly the clippy team can't seem to figure out how to allow enabling/disabling lints in their config file (https://github.com/rust-lang/cargo/issues/5034)
	# So we have to do it with CLI flags.
	cargo clippy -- --no-deps --deny warnings -D clippy::unwrap_used
	cargo clippy --tests -- --no-deps --deny warnings -A clippy::unwrap_used

.PHONY: build
build:
	# Use --all-targets to ensure that all of the benchmarks compile.
	cargo build --all-targets --all-features

.PHONY: test
test:
	RUST_BACKTRACE=1 cargo test
	# Amazingly, `--all-targets` causes doc-tests not to run.
	RUST_BACKTRACE=1 cargo test --doc

.PHONY: test-ignored
test-ignored:
	# This is how Rust suggests skipping expensive tests unless you really want them to run:
	# https://doc.rust-lang.org/book/ch11-02-running-tests.html#ignoring-some-tests-unless-specifically-requested
	cargo test -- --ignored

.PHONY: build-release
build-release:
	cargo build --release
