# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4](https://github.com/github/rust-gems/compare/bpe-v0.2.3...bpe-v0.2.4) - 2026-11-26

### Other

- Remove RustSec dependencies from benchmarks
- Enable benchmark library regression tests
- Fix benchmark tokenizer API calls
- Replace unmaintained tokenizer benchmark dependencies
- Replace unwrap() calls in validate_daac_bytes_checks_invariants test
- Use slice try_into in read_u32_le
- Avoid unwrap() in validate_daac_bytes to satisfy clippy::unwrap_used
- Validate DAAC deserialization and check token bounds in bpe

## [0.2.3](https://github.com/github/rust-gems/compare/bpe-v0.2.2...bpe-v0.2.3) - 2026-10-07

### Other

- Publish crates to crates.io via trusted publishing
- Update base64 requirement from 0.22 to 0.23
