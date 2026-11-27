# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0](https://github.com/github/rust-gems/compare/geo_filters-v0.4.1...geo_filters-v0.5.0) - 2026-11-26

### Other

- Simplify geo_filters review changes
- Replace geo_filters once_cell with LazyLock
- Replace private once_cell lazy with std
- Link BITS_PER_BLOCK in bitchunks doc comments
- Document parameter constraints and use BITS_PER_BLOCK in mask_size panic message
- Fix remaining medium and low code-scanning security alerts
- Refactor geo_filters unaligned storage into Unaligned<T> wrapper
- Nest PartialOrd match branches in GeoDiffCount
- Use zero-copy unaligned bucket reads for GeoDiffCount::msb
- Use zero-copy unaligned block reads in BitVec
- Avoid unaligned slice casts in GeoDiffCount and BitVec::from_bytes

### Changed

- **Breaking:** `Lookups::get_lookups()` now returns `std::sync::LazyLock` values instead of `once_cell::sync::Lazy`.
- Remove the direct dependency on `once_cell`.

## [0.4.1](https://github.com/github/rust-gems/compare/geo_filters-v0.4.0...geo_filters-v0.4.1) - 2026-10-07

### Other

- Use the standard library's version of the Euler-Mascheroni constant
