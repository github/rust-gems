# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/github/rust-gems/compare/hash-sorted-map-v0.1.0...hash-sorted-map-v0.1.1) - 2026-11-26

### Other

- Fix needless range loop in hash sorting
- Document why group_ops uses from_le_bytes instead of from_ne_bytes
- Update ctrl tags immediately during sort_by_hash insertion sort
- Fix panic-unwind double-drop in sort_by_hash and big-endian group_ops
