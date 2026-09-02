---
name: update-deps
description: Keep Cargo, npm, and GitHub Actions dependencies current through the repository's guarded dependency automation.
user-invocable: true
---

# Update Dependencies

Maintain one stable draft pull request per ecosystem without giving the Copilot CLI GitHub write credentials.

## Repository inventory

The workspace has 14 Cargo manifests: the virtual root, eight published crates (`bpe`, `bpe-openai`, `casefold`, `consistent-choose-k`, `geo_filters`, `hash-sorted-map`, `sparse-ngrams`, and `string-offsets`), and five benchmark/test support packages.

`Cargo.lock` is intentionally ignored and is not a durable dependency record. Cargo updates must change dependency requirements in the workspace manifests with the pinned `cargo-edit` version used by `.github/workflows/update-dependencies.yaml`; `cargo update` alone is not an update.

| Ecosystem | Durable files |
|---|---|
| Cargo | `crates/**/Cargo.toml` |
| npm | `crates/string-offsets/js/package.json`, `crates/string-offsets/js/package-lock.json` |
| GitHub Actions | `.github/workflows/*.yaml`, `.github/workflows/*.yml` |

## Automation contract

`.github/workflows/update-dependencies.yaml` runs at the Thursday 06:17 UTC fleet slot. Its matrix is deliberately serial in this order: Cargo, npm, GitHub Actions.

Roll out this repository-local automation before removing any existing dependency-update coverage, so there is no gap.

Each ecosystem run has two trust domains:

1. The read-only `generate` job runs a pinned updater, captures its deterministic patch, invokes a checksum-verified Copilot CLI release, runs machine validation, and uploads immutable patch and metadata artifacts.
2. The `apply` job never executes agent output. It validates both patches, checks deterministic and agent path allowlists independently, refuses artifacts generated from any commit other than its current `origin/main`, refuses non-bot history on the reserved branch, applies the final patch as data, and uses force-with-lease only on that reserved branch.

Manual dispatches are accepted only from `main`. Both jobs explicitly check out `main`, and the trusted scripts require `HEAD` to equal `refs/remotes/origin/main`.

The Copilot CLI is started with `--add-dir .` so this project skill is loaded as trusted local configuration. It receives no shell or GitHub MCP tool, cannot ask questions, and has only read/search/edit/web tools. It may repair consumer code, assess risk, and write the proposed PR title/body to a temporary in-tree handoff directory that the trusted script removes before snapshotting. It must not run git, push, create PRs, edit dependency manifests, or edit workflows.

## Ecosystem behavior

### Cargo

- Run the pinned `cargo-edit` release with compatible, incompatible, pinned, and recursive upgrades enabled.
- Delete the ignored generated `Cargo.lock`; only manifest changes are durable.
- The agent may repair Rust source files under `crates/**`.
- Validate with `make lint`, `make test`, and `make build`.

### npm

- Run the pinned `npm-check-updates` release with `--target latest` in `crates/string-offsets/js`, then regenerate `package-lock.json` with lifecycle scripts disabled. Direct ranges must move across major versions.
- Preserve and commit `package-lock.json`.
- Dependency updates, installs, builds, tests, and lifecycle code run without `NPM_TOKEN` or `NODE_AUTH_TOKEN`; the package uses only the public registry.
- Install the workflow's pinned `wasm-pack` before validation; never rely on the Makefile's unpinned fallback installation.
- The agent may repair tracked JavaScript consumer/test files under `crates/string-offsets/js/**`, excluding `package.json` and `package-lock.json`.
- Validate with `make lint`, `make test`, `make build`, and `make build-js`.

### GitHub Actions

- Run the pinned `pinact` release with `--update` and a 14-day minimum release age.
- Keep every action reference pinned to a full commit SHA with its version annotation.
- Agent edits are forbidden; the agent only summarizes the deterministic update.
- Validate workflow YAML parsing and offline SHA pinning with `pinact --fix=false --no-api`.

## Pull request behavior

- Branches are stable: `automation/dependencies/cargo`, `automation/dependencies/npm`, and `automation/dependencies/github-actions`.
- Reuse the workflow's own open draft PR for the branch. Refuse a non-draft PR, a PR by another author, a different base, multiple open PRs, or any non-bot commit on the reserved branch.
- A clean diff is a successful no-op: do not push, create, close, or edit a PR.
- Never mark a PR ready, merge it, close superseded PRs, or request review. `CODEOWNERS` routes changes to `@github/blackbird-reviewers`.
- After creating or updating a draft PR, explicitly dispatch `ci.yaml` on the reserved branch. This is nonblocking and does not depend on recursive workflow events.
- Repair is bounded to three agent passes. Missing output, allowlist violations, failed final validation, and unexpected branch/PR state are explicit failures.

## Authentication

The read-only generator uses `GITHUB_TOKEN` with `copilot-requests: write`, which bills Copilot usage to the organization. GitHub write authentication belongs only to the separate trusted apply job and must never be exposed to the Copilot CLI.

## Validation commands

```bash
make lint
make test
make build
make build-js # npm/WASM changes
.github/scripts/test-dependency-automation
```
