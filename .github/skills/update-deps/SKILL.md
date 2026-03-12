---
name: update-deps
description: Keep dependencies up-to-date. Discovers outdated deps via dependabot alerts/PRs, creates one PR per ecosystem, iterates until CI is green, then assigns for review.
user-invocable: true
---

# Update Dependencies

Automate the full dependency update lifecycle: discover what's outdated, apply updates grouped by ecosystem, fix breakage, get CI green, and hand off for human review.

## Repository context

This is a Rust workspace containing utility crates published to crates.io. All dependency update PRs target the **`main`** branch.

Dependabot is configured (`.github/dependabot.yaml`) to open PRs against `main` on the 2nd of each month. This skill gathers individual dependabot PRs, combines updates by ecosystem, fixes any breakage, gets CI green, and creates consolidated PRs for human review.

### Crates in this workspace

| Crate | Description |
|---|---|
| **bpe** | Fast byte-pair encoding |
| **bpe-openai** | OpenAI tokenizers built on bpe |
| **geo_filters** | Probabilistic cardinality estimation |
| **string-offsets** | UTF-8/UTF-16/Unicode position conversion (with WASM/JS bindings) |

Supporting packages (not published): `bpe-tests`, `bpe-benchmarks`.

### Ecosystems in this repo

| Ecosystem | Directories | Notes |
|---|---|---|
| **cargo** | `/` (workspace root) | All Rust deps managed at workspace level via `Cargo.lock` |
| **github-actions** | `.github/workflows/` | CI and publish workflows |
| **npm** | `crates/string-offsets/js/` | JS bindings for string-offsets (WASM) |

### Build and validation commands

```bash
make build     # cargo build --all-targets --all-features
make build-js  # npm run compile in crates/string-offsets/js
make lint      # cargo fmt --check + cargo clippy (deny warnings, forbid unwrap_used)
make test      # cargo test + doc tests
```

CI runs on `ubuntu-latest` with the `mold` linker. The lint job depends on build.

## Workflow

### 1. Assess repo state

Determine the repo identity and confirm the target branch.

```bash
git remote get-url origin   # extract owner/repo
git fetch origin main
git rev-parse --verify origin/main
```

Detect which ecosystems have pending updates:

```bash
[ -f Cargo.toml ] && echo "cargo"
ls .github/workflows/*.yml .github/workflows/*.yaml 2>/dev/null && echo "github-actions"
[ -f crates/string-offsets/js/package.json ] && echo "npm"
```

Report discovered ecosystems to the user.

### 2. Gather dependency intelligence

Fetch open dependabot PRs:

```bash
gh pr list --author 'app/dependabot' --base main --state open --json number,title,headRefName,labels --limit 100
```

Fetch open dependabot alerts:

```bash
gh api /repos/{owner}/{repo}/dependabot/alerts --jq '[.[] | select(.state=="open") | {number: .number, package: .security_vulnerability.package.name, ecosystem: .security_vulnerability.package.ecosystem, severity: .security_advisory.severity, summary: .security_advisory.summary}]'
```

For ecosystems without dependabot coverage or when running ad-hoc, use native tooling:

- **cargo:** `cargo update --dry-run`
- **npm:** `cd crates/string-offsets/js && npm outdated --json`

Also fetch the advisory URLs for any security-related updates. Individual alert details are at `https://github.com/{owner}/{repo}/security/dependabot/{alert_number}`. Fetch alert numbers and GHSA IDs via:

```bash
gh api /repos/{owner}/{repo}/dependabot/alerts --jq '[.[] | {number: .number, state, package: .security_vulnerability.package.name, ecosystem: .security_vulnerability.package.ecosystem, severity: .security_advisory.severity, ghsa_id: .security_advisory.ghsa_id, summary: .security_advisory.summary}]'
```

Include both open and auto_dismissed/dismissed alerts — the update may resolve alerts in any state.

Cross-reference and group all updates by ecosystem. Present a summary to the user:

- How many updates per ecosystem
- Which have security alerts (with severity, GHSA IDs, and advisory links)
- Which dependabot PRs already exist

**Flag high-risk upgrades.** Before proceeding, explicitly call out upgrades that carry elevated risk:

- **Major version bumps** — likely contain breaking API changes
- **Packages with wide blast radius** — for this repo, pay special attention to: `serde`, `itertools`, `regex-automata`, `wasm-bindgen`, `criterion`, and the Rust toolchain itself
- **Multiple major bumps in the same PR** — each major bump multiplies the risk; consider splitting them

Present the risk assessment to the user and recommend which upgrades to include vs. defer. When in doubt, prefer a smaller, safe update over an ambitious one that might break.

### 3. Create branch and apply updates

For each selected ecosystem, starting from `main`:

```bash
git checkout main
git pull origin main
git checkout -b deps/{ecosystem}-updates-$(date +%Y-%m-%d)
```

Apply updates using ecosystem-appropriate tooling:

**cargo:**

```bash
cargo update
# For major bumps, edit Cargo.toml version constraints then:
cargo check
```

This is a Cargo workspace — always run from the repo root. All crate `Cargo.toml` files are in `crates/`. The `Cargo.lock` at the root is the single source of truth.

**npm:**

```bash
cd crates/string-offsets/js
npm update
npm install
```

**github-actions:**

- Parse workflow YAML files in `.github/workflows/` for `uses:` directives
- For each action with an outdated version (from dependabot PRs/alerts), update the SHA or version tag
- Be careful to preserve comments and formatting

### 4. Build, lint, and test locally

Always run:

```bash
make lint      # cargo fmt --check + clippy with deny warnings
make test      # cargo test with backtrace
make build     # full workspace build (all targets, all features)
```

If npm dependencies changed:

```bash
make build-js  # npm compile for string-offsets JS binding
```

**If the build/lint/test fails:**

1. Read the error output carefully
2. Analyze what broke — likely API changes, type errors, or deprecation removals
3. Make the necessary code changes to fix the breakage
4. Run the pipeline again
5. Repeat up to 3 times

If still failing after 3 iterations, report the situation to the user and ask for guidance. Do not push broken code.

### 5. Commit and push

Stage all changes and commit with a descriptive message:

```bash
git add -A
git commit -m "chore(deps): update {ecosystem} dependencies

Updated packages:
- package-a: 1.0.0 → 2.0.0
- package-b: 3.1.0 → 3.2.0

{If code changes were needed:}
Fixed breaking changes:
- Updated X API usage for package-a v2

Supersedes: #{dependabot_pr_1}, #{dependabot_pr_2}
"
```

Push the branch:

```bash
git push -u origin HEAD
```

### 6. Create the PR

**Title:** `chore(deps): update {ecosystem} dependencies`

**Body should include:**

- List of updated dependencies with version changes (old → new)
- Any security alerts resolved — for each, link to the specific dependabot alert (`https://github.com/{owner}/{repo}/security/dependabot/{alert_number}`) and the GHSA advisory (`https://github.com/advisories/GHSA-xxxx-xxxx-xxxx`), along with severity and summary
- **High-risk changes flagged for reviewer attention** (major version bumps, wide-blast-radius packages)
- Code changes made to fix breakage (if any)
- References to superseded dependabot PRs
- Note that this was generated by the update-deps skill

Write the body to a temp file and create the PR **targeting `main`**:

```bash
gh pr create --title "chore(deps): update {ecosystem} dependencies" --body-file /tmp/deps-pr-body.md --base main
rm /tmp/deps-pr-body.md
```

### 7. Monitor CI and iterate on failures

Watch the PR's checks:

```bash
gh pr checks {pr_number} --watch --fail-fast
```

**If checks fail:**

1. Get the failed run details:

```bash
gh run list --branch {branch} --status failure --json databaseId,name --limit 1
gh run view {run_id} --log-failed
```

2. Analyze the failure — CI runs on `ubuntu-latest` with `mold` linker, which may differ from local builds.

3. Fix the issue locally, commit, and push:

```bash
git add -A
git commit -m "fix: resolve CI failure in {ecosystem} dep update

{Brief description of what failed and why}"
git push
```

4. Monitor again. Repeat up to 3 iterations total.

5. If still failing after 3 pushes, report to the user with the failure details and ask for help.

### 8. Close superseded dependabot PRs

For each dependabot PR that this update supersedes:

```bash
gh pr close {dependabot_pr_number} --comment "Superseded by #{new_pr_number} which includes this update along with other {ecosystem} dependency updates."
```

### 9. Assign for review

Determine the current user:

```bash
gh api user --jq '.login'
```

Request review:

```bash
gh pr edit {pr_number} --add-reviewer {user_login}
```

Report the final PR URL and a summary of what was done.

## Guidelines

- **All PRs target `main`.** There is no separate dev branch.
- **Never push to `main` directly.** Always work on a feature branch.
- **Never push code that doesn't pass `make lint` and `make test`.** If you can't fix it in 3 tries, stop and ask.
- **Be conservative with major version bumps.** If a major version update breaks things and the fix isn't obvious, skip that package and note it in the PR description.
- **Preserve lockfiles.** Always regenerate `Cargo.lock` and `package-lock.json` after updating — don't just edit manifests.
- **One ecosystem at a time.** Complete the full cycle (update → build → push → PR → CI green) for one ecosystem before moving to the next.
- **If no updates are needed** for an ecosystem, skip it and tell the user.
- **Security alerts take priority.** Address security alerts first within each ecosystem.
- **Clippy is strict.** This repo forbids `unwrap_used` outside tests and denies all warnings. New dependency versions may trigger new clippy lints — fix them.

## Edge cases

- **Cargo workspace:** All Rust dependencies are managed at the workspace root. Always run `cargo update` and `cargo check` from the repo root.
- **npm is scoped to string-offsets:** The only npm package is in `crates/string-offsets/js/`. Don't look for npm elsewhere.
- **WASM builds:** After updating `wasm-bindgen` or related deps, verify `make build-js` still works — WASM toolchain version mismatches are common.
- **Rate limits:** If `gh api` hits rate limits, wait and retry. Report to user if persistent.
- **Nothing to update:** Report cleanly and move to the next ecosystem (or exit).
- **Merge conflicts on push:** Rebase on `main` and retry: `git fetch origin main && git rebase origin/main`.
- **Branch already exists:** If `deps/{ecosystem}-updates-{date}` already exists, append a counter or ask user.
