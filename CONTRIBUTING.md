# Contributing

Hi there! We're thrilled that you'd like to contribute to this project. Your help is essential for keeping it great.

Contributions to this project are [released](https://help.github.com/articles/github-terms-of-service/#6-contributions-under-repository-license) to the public under the [project's open source license][license].

Please note that this project is released with a [Contributor Code of Conduct][code-of-conduct]. By participating in this project you agree to abide by its terms.

## Prerequisites for running and testing code

These are one time installations required to be able to test your changes locally as part of the pull request (PR) submission process.

1. Install Rust [through download](https://www.rust-lang.org/learn/get-started) | [through Homebrew](https://formulae.brew.sh/formula/rustup-init)

## Submitting a pull request

1. [Fork][fork] and clone the repository
2. Make sure the tests pass on your machine: `make test`
3. Make sure linter passes on your machine: `make lint`
4. Create a new branch: `git checkout -b my-branch-name`
5. Make your change, add tests, and make sure the tests and linter still pass
6. Push to your fork and [submit a pull request][pr]
7. Pat yourself on the back and wait for your pull request to be reviewed and merged.

Here are a few things you can do that will increase the likelihood of your pull request being accepted:

- Follow the [style guide][style].
- Write tests.
- Keep your change as focused as possible. If there are multiple changes you would like to make that are not dependent upon each other, consider submitting them as separate pull requests.
- Write a [good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

## Releasing a crate

Crates are published to [crates.io](https://crates.io) with
[Trusted Publishing](https://crates.io/docs/trusted-publishing), so there is no long-lived API
token stored in this repository. The `Publish crates` workflow exchanges its GitHub OIDC identity
for a token that is revoked when the run ends.

1. Bump `version` in the crate's `Cargo.toml` and merge that change to `main`.
2. Run the [`Publish crates`](../../actions/workflows/publish-crates.yaml) workflow via
   *Run workflow*, pick the crate, and optionally tick *dry-run* first to package and verify it
   without uploading. If a newly added crate isn't in the dropdown yet, type its name into
   *crate_name_override* instead — and add it to the dropdown in `publish-crates.yaml` while
   you're there.

A crate has to be published manually once before crates.io will let you configure a trusted
publisher for it. Configure it at `https://crates.io/crates/<crate>/settings/trusted-publishing`
with repository `github/rust-gems`, workflow `publish-crates.yaml`, and environment `crates-io`.
The environment name must match the workflow's `environment:` exactly or the token exchange fails.

## Resources

- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [Using Pull Requests](https://help.github.com/articles/about-pull-requests/)
- [GitHub Help](https://help.github.com)

[fork]: https://github.com/github/rust-algos/fork
[pr]: https://github.com/github/rust-algos/compare
[style]: https://doc.rust-lang.org/nightly/style-guide/
[code-of-conduct]: CODE_OF_CONDUCT.md
[license]: LICENSE
