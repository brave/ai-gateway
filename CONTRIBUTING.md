# Contributing to AI Gateway

Thanks for your interest in contributing! This document covers the basics of getting set up, the review process, and the conventions this repo follows.

## Before you start

For anything beyond a small fix (new features, architectural changes, new backends), please open an issue first to discuss the approach before investing time in a PR. For small bug fixes and docs improvements, feel free to open a PR directly.

## Getting set up

See the [README](README.md) for prerequisites, the Quickstart, and how to configure models/backends for local development. In short:

```sh
make setup   # builds the image, copies .env.example -> .env
make start   # runs the server via Docker
```

## Making changes

1. Fork the repo and create a branch off `main`.
2. Make your changes, following the code style below.
3. Add or update tests for any behavior change (see "Testing").
4. Update the [README](README.md) or the relevant [doc](docs/) if you're changing documented behavior or configuration.
5. Open a PR against `main`. The [PR template](.github/pull_request_template.md) will walk you through what reviewers expect - fill it out, including a clear summary and a description of how you tested the change.
6. Squash intermediate/fixup commits before requesting review; keep the history readable.

A member of the [CODEOWNERS](.github/CODEOWNERS) team will review your PR. CI (tests, lint, formatting, and dependency/security scanning) must pass before merge.

## Versioning

`ai-gateway` is tagged independently with [SemVer](https://semver.org), for self-hosters tracking breaking changes and new features. It has no required version dependency on `aichat-internal` - `ai-gateway` runs standalone when `internal_settings.internal_api_enabled=False`. If a change to the (internal-only) API contract between the two ever requires a minimum `aichat-internal` version, that's called out in the relevant PR/changelog entry rather than enforced via matching tags.

Release notes and the next version number are drafted automatically by [Release Drafter](https://github.com/release-drafter/release-drafter) as changes land on `prod` - see the running draft under [Releases](../../releases). It labels PRs from their title prefix (`feat:`/`fix:`/`chore:`/etc.) to pick a category and decide whether the next release is a major, minor, or patch bump; add a `major`/`breaking` label by hand for breaking changes. Releases are only cut against `prod` (see the `main` -> `prod` release process in `aichat-ops`), so the draft only updates on pushes to `prod`, and publishing it tags `prod`'s tip. Bump `pyproject.toml`'s `version` to match in the release PR, since it isn't updated automatically (the running server reports its version from the git SHA, not from `pyproject.toml`).

## Code style

- Formatting is enforced with [`black`](https://black.readthedocs.io/) - run `poetry run black .` before committing, or your PR will fail the `Check formatting` CI job.
- Linting uses `pylint` (currently restricted to catching undefined names - `pylint -d all -e E0602 ./aichat/`).
- Prefer clear, descriptive names over comments; only add comments that explain non-obvious intent, trade-offs, or constraints - avoid comments that just restate what the code does.

## Testing

Run the full suite via Docker Compose (this is what CI runs):

```sh
make test-image   # build the test image
make test-docker  # run all tests
```

To run a specific test file or test:

```sh
make test-docker TEST_ARG="test/aichat/serve/conversation_api_test.py::test_v1_conversation_max_tokens_config"
```

New functionality should come with test coverage. If you're changing prompt-generation logic covered by `syrupy` snapshots, regenerate them explicitly and review the diff rather than blindly accepting it:

```sh
make test-docker TEST_ARG="--snapshot-update <path/to/test.py>"
```

## Dependencies

Dependencies are managed with [Poetry](https://python-poetry.org) (`pyproject.toml` / `poetry.lock`). Pin exact versions for new dependencies (matching the existing style) and run `poetry lock` after editing `pyproject.toml` so the lock file stays in sync - CI checks this with `poetry check`.

## Issue and PR triage

This repo is maintained by a small internal team alongside their other work, so there's no fixed response-time SLA for issues or community PRs - they're triaged best-effort, with production priorities coming first. Stale PRs that go without author follow-up may be closed.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Please do not open a public GitHub issue for security vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under the [Mozilla Public License 2.0](LICENSE), the same license that covers the rest of this project.
