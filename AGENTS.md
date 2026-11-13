# ai-gateway Agent Guidelines

## Public repository

- This repo is open source. Treat all code, comments, docs, commit messages, PR/issue text, test data, and logs as public.
- Never reference internal or non-public services, their logic, endpoints, hostnames, infrastructure, or business requirements. This includes `aichat-internal` and `aichat-ops`.
- Never commit secrets, API keys, tokens, real user data, or internal URLs. Use obvious placeholders in examples, fixtures, and `.env.example`.
- Describe behavior in generic, self-contained terms (what the gateway does), not why a private system needs it.

## Code style

- No code comments unless absolutely necessary. Prefer clear names and small functions.
- For completely new sizable functionality, evaluate creating a service file in `aichat/serve/services` instead of adding it to the gateway.
- If a comment is unavoidable, keep it to one short line explaining why, not what.
- Imports at the top of the file only. No inline imports.
- Match existing patterns, naming, and module layout before introducing new ones.
- Keep changes minimal and scoped to the request. No drive-by refactors.

## Tests

- Add Gherkin-style BDD tests for every new feature and behavior change, using `pytest-bdd`.
- Put scenarios in `test/aichat/serve/bdd/features/<name>.feature` with steps in `test/aichat/serve/bdd/<name>_test.py`, following the existing files.
- Cover the happy path plus key error and edge cases.
- Reuse shared fixtures and helpers (`conftest.py`, `helpers.py`) instead of duplicating setup.
- Feature files and test data must follow the public-repo rules above.
- Run the relevant tests before finishing (`make test` or `poetry run pytest <path>`).

## APIs

- `conversation_api.py` is the deprecated V1 API and `open_ai_api.py` is the V2 API. New work targets V2;

## Git

- Never attribute commits or PRs to Cursor or an AI tool.

## Dependencies
- Never add dependencies without a good reason.
- Dependencies should be secure and well-maintained.
