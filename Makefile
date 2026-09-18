.PHONY: clean test setup start start-native test-image test-docker smoke-test

test:
	pytest -vv
	TEST_LANGUAGES=en coverage run --data-file /tmp/.coverage.en --context en -m pytest --snapshot-warn-unused aichat/conversation/prompt_test.py
	coverage combine
	coverage report --contexts en --fail-under 97
clean:
	rm -f .coverage .coverage.* coverage.xml

# Build the image and create .env if it doesn't already exist (see README.md Quickstart).
setup:
	docker build -t ai-gateway .
	test -f .env || cp .env.example .env

# Load .env and start the server via Docker.
start:
	@test -f .env || { echo ".env not found - run: make setup" >&2; exit 1; }
	set -a && . ./.env && set +a && ./scripts/start-local.sh

# Same as `start`, but runs directly on the host instead of Docker.
start-native:
	@test -f .env || { echo ".env not found - run: make setup" >&2; exit 1; }
	set -a && . ./.env && set +a && ./scripts/start-local-native.sh

# Build the Docker image used by test-docker (see README.md Testing).
test-image:
	docker build --target test-builder -t ai-gateway-test .

# Run the test suite via docker-compose. Rebuilds the image first so it never
# runs stale code. Pass TEST_ARG for pytest args, e.g.
# `make test-docker TEST_ARG="test/aichat/serve/conversation_api_test.py"`.
test-docker: test-image
	docker compose -f docker-compose.test.yml down
	TEST_ARG="$(TEST_ARG)" docker compose -f docker-compose.test.yml up --exit-code-from ai-gateway

# Public runnability smoke test: builds and starts the server exactly as a
# fresh clone would (only .env.example) and confirms model:"automatic"
# actually works end to end. See scripts/smoke-test.sh.
smoke-test:
	./scripts/smoke-test.sh
