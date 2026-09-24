.PHONY: test test-image test-docker start

test:
	pytest -vv

test-image:
	docker build --target test-builder -t aichat-media-test .

test-docker: test-image
	docker compose -f docker-compose.test.yml down
	TEST_ARG="$(TEST_ARG)" docker compose -f docker-compose.test.yml up --exit-code-from aichat-media

start:
	docker build -t aichat-media .
	docker run --rm -p 8002:8000 aichat-media
