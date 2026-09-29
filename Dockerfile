# ── shared base: Python + Poetry ────────────────────────────────────────────
FROM python:3.14-slim AS base

ENV POETRY_VERSION=2.2.1 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1 \
  UV_EXCLUDE_NEWER="7 days" \
  PATH="/root/.local/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /bin/
RUN uv tool install --no-cache "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# ── production dependencies ──────────────────────────────────────────────────
FROM base AS builder

RUN poetry install --no-root --only main

# Pre-download tiktoken BPE vocabulary files so images have no outbound
# network dependency on first import.  Both encodings are needed:
#   - o200k_base  (default tokenizer, used at module level in conversation/utils.py and pdf.py)
#   - cl100k_base (used in serve/utils.py:count_tokens)
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')"

# ── test image (extends builder with test dependencies) ──────────────────────
FROM builder AS test-builder

RUN poetry install --no-root --with test

COPY LICENSE README.md ./
COPY .coveragerc ./
COPY test /app/test
COPY aichat /app/aichat
RUN poetry install --only-root

# ── production runtime ───────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

ENV VIRTUAL_ENV=/app/.venv \
  PATH="/app/.venv/bin:$PATH" \
  TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.tiktoken_cache /app/.tiktoken_cache

COPY aichat /app/aichat

WORKDIR /app

ARG COMMIT=unknown
ENV AICHAT_GIT_SHA=$COMMIT

RUN useradd -r -u 1001 -g root appuser \
    && chown -R appuser /app
USER appuser

CMD [ "python", "-m", "uvicorn", "aichat.serve.api_server:app" ]
