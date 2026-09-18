# Configuring models

The server has no models available until `MODELS` is set (see `.env.example`). `MODELS` is a JSON object mapping a model key to its config; `MODEL_TRIAGING` maps categories to `{premium, non-premium}` model picks for `automatic` routing (and needs a `"conversation_title"` entry for conversation-title requests to work - see below). See `docker-compose.base.yml` for a larger example covering multiple backends (`litellm`, `vllm`, `bedrock`).

Any local OpenAI-compatible server works with `"backend": "litellm"` — set `address` to its base URL and `upstream_model` to the model name it serves. This covers [Ollama](https://ollama.com) (the `.env.example` default) and [LM Studio](https://lmstudio.ai) (using whatever model identifier is loaded in its local server, e.g. `curl http://localhost:1234/v1/models` while LM Studio is running). Since the server itself runs inside Docker via the Quickstart, `address` needs to reach your host machine, not the container - use `http://host.docker.internal:11434/v1` (Ollama) or `http://host.docker.internal:1234/v1` (LM Studio), not `localhost`. Use `localhost` only when running via `./scripts/start-local-native.sh` instead.

For a model to appear in `GET /v1/models` (the browser-facing endpoint, as opposed to the OpenAI-compatible `/v1/chat/completions`), it additionally needs `category` (`"chat"` or `"summary"`), `maker`, `friendly_name`, and `description` (an object keyed by language code, e.g. `{"en": "..."}`) - a model missing these will still work for chat completions but won't be listed.

Small local models often have a much smaller context window loaded than the `conversation_token_limit` you declare for them - Leo's system prompt alone can be a meaningful fraction of a small model's context, so even short requests may fail with a context-length error from the backend. If you hit this, either raise the context length for the model in Ollama/LM Studio or lower `conversation_token_limit`/`max_tokens` in `MODELS` to match.

> Auto-loading a set of sane default models when `MODELS` is unset is not yet implemented - contributions welcome.

## Tips for when running locally

- `./scripts/start-local.sh` runs `docker run -it`, which requires a real terminal. It'll fail with `the input device is not a TTY` if run from a script, CI job, or other non-interactive shell - run it from an actual terminal, or adapt the `docker run` flags for your context.

- You'll see an `ERROR`-level "Error parsing Androcles response... skipping" log line on every chat request. This comes from Dynamic Leo's request triage step, which is enabled by default (`ENABLE_DYNAMIC_LEO=true`) but has no triage model configured out of the box (`ANDROCLES_MODEL_ADDRESS` is unset). It's caught internally and doesn't affect the response - set `ENABLE_DYNAMIC_LEO=false` if the noise is distracting.

- Conversation-title requests (an OpenAI-protocol message with a `brave-conversation-title` content part) currently 500 unless `MODEL_TRIAGING` has a `"conversation_title"` key pointing at a configured model - the `TITLE_MODEL` variable is not used for this. This is tracked as a bug to fix; for now, add that key if you need conversation titling.

- You'll see a one-time `WARNING`-level "Alignment model 'llama-4-maverick' not found in configured models" / "Prompt injection model..." log line at startup. These come from the agent alignment/prompt-injection scanners, which are enabled by default (`alignment_checking_enabled`/`prompt_injection_scanning_enabled`) but default to a model name (`llama-4-maverick`) that isn't configured out of the box. They only run during tool-call/agent flows (not plain chat completions), and fall back to allowing the action when unavailable - harmless for local chat testing.

- The example commands in [Trying other endpoints](endpoints.md) for image generation, TTS, and embeddings reference model keys (`z-image-turbo`, `kokoro`, `text_embedding`) that aren't part of the Quickstart's default single-model `MODELS` config, so they'll 404 with `"model is not supported"` unless you add matching entries with `"type": "image"`/`"tts"`/`"embedding"` to `MODELS`.
