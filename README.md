# AI Gateway

Backend for Brave's AI Chat (Leo). An OpenAI-compatible API server that routes chat completions to configurable LLM backends (local, vLLM, LiteLLM, Bedrock, Anthropic) with support for MCP tools, deep research, and more.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (including Compose) - used by `make setup`/`start`/`test-docker`.
- `make` - included by default on macOS and most Linux distros.
- A backend to actually generate responses, e.g. [Ollama](https://ollama.com) (the `.env.example` default) or [LM Studio](https://lmstudio.ai) running locally - see [Configuring models](docs/configuring-models.md).

`make start-native` additionally requires [Poetry](https://python-poetry.org) and Python 3.12-3.14 on the host, since it runs outside Docker.

## Quickstart

Use `make` to build the image and copy the example environment file:

```sh
make setup
```

`.env` defaults `MODELS` to a local Ollama instance (`ollama pull qwen2.5:14b`); edit it to point at a different backend if needed. If you edit the JSON values, keep them single-quoted (e.g. `MODELS='{"...":"..."}'`) - unquoted double quotes get stripped when the variables are loaded into the shell, producing invalid JSON. Then start the server:

```sh
make start
```

Or run directly on the host instead of Docker (e.g. to use stdio MCP servers via a command not baked into the image, like NPX):

```sh
make start-native
```

Check that it's up:

```sh
curl http://127.0.0.1:8000/v1/models
```

This should return a JSON array containing your configured model(s). If it returns `[]`, see [Configuring models](docs/configuring-models.md) - a few metadata fields are required for a model to show up here, beyond what's needed to route requests to it.

### For Brave Developers

Checkout [internal setup](https://github.com/brave/ai-gateway-ops#aichat-server-development-for-brave-employees) for connecting a local browser build, premium access, and other Brave-internal instructions.

## Testing

Build the test image:

```sh
make test-image
```

Run all the tests with docker-compose:

```sh
make test-docker
```

<details>

<summary>Additional Test Commands</summary>

or run a specific test

```sh
make test-docker TEST_ARG="test/aichat/serve/conversation_api_test.py::test_v1_conversation_max_tokens_config"
```

Instruct Syrupy to generate new snapshots based on new outputs

```sh
make test-docker TEST_ARG="--snapshot-update test/aichat/conversation/prompt_test.py test/aichat/conversation/retrieve/functions_test.py"
```

</details>

## Roadmap

In addition to the ai-gateway repo, our roadmap includes open sourcing some other Brave built internal services used within ai-gateway. This includes the following:

- Brave Deep Research MCP
- Leo Private Analytics
- Search Orchestration MCP

## Learn more

- [Configuring models](docs/configuring-models.md) - backends, model metadata, common local-setup gotchas
- [Configuring MCP Servers](docs/mcp-servers.md) - MCP server config and custom handlers
- [Premium model access](docs/premium-access.md) - how premium vs. free requests are distinguished and what it affects
- [Trying other endpoints](docs/endpoints.md) - images, TTS, embeddings, regression tests, local browser builds
- [Deep Research](docs/deep-research.md) - enabling the deep research MCP tool
- [Bedrock](docs/bedrock.md) - using AWS Bedrock-backed models
- [Conversation Sharing](docs/conversation-sharing.md) - the Leo conversation-sharing endpoints
- [Rate Limiting](docs/rate-limiting.md) - rate limiting behavior and configuration

