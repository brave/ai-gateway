# Trying other endpoints

When `INTERNAL_MODELS_API_KEY` is set (typical Brave deploys), passthrough model routes require a shared team key:

```sh
-H "Authorization: Bearer $INTERNAL_MODELS_API_KEY"
```

That applies to embeddings, TTS, STT (`/v1/audio/transcriptions`), System One decisions (`/v1/systemone`), and image generation. Chat completions use the separate service-key flow via `aichat-internal`. Leave the env var unset locally to skip this gate.

Fetch the list of available models:

```sh
curl http://127.0.0.1:8000/v1/models
```

The endpoint returns model descriptions localised to the caller's preferred language. Pass an `Accept-Language` header using the standard HTTP format — the server picks the highest-priority language it has a translation for and falls back to English if none match:

```sh
curl http://127.0.0.1:8000/v1/models -H "Accept-Language: fr-CH, fr;q=0.9, en;q=0.8"
```

Try the image generation API:

```sh
curl -X POST http://127.0.0.1:8000/v1/images/generations -H "Content-Type: application/json" -d "@example/image_generation.json"  | jq -r '.data[0].b64_json' | base64 -d > lion.png
```

Try the TTS (text-to-speech) API:

```sh
curl -X POST http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" -d "@example/tts_speech.json" --output speech.mp3
```

Try the embeddings API:

```sh
curl -X POST http://127.0.0.1:8000/v1/embeddings -H "Content-Type: application/json" -d "@example/embeddings.json"
```

Try the System One API (`state` + `questions`; requires a `system_one` entry in `MODELS` with Triton `address`, `endpoint`, and `method`):

```sh
curl -X POST http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d "@example/system_one.json"
```

## Running the regression tests

Run the regression tests against all endpoints:

```sh
cd example && python -u regression_test.py
```

Use `--minimal` to test a representative subset of models (faster):

```sh
cd example && python -u regression_test.py --minimal
```

<details>

<summary>Additional Regression Test Options</summary>

| Flag                                   | Description                                        |
| -------------------------------------- | -------------------------------------------------- |
| `--skip MODEL_KEY [...]`               | Skip one or more models by key                     |
| `--mode streaming\|non-streaming`      | Run only streaming or non-streaming tests          |
| `--protocol openai\|mcp\|conversation` | Run tests for a specific protocol only             |
| `--url URL`                            | Server base URL (default: `http://localhost:8000`) |
| `--timeout SECONDS`                    | Per-request timeout in seconds (default: 120)      |

</details>

## Running with a local browser build

Start the browser with

```sh
npm run start -- --ai-chat-server-url="http://127.0.0.1:8000"
```
