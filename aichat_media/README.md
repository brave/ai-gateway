# aichat-media

Sandboxed media parsing service for Brave's AI Chat. Runs the Landlock/rlimit
sandboxed workers that:

- Parse PDFs to split/extract text for Bedrock's page limits (`/v1/pdf/analyze`)
- Decode PCM WAV uploads for speech-to-text (`/v1/stt/decode`)

This is called synchronously over HTTP by `ai-gateway` (the `aichat` package)
during request handling. It is deployed as its own EKS Deployment so its
CPU/memory can be provisioned and scaled independently of the main chat
completions service.

See `aichat_media/media_sandbox/` for the sandboxing implementation
(Landlock, rlimits, subprocess worker pool, JSON-lines IPC protocol).
