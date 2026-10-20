# Bedrock

AWS Bedrock is a serverless API for interacting with a suite of AWS supported models and tools.

- Supported Foundational Models: <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

The ai-gateway server currently supports hitting Bedrock models via the `/conversation` endpoint for message types `"chatMessage"` and `"uploadImage"`.

To add a new Bedrock model, prefix the model name with `bedrock-` (e.g. `bedrock-us.meta.llama3-2-11b-instruct-v1:0`) and add an entry to your `MODELS` JSON config with `"backend": "bedrock"` (see `.env.example` / `docker-compose.base.yml` for examples). Requires your own AWS credentials with Bedrock access.

## Guardrails

Guardrails are optional. Create an Amazon Bedrock guardrail with the policies you want (for example a content filter for sexual content), publish a numbered version, and set `BEDROCK_GUARDRAIL_ID` and `BEDROCK_GUARDRAIL_VERSION`. When either is unset the gateway behaves as before.

The guardrail is applied per model deployment, so non-Bedrock models and fallbacks are unaffected. Models using the Converse API (`bedrock/`) receive it through litellm's `guardrailConfig`. `BEDROCK_GUARDRAIL_STREAM_MODE` (`sync` or `async`, default `sync`) sets the streaming mode. `bedrock_mantle` models are not covered: the `bedrock-mantle` endpoint does not support guardrails.

Use a numbered version, not `DRAFT`. The guardrail must be in the same region as the Bedrock calls, and the caller needs `bedrock:ApplyGuardrail` on the guardrail (and on its guardrail profile if cross-region inference is enabled).
