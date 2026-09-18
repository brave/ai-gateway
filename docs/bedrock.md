# Bedrock

AWS Bedrock is a serverless API for interacting with a suite of AWS supported models and tools.

- Supported Foundational Models: <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

The ai-gateway server currently supports hitting Bedrock models via the `/conversation` endpoint for message types `"chatMessage"` and `"uploadImage"`.

To add a new Bedrock model, prefix the model name with `bedrock-` (e.g. `bedrock-us.meta.llama3-2-11b-instruct-v1:0`) and add an entry to your `MODELS` JSON config with `"backend": "bedrock"` (see `.env.example` / `docker-compose.base.yml` for examples). Requires your own AWS credentials with Bedrock access.
