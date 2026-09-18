# Premium model access

The server distinguishes premium from free requests by checking two things:

1. **Host check** — whether the HTTP `Host` or `X-Forwarded-Host` header matches `AI_CHAT_PREMIUM_HOST`
2. **SKU credential check** — whether a valid Leo Premium SKU cookie is present

Being identified as a premium request affects:

- **Model availability** — models marked `"free": false` in the models config are silently downgraded to `automatic` for non-premium requests
- **`automatic` model routing** — premium requests use the `premium` model from the triaging config; non-premium requests use the `non-premium` model (with a limited daily allowance before that kicks in)
- **Token limits** — premium requests use `conversation_token_limit_premium` and `max_tokens_premium`
- **Page limits** — premium requests use `max_pages_premium` for document retrieval
- **Rate limiting** — premium and free requests are tracked against different tiers

`start-local.sh` sets `ENV=local` (bypasses the SKU credential check) and `AI_CHAT_PREMIUM_HOST=127.0.0.1:8000` (matches the Host header sent by local clients), so all models and premium limits are available by default when developing locally.

To test specific auth scenarios, remove or change the relevant variable in `start-local.sh`:

| Scenario                                  | Change                                                                                               |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Test non-premium model routing and limits | Remove `-e AI_CHAT_PREMIUM_HOST=...`                                                                 |
| Test SKU credential validation            | Change `ENV` from `local` to any other value (requires a real SKU credential and payment API access) |
