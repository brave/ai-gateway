# Conversation Sharing

Conversation sharing lets a Brave browser user share a Leo conversation as a read-only link. The browser encrypts the conversation locally before upload; the server stores and returns an opaque ciphertext and never has access to the plaintext.

Shares can be deleted before their 7-day expiry via `POST /v1/share/delete`, using a `deletion_id` returned alongside the `share_id` at creation time. Possessing a `share_id` alone (e.g. a leaked share link) is not sufficient to delete a share.

## Endpoints

| Endpoint                   | Auth          | Description                                                                                                                        |
| -------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/share`           | `X-Brave-Key` | Accepts a base64-encoded ciphertext, stores it in S3, returns a UUID `share_id` and a UUID `deletion_id`. Rate-limited per IP.     |
| `GET /v1/share/{share_id}` | None          | Returns the stored ciphertext for a given share ID.                                                                                |
| `POST /v1/share/delete`    | `X-Brave-Key` | Accepts a `deletion_id`, deletes the corresponding share. Returns `404` if the `deletion_id` doesn't resolve. Rate-limited per IP. |

## Configuration

| Environment variable          | Default   | Description                                                                                                           |
| ----------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------- |
| `share_s3_bucket`             | _(empty)_ | S3 bucket name for storing share objects. **Required** — the endpoints return 500 until this is set.                  |
| `share_viewer_origin`         | _(empty)_ | Allowed origin for CORS on `GET /v1/share/{share_id}`, e.g. `https://brave.ai`. No CORS headers are emitted if unset. |
| `share_max_ciphertext_bytes`  | `1048576` | Maximum decoded ciphertext size in bytes (default 1 MB).                                                              |
| `share_create` _(rate limit)_ | `10`      | Daily per-IP create limit. Configured in `rate_limiting_settings.rate_limits`.                                        |
| `share_delete` _(rate limit)_ | `10`      | Daily per-IP delete limit. Configured in `rate_limiting_settings.rate_limits`.                                        |

The S3 bucket needs a lifecycle policy to expire objects after 7 days (covering both the `shares/*` and `deletions/*` prefixes), and an IAM policy granting the server `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject` on those prefixes plus `s3:ListBucket` on the bucket so missing-object lookups return 404 rather than 403.

## Emergency deletion

`scripts/emergency_delete_shares.py` is a standalone incident-response tool, not part of the deployed application. It deletes `shares/{share_id}` objects directly from S3 given only a list of share IDs, for cases where shares have leaked externally and no `deletion_id` is available. See the script's module docstring for usage; it requires AWS credentials with `s3:DeleteObject` on `shares/*` in the target bucket.
