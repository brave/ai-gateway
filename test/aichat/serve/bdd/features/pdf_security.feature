# PDF trust-boundary security tests.
#
# KNOWN BYPASSES (see review of PR #79 follow-ups):
#
# B1. Main /v1/chat/completions PDF-limit bypass via MIME parameters.
#     aichat/serve/services/pdf.py `_is_pdf_file_data` matched the exact,
#     case-sensitive prefix "data:application/pdf;base64,". Data URLs with
#     MIME parameters (";name=a.pdf", ";charset=binary", ";base64;x=y") skip
#     sandbox analysis, page/token truncation and the 20MB size check, yet
#     litellm's BedrockImageProcessor._parse_base64_image (factory.py:3449)
#     strips parameters and emits the identical Bedrock document(format=pdf).
#     -> FIXED (detection) in PR #83; remaining escapes (oversize, op
#        errors, non-PDF data urls) are removed by the fail-closed gate
#        (features/data_url_gate.feature).
#
# B2. API-key chat endpoint (api_key_chat_api.py) drops list content entirely
#     (messages carry a pydantic ValidatorIterator that is consumed by
#     per-message validation, so the backend receives an exhausted iterator
#     with 0 parts). The same endpoint forwards raw unvalidated dicts, so any
#     materialization re-opens file_id smuggling: litellm fetches the
#     attacker-controlled URL (public SSRF, uncapped body) and converts it to
#     a native PDF document. No preprocess_messages / PDF limits run here.
#     -> FIXED: raw body messages are validated once and materialized
#        (model_dump strips file_id/format), then gated + PDF-limited.
#
# B3. Conversation-title path forwards unprocessed PDFs. REFUTED at endpoint
#     level: empty-text title parts are rejected with 400 before augment
#     (conversation_title.py `_title_part_has_no_text`), and a truthy title
#     part makes augment replace the whole history. Endpoint-level regression
#     lives in features/data_url_gate.feature ("The title path never
#     forwards pdf parts").
#
# Not directly testable as scenarios (design risks, tracked in review):
# - pdf.py:147-154,159-164 fail-open: oversize >20MB and SandboxOpError PDFs
#   pass through un-truncated (20MB cap is advisory only).
# - utils.py:256-296 + conversation_settings.py:17: native PDF file parts
#   count as file_token_estimate (default 0) tokens -> invisible to
#   trimming/compaction/max-token checks.
# - passthrough_api.py: internal-key endpoint forwards messages verbatim with
#   no PDF boundary; auth is a no-op when INTERNAL_MODELS_API_KEY is unset.
# - media-processor (companion repo): caller-controlled analysis params, no
#   auth/body limit, pool timeout kills worker and fails queued requests.
Feature: PDF trust boundary
  Every client-supplied PDF that reaches a provider must first pass
  media-sandbox processing, regardless of how its data URL is written.

  Background:
    Given a pdf payload

  Scenario Outline: PDF data URLs carrying MIME parameters are still PDFs
    When a file part with data URL "data:application/pdf<params>" is checked
    Then it is classified as a pdf

    Examples:
      | params                  |
      | ;name=a.pdf;base64,     |
      | ;charset=binary;base64, |

  Scenario: An empty file_data is not a pdf
    When a file part with data URL "" is checked
    Then it is not classified as a pdf

  Scenario: MIME-parameter PDFs are processed by the sandbox
    When a user message carries a pdf with data URL "data:application/pdf;name=a.pdf;base64,"
    And pdf limits are enforced
    Then the sandbox was invoked
