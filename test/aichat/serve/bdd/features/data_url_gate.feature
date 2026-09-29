# Data URL fail-closed gate (security follow-up to PR #79 / #83).
#
# Rule: any client data URL that is not passed to the media processor is
# removed from the message before it reaches a provider. Known safe binary
# MIME types (image/png, jpeg, gif, webp — verified against litellm 1.93.0
# converters) bypass the processor; everything else must be confirmed by
# the processor's magic-byte sniff (/v1/media/inspect) or is removed.
# The PDF sandbox flow is fail-closed: oversize, SandboxOpError and
# unexpected processing failures remove the PDF part instead of passing it
# through raw.
Feature: Data URL fail-closed gate
  No data URL reaches a provider without media-processor handling.

  Background:
    Given a pdf payload

  Scenario Outline: Processing failures remove the pdf part
    Given a user message with a canonical pdf file part
    And the sandbox <outcome>
    When pdf limits are enforced
    Then the pdf part is removed

    Examples:
      | outcome           |
      | rejects the pdf   |
      | dies unexpectedly |

  Scenario: An oversized pdf is removed
    Given a user message with a canonical pdf file part
    And the pdf size limit is 0 MB
    When pdf limits are enforced
    Then the pdf part is removed

  Scenario: A sandbox worker death still fails the request
    Given a user message with a canonical pdf file part
    And the sandbox worker dies
    When pdf limits are enforced
    Then a hard sandbox error is raised


  Scenario: A known safe image file part is kept
    When the gate runs on a message with a "data:image/png;base64," part
    Then the part is kept

  Scenario: A safe binary payload rescued by the processor pass
    Given the media processor confirms a safe binary payload
    When the gate runs on a message with a "data:application/octet-stream;base64," part
    Then the part is kept

  Scenario: A text payload is removed
    Given the media processor rejects the payload
    When the gate runs on a message with a "data:text/plain;base64," part
    Then the part is removed

  Scenario: The gate fails closed when the processor is unavailable
    Given the media processor is unavailable
    When the gate runs on a message with a "data:image/svg+xml;base64," part
    Then the part is removed

  Scenario: The api-key endpoint forwards list content intact
    When an api-key chat request with one text part is handled
    Then the backend received 1 content part

  Scenario: The api-key endpoint removes unprocessed data urls
    When an api-key chat request with a "data:text/plain;base64," file part is handled
    Then the backend received 0 content part

  Scenario: The api-key endpoint processes pdfs through the sandbox
    Given a canonical pdf file part
    When an api-key chat request with the pdf file part is handled
    Then the sandbox was invoked

  Scenario: The title path never forwards pdf parts
    When a title request carries a pdf part and a title part with text
    Then the title model saw only the synthetic transcript

  Scenario: The passthrough endpoint removes unprocessed data urls
    When a passthrough request with a "data:text/plain;base64," file part is handled
    Then the backend received 0 content part