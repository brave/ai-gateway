Feature: Deprecated conversation endpoint and litellm passthrough
  POST /v1/conversation is deprecated: it returns a static localized
  upgrade notice without auth, rate limiting or LLM calls.
  POST /v1/passthrough forwards OpenAI-compatible requests straight to
  the litellm backend and streams raw chunks back.

  Scenario Outline: The deprecation notice is localized or falls back to English
    Given a conversation client with system language "<system_language>"
    When the deprecation notice is built
    Then the notice equals "<expected>"

    Examples:
      | system_language | expected          |
      | none            | en_only           |
      | en-US           | en_only           |
      | fr-FR           | fr_then_en        |
      | xx-YY           | en_only           |

  Scenario: Non-streaming conversation requests return a static notice
    Given a conversation client without credentials
    When a non-streaming conversation request is sent for language "de"
    Then the conversation response is a completion notice for "de"

  Scenario: Streaming conversation requests emit one SSE completion event
    Given a conversation client without credentials
    When a streaming conversation request is sent for model "mixtral-8x7b-instruct"
    Then the conversation stream ends with a DONE sentinel
    And the first conversation event is an English completion notice

  Scenario Outline: Passthrough rejects malformed or unsupported requests
    Given a passthrough request with <problem>
    When the passthrough request is processed
    Then a passthrough HTTP error with status 400 is raised

    Examples:
      | problem              |
      | a non-object body    |
      | a non-bool prompt_caching |
      | an unknown backend   |

  Scenario: Non-streaming passthrough returns the backend response
    Given a passthrough request for model "mixtral-8x7b-instruct" without streaming
    And a prompt caching flag of False
    And extra parameters "temperature=0.5, max_tokens=64"
    When the passthrough request is processed
    Then the backend received the prompt caching flag
    And the passthrough response is the backend completion

  Scenario: Sampling params are pruned then clamped for Claude upstreams
    Given a passthrough request for model "mixtral-8x7b-instruct" without streaming
    And a prompt caching flag of None
    And extra parameters "temperature=0.5"
    When the passthrough request is processed
    Then claude sampling params were applied to the backend params

  Scenario: Streaming passthrough emits SSE chunks and a DONE sentinel
    Given a streaming passthrough request for model "mixtral-8x7b-instruct"
    And a backend streaming two serializable chunks
    When the passthrough request is processed
    And the passthrough stream is consumed
    Then the stream contains both chunk payloads
    And the stream ends with a DONE sentinel

  Scenario: Unserializable streaming chunks are skipped with a warning
    Given a streaming passthrough request for model "mixtral-8x7b-instruct"
    And a backend streaming one unserializable chunk and one serializable chunk
    When the passthrough request is processed
    And the passthrough stream is consumed
    Then the stream contains only the serializable chunk payload
    And the stream ends with a DONE sentinel

  Scenario: Backend error payloads become HTTP errors
    Given a passthrough request for model "mixtral-8x7b-instruct" without streaming
    And a backend error with code "50200"
    When the passthrough request is processed
    Then a passthrough HTTP error with status 502 is raised