Feature: API-key chat completion usage settlement
  Provider-reported token usage for API-key chat completions is settled
  once per request, after the response is delivered.

  Background:
    Given the internal API is enabled
    And the upstream model is served by the "bedrock" backend
    And the upstream reports usage of 13 prompt, 12 completion, 3 cache write, 4 cache read and 5 reasoning tokens

  Scenario: A non-streaming completion settles the reported usage
    When a non-streaming api-key chat completion is handled
    Then usage was settled once
    And the settled usage carried the api key, model and provider "bedrock"
    And the settled usage carried 13 prompt, 12 completion, 3 cache write, 4 cache read and 5 reasoning tokens
    And the settled transaction id starts with "txn_"
    And the settle outcome "settled" was recorded

  Scenario: A streaming completion settles usage from the final chunk
    When a streaming api-key chat completion is handled
    Then usage was settled once
    And the settled usage carried 13 prompt, 12 completion, 3 cache write, 4 cache read and 5 reasoning tokens
    And the stream ends with a DONE sentinel

  Scenario: Streaming usage is hidden from clients that did not request it
    When a streaming api-key chat completion is handled
    Then the client stream contains no usage
    And usage was settled once

  Scenario: Streaming usage is forwarded to clients that requested it
    Given the client requests stream usage
    When a streaming api-key chat completion is handled
    Then the client stream contains the usage
    And usage was settled once

  Scenario: Upstream usage is always requested for streams
    Given the client disables stream usage
    When a streaming api-key chat completion is handled
    Then the upstream request asked for stream usage
    And the client stream contains no usage
    And usage was settled once

  Scenario: A client disconnect still settles usage from the rest of the stream
    When a streaming api-key chat completion is disconnected after the first chunk
    Then usage was settled once
    And the settled usage carried 13 prompt, 12 completion, 3 cache write, 4 cache read and 5 reasoning tokens

  Scenario: A disconnect while waiting for the upstream still settles usage once
    When a streaming api-key chat completion is disconnected while waiting for the upstream
    Then usage was settled once
    And the settled usage carried 13 prompt, 12 completion, 3 cache write, 4 cache read and 5 reasoning tokens

  Scenario: A non-streaming disconnect while waiting for the upstream still settles usage
    When a non-streaming api-key chat completion is disconnected while waiting for the upstream
    Then usage was settled once
    And the settled usage carried 13 prompt, 12 completion, 3 cache write, 4 cache read and 5 reasoning tokens

  Scenario: A completion without usage is not settled
    Given the upstream reports no usage
    When a non-streaming api-key chat completion is handled
    Then usage was not settled
    And the settle outcome "missing_usage" was recorded

  Scenario: An unavailable settle endpoint is recorded
    Given aichat-internal cannot settle usage
    When a non-streaming api-key chat completion is handled
    Then usage was settled once
    And the settle outcome "unavailable" was recorded

  Scenario: Usage is not settled when the internal API is disabled
    Given the internal API is disabled
    When a non-streaming api-key chat completion is handled
    Then usage was not settled

  Scenario: Each request settles with its own transaction id
    When 2 non-streaming api-key chat completions are handled
    Then usage was settled 2 times with distinct transaction ids

  Scenario Outline: The settled provider is derived from the model backend
    Given the upstream model is served by the "<backend>" backend at "<api_base>"
    When a non-streaming api-key chat completion is handled
    Then the settled usage carried the api key, model and provider "<provider>"

    Examples:
      | backend        | api_base                  | provider    |
      | bedrock        | none                      | bedrock     |
      | bedrock_mantle | none                      | bedrock     |
      | litellm        | https://cloud-api.near.ai | near        |
      | litellm        | http://vllm:8000/v1       | self_hosted |
