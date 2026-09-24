Feature: Chat completions endpoint
  POST /v1/chat/completions resolves the model, validates the request,
  prepares tools, augments prompts and forwards the conversation to the
  backend. It returns a streaming SSE response or a JSON completion.

  Background:
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled

  Scenario: Non-streaming completion returns backend content
    Given a fake backend for model "test-model" replying "Hello there"
    And a non-streaming chat request for model "test-model"
    When the chat completion is processed
    Then the JSON completion contains "Hello there"

  Scenario: Streaming completion returns an SSE stream
    Given a fake backend for model "test-model" streaming a chunk with content "Hi"
    And a streaming chat request for model "test-model"
    When the chat completion is processed
    Then the result is a streaming SSE response

  Scenario: Backend failure surfaces an HTTP error
    Given a fake backend for model "test-model" failing with code 50200
    And a non-streaming chat request for model "test-model"
    When the chat completion is processed
    Then an HTTP error with status 502 is raised

  Scenario: Tools are stripped for models without tool support
    Given a fake backend for model "test-model" replying "ok"
    And the model config does not support tools
    And a non-streaming chat request for model "test-model" with a get_weather tool
    When the chat completion is processed
    Then the backend received no tools

  Scenario: NEAR model requests authenticate with the NEAR API key
    Given a fake backend for model "near-test" replying "greetings"
    And a non-streaming chat request for model "near-test"
    When the chat completion is processed
    Then the backend params include the NEAR API key

  Scenario: Streaming a NEAR model verifies the ohttp config
    Given a fake backend for model "near-test" streaming a chunk with content "hi"
    And NEAR ohttp verification returns True
    And a streaming chat request for model "near-test"
    When the chat completion is processed
    Then the stream headers contain "Brave-NEAR-verified" set to "true"

  Scenario: Conversation title requests bypass the completion pipeline
    Given a fake backend for model "test-model" replying "unused"
    And a non-streaming chat request carrying a brave-conversation-title part
    When the chat completion is processed
    Then the title service answered and model selection never ran

  Scenario: Too many conversation rounds are rejected
    Given request validation is permissive
    And the conversation rounds maximum is 1
    And a fake backend for model "test-model" replying "unused"
    And a non-streaming chat request for model "test-model" with 3 user messages
    When the chat completion is processed
    Then a JSON error response with status 413 is returned

  Scenario: Pathologically large conversations are rejected
    Given request validation is permissive
    And the absolute token ceiling is 10
    And token trimming reports 9999 tokens
    And a fake backend for model "test-model" replying "unused"
    And a non-streaming chat request for model "test-model"
    When the chat completion is processed
    Then a JSON error response with status 413 is returned

  Scenario: Compaction start yields a compaction_starting event
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a fake backend for model "test-model" streaming a chunk with content "hi"
    And compaction will be triggered
    And compaction produces metadata for messages "[{"role": "user", "content": "sum"}]"
    And a streaming chat request for model "test-model"
    When the chat completion is processed
    Then the first stream event mentions compaction_starting
    And the stream ends with a DONE sentinel

  Scenario: Compaction failure yields a compaction_failed event
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And compaction will be triggered
    And compaction blows up
    And a streaming chat request for model "test-model"
    When the chat completion is processed
    Then the stream mentions compaction_failed
    And the stream ends with a DONE sentinel

  Scenario: Active model override replaces the response model with the placeholder
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a fake backend for model "test-model" replying with model "test-model"
    And an active model override without premium fallback
    And a non-streaming chat request for model "test-model"
    When the chat completion is processed
    Then the JSON completion model equals the placeholder model

  Scenario: Usage chunks emit a content receipt
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a streaming chat request for model "test-model"
    When the stream pipeline processes a usage chunk for 42 total tokens
    Then a content receipt event for 42 total tokens is emitted
    And a usage chunk for 42 total tokens is emitted

  Scenario: Content filter stop reasons emit a user-facing message
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a streaming chat request for model "test-model"
    When the stream pipeline processes a chunk stopped by content_filter
    Then the accumulated content is the content filter message
    And the finish reason is preserved as content_filter

  Scenario: Empty streams record an empty-response metric
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a streaming chat request for model "test-model"
    When the stream pipeline processes no chunks at all
    Then the stream still ends with a DONE sentinel

  Scenario: Client tool calls stop the stream and wait for the client
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a streaming chat request for model "test-model"
    When the stream pipeline processes a completed client tool call
    Then the tool call is forwarded to the client
    And no follow-up conversation is executed

  Scenario: MCP tool calls execute server-side and continue the conversation
    Given request validation is permissive
    And prompts are stubbed out
    And MCP integration is disabled
    And a streaming chat request for model "test-model"
    When the stream pipeline processes a completed MCP tool call with result "tool output"
    Then the follow-up conversation is executed with the tool result