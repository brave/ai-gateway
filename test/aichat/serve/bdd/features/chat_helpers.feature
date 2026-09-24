Feature: Chat completion helper functions
  Helpers used by the chat completions pipeline: request helpers, tool
  preparation, message augmentation and stream pipeline events.

  Scenario Outline: The last user message content is extracted
    Given a message list with last user content <description>
    When the last user message content is extracted
    Then the extracted content is <expected>

    Examples:
      | description            | expected    |
      | a_plain_string         | hi there    |
      | a_list_of_text_parts   | hello world |
      | a_list_of_part_objects | part text   |
      | an_assistant_message   | ignored     |
      | no_messages            | none        |
      | only_empty_list        | none        |

  Scenario Outline: Media content in messages is detected
    Given a message list containing <media> content parts
    When the media content is detected
    Then the detected media type is <media_type>

    Examples:
      | media     | media_type |
      | an_image  | vision     |
      | a_file    | file       |
      | a_file_url | file      |
      | a_video   | video      |
      | an_audio  | audio      |
      | text_only | none       |

  Scenario: Metrics state tracks first token and tool use observation
    When the metrics state is created with start time 5.0
    Then the metrics state starts at 5.0 with first token and tool use unrecorded

  Scenario Outline: Mid-stream fallback errors are recognized through the chain
    Given an error chain whose root is named <root_name>
    When the mid-stream fallback check runs
    Then the fallback detection result is <result>

    Examples:
      | root_name              | result |
      | MidStreamFallbackError | True   |
      | ValueError             | False  |

  Scenario: Unknown response payloads cannot be translated
    When an unregistered response payload is translated
    Then no SSE translation is produced

  Scenario Outline: Tools are prepared from client tools and MCP tools
    Given MCP integration is <mcp_enabled> and the model tool support is <tool_support>
    And MCP initialization returns tools "deep_research, web_search"
    And the client sends tool "client_tool"
    And the request excludes MCP tools "<exclude>"
    And the request includes MCP tools "<include>"
    And the client advertises deep research <deep_research>
    When tools are prepared for the request
    Then the prepared tool names are "<tool_names>"

    Examples:
      | mcp_enabled | tool_support | exclude       | include    | deep_research | tool_names              |
      | False       | True         | none          | none       | True          | client_tool             |
      | True        | True         | all           | none       | True          | client_tool             |
      | True        | True         | deep_research | none       | True          | client_tool, web_search |
      | True        | True         | none          | web_search | True          | client_tool, web_search |
      | True        | True         | none          | none       | False         | client_tool, web_search |
      | True        | False        | none          | none       | True          | client_tool             |

  Scenario: MCP fetch failures degrade to client tools only
    Given MCP integration is True and the model tool support is True
    And MCP initialization fails with a network error
    And the client sends tool "client_tool"
    When tools are prepared for the request
    Then the prepared tool names are "client_tool"

  Scenario: Malformed MCP tool responses degrade to client tools only
    Given MCP integration is True and the model tool support is True
    And MCP initialization fails with malformed JSON
    And the client sends tool "client_tool"
    When tools are prepared for the request
    Then the prepared tool names are "client_tool"

  Scenario: Messages are augmented by every registered prompt
    Given a free token limit of 100
    And a prompt that appends "LEO" to the messages
    And a single user message "hello"
    When the messages are augmented without premium access
    Then the augmented messages contain "LEO"
    And the PDF limit check received the free token limit

  Scenario: Premium requests use the premium token limit
    Given a premium token limit of 200
    And a prompt that appends "LEO" to the messages
    And a single user message "hello"
    When the messages are augmented with premium access
    Then the PDF limit check received the premium token limit

  Scenario: Empty streams record no chunks as the empty reason
    Given a streaming chat request for model "test-model"
    When the stream pipeline processes no chunks at all
    Then the stream still ends with a DONE sentinel

  Scenario: Abandoned tool calls record an incomplete tool call empty reason
    Given a streaming chat request for model "test-model"
    When the stream pipeline processes an abandoned tool call
    Then the stream still ends with a DONE sentinel

  Scenario: Prompt injection scan events wrap the content chunks
    Given a streaming chat request for model "test-model"
    And a pending prompt injection scan that completes with a result
    When the stream pipeline processes a chunk with content "hi"
    Then the scan started event is streamed first
    And the scan result event is streamed last before DONE

  Scenario: Completed inline searches are streamed alongside content
    Given a streaming chat request for model "test-model"
    And an inline search completing during the stream
    When the stream pipeline processes a chunk with content "hi"
    Then the inline search result is streamed in the events

  Scenario: Chunk models echo the requested model and drop spaces
    Given a streaming chat request for model "test-model"
    When the stream pipeline processes a chunk with model "hosted vllm test" and requested model "clean-model"
    Then the streamed chunk model equals "clean-model"

  Scenario: Mid-stream fallback errors recover with an error chunk
    Given a streaming chat request for model "test-model"
    And a stream that fails with a MidStreamFallbackError
    When the stream pipeline is consumed
    Then the stream contains a recovered error chunk
    And the stream still ends with a DONE sentinel

  Scenario: Deep research tool calls delegate to the streaming research pipeline
    Given a streaming chat request for model "test-model" with deep research capability
    And deep research tracking is armed
    And a completed tool call for "deep_research"
    When the stream pipeline is consumed
    Then the streaming research handler ran for "deep_research"
    And the stream still ends with a DONE sentinel

  Scenario: Requests without deep research tools record the capability metric
    Given a streaming chat request for model "test-model" with deep research capability
    And deep research tracking is armed
    When the stream pipeline processes a chunk with content "hi"
    Then the deep research capability metric sees the capability unused

  Scenario: Server-side tool events stream while results continue the conversation
    Given a streaming chat request for model "test-model"
    When the stream pipeline processes an MCP tool call that streams an event and returns a result
    Then the streamed tool event is forwarded
    And the follow-up conversation is executed with the tool result
    And the stream still ends with a DONE sentinel