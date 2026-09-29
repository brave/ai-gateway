Feature: MCP tool execution
  Tool execution streaming events and streaming-tool follow-up calls.

  Background:
    Given an MCP tool execution harness

  Scenario: Error list results produce error events
    Given a tool call for search
    And the executor returns an error list result
    When the tools are executed and streamed
    Then an error tool message with content "Tool exploded" is yielded
    And a tool error event is yielded
    And an error output chunk is yielded

  Scenario Outline: Tool result processing variants
    Given a raw tool result of kind <kind>
    When the tool result is processed
    Then the processed content is <expected>

    Examples:
      | kind            | expected             |
      | falsy           | empty                |
      | dict_content    | content plus newline |
      | dict_no_content | json dump            |
      | plain           | string conversion    |

  Scenario: Output chunk creation without parts returns None
    When an output chunk is created with no parts
    Then the output chunk is None

  Scenario: Output chunk creation failure returns None
    Given a broken chunk constructor
    When an output chunk is created with parts
    Then the output chunk is None

  Scenario: Error chunk creation failure returns None
    Given a broken chunk constructor
    When an error chunk is created
    Then the error chunk is None

  Scenario: Web sources formatting without handler
    Given no brave search handler registered
    When web sources are formatted
    Then the fallback count text is used

  Scenario: Web sources formatting via search handler
    Given a registered search handler
    And one client web source
    When web sources are formatted
    Then the formatted handler text is used

  Scenario: Tool end event is emitted
    When a tool end event is created for call_9 done_tool
    Then the end event carries tool_end

  Scenario: Simplifying dict tool parts
    Given a tool message with a dict part without text
    When the tool message is simplified
    Then the part is stringified into the content

  Scenario: Simplifying unknown part types
    Given a tool message with an unknown object part
    When the tool message is simplified
    Then the unknown part is skipped

  Scenario: Simplifying non string content
    Given a tool message with integer content
    When the tool message is simplified
    Then the content is the string form

  Scenario: Malformed streaming tool arguments are rejected
    Given a streaming tool call with broken arguments
    When the streaming tool is executed and proxied
    Then a tool start event is yielded first
    And an unknown streaming tool error follows

  Scenario: Deep research events are proxied
    Given a streaming deep research call
    When the streaming tool proxies events
    Then the deep research events are forwarded

  Scenario: Follow-up call after streaming tool results
    Given a streaming tool call that returns a result
    And a backend and request ready for follow-up
    When the streaming tool calls are handled
    Then a follow-up converse call happens
    And the follow-up response is streamed

  Scenario: Deep research only skips the follow-up call
    Given only deep research streaming calls
    When the streaming tool calls are handled
    Then no follow-up converse call happens

  Scenario: Remaining MCP calls skip the follow-up
    Given a streaming tool call that returns a result
    And an outstanding MCP tool call
    When the streaming tool calls are handled
    Then no follow-up converse call happens
