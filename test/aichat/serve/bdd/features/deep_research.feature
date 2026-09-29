Feature: Deep research streaming
  Deep research executor, service orchestration and handler output.

  Background:
    Given a deep research harness

  Scenario: Disabled deep research yields an error event
    Given deep research is disabled
    When the stream is executed for "why is the sky blue"
    Then a disabled error event is yielded

  Scenario Outline: Transport failures yield error events
    Given the deep research transport raises <failure>
    When the stream is executed
    Then an unreachable or timeout error event is yielded

    Examples:
      | failure            |
      | timeout            |
      | connect error      |
      | generic request    |

  Scenario: Leftover buffer junk is ignored
    Given a stream with a trailing non json fragment
    When the stream is executed
    Then no error is raised and events are yielded

  Scenario: Missing query argument errors immediately
    Given a deep research call without a query
    When deep research is executed and streamed
    Then a missing query error event is yielded

  Scenario: Final answer routes through the handler
    Given a registry containing the deep research handler
    And streamed events ending with a final answer and citations
    When deep research is executed and streamed
    Then a web sources chunk precedes the completion
    And the completion chunk contains spaced citations
    And the tool message holds the formatted answer

  Scenario: Final answer falls back without a handler
    Given a registry without the deep research handler
    And streamed events ending with a final answer
    When deep research is executed and streamed
    Then the completion chunk is yielded
    And the tool message uses plaintext citations

  Scenario: No final answer produces a marker chunk
    Given a registry containing the deep research handler
    And streamed events without a final answer
    When deep research is executed and streamed
    Then a deep research placeholder output chunk is yielded
    And the tool message says no final answer

  Scenario: Handler message content with neither sources nor content
    Given a formatted deep research result with no sources
    When the deep research tool message content is built
    Then the deep research content is "Deep research completed"
