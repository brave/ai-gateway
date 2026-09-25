Feature: Bedrock message tooling edge paths
  Bedrock adapters sanitize tool ids, split mixed turns, and format tools.

  Background:
    Given the bedrock edge harness

  Scenario Outline: Assistant content emptiness detection
    When the assistant content variant <variant> is checked
    Then non-emptiness is <result>

    Examples:
      | variant                       | result |
      | plain text                    | true   |
      | whitespace only               | false  |
      | list with text part           | true   |
      | list with content part        | true   |
      | list with truthy object part  | true   |
      | list with empty parts         | false  |
      | dict part without text        | false  |
      | numeric content               | true   |

  Scenario: Splitting an empty message list is a no-op
    When an empty message list is split
    Then the split result is empty

  Scenario: Mixed assistant turn is split into canonical order
    Given an assistant turn with content and tool calls followed by a tool result
    When the messages are split
    Then the tool calls come first without content
    And the original content becomes a trailing assistant message

  Scenario: Non-string continuation arguments are dropped
    Given tool calls where a continuation has non-string arguments
    When the split tool calls are merged
    Then the continuation is dropped and the warning is logged

  Scenario: Pairing validation returns the original on internal failure
    Given a non-iterable message list for pairing validation
    When the tool call result pairing is validated
    Then the original messages are returned unchanged

  Scenario: Tool formatting handles a non-dict function definition
    Given a tool whose dump contains a plain object function
    When the tools are formatted for bedrock
    Then the formatted tool keeps a function mapping

  Scenario: Tool formatting handles a mapping function definition
    Given a tool whose dump contains a keys-style function
    When the tools are formatted for bedrock
    Then the formatted tool keeps a function mapping
