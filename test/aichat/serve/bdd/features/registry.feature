Feature: MCP registry
  MCPServerRegistry stores handlers and formats results through configs.

  Background:
    Given an empty MCP registry

  Scenario: MCP content parsing extracts text items
    Given a base handler and a result with two text items and one non text
    When the MCP content is parsed
    Then only the two text items are returned

  Scenario Outline: Content parsing falls back to string conversion
    Given a base handler and a result of kind <kind>
    When the MCP content is parsed
    Then the parsed content is a single string

    Examples:
      | kind       |
      | no_content |
      | not_a_dict |

  Scenario: Base class defaults
    Given a base handler
    Then base tool guidance is empty
    And there are no augmented tools
    And base tool message content echoes the result
    And base output content parts are empty

  Scenario: Formatting failure returns an error payload
    Given a config whose formatter always raises
    When the config formats a result
    Then a formatting failed error payload is returned

  Scenario: Validation failure returns False
    Given a config whose validator always raises
    When the config validates a result
    Then the validation returns False

  Scenario: Servers listed regardless of enabled state
    Given a registered server brave_search enabled
    And a registered disabled server using the default handler
    When all servers are listed
    Then two servers are returned
    When only enabled servers are listed
    Then one server is returned

  Scenario: Tool start message from handler or fallback
    Given a registered search handler
    When a start message is requested for brave_search
    Then the handler message is used
    When a start message is requested for missing_server
    Then the fallback start message is used
