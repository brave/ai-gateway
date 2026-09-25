Feature: Default MCP server handler
  DefaultMCPServerHandler passes generic MCP results through.

  Background:
    Given a default server handler

  Scenario Outline: Validation accepts only non-None
    When the handler validates a result of kind <kind>
    Then the result is <validity>

    Examples:
      | kind     | validity |
      | dict     | valid    |
      | text     | valid    |
      | nothing  | invalid  |

  Scenario: Title and url payloads are flattened
    Given a raw default result with json content containing title and url
    When the result is formatted for tool web_fetch
    Then the default content is "Page Title\nhttps://p.example\nSome snippet"
    And the raw result is preserved

  Scenario: Other JSON payloads are pretty printed
    Given a raw default result with json content without title
    When the result is formatted for tool web_fetch
    Then the default content is pretty printed json

  Scenario: Non JSON texts pass through verbatim
    Given a raw default result with plain text content
    When the result is formatted for tool web_fetch
    Then the default content equals the text

  Scenario: Multiple texts are joined with blank lines
    Given a raw default result with two text items
    When the result is formatted for tool web_fetch
    Then the default content joins the two texts

  Scenario: Result without text content
    Given a raw default result with no text items
    When the result is formatted for tool web_fetch
    Then the default content is empty

  Scenario: Default handler metadata
    Then the server name is default
    And tool guidance is empty
    And there are no augmented tools
