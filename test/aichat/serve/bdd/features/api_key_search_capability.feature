Feature: API-key search capability
  Search flags on API-key chat completions require the web_search capability.

  Scenario Outline: A search flag is rejected without the web_search capability
    Given the internal API is enabled
    And verify allows the key with capabilities "chat"
    When a streaming api-key chat completion is requested with "<flag>"
    Then the response status is 403
    And the error code is "permission_denied"
    And search tools were not loaded
    And no completion was served

    Examples:
      | flag                |
      | brave_inline_search |
      | brave_search_tools  |

  Scenario: Search tools are loaded with the web_search capability
    Given the internal API is enabled
    And verify allows the key with capabilities "chat,web_search"
    When a streaming api-key chat completion is requested with "brave_search_tools"
    Then the response status is 200
    And search tools were loaded
    And the completion was served with the search tool

  Scenario: Inline search is served with the web_search capability
    Given the internal API is enabled
    And verify allows the key with capabilities "chat,web_search"
    When a streaming api-key chat completion is requested with "brave_inline_search"
    Then the response status is 200
    And search tools were not loaded
    And a completion was served

  Scenario: Missing capabilities default to chat
    Given the internal API is enabled
    And verify allows the key without capabilities
    When a streaming api-key chat completion is requested with "brave_search_tools"
    Then the response status is 403
    And no completion was served

  Scenario: Empty capabilities default to chat
    Given the internal API is enabled
    And verify allows the key with empty capabilities
    When a streaming api-key chat completion is requested with "brave_search_tools"
    Then the response status is 403
    And no completion was served

  Scenario: Chat without search needs only the chat capability
    Given the internal API is enabled
    And verify allows the key with capabilities "chat"
    When a streaming api-key chat completion is requested without search
    Then the response status is 200
    And a completion was served

  Scenario: Configured default capabilities allow search when verification is disabled
    Given the internal API is disabled
    And the default key capabilities are "chat,web_search"
    When a streaming api-key chat completion is requested with "brave_search_tools"
    Then verify was not called
    And the response status is 200
    And search tools were loaded

  Scenario: Configured default capabilities deny search when verification is disabled
    Given the internal API is disabled
    And the default key capabilities are "chat"
    When a streaming api-key chat completion is requested with "brave_search_tools"
    Then verify was not called
    And the response status is 403
    And no completion was served
