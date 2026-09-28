Feature: Conversation title completion
  The deprecated conversation title path reuses the streaming stack.

  Background:
    Given the conversation title harness

  Scenario: Empty title part is rejected
    Given a title request with an empty title part
    When the conversation title completion runs
    Then the response is a 400 error response
    And the invalid title request metric increased

  Scenario: Non-streaming title completion returns the backend response
    Given a title request with a filled title part
    And the title backend returns a completion
    When the conversation title completion runs without streaming
    Then the returned response carries the backend content

  Scenario: Backend error dict raises an HTTP error
    Given a title request with a filled title part
    And the title backend returns an error dict with code 50201
    When the conversation title completion runs without streaming
    Then an HTTP error with status 502 is raised

  Scenario: Streaming title completion proxies chunks
    Given a title request with a filled title part
    And the title backend returns a completion
    When the conversation title completion runs with streaming
    Then the streamed title chunks carry the backend content

  Scenario Outline: Title part detection
    Given a message list where <layout>
    When the last message is checked for a title part
    Then the title presence is <present>

    Examples:
      | layout                 | present |
      | user_with_title        | true    |
      | assistant_last         | false   |
      | string_content         | false   |
      | dict_title_part        | true    |

  Scenario: Title text extraction skips non-user and non-title parts
    Given a message list where assistant_then_user
    When the title part text is inspected
    Then the title part has usable text
