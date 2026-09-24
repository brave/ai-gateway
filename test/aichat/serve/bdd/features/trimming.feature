Feature: Token trimming edge paths
  Trimming truncates oversized content until messages fit the token budget.

  Background:
    Given the trimming harness

  Scenario Outline: Trimmable parts are truncated to fit
    Given messages whose real token count is <real_tokens> but the budget is <budget>
    And a trimmable part of type <part_type> carrying the field <field>
    When the messages are trimmed to fit
    Then the part content is truncated with a trimming note
    And <trimmed> tokens were removed

    Examples:
      | real_tokens | budget | part_type          | field    | trimmed |
      | 5000        | 200    | brave-page-text    | content  | >0      |
      | 5000        | 200    | brave-search-results | text   | >0      |

  Scenario: Non-string tool content is skipped
    Given messages whose real token count is 5000 but the budget is 200
    And a tool message whose content is a list
    When the messages are trimmed to fit
    Then the tool message content is unchanged

  Scenario: Tool trimming failure is logged and skipped
    Given messages whose real token count is 5000 but the budget is 200
    And a tool message with string content
    And text truncation always fails
    When the tool messages are trimmed directly
    Then no tokens were removed

  Scenario: Pass 1 stops when the excess is exhausted
    Given messages whose real token count is 5000 but the budget is 200
    And two trimmable parts where the first consumes the whole excess
    When the messages are trimmed to fit
    Then the second trimmable part is unchanged

  Scenario: Trimmable part without a content field is skipped
    Given messages whose real token count is 5000 but the budget is 200
    And a trimmable part of type brave-page-text carrying neither field
    When the messages are trimmed to fit
    Then the part content is unchanged

  Scenario: Trimmable part with empty content is skipped
    Given messages whose real token count is 5000 but the budget is 200
    And a trimmable part of type brave-page-text carrying an empty content field
    When the messages are trimmed to fit
    Then the part content is unchanged

  Scenario: Oversized plain text parts are capped in pass three
    Given messages whose real token count is 5000 but the budget is 200
    And a plain text part limited to 30 tokens
    When the messages are trimmed to fit
    Then the plain text part is truncated with a trimming note

  Scenario: Text part trimming failure is logged and skipped
    Given messages whose real token count is 5000 but the budget is 200
    And a plain text part limited to 30 tokens
    And text truncation always fails
    When the messages are trimmed to fit
    Then a text part failure is logged

  Scenario: Remaining excess logs a per-role breakdown
    Given messages whose real token count is 5000 but the budget is 10
    And a plain text part limited to 30 tokens
    When the messages are trimmed to fit
    Then an insufficient trimming warning is logged with a role breakdown

  Scenario: Maybe trim applies metrics when trimming occurs
    Given messages whose real token count is 5000 but the budget is 200
    And a trimmable part of type brave-page-text carrying the field content
    When maybe trim runs for a model
    Then trimming metrics are recorded
    And trimmed tokens are reported

  Scenario: Maybe trim is a no-op within the limit
    Given messages whose real token count is 100 but the budget is 5000
    When maybe trim runs for a model
    Then the original messages are returned with zero trimmed tokens