Feature: Deep research citation formatting
  Citation helpers build citation maps and format answers.

  Scenario: URLs are encoded for markdown
    Then a url with spaces and parens encodes percent tokens

  Scenario Outline: Citation map entries
    Given citations <citations>
    When the citation map is built
    Then the map has <count> entries
    And each entry carries url encoded url and hostname

    Examples:
      | citations                        | count |
      | numbered and linked              | 1     |
      | missing number                   | 0     |
      | missing url                      | 0     |
      | hostnameless url falls back      | 1     |

  Scenario: Answers gain a separator and spaced citations
    Given an answer "The sky is blue[1] indeed"
    When the answer is formatted with citations
    Then the answer starts with a separator
    And citation markers are preceded by spaces

  Scenario Outline: Plaintext citations
    Given an answer "Final verdict" with <citations>
    When plaintext citations are appended
    Then the plaintext result <expectation>

    Examples:
      | citations           | expectation                 |
      | numbered and linked | lists sources               |
      | none                | ends with the raw answer    |