Feature: Search MCP server handler
  SearchServerHandler validates, formats and presents Brave search MCP results.

  Background:
    Given a search server handler

  Scenario: Valid result has dict with content list
    When the handler validates a result of kind dict_content
    Then the result is valid

  Scenario: Invalid results are rejected
    When the handler validates a result of kind <kind>
    Then the result is invalid

    Examples:
      | kind          |
      | non_dict      |
      | dict_no_content |

  Scenario: Rich weather data extracted
    Given a search data row with provider url weather name text_name
    When rich weather data is extracted
    Then the extraction succeeds
    And the search data has title "Weather JSON Data from Test" url "https://w.example" and page content containing the weather

  Scenario Outline: Unusable weather payloads rejected
    Given a weather payload variant <variant>
    When rich weather data is extracted
    Then the extraction fails

    Examples:
      | variant          |
      | no_provider      |
      | blank_url        |
      | missing_weather  |
      | provider_not_dict |

  Scenario: Rich currency data extracted
    Given a currency payload with provider url conversion and name
    When rich currency data is extracted
    Then the extraction succeeds
    And the currency payload gains title url and page content

  Scenario Outline: Unusable currency payloads rejected
    Given a currency payload variant <variant>
    When rich currency data is extracted
    Then the extraction fails

    Examples:
      | variant              |
      | provider_missing     |
      | blank_url            |
      | currency_missing     |
      | conversion_missing   |

  Scenario: JSON list of results is formatted
    Given a raw search result with json list of two results
    When the result is formatted for tool brave_web_search
    Then two sources are produced
    And the content contains search_result markup
    And the search results text is joined
    And no queries are attached

  Scenario: Results wrapper dict is expanded
    Given a raw search result wrapping results list
    When the result is formatted for tool brave_web_search
    Then one source is produced

  Scenario: Non-list results value is kept without a url
    Given a raw search result with scalar results value
    When the result is formatted for tool brave_web_search
    Then zero sources are produced

  Scenario: Multiline JSONL text is parsed line by line
    Given a raw search result with two jsonl lines and one broken line
    When the result is formatted for tool brave_web_search
    Then two sources are produced

  Scenario: Blank text items are skipped
    Given a raw search result with blank and empty texts
    When the result is formatted for tool brave_web_search
    Then zero sources are produced
    And the fallback summary is returned

  Scenario: Rich weather result becomes a source and rich result
    Given a raw search result containing a rich weather entry
    When the result is formatted for tool brave_web_search
    Then one source is produced
    And rich results are attached
    And the weather source has the provider title and url

  Scenario: Unusable rich entries produce no source
    Given a raw search result containing an unusable rich weather entry
    When the result is formatted for tool brave_web_search
    Then zero sources are produced
    And rich results are still recorded

  Scenario: Rich currency result becomes a source
    Given a raw search result containing a rich currency entry
    When the result is formatted for tool brave_web_search
    Then one source is produced
    And rich results are attached

  Scenario: Rich entry with unknown subtype is dropped
    Given a raw search result containing a rich entry with unknown subtype
    When the result is formatted for tool brave_web_search
    Then zero sources are produced

  Scenario: Source fields include favicons and snippets
    Given a raw search result with meta_url favicon and extra snippets
    When the result is formatted for tool brave_web_search
    Then the source carries meta_url favicon extra snippets and snippet

  Scenario: Non-list extra snippets are dropped
    Given a raw search result with scalar extra snippets and a description
    When the result is formatted for tool brave_web_search
    Then the source carries description as snippet and no extra snippets

  Scenario: Summary lists titles and counts more
    Given a raw search result with four results
    When the result is formatted for tool brave_web_search
    Then the summary names three titles and mentions 1 more

  Scenario: Queries from result top level are attached
    Given a raw search result with top level queries
    When the result is formatted for tool brave_web_search
    Then the queries are attached

  Scenario: Queries from structured content are attached
    Given a raw search result with structured content queries
    When the result is formatted for tool brave_web_search
    Then the queries are attached

  Scenario Outline: Tool start messages
    Given a tool start for <tool> with query <query>
    When the tool start message is requested
    Then the start message is "<message>"

    Examples:
      | tool              | query          | message                     |
      | brave_web_search  | norway capital | Searching for: norway capital |
      | brave_web_search  | none           | Searching the web...        |
      | brave_news_search | ai regulation  | Searching news for: ai regulation |
      | brave_news_search | none           | Searching news...           |
      | other_tool        | none           | Running other_tool...       |

  Scenario: Tool guidance and augmented tools
    When tool guidance is requested
    Then guidance covers brave_web_search and brave_news_search
    And there are no augmented tools

  Scenario: Tool message content from sources
    Given a formatted result with two sources and queries
    When the tool message content is requested
    Then the content is a web sources part list

  Scenario: Tool message content from rich results only
    Given a formatted result with only rich results
    When the tool message content is built
    Then the content is a text part summary

  Scenario: Tool message content falls back to text
    Given a formatted result with plain content "search finished"
    When the tool message content is built
    Then the tool message content is "search finished"

  Scenario: Tool message content with empty content
    Given a formatted result with no sources and blank content
    When the tool message content is built
    Then the tool message content is "Tool execution completed"

  Scenario: Output content parts with sources
    Given a formatted result with two sources and a query list
    When the output content parts are built
    Then one web sources output part is produced
    And the output part carries the query list

  Scenario: Output content parts without sources
    Given a formatted result with no sources
    When the output content parts are built
    Then the output parts are empty

  Scenario Outline: Formatting web sources content
    Given web sources payload kind <kind>
    When web sources content is formatted
    Then the formatted web sources text <expectation>

    Examples:
      | kind                | expectation                    |
      | basic               | has one citation block         |
      | with_page_content   | embeds full page content       |
      | with_extra_snippets | embeds additional snippets     |
      | empty               | reports zero results           |

  Scenario: Handler names the search server
    Then the server name is brave_search