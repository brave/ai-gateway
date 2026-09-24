Feature: Brave search proxy and inline search helper
  The rhfetch rich-results proxy, its CORS handling and the inline search task helper.

  Background:
    Given the search proxy harness

  Scenario Outline: CORS headers only apply to allowlisted origins
    Given allowed cors origins are configured
    When the cors headers are computed for origin <origin>
    Then the cors headers are <expected>

    Examples:
      | origin          | expected      |
      | https://a.com   | present       |
      | https://evil.io | absent        |
      | none            | absent        |

  Scenario: The preflight for rhfetch echoes the cors headers
    Given allowed cors origins are configured
    When the rhfetch preflight is requested from "https://a.com"
    Then the preflight carries the allow-all cors headers

  Scenario: The preflight for rhfetch skips cors headers for unknown origins
    Given allowed cors origins are configured
    When the rhfetch preflight is requested from "https://evil.io"
    Then the preflight carries no cors headers

  Scenario: A rich fetch result is proxied with the subscription token
    Given allowed cors origins are configured
    And the upstream returns 200 with {"rich": true}
    When the rich fetch for "topic/news" is requested from "https://a.com"
    Then the proxied body is {"rich": true} and the token header was sent

  Scenario: A non-200 rich fetch raises an upstream failure
    Given allowed cors origins are configured
    And the upstream returns 503
    When the rich fetch for "topic/news" is requested from "https://a.com"
    Then the rich fetch raises the upstream failure

  Scenario: A transport error on rich fetch reports a failure
    Given allowed cors origins are configured
    And the upstream transport drops
    When the rich fetch for "topic/news" is requested from "https://a.com"
    Then the rich fetch raises the upstream failure

  Scenario: Completed inline search tasks are popped and pending ones retained
    Given the inline search helper with one finished and one pending task
    When the completed searches are popped
    Then the finished result is returned and the pending task remains

  Scenario: A failed inline search task is dropped with a warning
    Given the inline search helper with one failed and one pending task
    When the completed searches are popped
    Then no results are returned and the pending task remains
