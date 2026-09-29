Feature: MCP integration helpers
  Shared MCP client lifecycle and per-request initialization.

  Background:
    Given MCP integration state is reset

  Scenario: Shared client roundtrip
    Given a shared MCP client is set
    Then the shared MCP client is returned
    When the shared client is cleared
    Then no shared MCP client remains

  Scenario: Client for request reuses the shared client
    Given a shared MCP client is set
    When a client is obtained for the request
    Then the shared client is reused

  Scenario: Shared client creation creates and stores a new client
    Given no shared MCP client
    When a client is obtained for the request
    Then a new MCP client is created

  Scenario Outline: Warmup skips when disabled or no enabled servers
    Given mcp enabled is <enabled> and enabled servers are <enabled_servers>
    When the shared MCP client warmup runs
    Then the warmup result is <outcome>

    Examples:
      | enabled | enabled_servers | outcome |
      | false   | yes             | none    |
      | true    | no              | none    |
      | false   | no              | none    |

  Scenario: Warmup starts stdio subprocesses and caches the client
    Given mcp enabled is true and enabled servers exist
    When the shared MCP client warmup runs
    Then a client is warmed up and cached as shared

  Scenario: Warmup failure degrades gracefully
    Given mcp enabled is true and enabled servers exist but startup fails
    When the shared MCP client warmup runs
    Then the warmup result is none

  Scenario Outline: Shared client shutdown
    Given a shared client that is <state>
    When the shared MCP client shuts down
    Then the close behaviour is <outcome>

    Examples:
      | state        | outcome              |
      | none         | skipped              |
      | healthy      | transports closed    |
      | failing      | error logged safely  |

  Scenario: Tool guidance comes from the client
    Given a shared client that provides guidance
    When tool guidance is fetched
    Then the guidance dictionary is returned

  Scenario: Tool guidance failure returns empty dict
    Given a guidance client that raises
    When tool guidance is fetched
    Then the guidance is empty
