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

  Scenario: Client for request creates a fallback client
    Given no shared MCP client
    When a client is obtained for the request
    Then a new MCP client is created

  Scenario Outline: Warmup skips when disabled or no stdio servers
    Given mcp enabled is <enabled> and stdio servers are <stdio>
    When the shared MCP client warmup runs
    Then the warmup result is <outcome>

    Examples:
      | enabled | stdio | outcome |
      | false   | yes   | none    |
      | true    | no    | none    |

  Scenario: Warmup starts stdio subprocesses and caches the client
    Given mcp enabled is true and stdio servers exist
    When the shared MCP client warmup runs
    Then a client is warmed up and stored as shared

  Scenario: Warmup failure degrades gracefully
    Given mcp enabled is true and stdio servers exist but startup fails
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
    Given a registry with handlers and a guidance client
    When tool guidance is fetched
    Then the guidance dictionary is returned

  Scenario: Tool guidance failure returns empty dict
    Given a guidance client that raises
    When tool guidance is fetched
    Then the guidance is empty