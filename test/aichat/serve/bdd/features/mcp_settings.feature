Feature: MCP settings coercion
  MCPSettings coerces the MCP_SERVERS environment value.

  Background:
    Given no MCP server environment variables

  Scenario: Default settings values
    When default MCP settings are constructed
    Then mcp is enabled by default
    And deep research is disabled with localhost url
    And deep research limits are 3 iterations 30 queries and 300 seconds

  Scenario Outline: Server list coercion
    Given an MCP servers value <raw_value>
    When MCP settings are constructed
    Then the coerced server list is <expected>

    Examples:
      | raw_value            | expected           |
      | none_value           | empty              |
      | empty_string         | empty              |
      | valid_json_list      | passthrough        |
      | dict_payload         | empty              |
      | json_list_string     | parsed             |
      | json_dict_string     | empty              |
      | broken_json_string   | empty              |
      | blank_string         | empty              |
      | number               | empty              |
