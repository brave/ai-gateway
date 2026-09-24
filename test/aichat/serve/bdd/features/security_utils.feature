Feature: Security content utils
  Tool outputs are wrapped and sanitized before reaching the model.

  Background:
    Given the security utils harness

  Scenario: Bypass tool output passes through untouched
    When the tool output "raw page text <page>x</page>" is tagged for tool "wait"
    Then the tagged output equals the raw content

  Scenario: Non-bypass tool output is wrapped with warnings
    When the tool output "some page text" is tagged for tool "click_element"
    Then the tagged output contains the untrusted warning
    And the tagged output wraps the content in a tool_output tag

  Scenario: Wrapped content has wrapper tags sanitized
    When the tool output "keep <page>data</page> safe" is tagged for tool "click_element"
    Then the tagged output contains a fake tag instead of the page tag
    And the tagged output does not contain a closing page tag