Feature: System One Triton response decoding
  System One generator helpers decode Triton passthrough responses.

  Background:
    Given the system one generator harness

  Scenario: A dict Triton response passes through
    Given the Triton response is the dict {"outputs": [{"name": "result_json", "data": ['{"answers": []}']}]}
    When the Triton response is converted to a dict
    Then the converted dict keeps the outputs

  Scenario: An object with a json method is converted
    Given the Triton response object exposes a json method
    When the Triton response is converted to a dict
    Then the converted dict keeps the outputs

  Scenario: An unexpected Triton response type raises
    Given the Triton response is an object with a non-callable json attribute
    When the Triton response is converted to a dict
    Then a ValueError about the response type is raised

  Scenario: A Triton result without outputs raises
    Given a Triton result dict without outputs
    When the result json is decoded
    Then a ValueError about missing outputs is raised

  Scenario: A Triton result output without data raises
    Given a Triton result whose output has no data
    When the result json is decoded
    Then a ValueError about missing data is raised