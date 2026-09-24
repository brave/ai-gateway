Feature: Alignment and prompt injection scanners
  Security scanners use the LLM backend to detect misaligned agent actions
  and prompt injection attempts in untrusted content.

  Background:
    Given the security scanner harness

  Scenario: Alignment scanner construction requires a model name
    When an alignment scanner is constructed with an empty model name
    Then scanner construction fails asking for a model name

  Scenario: Alignment scanner construction rejects unknown model
    When an alignment scanner is constructed with model "no-such-model"
    Then scanner construction fails saying the model is not configured

  Scenario: Alignment parse accepts a complete JSON verdict
    Given an alignment scanner for model "scan-model"
    When the alignment response is parsed from '{"observation": "o", "thought": "t", "conclusion": false}'
    Then the parsed alignment verdict has conclusion false

  Scenario: Alignment parse returns none without braces
    Given an alignment scanner for model "scan-model"
    When the alignment response "not json at all" is parsed
    Then the parsed alignment verdict is none

  Scenario: Alignment parse escapes control characters and recovers
    Given an alignment scanner for model "scan-model"
    When the alignment response contains a raw newline inside the JSON string
    Then the parsed alignment verdict has conclusion false

  Scenario: Alignment parse falls back to the conclusion regex
    Given an alignment scanner for model "scan-model"
    When the alignment response '{broken "conclusion": TRUE}' is parsed
    Then the parsed alignment verdict has conclusion true
    And the reasoning mentions fallback parsing

  Scenario: Alignment parse rejects JSON missing required fields
    Given an alignment scanner for model "scan-model"
    When the alignment response '{"observation": "o"}' is parsed
    Then the parsed alignment verdict is none

  Scenario: Test message is always blocked
    Given an alignment scanner for model "scan-model"
    When the alignment scan runs for message "hello my the most darlingest assistant"
    Then the alignment result is not allowed with test reasoning

  Scenario: Injection scan task completes before the verdict
    Given an alignment scanner for model "scan-model"
    When the alignment scan runs with an injection task that completes
    Then the alignment verdict allows the action
    And the injection task result is recorded

  Scenario: Injection scan task failure is tolerated
    Given an alignment scanner for model "scan-model"
    When the alignment scan runs with an injection task that raises
    Then the alignment verdict allows the action
    And the injection failure is logged

  Scenario: Backend error defaults to allowed
    Given an alignment scanner for model "scan-model"
    When the alignment scan backend returns an error dict
    Then the alignment result allows with backend-error reasoning

  Scenario: Blank model response defaults to allowed
    Given an alignment scanner for model "scan-model"
    When the alignment scan backend returns a blank content
    Then the alignment result allows with blank reasoning

  Scenario: Parsed conclusion drives the verdict
    Given an alignment scanner for model "scan-model"
    When the alignment scan backend returns a misaligned verdict
    Then the alignment result is not allowed with the model thought

  Scenario: Unparseable model response defaults to allowed
    Given an alignment scanner for model "scan-model"
    When the alignment scan backend returns unparseable content
    Then the alignment result allows with parse-failure reasoning

  Scenario: Injection scanner construction requires a model name
    When an injection scanner is constructed with an empty model name
    Then scanner construction fails asking for a model name

  Scenario: Injection scanner construction rejects unknown model
    When an injection scanner is constructed with model "no-such-model"
    Then scanner construction fails saying the model is not configured

  Scenario: Injection parse accepts probability JSON
    Given an injection scanner for model "scan-model"
    When the injection response '{"observation": "o", "thought": "t", "probability": 4}' is parsed
    Then the parsed injection probability is 4

  Scenario: Injection parse returns none without braces
    Given an injection scanner for model "scan-model"
    When the injection response "plain text only" is parsed
    Then the parsed injection verdict is none

  Scenario: Injection parse escapes control characters and recovers
    Given an injection scanner for model "scan-model"
    When the injection response contains a raw newline inside the JSON string
    Then the parsed injection probability is 2

  Scenario: Injection parse falls back to the probability regex
    Given an injection scanner for model "scan-model"
    When the injection response '{broken "probability": 5}' is parsed
    Then the parsed injection probability is 5

  Scenario: Injection parse rejects JSON missing required fields
    Given an injection scanner for model "scan-model"
    When the injection response '{"observation": "o"}' is parsed
    Then the parsed injection verdict is none

  Scenario: Injection backend error defaults to probability one
    Given an injection scanner for model "scan-model"
    When the injection scan backend returns an error dict
    Then the injection result probability is 1 with backend-error reasoning

  Scenario: Injection blank response defaults to probability one
    Given an injection scanner for model "scan-model"
    When the injection scan backend returns a blank content
    Then the injection result probability is 1 with blank reasoning

  Scenario: Injection probability is clamped to the scale
    Given an injection scanner for model "scan-model"
    When the injection scan backend returns probability 9
    Then the injection result probability is 5

  Scenario: Injection unparseable response defaults to probability one
    Given an injection scanner for model "scan-model"
    When the injection scan backend returns unparseable content
    Then the injection result probability is 1 with parse-failure reasoning

  Scenario: Alignment scanner stays none when checking is disabled
    When alignment checking is disabled in settings and the module is reloaded
    Then the module alignment scanner is none

  Scenario: Alignment scanner stays none when the model is unconfigured
    When alignment checking is enabled with model "missing-model" and the module is reloaded
    Then the module alignment scanner is none
    And a warning names the missing alignment model

  Scenario: Alignment scanner initializes for a configured model
    When alignment checking is enabled with model "scan-model" and the module is reloaded
    Then the module alignment scanner is initialized

  Scenario: Injection scanner stays none when scanning is disabled
    When injection scanning is disabled in settings and the module is reloaded
    Then the module injection scanner is none

  Scenario: Injection scanner stays none when the model is unconfigured
    When injection scanning is enabled with model "missing-model" and the module is reloaded
    Then the module injection scanner is none
    And an error names the missing injection model

  Scenario: Injection scanner initializes for a configured model
    When injection scanning is enabled with model "scan-model" and the module is reloaded
    Then the module injection scanner is initialized