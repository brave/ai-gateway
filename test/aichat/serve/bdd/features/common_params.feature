Feature: Common completion parameters and request checks
  Shared dependency plumbing for completion endpoints: model resolution,
  model override, bearer tokens and cross-endpoint request validation.

  Scenario Outline: Premium models downgrade on the non-premium endpoint
    Given a model catalog with free "mixtral-8x7b-instruct" and premium "premium-x"
    And the request asks for model "<requested>"
    And the host is premium <is_premium>
    When the model is resolved
    Then the resolved model is "<resolved>"

    Examples:
      | requested                | is_premium | resolved                |
      | premium-x                | False      | automatic               |
      | mixtral-8x7b-instruct    | False      | mixtral-8x7b-instruct   |
      | premium-x                | True       | premium-x               |

  Scenario Outline: Model overrides swap in fallbacks
    Given an override state <override> with premium fallback <premium_fallback> and fallback <fallback>
    And the host is premium <is_premium>
    When the model override is applied to "premium-x"
    Then the resulting model is "<result>"

    Examples:
      | override | premium_fallback | fallback    | is_premium | result    |
      | off      | none             | none        | False      | premium-x |
      | on       | none             | none        | True       | premium-x |
      | on       | free-fallback    | none        | True       | free-fallback |
      | on       | none             | free-fb     | False      | free-fb   |
      | on       | none             | none        | False      | automatic |

  Scenario: Completion common params carry request state
    Given a request for model "automatic" with events and messages
    When the completion common params are built without an override
    Then the raw request state records the model, events and messages
    And the common params mark the automatic request

  Scenario: Requests without a valid premium credential are rejected
    Given a premium host without a valid SKU credential
    When the common request checks run for model "premium-x"
    Then a JSON error response with status 401 is returned

  Scenario Outline: Cross-endpoint request checks reject invalid requests
    Given a model catalog with free "mixtral" and premium "premium-x"
    And the common check state <state>
    When the common request checks run for model "<model>"
    Then a JSON error response with status <status> is returned

    Examples:
      | state                  | model        | status |
      | invalid_services_key   | mixtral      | 401    |
      | request_not_allowed    | mixtral      | 400    |
      | unknown_model          | mystery-m    | 404    |
      | premium_model_free_user | premium-x   | 403    |

  Scenario: Rate limited requests are rejected with a 429
    Given a model catalog with free "mixtral" and premium "premium-x"
    And rate limiting is enabled and the verdict denies the request
    When the common request checks run for model "mixtral"
    Then a JSON error response with status 429 is returned

  Scenario: Rate limit fallback models are propagated
    Given a model catalog with free "mixtral" and premium "premium-x"
    And rate limiting is enabled and the verdict proposes "fallback-m"
    When the common request checks run for model "mixtral"
    Then the common model becomes "fallback-m"

  Scenario: Content agent support unlocks premium models for free users
    Given a model catalog with free "mixtral" and premium "premium-x"
    And premium-x supports the content agent
    And the request advertises the content agent capability
    When the common request checks run for model "premium-x"
    Then the common request checks pass

  Scenario: Automatic models are allowed within the daily limit
    Given a model catalog with free "mixtral" and premium "premium-x"
    And the automatic daily limit peek reports 1 of 10 used
    When the common request checks run for model "automatic"
    Then the common request checks pass

  Scenario: Automatic models beyond the daily limit are rejected
    Given a model catalog with free "mixtral" and premium "premium-x"
    And the automatic daily limit peek reports 11 of 10 used
    When the common request checks run for model "automatic"
    Then a JSON error response with status 403 is returned

  Scenario: Free content agent requests respect the content agent rate limit
    Given a model catalog with free "mixtral" and premium "premium-x"
    And rate limiting is enabled
    And the request advertises the content agent capability
    And the content agent rate limit is exhausted
    When the common request checks run for model "mixtral"
    Then a JSON error response with status 429 is returned

  Scenario Outline: Rate limit peeks decide limits by count
    Given a rate limit peek reporting exceeded <exceeded> count <count> of limit <limit>
    When the peek is evaluated
    Then the peek verdict is <verdict>

    Examples:
      | exceeded | count | limit | verdict |
      | True     | 0     | 10    | False   |
      | False    | 5     | 10    | True    |
      | False    | 11    | 10    | False   |

  Scenario Outline: Bearer tokens are extracted from authorization headers
    Given an authorization header "<header>"
    When the bearer token is extracted
    Then the extracted token is <token>

    Examples:
      | header               | token    |
      | Bearer abc123        | abc123   |
      | bearer abc123        | abc123   |
      | Basic abc            | none     |
      | <empty>              | none     |

  Scenario Outline: Internal models API keys are checked
    Given the internal models API key is <configured>
    And an authorization header "<header>"
    When the internal models API key is validated
    Then the validation result is <result>

    Examples:
      | configured | header              | result |
      | unset      | Bearer anything     | True   |
      | secret123  | Bearer secret123    | True   |
      | secret123  | Bearer wrong        | False  |
      | secret123  | <empty>             | False  |

  Scenario: Error responses follow the Anthropic error envelope
    When an error response is created for the invalid SKU credential code and message "no SKU"
    Then the JSON error status is 401
    And the error payload type is "error"
    And the error message is "no SKU"
    And the error type is "40104"

  Scenario: Missing API key configuration skips enforcement
    Given the internal models API key is unset
    When the internal models API key is required
    Then no error response is produced
