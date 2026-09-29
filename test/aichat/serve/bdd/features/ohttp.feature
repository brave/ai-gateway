Feature: OHTTP config and relay endpoints
  Near-TEE e2ee model configuration lookup and request relay.

  Background:
    Given the ohttp harness

  Scenario: The ohttp config is returned for an e2ee-capable model
    Given a model "near-model" with e2ee support
    And near verification returns the ohttp config
    When the ohttp config for "near-model" is fetched
    Then the ohttp config response is the near config

  Scenario: The ohttp config rejects unknown models
    Given no models are configured
    When the ohttp config for "ghost" is fetched
    Then the ohttp config response is a model not found error

  Scenario: The ohttp config rejects models without e2ee support
    Given a model "plain-model" without e2ee support
    When the ohttp config for "plain-model" is fetched
    Then the ohttp config response is a model not found error

  Scenario: The ohttp config reports an internal error when verification fails
    Given a model "near-model" with e2ee support
    And near verification fails
    When the ohttp config for "near-model" is fetched
    Then the ohttp config response is an internal error

  Scenario: The ohttp config rejects invalid services keys
    When the ohttp config for "near-model" is fetched with an invalid services key
    Then the ohttp config response is an invalid auth key error

  Scenario: The relay proxies the request upstream and streams the response
    Given a model "near-model" with e2ee support
    And near verification returns the ohttp config
    And the upstream relay returns "ohttp-response-bytes"
    When the relay for "near-model" is posted with "ohttp-request-bytes"
    Then the relay response is "ohttp-response-bytes" with media type "message/ohttp-res"
    And the upstream request hit "https://example.near.ai/ohttp" with the bearer token

  Scenario: The relay rejects unknown models
    Given no models are configured
    When the relay for "ghost" is posted with "payload"
    Then the relay response is a model not found error

  Scenario: The relay rejects models without e2ee support
    Given a model "plain-model" without e2ee support
    When the relay for "plain-model" is posted with "payload"
    Then the relay response is a model not found error

  Scenario: The relay reports an internal error when verification fails
    Given a model "near-model" with e2ee support
    And near verification fails
    When the relay for "near-model" is posted with "payload"
    Then the relay response is an internal error

  Scenario: The relay reports an internal error when the near api key is missing
    Given a model "near-model" with e2ee support
    And near verification returns the ohttp config
    And the near api key is not configured
    When the relay for "near-model" is posted with "payload"
    Then the relay response is an internal error

  Scenario: The relay returns a bad gateway when the upstream is unreachable
    Given a model "near-model" with e2ee support
    And near verification returns the ohttp config
    And the upstream relay connection is refused
    When the relay for "near-model" is posted with "payload"
    Then the relay response is a bad gateway error

  Scenario: The relay returns the common params error before contacting upstream
    Given a model "near-model" with e2ee support
    And near verification returns the ohttp config
    And the common params check fails
    When the relay for "near-model" is posted with "payload"
    Then the relay response is the common params error
