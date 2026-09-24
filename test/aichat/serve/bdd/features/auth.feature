Feature: Request authentication verdicts
  The gateway consults aichat-internal for auth verdicts, premium
  credentials and service keys; self-host mode bypasses all of it.

  Background:
    Given the gateway app

  Scenario: Auth verdict is skipped without the internal API
    Given the internal API is disabled
    When an auth verdict is requested for a chat payload
    Then no verdict is produced

  Scenario: Auth verdict forwards only non-message metadata
    Given the internal API is enabled
    And aichat-internal answers auth requests with service_key_allowed true
    When an auth verdict is requested for a chat payload
    Then the verdict is returned
    And the auth request carried the body digest
    And the auth request metadata excluded "messages"

  Scenario: Idempotency key is stable for identical payloads
    Given messages for alice and model "llama-2-13b-chat"
    When the idempotency key is created twice
    Then both idempotency keys match

  Scenario: Idempotency key changes with the payload
    Given messages for alice and model "llama-2-13b-chat"
    When the idempotency key is created twice
    And the idempotency key is created for model "mixtral-8x7b-instruct"
    Then the keys for different payloads differ

  Scenario: Idempotency key is random without messages or model
    When the idempotency key is created without messages or model
    Then the idempotency key starts with "rand_"

  Scenario: Premium host is detected from the host header
    Given the premium host header "leo.brave.com"
    When the premium host is checked
    Then the request is recognized as premium

  Scenario: Premium host is detected from the forwarded host header
    Given the forwarded host header "leo.brave.com"
    When the premium host is checked
    Then the request is recognized as premium

  Scenario: Premium credential always passes in the local environment
    Given the environment is local
    And the internal API is enabled
    When the premium credential is checked without a cookie
    Then the premium credential is accepted

  Scenario: Self-host mode denies premium credentials
    Given the environment is production
    And the internal API is disabled
    When the premium credential is checked with cookie "sku-1"
    Then the premium credential is denied

  Scenario: Missing premium cookie denies the credential
    Given the environment is production
    And the internal API is enabled
    When the premium credential is checked without a cookie
    Then the premium credential is denied

  Scenario: Premium backend unavailability returns service unavailable
    Given the environment is production
    And the internal API is enabled
    And aichat-internal cannot answer sku requests
    When the premium credential is checked with cookie "sku-1"
    Then the request is rejected with 503

  Scenario: Brave key check passes without the internal API
    Given the internal API is disabled
    When the brave key is checked
    Then the brave key is accepted

  Scenario: Brave key check fails closed when no verdict exists
    Given the internal API is enabled
    And no auth verdict exists
    When the brave key is checked
    Then the request is rejected with 500

  Scenario: Brave key check follows the verdict
    Given the internal API is enabled
    And the auth verdict says brave_key_allowed false
    When the brave key is checked
    Then the brave key is rejected

  Scenario: Service key check passes without the internal API
    Given the internal API is disabled
    When the service key is checked
    Then the service key is accepted

  Scenario: Service key check fails closed when no verdict exists
    Given the internal API is enabled
    And no auth verdict exists
    When the service key is checked
    Then the request is rejected with 500

  Scenario: Service key verdict overrides the model selection state
    Given the internal API is enabled
    And the auth verdict carries the service key state
    When the service key is checked
    Then the service key is accepted
    And the request state carries "service_key_id" "sk-1"
    And the request state carries model override true with fallback "fallback-model"
    And the request state carries premium fallback "premium-fallback"
    And the request state carries request_allowed false

  Scenario: Request allowed defaults to true for older verdicts
    Given the internal API is enabled
    And an auth verdict without a request_allowed field
    When the service key is checked
    Then the request state carries request_allowed true