Feature: aichat-internal HTTP client
  Outbound calls to the internal service retry idempotent failures and
  report unavailability instead of raising.

  Background:
    Given aichat-internal base url "http://aichat-internal.test"

  Scenario: Auth verify posts the verdict request to /1/auth
    When auth verify is sent
    Then aichat-internal received a POST to "/1/auth"
    And the auth verify payload was forwarded

  Scenario: Sku verify posts credentials to /1/sku_verification
    When sku verify is sent with credential "sku-1"
    Then aichat-internal received a POST to "/1/sku_verification"
    And the sku verify payload carried the credential and idempotency key

  Scenario: API key verify posts the key to /1/api_key/verify
    When api key verify is sent with key "sk_test_k1_s1_abcdefgh"
    Then aichat-internal received a POST to "/1/api_key/verify"
    And the api key verify payload carried the key

  Scenario: API key verify is retried after a broken response
    Given aichat-internal breaks the first response
    When api key verify is sent with key "sk_test_k1_s1_abcdefgh"
    Then the api key verdict succeeded on the second attempt

  Scenario: API key verify returns nothing when aichat-internal is unavailable
    Given aichat-internal answers 503
    When api key verify is sent with key "sk_test_k1_s1_abcdefgh"
    Then the verdict request returns nothing

  Scenario: Rate limit salts are fetched from /1/rate_limit_salts
    When rate limit salts are requested
    Then aichat-internal received a GET to "/1/rate_limit_salts"

  Scenario: Rate limit checks post the hashed identity to /1/rate_limit
    When a rate limit check is sent for model "llama-2-13b-chat"
    Then aichat-internal received a POST to "/1/rate_limit"
    And the rate limit payload carried the hashed identity

  Scenario: Non-200 responses report unavailability and return nothing
    Given aichat-internal answers 503
    When auth verify is sent
    Then the verdict request returns nothing

  Scenario: Idempotent reads are retried after a broken response
    Given aichat-internal breaks the first response
    When rate limit salts are requested
    Then the salts request succeeded on the second attempt

  Scenario: Idempotent auth verdicts are retried after a broken response
    Given aichat-internal breaks the first response
    When auth verify is sent
    Then the auth verdict succeeded on the second attempt

  Scenario: Non-idempotent writes are not retried on read errors
    Given aichat-internal breaks the first response
    When a rate limit check is sent for model "llama-2-13b-chat"
    Then the rate limit check returns nothing after one attempt

  Scenario: Non-idempotent writes are not retried on protocol errors
    Given aichat-internal aborts the first response with a protocol error
    When a rate limit check is sent for model "llama-2-13b-chat"
    Then the rate limit check returns nothing after one attempt

  Scenario: Idempotent reads are retried after a protocol error
    Given aichat-internal aborts the first response with a protocol error
    When rate limit salts are requested
    Then the salts request succeeded on the second attempt

  Scenario: Persistent transport failures give up after two attempts
    Given aichat-internal breaks every response
    When rate limit salts are requested
    Then the salts request returns nothing after two attempts
