Feature: Rate limiting
  Free-tier traffic is rate limited per IP through aichat-internal; the
  gateway fails closed when the internal service is unavailable.

  Background:
    Given the gateway app

  Scenario: IP hashing is deterministic and salt-sensitive
    When the ip "192.168.1.100" is hashed with salt "s"
    And the ip "192.168.1.100" is hashed with salt "t"
    Then the hash is stable across calls
    And hashing with salt "t" produces a different hash

  Scenario: A missing ip falls back to localhost
    When the ip "UNKNOWN" is hashed with salt "s"
    And the ip "127.0.0.1" is hashed with salt "s"
    Then the hash equals the hash of "127.0.0.1" with salt "s"

  Scenario: Salts are fetched once per epoch
    Given the internal API is enabled
    When five salt lookups happen in a row
    Then aichat-internal was asked for salts exactly once

  Scenario: Forced rate limit error denies the request
    Given the internal API is enabled
    And model "m" is free with limit 10 per 60 seconds
    When a rate limit is checked with query "force-error-rate-limit-user"
    Then the rate limit verdict denies

  Scenario: Premium host requests bypass the free-model bucket
    Given the internal API is enabled
    And model "m" is free
    When a rate limit is checked from a premium host without a credential
    Then the rate limit verdict allows

  Scenario: Paid models bypass the free rate limiter
    Given the internal API is enabled
    And model "m" is not free
    When a rate limit is checked
    Then the rate limit verdict allows

  Scenario: Content agent requests bypass the free rate limiter
    Given the internal API is enabled
    And model "m" is free
    When a content agent rate limit is checked
    Then the rate limit verdict allows

  Scenario: Free models are checked against aichat-internal
    Given the internal API is enabled
    And model "m" is free with limit 10 per 60 seconds
    And aichat-internal allows the rate limit check
    When a rate limit is checked
    Then the rate limit verdict allows
    And the rate limit payload carried the model config

  Scenario: Free models without rate limit config are denied
    Given the internal API is enabled
    And model "m" is free without limits
    When a rate limit is checked
    Then the rate limit verdict denies

  Scenario: Internal unavailability denies free-model traffic
    Given the internal API is enabled
    And model "m" is free with limit 10 per 60 seconds
    And aichat-internal is unavailable
    When a rate limit is checked
    Then the rate limit verdict denies

  Scenario: Premium credentials are capped by aichat-internal
    Given the internal API is enabled
    And model "m" is not free
    And aichat-internal allows the rate limit check
    When a premium rate limit is checked
    Then the rate limit verdict allows
    And the premium bucket was queried with no local cap

  Scenario: Premium daily cap denials propagate
    Given the internal API is enabled
    And model "m" is not free
    And aichat-internal denies the rate limit check
    When a premium rate limit is checked
    Then the rate limit verdict denies

  Scenario: Content agent limits deny when aichat-internal is down
    Given the internal API is enabled
    And aichat-internal is unavailable
    When a content agent limit is checked
    Then the content agent verdict denies

  Scenario: A content agent limit follows the aichat-internal verdict
    Given the internal API is enabled
    And aichat-internal allows the content agent check
    When a content agent limit is checked
    Then the content agent verdict allows

  Scenario: A content agent limit is denied without an http client
    When a content agent limit is checked without an http client
    Then the content agent verdict denies

  Scenario: Automatic mode peek reports the internal counters
    Given rate limiting is enabled
    And the internal API is enabled
    And the automatic daily limit is 3
    And aichat-internal peek reports count 2 of 3
    And explicit rate limit salts "salt-now" and "salt-prior"
    When the automatic mode daily limit is peeked
    Then the peek result is count 2 exceeded false limit 3
    And the peek carried both hashed ip keys

  Scenario: Automatic mode peek skips the internal check when disabled
    Given rate limiting is disabled
    When the automatic mode daily limit is peeked
    Then the peek result is count 0 exceeded false limit 0

  Scenario: Automatic mode peek fails closed when aichat-internal is down
    Given rate limiting is enabled
    And the internal API is enabled
    And the automatic daily limit is 3
    And aichat-internal is unavailable
    When the automatic mode daily limit is peeked
    Then the peek result is count 0 exceeded true limit 0

  Scenario: Automatic mode increment follows the internal verdict
    Given rate limiting is enabled
    And the internal API is enabled
    And the automatic daily limit is 3
    And aichat-internal denies the automatic mode check
    When the automatic mode count is incremented
    Then the increment verdict denies

  Scenario: Automatic mode increment is skipped when rate limiting is disabled
    Given rate limiting is disabled
    When the automatic mode count is incremented
    Then the increment verdict allows

  Scenario: Automatic mode increment fails closed when aichat-internal is down
    Given rate limiting is enabled
    And the internal API is enabled
    And the automatic daily limit is 3
    And aichat-internal is unavailable
    When the automatic mode count is incremented
    Then the increment verdict denies

  Scenario: Route limits follow the internal verdict
    Given rate limiting is enabled
    And the internal API is enabled
    And aichat-internal denies the route check
    When a route limit is checked for "/v1/embeddings"
    Then the route limit verdict denies

  Scenario: Route limits fail closed when aichat-internal is down
    Given rate limiting is enabled
    And the internal API is enabled
    And aichat-internal is unavailable
    When a route limit is checked for "/v1/embeddings"
    Then the route limit verdict denies

  Scenario: Route limits are skipped when rate limiting is disabled
    Given rate limiting is disabled
    When a route limit is checked for "/v1/embeddings"
    Then the route limit verdict allows

  Scenario: The route decorator raises 429 when limited
    Given rate limiting is enabled
    And the internal API is enabled
    And aichat-internal denies the route check
    When a decorated handler with limit 5 is called
    Then the handler raised 429 "Daily rate limit exceeded for /v1/demo"

  Scenario: The route decorator is a no-op without limit or config key
    Given rate limiting is enabled
    And the internal API is enabled
    When a decorated handler without configuration is called
    Then the handler returned "ok"
