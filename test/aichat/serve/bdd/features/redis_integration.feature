Feature: Redis integration
  The gateway caches NEAR OHTTP configurations in Redis through a shared
  async connection pool. These scenarios run against a live Redis and skip
  when no Redis is reachable; run them natively by pointing REDIS_HOST at
  a local instance.

  Background:
    Given a reachable redis
    And the cache is empty

  Scenario: The connection pool hands out a shared working pool
    When the pool is created
    And the pool is requested again
    Then the same pool is returned
    And a client from the pool round-trips a value

  Scenario: A discarded pool is rebuilt on the next request
    When the pool is created
    And the pool is discarded
    And the pool is requested again
    Then a new pool is returned
    And a client from the new pool round-trips a value

  Scenario: A verified attestation is cached in redis and reused
    Given a fresh attestation for "near-model" that verifies cleanly
    When the ohttp config is verified and fetched for "near-model"
    Then the built config is returned
    And it is stored in redis with a ttl
    When the ohttp config is verified and fetched again for "near-model"
    Then the same config is returned without a second fetch

  Scenario: A config cached by an earlier run is served without contacting the upstream
    Given the redis cache holds the ohttp config for "near-model"
    When the ohttp config is verified and fetched for "near-model"
    Then the cached config is returned without a fetch

  Scenario: A failed verification caches nothing in redis
    Given a fresh attestation for "near-model" that fails verification
    When the ohttp config is verified and fetched for "near-model"
    Then no ohttp config is returned
    And redis holds no cached config
