Feature: NEAR TEE attestation
  Signature verification, image digest extraction, S3 pre-verification checks
  and the cached ohttp config lookup.

  Background:
    Given the near harness

  Scenario Outline: OHTTP configs are built from verified attestations
    Given a model "near-model" addressed at "https://addr/v1/"
    And a key config hex of <key_hex>
    When the ohttp config is built from the attestation
    Then the built config result is <built>

    Examples:
      | key_hex | built |
      | deadbeef | decodable |
      | zzzz | none |

  Scenario: The ohttp signature verifies against the signing key
    Given a genuine ed25519 ohttp attestation
    When the ohttp signature is verified
    Then the signature is valid

  Scenario: A tampered ohttp signature fails verification
    Given a genuine ed25519 ohttp attestation
    When the ohttp signature is verified after tampering
    Then the signature is invalid

  Scenario: The latest compose manager image is extracted
    Given compose actions ending with "compose_manager_started"
    When the last compose manager image is extracted
    Then the bare digest "abc123" is returned

  Scenario: A compose manager image without a sha256 digest is rejected
    Given compose actions with a tagged but undigested image
    When the last compose manager image is extracted
    Then no compose manager digest is returned

  Scenario: Target digests are extracted from the gateway compose file
    Given a gateway compose file with all four target images digested
    When the target digests are extracted
    Then all four target digests are returned

  Scenario: Target digest extraction fails on digest count mismatch
    Given a gateway compose file with only two target images
    When the target digests are extracted
    Then no target digests are returned

  Scenario: Images without a registry prefix are recognized
    Given a gateway compose file without a registry prefix
    When the target digests are extracted
    Then all four target digests are returned

  Scenario Outline: S3 head objects report existence and 404s
    Given an s3 head client that reports <presence>
    When a head object is performed for "signing-keys" "key-1"
    Then the head result is <result>

    Examples:
      | presence   | result |
      | the object | True   |
      | nothing    | False  |

  Scenario Outline: Attestations are verified against the pre-verified registries
    Given a real ohttp attestation with signature <sig_state>
    And signing keys <keys_state>
    And a verified compose manager image <compose_state>
    When the attestation is verified
    Then verification is <verdict>

    Examples:
      | sig_state | keys_state | compose_state | verdict |
      | valid     | registered | verified      | successful |
      | valid     | registered | missing       | failed |
      | tampered  | registered | verified      | failed |
      | valid     | unregistered | verified    | failed |

  Scenario: A non-404 s3 head error is re-raised
    Given an s3 head client that raises AccessDenied
    When a head object is performed for "pre-verified" "digest-x"
    Then the head raises ClientError

  Scenario: A cached ohttp config short-circuits verification
    Given the redis cache holds the ohttp config for "near-model"
    And a model "near-model" addressed at "https://addr/v1/"
    When the ohttp config is verified and fetched for "near-model"
    Then the cached config is returned without contacting the upstream

  Scenario: A fresh attestation is verified, built and cached
    Given a fresh attestation for "near-model" that verifies cleanly
    And a model "near-model" addressed at "https://addr/v1/"
    When the ohttp config is verified and fetched for "near-model"
    Then the built config is returned and stored in redis

  Scenario: A failed verification returns no ohttp config
    Given a fresh attestation for "near-model" that fails verification
    And a model "near-model" addressed at "https://addr/v1/"
    When the ohttp config is verified and fetched for "near-model"
    Then no ohttp config is returned
