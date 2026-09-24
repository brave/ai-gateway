Feature: Share storage API
  End-to-end encrypted conversation shares persisted in S3 with deletion sidecars.

  Background:
    Given the share harness

  Scenario: A share is created and stored with its deletion sidecar
    When a share is created with a valid ciphertext
    Then the share is stored at status 201 with distinct share and deletion ids
    And the sidecar and share objects were written to the bucket

  Scenario: Share creation rejects invalid services keys
    When a share is created with a valid ciphertext and an invalid services key
    Then the response is an invalid auth key error

  Scenario: Share creation rejects a missing ciphertext
    When a share is created without a ciphertext
    Then the response is a bad request error

  Scenario: Share creation rejects ciphertext that is not valid base64
    When a share is created with a non-base64 ciphertext
    Then the response is a bad request error

  Scenario: Share creation rejects oversized ciphertext
    When a share is created with a ciphertext over the size limit
    Then the response is a bad request error

  Scenario: Share creation rejects invalid json bodies
    When a share is created with an invalid json body
    Then the response is a bad request error

  Scenario: Share creation reports a storage failure
    Given the s3 put fails
    When a share is created with a valid ciphertext
    Then the response is an internal server error

  Scenario: A stored share is retrieved
    When the share "share-123" is fetched
    Then the ciphertext "ciphertext-body" is returned

  Scenario: A missing share reports not found
    Given the s3 get reports NoSuchKey
    When the share "share-404" is fetched
    Then the response is a share not found error

  Scenario: A retrieval failure reports an internal error
    Given the s3 get raises an unexpected error
    When the share "share-err" is fetched
    Then the response is an internal server error

  Scenario: Fetched shares carry the viewer cors header
    Given the share viewer origin is configured
    When the share "share-cors" is fetched
    Then the response carries the viewer cors header

  Scenario: A share is deleted through its deletion sidecar
    When the deletion "deletion-1" is posted
    Then the response is 200 and the share and sidecar objects are deleted

  Scenario: Deleting an unknown deletion id reports not found
    Given the sidecar get reports NoSuchKey
    When the deletion "deletion-404" is posted
    Then the response is a share not found error

  Scenario: A sidecar read failure reports an internal error
    Given the sidecar get raises an unexpected error
    When the deletion "deletion-err" is posted
    Then the response is an internal server error

  Scenario: A sidecar client error other than not-found reports an internal error
    Given the sidecar get reports AccessDenied
    When the deletion "deletion-denied" is posted
    Then the response is an internal server error

  Scenario: Deleting with a missing deletion id is rejected
    When the deletion is posted without a deletion id
    Then the response is a bad request error

  Scenario: The options preflight lists the allowed methods
    When the preflight for "share-1" is requested
    Then the preflight lists "GET, OPTIONS" and the share content type
