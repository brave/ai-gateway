Feature: Service readiness and version info
  Orchestrators probe the gateway to know when it can serve traffic and
  which build is running.

  Scenario: Ready endpoint reports ready after startup
    Given the gateway has fully started
    When the client requests GET /health/ready
    Then the response status is 200
    And the response body is
        """
        {"status": "ready"}
        """

  Scenario: Ready endpoint reports not ready before startup completes
    Given the gateway has not finished starting
    When the client requests GET /health/ready
    Then the response status is 503
    And the response body is
        """
        {"status": "not ready"}
        """

  Scenario: Info endpoint reports the running version
    Given the gateway reports version "abc1234"
    When the client requests GET /info
    Then the response status is 200
    And the response body is
        """
        {"version": "abc1234"}
        """