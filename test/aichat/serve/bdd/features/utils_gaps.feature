Feature: Serve utils remaining branches
  Edge branches in aichat/serve/utils.py not exercised elsewhere.

  Background:
    Given the utils gaps harness

  Scenario: Upstream mapping skips non-llm model configs
    Given model settings contain a non-llm config "embed-only"
    When the upstream to model id mapping is refreshed
    Then "embed-only" is not in the mapping keys

  Scenario: Normalize model falls back to last path segment
    Given model settings contain upstream "mixtral-instruct-awq" for "mixtral-free"
    When normalize model name is called with "hosted_vllm/casperhansen/mixtral-instruct-awq"
    Then the normalized model is "mixtral-free"

  Scenario: Malloc trim helper returns a callable on success
    Given find_library returns "c" and CDLL exposes malloc_trim
    When get malloc trim runs
    Then a callable is returned

  Scenario: Periodic malloc trim loop invokes trim
    Given malloc trim is stubbed as a counting function
    When the periodic malloc trim loop runs briefly
    Then the trim function was called at least once

  Scenario: Token estimate returns zero for unknown dict content
    When token estimate is computed for a dict content with type "custom-thing"
    Then the token estimate is 0
