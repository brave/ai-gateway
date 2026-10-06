Feature: vLLM deployment pool routing
  LitellmBackend classifies pooled models and sets LiteLLM router tags when
  DEPLOYMENT_POOLS_ENABLED is on. With the flag off, traffic uses the default
  address but shadow metrics still record the would-be pool.

  Scenario: Shadow mode does not tag router requests
    Given model "qwen-14b-instruct" has deployment pools configured
    And deployment pool routing is disabled
    And a short user message for pool routing
    When the litellm backend runs chat completion
    Then the router completion has no pool tags

  Scenario: Active mode tags short requests
    Given model "qwen-14b-instruct" has deployment pools configured
    And deployment pool routing is enabled
    And a short user message for pool routing
    When the litellm backend runs chat completion
    Then the router completion includes pool tag "pool:short"

  Scenario: Active mode tags long requests
    Given model "qwen-14b-instruct" has deployment pools configured
    And deployment pool routing is enabled
    And a long user message for pool routing
    When the litellm backend runs chat completion
    Then the router completion includes pool tag "pool:long"

  Scenario: Shadow mode increments deployment pool metrics
    Given model "qwen-14b-instruct" has deployment pools configured
    And deployment pool routing is disabled
    And deployment pool shadow metrics baseline is captured
    And a short user message for pool routing
    When the litellm backend runs chat completion
    Then deployment_pool_route_total increases for pool "short_text" mode "shadow"
