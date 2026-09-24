Feature: Model selection for chat requests
  Automatic requests are triaged to premium and non-premium models via
  configuration, request shape and Androcles classification. Explicit
  models pass through unless unknown or ensembles.

  Scenario: Content agent capability with a supported model keeps the model
    Given model "leo-think" with content agent support
    When a model is selected with content agent capability
    Then the selected model is "leo-think"

  Scenario: Content agent capability without support uses the default agent model
    Given model "basic" without content agent support
    And a content agent default model "agent-model"
    When a model is selected for "automatic" with content agent capability
    Then the selected model is "agent-model"

  Scenario: Explicit free model passes through unchanged
    Given model "leo-hw" exists
    When a model is selected for "leo-hw"
    Then the selected model is "leo-hw"

  Scenario: Unknown model falls back to automatic triage
    Given no models configured beyond triaging defaults
    And triaging default premium "premium-a" and non-premium "free-a"
    When a model is selected for "does-not-exist"
    Then the selected model is "free-a"

  Scenario: Explicit ensemble model resolves a weighted pick
    Given model "ensemble-x" with weighted members "a-model:90, b-model:10"
    And the random source favors the first option
    When a model is selected for "ensemble-x"
    Then the selected model is "a-model"

  Scenario: Weighted ensemble favors the heavier member
    Given model "ensemble-x" with weighted members "a-model:10, b-model:90"
    And the random source favors the second option
    When a model is selected for "ensemble-x"
    Then the selected model is "b-model"
    And the picker weighed the options as 10,90

  Scenario: Automatic request for premium users selects the premium model
    Given triaging default premium "premium-a" and non-premium "free-a"
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-a"

  Scenario: Automatic request for free users selects the non-premium model
    Given triaging default premium "premium-a" and non-premium "free-a"
    When a model is selected for "automatic" without premium access
    Then the selected model is "free-a"

  Scenario: Automatic free requests under the daily allowance are treated premium
    Given triaging default premium "premium-a" and non-premium "free-a"
    And the automatic daily count check allows the request
    When a model is selected for "automatic" without premium access and rate key "1.2.3.4"
    Then the selected model is "premium-a"
    And the daily count was incremented

  Scenario: Automatic free requests over the daily allowance stay non-premium
    Given triaging default premium "premium-a" and non-premium "free-a"
    And the automatic daily count check denies the request
    When a model is selected for "automatic" without premium access and rate key "1.2.3.4"
    Then the selected model is "free-a"
    And the daily count was incremented

  Scenario: Premium triaged model from an ensemble resolves a weighted pick
    Given triaging default premium "ensemble-x" and non-premium "free-a"
    And model "ensemble-x" with weighted members "a-model:90, b-model:10"
    And the random source favors the first option
    When a model is selected for "automatic" with premium access
    Then the selected model is "a-model"

  Scenario: Summary requests route to the summary triage category
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging summary premium "premium-s" and non-premium "free-s"
    And brave summary generation is enabled
    And a user message carrying a "brave-request-summary" part
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-s"

  Scenario: Summary requests are skipped when brave summary is disabled
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging summary premium "premium-s" and non-premium "free-s"
    And brave summary generation is disabled
    And a user message carrying a "brave-request-summary" part
    When a model is selected for "automatic"
    Then the selected model is "free-a"

  Scenario: Media content routes to the media triage category
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging vision premium "premium-v" and non-premium "free-v"
    When a model is selected for "automatic" with media type "vision"
    Then the selected model is "free-v"

  Scenario: Tab focus parts route to the tab focus triage category
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging tab_focus premium "premium-t" and non-premium "free-t"
    And a user message carrying a "brave-filter-tabs" part
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-t"

  Scenario: Long conversations route to the long context triage category
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging long_context premium "premium-l" and non-premium "free-l"
    And a long conversation over the token threshold
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-l"

  Scenario: Androcles prefetch task routes to its triage category
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging coding premium "premium-c" and non-premium "free-c"
    And a prefetch with task type "coding"
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-c"

  Scenario: Androcles prefetch without a task type routes to default
    Given triaging default premium "premium-a" and non-premium "free-a"
    And a prefetch without a task type
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-a"

  Scenario Outline: Live androcles classification routes triaged requests
    Given triaging default premium "premium-a" and non-premium "free-a"
    And triaging <task> premium "premium-c" and non-premium "free-c"
    And androcles classifies the request as "<task>"
    And a last user message "<message>"
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-c"
    And androcles classified the last user message

    Examples:
      | task     | message                             |
      | coding   | write me a sorting algorithm        |
      | vision   | describe this photo                 |
      | language | translate this paragraph for me     |

  Scenario: Unrecognized androcles classification routes to default
    Given triaging default premium "premium-a" and non-premium "free-a"
    And androcles classification returns nothing
    And a last user message "hello there"
    When a model is selected for "automatic" with premium access
    Then the selected model is "premium-a"
    And androcles classified the last user message

  Scenario: Missing triage category falls back to the first configured model
    Given model "fallback-first" exists
    And no triaging configuration
    When the triage models for "vision" are resolved
    Then the resolved mapping is premium "fallback-first" and non-premium "fallback-first"

  Scenario: Missing triage configuration without models raises
    Given no models and no triaging configuration
    When the triage models for "vision" are resolved
    Then a triage configuration error is raised

  Scenario Outline: The last user message content type detection
    Given a message history with <description>
    When the last user message is checked for type "<part_type>"
    Then the detection result is <result>

    Examples:
      | description                       | part_type             | result |
      | a tab focus part                  | brave-filter-tabs     | True   |
      | a summary part                    | brave-request-summary | True   |
      | plain string content              | brave-filter-tabs     | False  |
      | an assistant message after user   | brave-filter-tabs     | True   |
      | no messages                       | brave-filter-tabs     | False  |
      | a user message with unknown parts | brave-filter-tabs     | False  |
      | missing content parts             | brave-filter-tabs     | False  |
