Feature: API-key chat completion model selection
  The verify endpoint decides which model serves an API-key chat completion.
  The response always reports the model the caller requested.

  Background:
    Given the internal API is enabled

  Scenario: The model returned by verify is served
    Given verify allows the key and selects model "served-model"
    When an api-key chat completion is requested for model "requested-model"
    Then verify was asked about model "requested-model"
    And the completion was served by model "served-model"
    And the response status is 200
    And the response reports model "requested-model"
    And usage was settled for model "served-model"

  Scenario: The requested model is served when verify selects none
    Given verify allows the key without selecting a model
    When an api-key chat completion is requested for model "requested-model"
    Then the completion was served by model "requested-model"
    And the response reports model "requested-model"
    And usage was settled for model "requested-model"

  Scenario: A model denied by verify is rejected before inference
    Given verify denies the key with reason "model_not_allowed"
    When an api-key chat completion is requested for model "requested-model"
    Then the response status is 400
    And the error code is "model_not_found"
    And no completion was served
    And no usage was settled
