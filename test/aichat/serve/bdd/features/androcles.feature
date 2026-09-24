Feature: Androcles task classification
  Androcles Triton inference classifies a user request into vision,
  language or coding tasks from output probabilities.

  Scenario: List inputs are flattened to their text parts
    Given an androcles input list with text parts "describe this"
    When the androcles inference runs with a router returning 5 probabilities
    Then the request payload carried the joined text "describe this"

  Scenario: Empty inputs are not classified
    Given an empty androcles input
    When the androcles inference runs
    Then no probabilities are returned

  Scenario: Non-string inputs are not classified
    Given an androcles input of 42
    When the androcles inference runs
    Then no probabilities are returned

  Scenario: Successful inference parses the router output
    Given a router that returns output data "0.1,0.2,0.3,0.95,0.1,0.1,0.1,0.1,0.1,0.1,0.1"
    When the androcles inference runs without a timeout
    Then the probabilities are "0.1,0.2,0.3,0.95,0.1,0.1,0.1,0.1,0.1,0.1,0.1"

  Scenario Outline: Task types are derived from dominant probabilities
    Given probabilities <probabilities>
    When the task type is derived
    Then the derived task is <task>

    Examples:
      | probabilities                                                        | task    |
      | 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.95,0.1,0.1                         | vision  |
      | 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.95                         | language |
      | 0.1,0.1,0.1,0.95,0.1,0.1,0.1,0.1,0.1,0.1,0.1                         | coding  |
      | 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1                          | none    |
      | 0.1,0.1,0.1                                                          | none    |

  Scenario: Missing probabilities derive no task
    Given no probabilities at all
    When the task type is derived
    Then the derived task is none

  Scenario Outline: Inference failures degrade to no probabilities
    Given a router that <router>
    When the androcles inference runs with a timeout
    Then no probabilities are returned

    Examples:
      | router          |
      | times out       |
      | raises an error |