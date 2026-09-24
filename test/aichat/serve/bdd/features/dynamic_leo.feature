Feature: Dynamic Leo signals and embeddings
  Dynamic Leo classifies user intent with keywords, embeddings, and androcles.

  Background:
    Given the dynamic leo harness

  Scenario: Empty phrase embedding input keeps its prefix off
    When the embedding input for a blank phrase is formatted
    Then the formatted input is empty

  Scenario Outline: Cosine similarity guards and values
    When the cosine similarity of <left> and <right> is computed
    Then the similarity result is <result>

    Examples:
      | left          | right         | result |
      | [1, 0]        | [1, 0]        | 1.0    |
      | [1, 0]        | [0, 1]        | 0.0    |
      | []            | [1, 0]        | 0.0    |
      | [1, 0]        | [1]           | 0.0    |
      | [0, 0]        | [1, 0]        | 0.0    |

  Scenario: Zero timeout awaits embeddings directly
    Given the embedding timeout is disabled
    When phrase vectors are fetched for two phrases
    Then the vectors map each phrase to an embedding

  Scenario: Embedding batch size mismatch skips the batch
    When phrase vectors are fetched but the backend returns fewer rows
    Then no vectors are recorded

  Scenario: Embedding backend failure skips the batch
    When phrase vectors are fetched and the backend raises
    Then no vectors are recorded

  Scenario: Rows without usable embeddings are skipped
    When phrase vectors are fetched with an embedding-free row
    Then only the usable phrase has a vector

  Scenario: Phrase vectors are cached per signature
    When phrase vectors are fetched twice for the same phrases
    Then the backend is called only once

  Scenario: Phrase vectors refetch when the signature changes
    When phrase vectors are fetched then refetched for other phrases
    Then the backend is called twice

  Scenario Outline: Query embedding reasons
    Given category phrases with stored vectors
    When embedding reasons are computed for a <query> query
    Then the matched categories are <matched>

    Examples:
      | query    | matched  |
      | matching | tracking |
      | other    | none     |

  Scenario: Empty query embedding inputs return no reasons
    When embedding match reasons are computed for blank text
    Then no embedding reasons are returned

  Scenario: Query embedding failure returns no reasons
    When embedding match reasons are computed and the backend raises
    Then no embedding reasons are returned

  Scenario: Category config cache returns the parsed categories twice
    When the dynamic leo config is parsed twice
    Then the cached categories object is reused

  Scenario: Merged triage probabilities take the elementwise max
    When the probabilities [0.1, 0.9] and [0.5, 0.2] are merged
    Then the merged probabilities are [0.5, 0.9]

  Scenario Outline: Embedding model resolution
    Given the explicit embedding model is <explicit>
    And the configured models are <models>
    When the embedding model id is resolved
    Then the resolved embedding model is <resolved>

    Examples:
      | explicit    | models                                | resolved    |
      | embed-x     | embed-x is embedding and chat is llm  | embed-x     |
      | chat-model  | embed-x is embedding and chat is llm  | embed-x     |
      | embed-x     | chat is llm                           | none        |
      | none        | chat is llm                           | none        |

  Scenario: Dynamic leo is a no-op when disabled
    When dynamic leo runs while disabled
    Then no prefetch is returned

  Scenario: Dynamic leo returns none without user text
    When dynamic leo runs without user text
    Then no prefetch is returned

  Scenario: Phrase prefetch failure degrades to no vectors
    Given dynamic leo with keywords and similar phrases
    When phrase prefetch raises while dynamic leo runs
    Then the prefetch completes with keyword categories only

  Scenario: Label reasons match configured label thresholds
    Given a category with the label Coding at threshold 0.5
    When label reasons are computed for probabilities hitting the label
    Then the matched category is reported with a labels reason

  Scenario: Category config parse failure returns none
    When the dynamic leo config is parsed from a list
    Then the parsed categories are none

  Scenario: Blank user content contributes no turns
    When the last user turns are extracted with a blank message
    Then no turns are returned

  Scenario Outline: Androcles label indexes
    When the androcles label index for <name> is looked up
    Then the label index is <index>

    Examples:
      | name                | index |
      | Math / Calculations | 9     |
      | Travel Planning     | 19    |
      | bogus               | none  |

  Scenario: Keyword reasons merge across turns
    Given dynamic leo with keywords in both turns
    When dynamic leo runs with keyword hits in the last and prior turns
    Then the matched categories merge both keyword reasons
