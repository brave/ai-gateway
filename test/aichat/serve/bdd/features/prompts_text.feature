Feature: Prompt text content augmenters
  File extracted text and PDF text prompt augmenters rewrite untrusted
  content parts into plain text for downstream models.

  Background:
    Given the prompt text harness

  Scenario: Extracted text augment rewrites a string content part
    Given a user message with a brave-file-extracted-text part "File & <stuff>"
    When file extracted text augment runs
    Then the message content is a single text part
    And the text part starts with "The following is extracted text from a file:"
    And the text part contains "File & <stuff>"

  Scenario: Extracted text augment joins list content text parts
    Given a user message with an extracted-text part holding list content "Alpha" and "Beta"
    When file extracted text augment runs
    Then the message content is a single text part
    And the text part is "The following is extracted text from a file:\nAlpha\nBeta"

  Scenario: Extracted text augment leaves other parts alone
    Given a user message with a text part "keep me" and an assistant reply
    When file extracted text augment runs
    Then the message count is 2
    And the first message content keeps "keep me"

  Scenario: PDF text augment rewrites a pdf text part
    Given a user message with a brave-pdf-text part "Doc & <notes>"
    When pdf text content augment runs
    Then the message content is a single text part
    And the text part starts with "The following is extracted text from PDF pages"
    And the text part contains "Doc & <notes>"

  Scenario: PDF text augment leaves other parts alone
    Given a user message with a text part "keep me"
    When pdf text content augment runs
    Then the message content is a single text part
    And the text part is "keep me"

  Scenario: Leo system prompt keeps an invalid training cutoff raw
    Given a leo messages list without a system message
    When leo system augment runs with training cutoff "not-a-date"
    Then the system message contains "Your training data ends in not-a-date."

  Scenario: Months since helper counts elapsed months
    When months since is computed for 2024-01-15 and 2024-04-20
    Then the result is 3

  Scenario: Abstract prompt augment raises
    When the abstract Prompt augment is called directly
    Then a NotImplementedError is raised

  Scenario: User memory trace text for scalar memory
    Given a user memory content part with scalar memory "food: pineapple"
    When the alignment trace text is computed
    Then the trace starts with "User memory attached by the browser agent"
    And the trace contains "food: pineapple"

  Scenario: User memory trace text for list memory
    Given a user memory content part with list memory "pets: cats, dogs"
    When the alignment trace text is computed
    Then the trace contains "pets:"
    And the trace contains "- cats"
    And the trace contains "- dogs"

  Scenario: User memory trace text without memory attribute
    Given a user memory content part with no memory attribute
    When the alignment trace text is computed
    Then the trace is none