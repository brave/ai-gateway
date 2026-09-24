Feature: Conversation compaction
  Old conversation turns are summarized by the Jaguar compaction model.

  Background:
    Given the compaction harness

  Scenario: Full compaction pipeline produces a summary
    Given a conversation with more turns than preserved and a tool call
    When the conversation is compacted with skip checks
    Then a compaction summary is produced
    And the compacted messages start with the summary message
    And compaction metadata reports the indices

  Scenario: Compaction check path triggers when over threshold
    Given a conversation with more turns than preserved turns
    When the conversation is compacted with checks
    Then a compaction summary is produced
    And the compaction trigger was logged

  Scenario: Compaction is skipped when nothing to compact
    Given a short conversation with only preserved turns
    When the conversation is compacted with checks
    Then the original messages are returned with no metadata

  Scenario: Large history is split into multiple chunks
    Given a conversation with more turns than preserved turns
    And a tiny chunk token limit
    When the conversation is compacted with skip checks
    Then a multi-chunk split is logged
    And section labels cover start and end

  Scenario: Failed final chunk falls back to the Haiku model
    Given a conversation with more turns than preserved turns
    And a jaguar backend that always fails
    When the conversation is compacted with skip checks
    Then a compaction summary is produced via the fallback model

  Scenario: A single chunk summary is generated directly
    Given a conversation with more turns than preserved turns
    When a single chunk summary is generated
    Then the summary text is returned

  Scenario: An existing summary is re-summarized into one message
    Given compacted messages containing an old summary
    When the compacted messages are re-summarized
    Then the rebuilt list has a single fresh summary message

  Scenario: Summary extraction returns raw content without context tags
    Given a summary message without a closing context tag
    When the summary text is extracted
    Then the extracted text equals the raw content

  Scenario: Empty compaction selection returns early
    Given a short conversation with only preserved turns
    When messages are selected for compaction
    Then nothing is selected for compaction