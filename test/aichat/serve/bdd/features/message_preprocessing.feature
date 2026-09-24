Feature: Message preprocessing for chat completions
  Before messages reach the model they are preprocessed per model
  capabilities: attachment limits are enforced, duplicate attachments
  are deduplicated and unsupported parts are stripped or noted.

  Scenario Outline: Oldest images are dropped beyond the image limit
    Given a model that supports images with image limit <limit>
    And messages carrying images "<layout>"
    When the messages are preprocessed
    Then the surviving image layout is "<layout>"

    Examples:
      | limit | layout     | layout        |
      | 2     | over_limit | under_limit   |
      | 3     | over_limit | over_limit    |

  Scenario: Bedrock document limits drop the oldest documents
    Given a bedrock model with document limit 1
    And messages carrying two documents and one image
    When the messages are preprocessed
    Then only the newest document survives

  Scenario: Non-bedrock models have no document limit
    Given a non-bedrock model with document limit none
    And messages carrying two documents and one image
    When the messages are preprocessed
    Then both documents survive

  Scenario: Bedrock duplicate document content keeps only the newest
    Given a bedrock model
    And two attachments with identical content and different names
    When the messages are preprocessed
    Then only the last attachment with that content survives

  Scenario: Duplicate attachment names are renamed with occurrence and hash
    Given any model
    And two attachments named "report.pdf" with different content
    When the messages are preprocessed
    Then the first attachment keeps its name
    And the second attachment is renamed like "report.pdf-1-<hash>"

  Scenario: Tool messages are stripped for models without tool support
    Given a model without tool support
    And a conversation with tool result and assistant tool calls
    When the messages are preprocessed
    Then tool results are removed and assistant tool calls are cleared
    And assistant-only messages without content are removed

  Scenario: File parts are replaced by text notes for models without file support
    Given a model without file support
    And a message with an inline file
    When the messages are preprocessed
    Then the file parts are replaced with explanatory text notes

  Scenario: Image parts are replaced by text notes for models without image support
    Given a model without image support
    And a message with an image
    When the messages are preprocessed
    Then the image is replaced with an explanatory text note