Feature: Truncation continuation
  Detects accidental early stops (finish_reason=stop with unfinished text or
  degenerate repetition) and recovers via retry or continuation, in both
  streaming and non-streaming pipelines.

  Scenario Outline: Longest repeated character run is measured
    Given the text "<text>"
    When the longest character run is measured
    Then the longest run length is <run>

    Examples:
      | text       | run |
      | aaabbc     | 3   |
      | aaaa       | 4   |
      | x          | 1   |
      | a b  c  d  | 1   |

  Scenario Outline: Degenerate repetition is detected
    Given the text "<text>"
    When degenerate repetition is checked with min run <min_run>
    Then the repetition verdict is <verdict>

    Examples:
      | text   | min_run | verdict |
      | aaaaaa | 6       | True    |
      | aaaaa  | 6       | False   |
      | hello  | 6       | False   |

  Scenario Outline: Accidental early stops are recognized
    Given the text "<text>"
    And a finish reason "<finish_reason>"
    When the truncated reply check runs
    Then the accidental stop verdict is <verdict>

    Examples:
      | text                                          | finish_reason | verdict |
      | This reply ends abruptly without punctuation  | stop          | True    |
      | All done, thanks!                             | stop          | False   |
      | short                                         | stop          | False   |
      | This reply ends abruptly without punctuation  | length        | False   |

  Scenario Outline: Truncation actions are resolved
    Given the text "<text>"
    And a finish reason "<finish_reason>"
    When the truncation action is resolved
    Then the action is <action>

    Examples:
      | text                                         | finish_reason | action   |
      | This reply ends abruptly without punctuation | stop          | continue |
      | aaaaaaaaaaaaaaa                              | stop          | retry    |
      | A complete sentence.                         | stop          | none     |
      | Whatever comes next                          | length        | none     |

  Scenario Outline: Continuation fixup requires the model flag
    Given a model config with truncation_continuation <flag>
    When the truncation fixup gate is evaluated
    Then the fixup is enabled <enabled>

    Examples:
      | flag  | enabled |
      | True  | True    |
      | False | False   |

  Scenario: Continuation messages append assistant partial and continuation prompt
    Given conversation messages "user hello"
    And a partial assistant reply "I was in the middle of"
    When the continuation messages are built
    Then the last message asks to continue exactly where it stopped
    And the assistant partial is preserved in the second-to-last message

  Scenario Outline: Response helpers inspect dicts and objects alike
    Given a completion response <description>
    When the response helpers are exercised
    Then the tool call detection is <tool_calls> and content is "<content>" and finish reason is "<finish_reason>"
    And the response content can be overwritten with "replaced"

    Examples:
      | description        | tool_calls | content | finish_reason |
      | a plain dict       | False       | partial | stop          |
      | an object response | False       | hi      | tool_calls    |
      | a dict with tools  | True        | hi      | tool_calls    |

  Scenario Outline: Non-streaming recovery passes through ineligible responses
    Given truncation continuation is enabled
    And a completion response <description>
    When non-streaming truncation recovery runs
    Then the response is returned unchanged

    Examples:
      | description                       |
      | an error dict                     |
      | a response carrying tool calls    |
      | a response without content        |
      | a complete non-truncated response |

  Scenario: Disabled truncation recovery passes the response through
    Given truncation continuation is disabled
    And a completion response a plain dict
    When non-streaming truncation recovery runs
    Then the response is returned unchanged

  Scenario: Excessive recovery depth passes the response through
    Given truncation continuation is enabled
    And the truncation depth budget is exhausted
    And a completion response a plain dict
    When non-streaming truncation recovery runs
    Then the response is returned unchanged

  Scenario: Degenerate non-streaming replies trigger a backend retry
    Given truncation continuation is enabled
    And a completion response that degenerately repeats
    And a fake recovery backend retrying "fresh answer"
    When non-streaming truncation recovery runs
    Then the backend was asked to converse once
    And the returned content is "fresh answer"

  Scenario: Cut-off non-streaming replies are continued with a follow-up call
    Given truncation continuation is enabled
    And a completion response that is cut off mid-sentence
    And a fake recovery backend continuing "and that is all"
    When non-streaming truncation recovery runs
    Then the backend was asked to converse once
    And the merged content contains the partial text and the continuation

  Scenario: Failed continuation calls keep the original response
    Given truncation continuation is enabled
    And a completion response that is cut off mid-sentence
    And a fake recovery backend failing its follow-up call
    When non-streaming truncation recovery runs
    Then the response is returned unchanged

  Scenario Outline: Streaming recovery is planned from stream state
    Given truncation continuation is <flag>
    And assistant text "<text>" with finish reason "<finish_reason>" and tool calls <tool_calls>
    When the stream truncation recovery is planned
    Then the planned action is <action>

    Examples:
      | flag  | text                                         | finish_reason | tool_calls | action   |
      | True  | This reply ends abruptly without punctuation | stop          | False      | continue |
      | True  | aaaaaaaaaaaaaaa                              | stop          | False      | retry    |
      | True  | partial                                      | stop          | True       | none     |
      | False | This reply ends abruptly without punctuation | stop          | False      | none     |
      | True  | <empty>                                      | stop          | False      | none     |

  Scenario: Recorded stream recovery emits the degenerate skip metric
    Given the recorded recovery action "retry" for model "test-model"
    Then the skipped-degenerate-stream metric was incremented

  Scenario: Recorded stream recovery emits the continue metric
    Given the recorded recovery action "continue" for model "test-model"
    Then the continue metric was incremented

  Scenario: Streaming continuation calls the backend with the near API key
    Given truncation continuation is enabled
    And a streaming request for model "near-test"
    And a fake continuation backend
    When the stream continuation runs
    Then the continuation backend received the NEAR API key
    And the continuation depth advanced to 1
    And the follow-up events are forwarded