Feature: OpenAI adapter security and stream helpers
  Alignment checking, trace building and prompt injection scanning for the
  OpenAI-compatible pipeline.

  Scenario Outline: Chunk emitters inject compaction metadata once
    Given a chunk emitter with compaction metadata <with_metadata>
    And a first chunk payload "<first_chunk>"
    When the chunk is emitted
    Then the emitted chunk carries compaction metadata <metadata_expected>
    And a second emission carries no metadata <second_expected>

    Examples:
      | with_metadata | first_chunk        | metadata_expected | second_expected |
      | yes           | data: {"id":1}     | yes               | yes             |
      | no            | data: {"id":1}     | no                | no              |
      | yes           | not-json-chunk     | no                | yes             |

  Scenario Outline: Long responses are truncated for the trace
    Given a trace response of length <length>
    When the response is truncated for the trace
    Then the truncated response starts with "..." <ellipsis> and keeps the tail <tail_kept>

    Examples:
      | length | ellipsis | tail_kept |
      | 2000   | yes      | yes       |
      | 20     | no       | yes       |

  Scenario Outline: Conversation traces describe every message role
    Given an openai trace with a <description>
    When the conversation trace is built
    Then the trace contains "<fragment>"

    Examples:
      | description                     | fragment                                        |
      | tool result                     | TOOL: tool output is omitted                    |
      | plain user message              | USER: hello there                               |
      | image attachment                | USER: Image uploaded by the user                |
      | assistant reply                 | ASSISTANT: partial answer                       |
      | assistant tool call             | SELECTED ACTION:                                |
      | system message                  | SYSTEM: be nice                                 |
      | final assistant response        | ASSISTANT: streamed tail                        |

  Scenario Outline: The original user message is extracted for scanning
    Given an openai message list with user content <description>
    When the original user message is extracted
    Then the extracted message is <extracted>

    Examples:
      | description            | extracted               |
      | a plain string         | hello there             |
      | text parts             | USER: hello there       |
      | attachment parts only  | User message is not found |
      | no user message        | User message is not found |

  Scenario Outline: Alignment checks consult the scanner and metrics
    Given the alignment scanner is <scanner_state>
    And a conversation with untrusted tool results
    When an alignment check runs for tool "web_search" with arguments "{}"
    Then the alignment verdict allowed is <allowed>

    Examples:
      | scanner_state | allowed |
      | available     | False   |
      | unavailable   | True    |

  Scenario Outline: Buffered tool call arguments are reassembled
    Given buffered chunks with tool arguments <description>
    When the tool arguments are extracted from the buffer
    Then the extracted arguments are "<arguments>"

    Examples:
      | description       | arguments     |
      | a single chunk    | {"q": "x"}    |
      | split chunks      | {"q": 1}      |
      | an empty buffer   | nothing       |

  Scenario Outline: Non-streaming alignment checks decorate tool calls
    Given a non-streaming response with <description>
    And the alignment scanner is available
    When the non-streaming alignment check runs
    Then the response tool calls carry alignment metadata <decorated>

    Examples:
      | description                     | decorated |
      | tool calls and untrusted content | yes      |
      | no choices                      | no        |
      | no tool calls                   | no        |
      | trusted content only            | no        |

  Scenario: Bypassed tools skip alignment checks in non-streaming responses
    Given a non-streaming response with tool calls and untrusted content
    And the tool is in the alignment bypass list
    And the alignment scanner is available
    When the non-streaming alignment check runs
    Then the response tool calls carry alignment metadata no

  Scenario Outline: Recent tool results decide injection scanning
    Given a message history with <description>
    When recent tool results are checked
    Then untrusted results are present <present>

    Examples:
      | description                            | present |
      | no messages                            | False   |
      | a plain user reply only                | False   |
      | bypassed tool results only             | False   |
      | unbypassed tool results                | True    |
      | tool results without an assistant call | True    |

  Scenario Outline: Latest tool results are extracted for scanning
    Given tool result messages with content <description>
    When the latest tool results are extracted
    Then the extracted content is <content>

    Examples:
      | description            | content              |
      | two string results     | first-SEP-second     |
      | a text part result     | part text      |
      | no tool results        | none           |

  Scenario Outline: Prompt injection scans start only when eligible
    Given prompt injection scanning is <enabled>
    And the request capability is <capability>
    And a message history with unbypassed tool results
    When the injection scan is launched
    Then the scan task started <started>

    Examples:
      | enabled | capability    | started |
      | True    | content_agent | yes     |
      | False   | content_agent | no      |
      | True    | chat          | no      |

  Scenario Outline: The injection scanner grades tool results
    Given the injection scanner is <scanner_state>
    And a message history with <history>
    When the injection scan task executes
    Then the scan probability is <probability>

    Examples:
      | scanner_state | history                  | probability |
      | available     | unbypassed tool results  | 4           |
      | unavailable   | unbypassed tool results  | 1           |
      | available     | no tool results          | 1           |

  Scenario Outline: Non-streaming responses carry the injection verdict
    Given a non-streaming response payload <payload>
    And an injection scan task that <task_state>
    When the non-streaming injection scan runs
    Then the response carries prompt_injection_scan <carried>

    Examples:
      | payload            | task_state      | carried |
      | a dict             | completes       | yes     |
      | a dict             | raises          | no      |
      | a dict             | absent          | no      |
      | an object response | completes       | yes     |

  Scenario Outline: Alignment buffering intercepts tool call streams
    Given alignment checking is enabled
    And a message history with unbypassed tool results
    And a stream chunk with tool call "web_search"
    And the buffer state is <state>
    When the alignment buffering handles the chunk
    Then buffering decision continue is <continue> with chunks <chunks>

    Examples:
      | state         | continue | chunks |
      | fresh         | True     | 0      |
      | finishing     | True     | 2      |
      | flushing      | False    | 2      |
      | passthrough   | False    | 0      |

  Scenario: Buffered tool calls are scanned and decorated
    Given alignment checking is enabled
    And a message history with unbypassed tool results
    And buffered tool call chunks for "web_search" with alignment verdict allowed False
    When the buffered tool call chunks are processed
    Then the streamed chunk carries the alignment verdict
    And the tracked tool call carries the alignment verdict