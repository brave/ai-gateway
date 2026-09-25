Feature: PDF text-extraction service
  Sandbox-backed PDF analysis and message preprocessing for file parts.

  Background:
    Given the pdf service harness

  Scenario Outline: PDF data URLs are detected
    When the file data <data_desc> is inspected
    Then the pdf detection result is <expected>

    Examples:
      | data_desc                  | expected |
      | is a pdf data url          | True     |
      | is a png data url          | False    |
      | is plain text              | False    |

  Scenario: PDF bytes survive a decode/encode round trip
    When pdf bytes are encoded to a data url and decoded back
    Then the decoded bytes match the original

  Scenario Outline: Extraction token budgets are computed safely
    When the conversation token limit is <limit>
    Then the max extraction tokens are <tokens>

    Examples:
      | limit    | tokens  |
      | 6400     | 4800    |
      | none     | 96000   |
      | 1000000  | 96000   |

  Scenario: PDF analysis forwards the sandbox payload
    When pdf bytes are analyzed with budget 4800
    Then the sandbox receives op "pdf_analyze" with encoding "o200k_base" max_allowed_pages 85

  Scenario: PDF analysis re-raises hard sandbox worker errors
    When the sandbox worker dies during pdf analysis
    Then the analysis raises SandboxWorkerError

  Scenario Outline: Sync result assembly handles sandbox analyses
    Given a pdf analysis <analysis_desc>
    When the analysis result is assembled
    Then the assembled result is <expected>

    Examples:
      | analysis_desc                     | expected                                    |
      | encrypted passthrough             | the original part                           |
      | all_text with content             | a brave-pdf-text part with the text          |
      | all_text without content          | the original part                           |
      | under budget without native limit | the original part                           |
      | native limit with overflow text   | a truncated file part plus a brave-pdf-text part |
      | native limit without overflow     | a truncated file part only                  |

  Scenario Outline: Content parts pipeline guards each part type
    When content parts <parts_desc> are processed
    Then the processed result is <expected>

    Examples:
      | parts_desc                 | expected                        |
      | contain only text parts    | the parts pass through unchanged |
      | contain a non-pdf file     | the parts pass through unchanged |
      | exceed the size limit      | the pdf part passes through unchanged |
      | hit a sandbox op error     | the pdf part passes through unchanged |
      | hit an unexpected error    | the pdf part passes through unchanged |

  Scenario: Content parts pipeline re-raises hard sandbox worker failures
    When the sandbox worker dies during content part processing
    Then processing raises SandboxWorkerError

  Scenario: Content parts pipeline processes pdf parts through the sandbox
    When a pdf content part is processed with a successful analysis
    Then the sandbox op was called once and the parts contain the brave-pdf-text

  Scenario Outline: Message-level pdf preprocessing touches only pdf-bearing user messages
    When messages <messages_desc> are preprocessed
    Then the sandbox processing <outcome>

    Examples:
      | messages_desc                        | outcome                          |
      | are assistant-only                   | is skipped entirely              |
      | have a user message with plain text  | is skipped entirely              |
      | have a user message with a pdf file  | runs on the pdf parts            |
