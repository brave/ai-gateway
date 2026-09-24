Feature: Serve utility helpers
  Low-level helpers shared across serve endpoints: SSE responses, token
  accounting, model name normalization and malloc trimming.

  Scenario Outline: Litellm model names normalize to friendly model ids
    Given a model catalog where "qwen-14b" maps upstream "Qwen/Qwen3-14B"
    And the bedrock model "titan" uses inference profile env "TITAN_PROFILE"
    And the environment variable "TITAN_PROFILE" is set to "titan-v2"
    When the litellm model "<litellm>" is normalized
    Then the normalized model is <normalized>

    Examples:
      | litellm              | normalized |
      | Qwen/Qwen3-14B       | qwen-14b   |
      | hosted_vllm/Qwen/Qwen3-14B | qwen-14b |
      | titan-v2             | titan      |
      | unknown-model        | none       |
      | <empty>              | none       |

  Scenario: Legacy prompt transcripts are parsed for the last user input
    Given a prompt transcript ending with the response seed and user question "why is the sky blue"
    When the last user input is parsed
    Then the parsed input is "why is the sky blue"

  Scenario: Prompt transcripts without the response seed parse to nothing
    Given a prompt transcript without the response seed
    When the last user input is parsed
    Then the parsed input is nothing

  Scenario: Static content streams as five character chunks
    Given static content "Hello there friend"
    When the static content is streamed with no delay
    Then the chunks reassemble to "Hello there friend"
    And no chunk is longer than 5 characters

  Scenario: Static content streams as SSE completion events
    Given static content "abc def"
    When the static content is streamed as SSE for model "mixtral-8x7b-instruct"
    Then every SSE event except the last is a completion with cumulative content
    And the final SSE event is the DONE sentinel

  Scenario: Free model names are parsed from the catalog
    Given a model catalog where "mixtral" is free, "premium-x" is not and "embedding-x" is a non-llm free model
    When the free model names are parsed
    Then the free models are "mixtral"

  Scenario: Rate limit keys bucket by epoch interval
    Given rate key "1.2.3.4" for model "mixtral" and interval 60
    When the rate limiting key is built twice
    Then the two keys are equal
    And the key contains the model and the rate key

  Scenario: SSE responses wrap raw generators with data framing
    Given a raw generator emitting "one" and "two"
    When the SSE response is consumed
    Then the events are "data: one", "data: two" and the DONE sentinel
    And the generator was closed

  Scenario: SSE responses carry custom headers
    Given a raw generator emitting "one"
    And a custom header "X-Trace" set to "trace-1"
    When the SSE response headers are inspected
    Then the header "X-Trace" is "trace-1"

  Scenario Outline: Token counting handles messages, parts and failures
    Given token counting input <description>
    When the tokens are counted
    Then the token count is greater than zero

    Examples:
      | description                      |
      | plain text messages              |
      | multimodal text parts            |
      | a broken tokenizer               |
      | calculate plain content          |
      | calculate non-text part          |
      | calculate tool calls             |

  Scenario Outline: Token estimates handle nested and typed content
    Given a custom tokenizer counting 3 tokens per encode
    And token estimate content <description>
    When the token count estimate is computed with the custom tokenizer
    Then the estimate is <estimate>

    Examples:
      | description               | estimate        |
      | a plain string            | 3               |
      | a nested list             | 6               |
      | a text dict               | 3               |
      | an image dict             | image_estimate  |
      | an unknown dict           | 0               |
      | an unrecognized object    | 0               |

  Scenario Outline: Malloc trimming resolves to a safe no-op
    Given libc lookup <libc_lookup>
    When malloc trim support is loaded
    Then the trim function is <trim_result>

    Examples:
      | libc_lookup     | trim_result |
      | missing         | none        |
      | load failure    | none        |
      | no trim symbol  | none        |
      | trim symbol     | callable    |

  Scenario: Periodic malloc trim sleeps between trims
    Given a counting trim function
    When the periodic trim loop runs for one tick
    Then the loop trims between ticks and exits cleanly

  Scenario: Periodic malloc trim skips trimming without libc
    Given no trim function is available
    When the periodic trim loop runs for one tick
    Then the loop exits cleanly without trimming

  Scenario: Custom tokenizers skip special tokens
    Given a custom tokenizer counting 3 tokens per encode
    When the token count estimate is computed for "hello world" with the custom tokenizer
    Then the estimate is 3
