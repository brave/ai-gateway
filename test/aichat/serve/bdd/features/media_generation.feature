Feature: Media service generators
  Triton/HTTP passthrough helpers backing the media endpoints.

  Background:
    Given the media generator harness

  # --- STT generator ---

  Scenario: STT decode asks the sandbox pool for a mono 16k float32 buffer
    When a WAV upload is decoded through the sandbox returning pcm for 16000 samples
    Then the decoded audio has 16000 samples and duration 250.0 milliseconds

  Scenario: STT decode rejects inconsistent sandbox payloads
    When the sandbox returns a sample count that does not match the buffer
    Then decoding raises ValueError "sandbox decode returned inconsistent sample count"

  Scenario Outline: Triton responses convert to dicts or raise
    When a triton response of kind "<kind>" is normalized
    Then the conversion <outcome>

    Examples:
      | kind        | outcome                          |
      | dict        | returns the dict unchanged       |
      | json_object | returns the json document        |
      | garbage     | raises ValueError                |

  Scenario Outline: Transcription extraction handles parakeet outputs
    Given a triton transcription payload <payload>
    When the transcription is extracted
    Then the extracted text is "<text>"

    Examples:
      | payload            | text        |
      | named_output_str   | hello there |
      | first_output_str   | fallback    |
      | bytes_list         | bytes text  |

  Scenario Outline: Transcription extraction rejects malformed payloads
    Given a malformed transcription payload <defect>
    When the transcription is extracted
    Then extraction raises ValueError

    Examples:
      | defect          |
      | no_outputs      |
      | missing_data    |
      | unexpected_type |

  Scenario: Transcribe audio sends an FP32 audio_signal tensor to Triton
    When audio samples are transcribed through the router
    Then the triton request has an audio_signal tensor of 4 float32 samples
    And the transcript "hi there" is returned

  Scenario: Transcribe upload computes duration from the target sample rate
    When an upload with 32000 samples is transcribed
    Then the transcript is "hi" and duration is 2000.0 milliseconds

  Scenario: Transcribe upload rejects empty transcripts
    When the triton backend returns blank text
    Then transcription raises RuntimeError "Transcription returned empty text"

  # --- TTS generator ---

  Scenario: Speech generation passes arguments to the router speech API
    When speech is generated with voice "af_heart"
    Then the router aspeech call receives the same arguments

  Scenario Outline: Speech streaming proxies the upstream model address
    Given a tts model config with address "<address>" and api key "<api_key>"
    When speech streaming is requested
    Then the upstream request posts to "<url>" with Authorization "Bearer <auth>"
    And the stream yields the upstream audio chunks

    Examples:
      | address            | api_key | url                              | auth  |
      | http://tts:9000/   | sk-1    | http://tts:9000/audio/speech     | sk-1  |
      | http://tts:9000    | none    | http://tts:9000/audio/speech     | empty |

  Scenario: Speech streaming fails loudly on non-200 upstream responses
    Given a tts model config with address "http://tts:9000/" and api key "sk-1"
    When the upstream speech endpoint returns status 500
    Then streaming raises RuntimeError "Upstream speech failed (500)"

  # --- Embeddings generator ---

  Scenario Outline: OpenAI embedding requests transform to Triton format
    When an embeddings request carries input <input_desc>
    Then the triton payload has shape [<count>, 1] and data rows <rows>

    Examples:
      | input_desc        | count | rows        |
      | a single string   | 1     | [['hello']] |
      | a list of strings | 2     | [['a','b']] |

  Scenario Outline: Triton embedding responses transform back to OpenAI format
    Given a triton embeddings response with <input_count> inputs and a 4-float data array
    When the response is transformed
    Then the OpenAI response has <input_count> vectors of dimension <dim> and usage tokens <tokens>

    Examples:
      | input_count | dim | tokens |
      | 2           | 4   | 9      |
      | 1           | 8   | 3      |

  Scenario: Triton embedding responses without outputs raise ValueError
    When the triton embeddings response has no outputs
    Then transforming raises ValueError "No outputs in Triton response"

  Scenario: Embeddings generation accepts non-dict objects with a non-callable json attribute
    When the triton passthrough returns an object whose json attribute is not callable
    Then the embeddings response keeps the object data

  # --- Image generation ---

  Scenario: Image generation passes arguments to the router image API
    When an image is generated with prompt "a cat"
    Then the router aimage_generation call receives the same arguments

  # --- System One generator ---

  Scenario Outline: SystemOne passthrough kwargs derive from the model config
    Given a systemone model config <config_desc>
    When passthrough kwargs are built for model "so-model"
    Then the kwargs are <kwargs>

    Examples:
      | config_desc                                     | kwargs                                                                                          |
      | with defaults                                   | {'method': 'POST', 'json': 'json-payload', 'model': 'so-model'}                                  |
      | with explicit method endpoint address and key   | {'method': 'GET', 'json': 'json-payload', 'model': 'so-model', 'endpoint': 'http://ep', 'api_base': 'http://addr', 'api_key': 'sk'} |

  Scenario Outline: JSON fields serialize deterministically
    When the field value <value_desc> is serialized
    Then the serialized value is <serialized>

    Examples:
      | value_desc                | serialized      |
      | a plain string            | 'state-1'       |
      | a dict                    | '{"a": 1}'      |

  Scenario Outline: Triton result_json outputs decode to backend payloads
    Given a triton result payload <payload>
    When the result json is decoded
    Then the decoded payload is <decoded>

    Examples:
      | payload             | decoded         |
      | named_string        | {'answers': 1}  |
      | first_output_string | {'answers': 2}  |
      | bytes_value         | {'answers': 3}  |
      | nested_list         | {'answers': 4}  |
      | non_string_number   | 5               |

  Scenario: SystemOne response formatting respects route_only and routing flags
    When the backend payload {'answers': ['a'], 'usage': {'input_tokens': 3, 'output_tokens': 4}, 'routing': 'router'} is formatted with route_only "<route_only>" and include_routing "<include_routing>"
    Then the formatted response is <expected>

    Examples:
      | route_only | include_routing | expected                                                          |
      | false      | true            | {'model': 'so-external', 'answers': ['a'], 'usage': {'input_tokens': 3, 'output_tokens': 4}, 'routing': 'router'} |
      | false      | false           | {'model': 'so-external', 'answers': ['a'], 'usage': {'input_tokens': 3, 'output_tokens': 4}}                      |
      | true       | true            | {'model': 'so-external', 'routing': 'router'}                    |

  Scenario: SystemOne response formatting rejects missing payloads
    When the backend payload <payload> is formatted with route_only "<route_only>"
    Then formatting raises ValueError "<message>"

    Examples:
      | payload                 | route_only | message                          |
      | none_routing            | true       | route_only response missing routing |
      | none_answers            | false      | response missing answers         |

  Scenario: SystemOne builds triton requests with override and route_only tensors
    When a triton request is built with state "st" questions "qu" model_override "ov" route_only True
    Then the request has tensors for state_json questions_json model_override and route_only
    And outputs request the result_json tensor

  Scenario: SystemOne run uses the passthrough kwargs and decodes the response
    When run_system_one executes for model "so-model" with state {"k": 1}
    Then the passthrough call model is "so-model"
    And the decoded response has answers ["a"]