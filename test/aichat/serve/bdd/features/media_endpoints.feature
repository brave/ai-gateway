Feature: Media API endpoints (STT, TTS, embeddings, image generation, system one)
  Validation, auth and dispatch behavior of the media-facing HTTP handlers.

  Background:
    Given the media test harness

  # --- STT /audio/transcriptions ---

  Scenario: Transcriptions reject unauthenticated callers
    When a transcription request arrives without credentials
    Then the response is a 401 error

  Scenario: Transcriptions require a model parameter
    When a transcription request has no model
    Then the response is a 400 error with "model parameter is required for transcription"

  Scenario Outline: Transcriptions reject unsupported models
    When a transcription request names the model "<model>" of type "<type>"
    Then the response is a 404 error with "<message>"

    Examples:
      | model     | type            | message                                        |
      | nope      | speech_to_text  | model is not supported - nope                  |
      | parakeet  | llm             | model does not support speech-to-text - parakeet |

  Scenario: Transcriptions reject upload read failures
    When reading the transcription upload raises a generic error
    Then the response is a 400 error with "Failed to read upload"

  Scenario: Transcriptions reject unsupported response formats
    When a transcription request asks for response_format "xml"
    Then the response is a 400 error with "unsupported response_format: xml"

  Scenario: Transcriptions report a 400 when transcription validation fails
    When the transcription backend raises ValueError "bad audio"
    Then the response is a 400 error with "bad audio"

  Scenario: Transcriptions report a 500 when transcription fails unexpectedly
    When the transcription backend raises RuntimeError "boom"
    Then the response is a 500 error with "Transcription failed: boom"

  Scenario Outline: Transcriptions render each response format
    When a transcription succeeds and asks for response_format "<format>"
    Then the transcription response has media type "<media_type>" and matches the "<format>" rendering

    Examples:
      | format       | media_type       |
      | text         | text/plain       |
      | verbose_json | application/json |
      | srt          | text/plain       |
      | vtt          | text/vtt         |

  Scenario: Transcriptions default to a JSON body with the transcript text
    When a transcription succeeds and asks for response_format "json"
    Then the transcription JSON body has text "hello world"

  # --- TTS /audio/speech ---

  Scenario Outline: Speech generation maps response formats to content types
    When a speech request asks for response_format "<fmt>"
    Then the speech audio is returned as "<content_type>"

    Examples:
      | fmt   | content_type |
      | none  | audio/mpeg   |
      | mp3   | audio/mpeg   |
      | opus  | audio/opus   |
      | aac   | audio/aac    |
      | flac  | audio/flac   |
      | pcm   | audio/pcm    |
      | wav   | audio/mpeg   |

  Scenario: Speech generation reports an internal error for invalid generator responses
    When the speech generator returns an object without audio content
    Then the response is a 500 error with "Invalid response format from speech generation"

  Scenario: Speech generation streams audio when requested
    When a speech request sets stream to true
    Then the response is a streaming audio response of type "audio/mpeg"

  # --- Embeddings /v1/embeddings ---

  Scenario Outline: Embeddings reject invalid requests
    When an embeddings request <defect> is submitted
    Then the response is a <status> error with "<message>"

    Examples:
      | defect                        | status | message                                    |
      | with no model                 | 400    | model parameter is required for embeddings |
      | with no input                 | 400    | input parameter is required for embeddings |
      | naming an unknown model       | 404    | model is not supported - nope              |
      | naming a non-embedding model  | 404    | model does not support embeddings          |
      | with invalid JSON             | 400    | Invalid JSON                               |

  Scenario: Embeddings report a 500 when generation fails
    When the embeddings generator raises RuntimeError "boom"
    Then the response is a 500 error with "Embeddings generation failed: boom"

  # --- Images /v1/images/generations ---

  Scenario Outline: Image generation rejects invalid requests
    When an image generation request <defect> is submitted
    Then the response is a <status> error with "<message>"

    Examples:
      | defect                  | status | message                                     |
      | with no model           | 400    | model parameter is required for image generation |
      | naming an unknown model | 404    | model is not supported - nope               |
      | with invalid JSON       | 400    | Invalid JSON                                |

  Scenario: Image generation reports a 500 when generation fails
    When the image generator raises RuntimeError "boom"
    Then the response is a 500 error with "Image generation failed: boom"

  # --- SystemOne /v1/systemone ---

  Scenario: SystemOne rejects unauthenticated callers
    When a systemone request arrives without credentials
    Then the response is a 401 error

  Scenario Outline: SystemOne rejects invalid requests
    When a systemone request <defect> is submitted
    Then the response is a <status> error with "<message>"

    Examples:
      | defect                                  | status | message                                |
      | with invalid JSON                       | 400    | Invalid JSON                           |
      | that is not a JSON object               | 400    | request body must be a JSON object     |
      | with no model                           | 400    | model parameter is required for systemone |
      | naming an unknown model                 | 404    | model is not supported - nope          |
      | naming a non-systemone model            | 404    | model does not support systemone       |
      | with no state                           | 400    | state parameter is required            |
      | with no questions                       | 400    | questions parameter is required        |
      | with an empty model_override            | 400    | model_override must be a non-empty string |
      | with a non-boolean route_only           | 400    | route_only must be a boolean           |
      | with a non-boolean include_routing      | 400    | include_routing must be a boolean      |

  Scenario: SystemOne runs inference with validated parameters
    When a systemone request carries model_override "  so-override  " and route flags
    Then the systemone response contains model "so-external"
    And run_system_one received model_override "so-override" route_only True include_routing True

  Scenario: SystemOne reports a 400 on validation errors from the backend
    When the systemone backend raises ValueError "routing missing"
    Then the response is a 400 error with "routing missing"

  Scenario: SystemOne reports a 500 when inference fails unexpectedly
    When the systemone backend raises RuntimeError "boom"
    Then the response is a 500 error with "System One inference failed: boom"
