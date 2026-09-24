Feature: Media sandbox primitives
  JSON-lines protocol, resource limits, worker loop, sandbox ops and landlock guards.

  Background:
    Given the sandbox harness

  Scenario: Protocol requests round trip through JSON lines
    When a request id 7 op "pdf_analyze" with args {"pdf_b64": "AAAA"} is encoded and decoded
    Then the decoded request has id 7 op "pdf_analyze" and args {"pdf_b64": "AAAA"}

  Scenario Outline: Protocol request encoding rejects oversized payloads
    When a request is encoded with <defect>
    Then encoding raises ProtocolError

    Examples:
      | defect                        |
      | a huge string argument       |
      | a huge serialized payload    |

  Scenario Outline: Protocol request decoding rejects malformed lines
    When a request line <line_desc> is decoded
    Then decoding raises ProtocolError

    Examples:
      | line_desc           |
      | that is invalid json |
      | missing the id field |

  Scenario: Protocol responses round trip ok and error payloads
    When an ok response and an error response are encoded and decoded
    Then both round trip with their values intact

  Scenario Outline: Protocol response decoding rejects malformed lines
    When a response line <line_desc> is decoded
    Then decoding raises ProtocolError

    Examples:
      | line_desc           |
      | that is invalid json |
      | missing the id field |

  Scenario: Base64 helpers round trip binary payloads
    When bytes are base64 encoded and decoded
    Then the bytes match the original

  Scenario: The op wall clock alarm raises the dedicated exception
    When the SIGALRM handler fires
    Then the raised error is OpWallClockExceeded

  Scenario Outline: Resource limits are lowered with platform fallbacks
    When rlimits are applied <limit_desc>
    Then the limit application <outcome>

    Examples:
      | limit_desc                          | outcome                                  |
      | with soft limits below hard limits  | sets the new soft and hard limits        |
      | with an unsupported limit value     | warns without raising on non-linux       |

  Scenario: Per-op limits arm the CPU budget and wall clock alarm
    When op limits are armed for 5 cpu seconds
    Then the alarm is set to 13 seconds
    And disarming clears the alarm

  Scenario: The worker scrubs non-allowlisted environment variables
    When the worker environment is scrubbed
    Then only allowlisted keys remain and bytecode writing is disabled

  Scenario: The worker writes structured error responses
    When an error response is written for request 4
    Then the output line is the json error payload for request 4

  Scenario Outline: The worker serve loop handles each input line
    Given a serve loop with handler for "ping"
    When the input line <line_desc> is processed
    Then the serve loop <outcome>

    Examples:
      | line_desc                        | outcome                                        |
      | is a valid ping request          | replies with a pong payload                    |
      | names an unknown op              | replies with a KeyError error response         |
      | is invalid json                  | replies with a ProtocolError for id -1         |
      | is non-utf8 bytes                | replies with a ProtocolError for id -1         |
      | is blank                         | produces no output and continues               |
      | reaches end of stream            | exits the loop                                 |

  Scenario: Sandbox initialization is a no-op guard on non-linux platforms
    When the sandbox initializes on non-linux
    Then initialization returns 0 without applying landlock

  Scenario Outline: STT decoding validates and normalizes WAV uploads
    When a WAV payload <wav_desc> is decoded
    Then the decode <outcome>

    Examples:
      | wav_desc                              | outcome                                                   |
      | is 16k mono int16                     | returns 16000 samples at the target rate                  |
      | is stereo 8k                          | downmixes and resamples to 16k                            |
      | is 8-bit unsigned                     | centers the samples around zero                           |
      | is 32-bit signed                      | normalizes int32 samples                                  |
      | is compressed                         | rejects compressed WAVE formats                           |
      | exceeds the duration limit            | rejects with the duration error                           |
      | has an invalid framerate              | rejects with invalid WAV parameters                       |
      | has a corrupt stereo frame buffer     | rejects with corrupt WAV frame buffer                     |
      | has an unsupported sample width       | rejects with the sample width error                       |
      | is empty after decoding               | rejects with empty audio                                  |
      | is malformed wave data                | rejects with the PCM WAV support message                  |

  Scenario Outline: PDF analysis handles page counts and token budgets
    Given a pdf document <pdf_desc>
    When the pdf is analyzed with budget <budget>
    Then the analysis <outcome>

    Examples:
      | pdf_desc                          | budget | outcome                                        |
      | with 2 blank pages                | 100    | reports 2 pages under the budget               |
      | with 90 blank pages               | 100    | extracts all text over the page limit          |
      | needing native truncation         | 20     | truncates to native pages with overflow text   |

  Scenario: PDF analysis reports encrypted documents for passthrough
    When the pdf analysis is asked about an encrypted document
    Then the analysis reports the encrypted passthrough

  Scenario Outline: Probe ops behave on non-linux hosts
    When the probe op <op> runs
    Then the probe result <outcome>

    Examples:
      | op            | outcome                                       |
      | probe_env     | lists the sorted environment keys             |
      | probe_fs_write| skips the write check on non-linux            |
      | spin_cpu      | spins for the requested duration              |

  Scenario: Landlock ABI probing returns zero on non-linux platforms
    When the landlock ABI is probed on non-linux
    Then the ABI is 0

  Scenario: Applying landlock on non-linux platforms is unsupported
    When landlock is applied on non-linux
    Then it raises LandlockUnavailable

  Scenario: The sandbox pool returns the shared singleton
    When the pool singleton is requested twice
    Then both requests return the same pool instance

  Scenario: The sandbox pool retries workers and fails loudly when all die
    When both pool workers fail hard during a call
    Then the call raises SandboxWorkerError "all 2 sandbox workers failed"

  Scenario: The sandbox pool surfaces op errors immediately
    When the first pool worker reports an op error
    Then the call raises SandboxOpError

  Scenario: A dead pool worker restarts on the next call
    When a pool worker dies and the next call succeeds after restart
    Then the call result is the restarted worker payload
    And a start failure raises SandboxWorkerError

  Scenario: Pool worker calls survive timeouts and transport drops
    When a worker call exceeds its wall clock
    Then the worker is killed and raises SandboxWorkerError
    And a transport drop marks the worker dead and raises SandboxWorkerError

  Scenario: Pool shutdown and reset clear worker state
    When the pool is shut down and reset
    Then the pool is empty and the singleton is cleared