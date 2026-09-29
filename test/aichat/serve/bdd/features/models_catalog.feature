Feature: Model catalog and configuration
  The /v1/models endpoint exposes the model catalog for browser clients
  and internal code resolves ModelConfig entries from model settings.

  Scenario Outline: Model configs are resolved from settings
    Given a model "cap-model" with capabilities "<caps>", type "<type>" and free <free>
    When the model config is resolved
    Then the config tool support is <tool_support>
    And the config free flag is <free>

    Examples:
      | caps         | type | tool_support | free  |
      | tools        | llm  | True         | True  |
      | tools,vision | llm  | True         | True  |
      | vision       | llm  | False        | True  |
      | tools        | llm  | True         | False |

  Scenario: Unsupported model types resolve to no config
    Given a model "embedding-1" with type "embedding"
    When the model config is resolved
    Then there is no model config

  Scenario: Models without explicit free flags default to free
    Given a model "simple" with no free flag
    When the model config is resolved
    Then the config free flag is True

  Scenario: Fallback models propagate to the config
    Given a model "main-model" falling back to "backup-model"
    When the model config is resolved
    Then the config fallback models are "backup-model"

  Scenario: Repeated config lookups reuse the cached config
    Given a model "cap-model" with capabilities "tools", type "llm" and free True
    When the model config is resolved twice
    Then both resolutions return the same config

  Scenario Outline: Bedrock fallback detection
    Given a model "cap-model" with backend "<backend>"
    And fallback model "backup-model" with backend "<fallback_backend>"
    When the bedrock fallback is checked
    Then the fallback verdict is <verdict>

    Examples:
      | backend | fallback_backend | verdict |
      | bedrock | vllm             | True    |
      | vllm    | bedrock          | True    |
      | vllm    | vllm             | False   |

  Scenario Outline: LiteLLM model names normalize to friendly ids
    Given a model "friendly-1" with upstream model "<upstream>"
    And inference profile env "MY_PROFILE_A" set to "profile-x"
    When the litellm name "<litellm_name>" is normalized
    Then the normalized model is "<normalized>"

    Examples:
      | upstream                          | litellm_name                      | normalized   |
      | org/friendly-upstream             | org/friendly-upstream             | friendly-1   |
      | org/friendly-upstream             | hosted_vllm/org/friendly-upstream | friendly-1   |
      | org/friendly-upstream             | unknown/thing                     | thing        |
      | org/friendly-upstream             | none                              | none         |
      | org/friendly-upstream             | friendly-upstream                 | friendly-upstream |

  Scenario: Bedrock inference profiles map through the environment
    Given a model "bedrock-1" with backend "bedrock" and inference profile env "MY_PROFILE_B"
    And inference profile env "MY_PROFILE_B" set to "profile-y"
    When the litellm name "profile-y" is normalized
    Then the normalized model is "bedrock-1"

  Scenario: Bedrock profile matching keeps the requested model id
    Given bedrock models "bedrock-1,bedrock-2,bedrock-3" sharing profile env "MY_PROFILE_A"
    And inference profile env "MY_PROFILE_A" set to "profile-x"
    When the litellm name "profile-x" is normalized with requested model "bedrock-2"
    Then the normalized model is "bedrock-2"

  Scenario: Model configs expose dict-style access
    Given a model "cap-model" with capabilities "tools", type "llm" and free True
    When the model config is resolved
    Then the config dict model id is "cap-model"
    And the config lookup for "tool_support" is True

  Scenario: Bedrock mantle upstream models map through the upstream name
    Given a model "mantle-1" with upstream model "org/mantle-upstream" on backend "bedrock_mantle"
    When the litellm name "org/mantle-upstream" is normalized
    Then the normalized model is "mantle-1"

  Scenario: Non-bedrock models carry no inference profile
    Given a model "vllm-1" with backend "vllm" and inference profile env "MY_PROFILE_C"
    And inference profile env "MY_PROFILE_C" set to "profile-z"
    When the litellm name "profile-z" is normalized
    Then the normalized model is "profile-z"

  Scenario: Model triaging without defaults logs a warning
    Given model triaging with only a "vision" category
    When the model settings are constructed
    Then a misconfiguration warning is logged

  Scenario: Model triaging with a default logs no warning
    Given model triaging with a "default" category
    When the model settings are constructed
    Then no misconfiguration warning is logged

  Scenario Outline: Accept-Language parsing for model listing
    Given a browser client requesting models with language "<accept_language>"
    When the models are listed
    Then the "basic" model display description is "<description>"

    Examples:
      | accept_language         | description     |
      | de-DE,de;q=0.9,en;q=0.8 | Ein Testmodell. |
      | fr                      | A test model.   |
      | nothing                 | A test model.   |

  Scenario: The models endpoint returns catalog entries
    Given a browser client with model "basic" defined
    When the models are listed with language "en"
    Then the catalog contains model "basic"
    And the catalog entry options name is "basic"
    And the "basic" catalog entry access is "basic_and_premium"
