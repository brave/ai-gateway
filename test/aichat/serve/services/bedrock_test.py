import pytest

from aichat.protocol.open_ai_protocol import Tool, ToolFunction
from aichat.serve.constants import CONTENT_FILTER_MESSAGE
from aichat.serve.services.bedrock import (
    MAX_CACHE_POINTS,
    add_cache_control_to_tools,
    apply_content_filter_to_completion_response,
    bedrock_settings,
    completion_content_for_finish_reason,
    filter_tool_call_result_pairing,
    format_tools_for_bedrock,
    get_cache_control_injection_points,
    get_prefix_cache_indices,
    map_tool_role_to_assistant,
    sanitize_tool_call_id_for_bedrock,
    sanitize_tool_name_for_bedrock,
)
from aichat.serve.utils import get_token_count_estimate


class TestCompletionContentForFinishReason:
    """Tests for completion_content_for_finish_reason."""

    def test_returns_content_filter_message_when_finish_reason_is_content_filter(self):
        result = completion_content_for_finish_reason("content_filter", None)
        assert result == CONTENT_FILTER_MESSAGE

    def test_returns_content_filter_message_when_content_filter_even_with_content(self):
        result = completion_content_for_finish_reason(
            "content_filter", "truncated or empty"
        )
        assert result == CONTENT_FILTER_MESSAGE

    def test_returns_original_content_when_finish_reason_is_stop(self):
        content = "Here is the answer."
        result = completion_content_for_finish_reason("stop", content)
        assert result == content

    def test_returns_original_content_when_finish_reason_is_length(self):
        content = "Partial response..."
        result = completion_content_for_finish_reason("length", content)
        assert result == content

    def test_returns_original_content_when_finish_reason_is_none(self):
        content = "Some content"
        result = completion_content_for_finish_reason(None, content)
        assert result == content

    def test_returns_none_when_finish_reason_and_content_both_none(self):
        result = completion_content_for_finish_reason(None, None)
        assert result is None

    def test_returns_original_content_when_finish_reason_is_tool_calls(self):
        content = ""
        result = completion_content_for_finish_reason("tool_calls", content)
        assert result == ""


class TestApplyContentFilterToCompletionResponse:
    """Tests for apply_content_filter_to_completion_response."""

    def test_replaces_message_content_when_finish_reason_is_content_filter(self):
        response = {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"role": "assistant", "content": ""},
                }
            ]
        }
        apply_content_filter_to_completion_response(response)
        assert response["choices"][0]["message"]["content"] == CONTENT_FILTER_MESSAGE
        assert response["choices"][0]["finish_reason"] == "content_filter"

    def test_sets_message_content_when_message_is_none(self):
        response = {"choices": [{"finish_reason": "content_filter", "message": None}]}
        apply_content_filter_to_completion_response(response)
        assert response["choices"][0]["message"]["content"] == CONTENT_FILTER_MESSAGE

    def test_does_not_mutate_when_finish_reason_is_stop(self):
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello"},
                }
            ]
        }
        original_content = response["choices"][0]["message"]["content"]
        apply_content_filter_to_completion_response(response)
        assert response["choices"][0]["message"]["content"] == original_content

    def test_does_not_mutate_when_finish_reason_is_length(self):
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "Truncated..."},
                }
            ]
        }
        apply_content_filter_to_completion_response(response)
        assert response["choices"][0]["message"]["content"] == "Truncated..."

    def test_no_op_when_response_is_not_dict(self):
        response = None
        apply_content_filter_to_completion_response(response)
        assert response is None

    def test_no_op_when_choices_is_empty(self):
        response = {"choices": []}
        apply_content_filter_to_completion_response(response)
        assert response["choices"] == []

    def test_no_op_when_choices_is_missing(self):
        response = {"id": "cmpl-123", "model": "test"}
        apply_content_filter_to_completion_response(response)
        assert "choices" not in response

    def test_makes_message_mutable_when_message_is_immutable_like(self):
        """Ensure we can set content even when message was a non-dict (e.g. from model_dump)."""
        response = {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"role": "assistant", "content": "x"},
                }
            ]
        }
        apply_content_filter_to_completion_response(response)
        assert response["choices"][0]["message"]["content"] == CONTENT_FILTER_MESSAGE


@pytest.fixture
def valid_openai_tool():
    """Mock valid OpenAI function tool"""
    return Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        ),
    )


@pytest.fixture
def openai_tool_missing_parameters():
    """Mock OpenAI tool with missing parameters"""
    return Tool(
        type="function",
        function=ToolFunction(
            name="simple_tool", description="A simple tool", parameters=None
        ),
    )


@pytest.fixture
def openai_tool_incomplete_parameters():
    """Mock OpenAI tool with incomplete parameters schema"""
    return Tool(
        type="function",
        function=ToolFunction(
            name="incomplete_tool",
            description="Tool with incomplete params",
            parameters={
                # Missing type and properties - this will be fixed by the service
                "additionalProperties": True
            },
        ),
    )


class TestFormatToolsForBedrock:
    def test_format_tools_for_bedrock_valid_openai_tools(self, valid_openai_tool):
        """Test formatting already valid OpenAI tools"""
        tools = [valid_openai_tool]

        result = format_tools_for_bedrock(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get current weather"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_format_tools_for_bedrock_missing_parameters(
        self, openai_tool_missing_parameters
    ):
        """Test formatting OpenAI tool with missing parameters"""
        tools = [openai_tool_missing_parameters]

        result = format_tools_for_bedrock(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "simple_tool"
        # Should have default parameters added
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert params["additionalProperties"] is True

    def test_format_tools_for_bedrock_incomplete_parameters(
        self, openai_tool_incomplete_parameters
    ):
        """Test formatting OpenAI tool with incomplete parameters schema"""
        tools = [openai_tool_incomplete_parameters]

        result = format_tools_for_bedrock(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "incomplete_tool"
        # Should have missing fields added
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert params["additionalProperties"] is True

    def test_format_tools_for_bedrock_adds_missing_additional_properties(self):
        """Test that additionalProperties is added even when parameters dict exists (fixes original bug)"""
        tool_missing_additional_properties = Tool(
            type="function",
            function=ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters={
                    "type": "object",
                    "properties": {"param1": {"type": "string"}},
                    # Missing additionalProperties - this was the bug in original code!
                },
            ),
        )
        tools = [tool_missing_additional_properties]

        result = format_tools_for_bedrock(tools)

        assert len(result) == 1
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"]["param1"]["type"] == "string"
        assert params["additionalProperties"] is True  # Should be added by our fix!

    def test_format_tools_for_bedrock_empty_list(self):
        """Test formatting empty tools list"""
        result = format_tools_for_bedrock([])
        assert result == []


def _content_with_tokens(target_tokens: int) -> str:
    """Build a string whose token count (per get_token_count_estimate) is at
    least target_tokens, using distinct words so BPE doesn't compress it.

    Counts incrementally per appended word (each space-prefixed ``wordN``
    tokenizes independently) to stay O(n); re-tokenizing the whole growing
    string each iteration would be O(n^2) at import time.
    """
    words: list[str] = []
    total = 0
    while total < target_tokens:
        word = f"word{len(words)}"
        total += get_token_count_estimate(word if not words else f" {word}")
        words.append(word)
    return " ".join(words)


# A chunk that on its own clears the cache minimum.
BIG = _content_with_tokens(bedrock_settings.bedrock_cache_min_tokens + 50)
# A chunk well under the minimum.
SMALL = "short message"


@pytest.fixture
def all_low_token_messages():
    """Messages whose cumulative total never reaches the cache minimum."""
    return [
        {"role": "user", "content": "Short message 1"},
        {"role": "assistant", "content": "Short message 2"},
        {"role": "user", "content": "Short message 3"},
    ]


class TestGetPrefixCacheIndices:
    def test_anchor_at_first_crossing(self):
        """The first index whose cumulative prefix clears the minimum is the anchor."""
        messages = [
            {"role": "user", "content": SMALL},  # 0: small
            {"role": "assistant", "content": BIG},  # 1: crosses here
            {"role": "user", "content": SMALL},  # 2
        ]
        result = get_prefix_cache_indices(messages, max_count=2)
        # Anchor is index 1 (first crossing); latest eligible is index 2.
        assert result == [1, 2]

    def test_returns_anchor_and_latest(self):
        """With several eligible messages, return the earliest and latest."""
        messages = [
            {"role": "user", "content": BIG},  # 0: crosses immediately
            {"role": "assistant", "content": BIG},  # 1
            {"role": "user", "content": BIG},  # 2
            {"role": "assistant", "content": BIG},  # 3
        ]
        result = get_prefix_cache_indices(messages, max_count=2)
        assert result == [0, 3]

    def test_max_count_one_returns_only_anchor(self):
        messages = [
            {"role": "user", "content": BIG},
            {"role": "assistant", "content": BIG},
        ]
        assert get_prefix_cache_indices(messages, max_count=1) == [0]

    def test_max_count_zero(self):
        messages = [{"role": "user", "content": BIG}]
        assert get_prefix_cache_indices(messages, max_count=0) == []

    def test_all_low_tokens_returns_empty(self, all_low_token_messages):
        assert get_prefix_cache_indices(all_low_token_messages) == []

    def test_empty_messages(self):
        assert get_prefix_cache_indices([]) == []

    def test_skips_messages_without_content(self):
        """Cumulative count ignores content-less messages and never targets them."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{}]},  # 0: no content
            {"role": "user", "content": BIG},  # 1: crosses
            {"role": "assistant", "content": None},  # 2: no content
        ]
        result = get_prefix_cache_indices(messages, max_count=2)
        # Only index 1 has content & clears the minimum.
        assert result == [1]

    def test_single_eligible_message_no_duplicate(self):
        """When anchor and latest collapse to one index, return it once."""
        messages = [{"role": "user", "content": BIG}]
        assert get_prefix_cache_indices(messages, max_count=2) == [0]

    def test_large_system_prompt_is_not_targeted(self):
        """A large system prompt counts toward the prefix total but never
        receives an index-based checkpoint (it is cached by role separately)."""
        messages = [
            {"role": "system", "content": BIG},  # 0: large, but cached by role
            {"role": "user", "content": SMALL},  # 1: prefix already clears min
            {"role": "assistant", "content": SMALL},  # 2
        ]
        result = get_prefix_cache_indices(messages, max_count=2)
        # Anchor is the first non-system message, not the system prompt at 0.
        assert 0 not in result
        assert result == [1, 2]


class TestGetCacheControlInjectionPoints:
    def test_system_message_is_cached(self):
        """A system message gets a role-targeted ephemeral point."""
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": SMALL},
        ]
        result = get_cache_control_injection_points(messages)
        assert result == [
            {"location": "message", "role": "system", "control": {"type": "ephemeral"}}
        ]

    def test_system_plus_history_points(self):
        """System point plus prefix points, all ephemeral and well-formed."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": BIG},
            {"role": "assistant", "content": BIG},
        ]
        result = get_cache_control_injection_points(messages)
        # 1 system point (by role) + anchor + latest history points.
        assert result[0] == {
            "location": "message",
            "role": "system",
            "control": {"type": "ephemeral"},
        }
        history = result[1:]
        assert [p["index"] for p in history] == [1, 2]
        for point in result:
            assert point["control"] == {"type": "ephemeral"}

    def test_cache_system_prompt_disabled(self):
        """No system point when cache_system_prompt=False."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": BIG},
        ]
        result = get_cache_control_injection_points(messages, cache_system_prompt=False)
        assert all("role" not in p for p in result)

    def test_reserved_points_shrink_budget(self):
        """reserved_points reduces how many checkpoints are emitted."""
        messages = [{"role": "user", "content": BIG} for _ in range(4)]
        # No reservation: anchor + latest = 2 points.
        assert len(get_cache_control_injection_points(messages)) == 2
        # Reserve all 4 (e.g. impossible tools scenario): nothing left.
        assert (
            get_cache_control_injection_points(
                messages, reserved_points=MAX_CACHE_POINTS
            )
            == []
        )

    def test_never_exceeds_max_cache_points(self):
        """System + history never exceeds the budget minus reservations."""
        messages = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": BIG} for _ in range(6)
        ]
        result = get_cache_control_injection_points(messages, reserved_points=1)
        assert len(result) <= MAX_CACHE_POINTS - 1

    def test_all_low_tokens_no_history_points(self, all_low_token_messages):
        """No system message and no big prefix => no points."""
        result = get_cache_control_injection_points(all_low_token_messages)
        assert result == []

    def test_empty_messages(self):
        assert get_cache_control_injection_points([]) == []

    def test_large_system_prompt_no_duplicate_point(self):
        """A large system prompt yields exactly one role point, not also an
        index point on the system message itself."""
        messages = [
            {"role": "system", "content": BIG},
            {"role": "user", "content": BIG},
            {"role": "assistant", "content": BIG},
        ]
        result = get_cache_control_injection_points(messages)
        role_points = [p for p in result if "role" in p]
        index_points = [p for p in result if "index" in p]
        assert len(role_points) == 1
        # No index checkpoint lands on the system message at index 0.
        assert all(p["index"] != 0 for p in index_points)

    def test_contentless_system_message_not_cached(self):
        """A system message with no content yields no role-targeted point."""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": SMALL},
        ]
        result = get_cache_control_injection_points(messages)
        assert all("role" not in p for p in result)


class TestAddCacheControlToTools:
    def test_tags_last_tool(self):
        tools = [{"function": {"name": "a"}}, {"function": {"name": "b"}}]
        result = add_cache_control_to_tools(tools)
        assert "cache_control" not in result[0]
        assert result[-1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_list_unchanged(self):
        assert add_cache_control_to_tools([]) == []

    def test_single_tool(self):
        tools = [{"function": {"name": "only"}}]
        result = add_cache_control_to_tools(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_does_not_mutate_input(self):
        """Tagging returns a new list/dict and leaves the input untouched, so
        shared tool definitions never leak a cache_control key."""
        last_tool = {"function": {"name": "b"}}
        tools = [{"function": {"name": "a"}}, last_tool]
        result = add_cache_control_to_tools(tools)
        # Input list and its dicts are unchanged...
        assert "cache_control" not in last_tool
        assert all("cache_control" not in t for t in tools)
        # ...while the returned copy carries the checkpoint.
        assert result[-1]["cache_control"] == {"type": "ephemeral"}
        assert result[-1] is not last_tool

    def test_non_dict_last_tool_unchanged(self):
        tools = [{"function": {"name": "a"}}, "not-a-dict"]
        assert add_cache_control_to_tools(tools) == tools


class TestMapToolRoleToAssistant:
    def test_maps_tool_to_assistant(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
            {"role": "assistant", "content": "Done"},
        ]
        result = map_tool_role_to_assistant(messages)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["tool_call_id"] == "call_1"
        assert result[2]["role"] == "assistant"

    def test_empty_messages(self):
        assert map_tool_role_to_assistant([]) == []

    def test_no_tool_messages(self):
        messages = [{"role": "user", "content": "Hi"}]
        assert map_tool_role_to_assistant(messages) == messages


class TestValidateToolCallResultPairing:
    def test_filter_tool_call_result_pairing_valid_conversation(self):
        """Test with properly paired tool calls and results"""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Sunny, 72°F",
            },
            {"role": "assistant", "content": "It's sunny and 72°F in NYC."},
        ]

        result = filter_tool_call_result_pairing(messages)

        # All messages should be preserved
        assert len(result) == 4
        assert result == messages

    def test_filter_tool_call_result_pairing_orphaned_tool_result(self):
        """Test with orphaned tool result (no matching tool call)"""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Sunny, 72°F",
            },
            # This tool result has no matching tool call
            {
                "role": "tool",
                "tool_call_id": "call_456",
                "content": "This is orphaned",
            },
            {"role": "assistant", "content": "Response"},
        ]

        result = filter_tool_call_result_pairing(messages)

        # Orphaned tool result should be removed
        assert len(result) == 4
        # Check that call_456 tool result was removed
        tool_call_ids_in_result = [
            msg.get("tool_call_id") for msg in result if msg.get("role") == "tool"
        ]
        assert "call_123" in tool_call_ids_in_result
        assert "call_456" not in tool_call_ids_in_result

    def test_filter_tool_call_result_pairing_malformed_json_scenario(self):
        """Test scenario where malformed JSON causes tool call to be filtered but result remains"""
        messages = [
            {"role": "user", "content": "Click something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_valid",
                        "type": "function",
                        "function": {
                            "name": "valid_tool",
                            "arguments": '{"param": "value"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_valid",
                "content": "Success",
            },
            # Simulating: malformed tool call was filtered out by Bedrock/litellm
            # but the tool result remains (this is the bug we're fixing)
            {
                "role": "tool",
                "tool_call_id": "call_malformed",
                "content": "Error: Failed to parse input JSON",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        # Should remove the orphaned tool result for the malformed call
        assert len(result) == 3
        tool_results = [msg for msg in result if msg.get("role") == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "call_valid"

    def test_filter_tool_call_result_pairing_multiple_tool_calls(self):
        """Test with multiple tool calls in same turn"""
        messages = [
            {"role": "user", "content": "Get weather and time"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": "Sunny",
            },
            {
                "role": "tool",
                "tool_call_id": "call_time",
                "content": "3:00 PM",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        # All messages should be preserved
        assert len(result) == 4
        assert result == messages

    def test_filter_tool_call_result_pairing_multi_turn_conversation(self):
        """Test with multiple turns of tool calls"""
        messages = [
            {"role": "user", "content": "First request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool1", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result 1"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool2", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
            {"role": "assistant", "content": "Second response"},
        ]

        result = filter_tool_call_result_pairing(messages)

        # All messages should be preserved
        assert len(result) == 8
        assert result == messages

    def test_filter_tool_call_result_pairing_multi_turn_with_orphan(self):
        """Test multi-turn where second turn has orphaned result"""
        messages = [
            {"role": "user", "content": "First request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool1", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result 1"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool2", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
            # Orphaned result from previous turn (call_1 no longer valid)
            {"role": "tool", "tool_call_id": "call_1", "content": "Stale result"},
        ]

        result = filter_tool_call_result_pairing(messages)

        # Should remove the stale result
        assert len(result) == 7
        tool_results = [msg for msg in result if msg.get("role") == "tool"]
        assert len(tool_results) == 2
        # Only call_1 (first turn) and call_2 (second turn) should remain
        tool_call_ids = [msg["tool_call_id"] for msg in tool_results]
        assert tool_call_ids == ["call_1", "call_2"]

    def test_filter_tool_call_result_pairing_empty_messages(self):
        """Test with empty messages list"""
        result = filter_tool_call_result_pairing([])
        assert result == []

    def test_filter_tool_call_result_pairing_no_tool_calls(self):
        """Test with conversation that has no tool calls"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]

        result = filter_tool_call_result_pairing(messages)

        # All messages should be preserved
        assert len(result) == 4
        assert result == messages

    def test_filter_tool_call_result_pairing_tool_call_without_id(self):
        """Test with tool call that has no ID (edge case)"""
        messages = [
            {"role": "user", "content": "Request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        # Missing 'id' field
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Result"},
        ]

        result = filter_tool_call_result_pairing(messages)

        # Tool result should be removed since no valid tool call ID exists
        assert len(result) == 2
        tool_results = [msg for msg in result if msg.get("role") == "tool"]
        assert len(tool_results) == 0

    def test_filter_tool_call_result_pairing_preserves_non_tool_messages(self):
        """Test that non-tool messages are always preserved"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
            # Orphaned tool result
            {"role": "tool", "tool_call_id": "call_orphan", "content": "Orphan"},
            {"role": "assistant", "content": "Final response"},
        ]

        result = filter_tool_call_result_pairing(messages)

        # Should have 6 messages (orphan removed)
        assert len(result) == 6
        # Verify all non-tool messages are preserved
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Hi!"

    def test_split_tool_call_continuation_merges_into_valid_json(self):
        """A continuation entry with empty name/id is merged onto the prior tool call."""
        messages = [
            {"role": "user", "content": "Search please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_split",
                        "type": "function",
                        "function": {
                            "name": "brave_web_search",
                            "arguments": '{"query": ["AI training data scarcity 20',
                        },
                    },
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": '26"], "country": "US"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_split", "content": "results"},
        ]

        result = filter_tool_call_result_pairing(messages, model_id="qwen-3-235b")

        assert len(result) == 3
        assistant = result[1]
        assert len(assistant["tool_calls"]) == 1
        merged_tc = assistant["tool_calls"][0]
        assert merged_tc["id"] == "call_split"
        assert merged_tc["function"]["name"] == "brave_web_search"
        assert (
            merged_tc["function"]["arguments"]
            == '{"query": ["AI training data scarcity 2026"], "country": "US"}'
        )
        # Tool result should still be paired
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_split"

    def test_split_tool_call_continuation_unrecoverable_is_dropped(self):
        """When merging produces invalid JSON, the continuation is dropped."""
        messages = [
            {"role": "user", "content": "Search please"},
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "brave_web_search",
                            "arguments": '{"query": ["unterminated',
                        },
                    },
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": " still bad",
                        },
                    },
                ],
            },
        ]

        result = filter_tool_call_result_pairing(messages, model_id="qwen-3-235b")

        # Continuation merge fails -> continuation dropped.
        # Prior entry alone has invalid JSON -> also dropped by the JSON pass.
        # Assistant message kept (content present) without tool_calls.
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "thinking..."
        assert "tool_calls" not in result[1]

    def test_restarted_json_tool_call_dropped(self):
        """A single tool call whose arguments contain restarted/duplicated JSON is dropped."""
        bad_arguments = (
            '{"memory": "User owns a 1988 Suzuki Samurai with 30x9.50R15 wheels'
            '{"memory": "User owns a 1988 Suzuki Samurai"}'
        )
        messages = [
            {"role": "user", "content": "Remember this"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "call_restart",
                        "type": "function",
                        "function": {
                            "name": "memory_storage_tool",
                            "arguments": bad_arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_restart",
                "content": "stored",
            },
        ]

        result = filter_tool_call_result_pairing(messages, model_id="qwen-3-235b")

        # Bad tool call dropped; assistant message preserved without tool_calls
        # because it still has content; tool result dropped (orphan).
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1].get("tool_calls") is None
        assert result[1]["content"] == "ok"

    def test_truncated_single_tool_call_dropped(self):
        """A single tool call with truncated JSON (no continuation entry) is dropped."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tooluse_TicbBd88bFvFdC42swdNRY",
                        "type": "function",
                        "function": {
                            "name": "brave_web_search",
                            "arguments": '{"query": ["Sypha Run lyrics"], "country',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tooluse_TicbBd88bFvFdC42swdNRY",
                "content": "results",
            },
        ]

        result = filter_tool_call_result_pairing(messages, model_id="qwen-3-235b")

        # Bad tool call dropped; assistant has no content -> message dropped
        # entirely; tool result orphaned -> dropped.
        assert result == []


class TestSanitizeToolCallIdForBedrock:
    """Tests for sanitize_tool_call_id_for_bedrock."""

    @pytest.mark.parametrize(
        "value", ["abc", "tool_123", "tool-call-id", "ABC_def-9", "a"]
    )
    def test_returns_input_when_already_valid(self, value):
        assert sanitize_tool_call_id_for_bedrock(value) == value

    @pytest.mark.parametrize("value", [None, "", 0, 123, []])
    def test_returns_input_when_falsy_or_non_string(self, value):
        assert sanitize_tool_call_id_for_bedrock(value) == value

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("call.123", "call_123"),
            ("call:abc", "call_abc"),
            ("call/abc", "call_abc"),
            ("call abc", "call_abc"),
            ("toolu_01.A:B/C", "toolu_01_A_B_C"),
            ("café", "caf_"),
        ],
    )
    def test_replaces_disallowed_characters_with_underscore(self, raw, expected):
        assert sanitize_tool_call_id_for_bedrock(raw) == expected

    def test_idempotent(self):
        once = sanitize_tool_call_id_for_bedrock("call.with:bad/chars")
        twice = sanitize_tool_call_id_for_bedrock(once)
        assert once == twice

    def test_truncates_overlong_id_to_64_chars(self):
        long_id = "a" * 100
        result = sanitize_tool_call_id_for_bedrock(long_id)
        assert len(result) == 64
        assert result.startswith("a" * 55 + "_")

    def test_truncation_matches_bedrock_charset(self):
        long_id = "call.with:bad/chars." * 10
        result = sanitize_tool_call_id_for_bedrock(long_id)
        assert len(result) <= 64
        assert all(c.isalnum() or c in "_-" for c in result)

    def test_truncation_is_deterministic(self):
        long_id = "toolu_" + "x" * 100
        assert sanitize_tool_call_id_for_bedrock(
            long_id
        ) == sanitize_tool_call_id_for_bedrock(long_id)

    def test_truncation_is_idempotent(self):
        once = sanitize_tool_call_id_for_bedrock("a" * 100)
        twice = sanitize_tool_call_id_for_bedrock(once)
        assert once == twice


class TestSanitizeToolNameForBedrock:
    """Tests for sanitize_tool_name_for_bedrock."""

    @pytest.mark.parametrize(
        "value", ["search", "brave_web_search", "deep-research", "abc123"]
    )
    def test_returns_input_when_already_valid(self, value):
        assert sanitize_tool_name_for_bedrock(value) == value

    @pytest.mark.parametrize("value", [None, "", 0, 42, []])
    def test_returns_input_when_falsy_or_non_string(self, value):
        assert sanitize_tool_name_for_bedrock(value) == value

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("server.tool", "server_tool"),
            ("namespace:tool", "namespace_tool"),
            ("path/to/tool", "path_to_tool"),
            ("with space", "with_space"),
            ("a.b:c/d e", "a_b_c_d_e"),
        ],
    )
    def test_replaces_disallowed_characters_with_underscore(self, raw, expected):
        assert sanitize_tool_name_for_bedrock(raw) == expected

    def test_truncates_overlong_name_to_64_chars(self):
        # e.g. a WebMCP tool name with a site URL appended.
        long_name = "brave_search_https_example_com_some_long_path_" + "x" * 40
        result = sanitize_tool_name_for_bedrock(long_name)
        assert len(result) == 64
        assert all(c.isalnum() or c in "_-" for c in result)

    def test_truncation_applied_after_charset_sanitization(self):
        long_name = "server.tool/" + "a" * 100
        result = sanitize_tool_name_for_bedrock(long_name)
        assert len(result) == 64
        assert all(c.isalnum() or c in "_-" for c in result)


class TestFilterToolCallResultPairingSanitization:
    """The pairing filter must rewrite Bedrock-invalid IDs/names symmetrically
    so that tool_use ↔ tool_result pairing is preserved after sanitization."""

    def test_sanitizes_id_on_both_assistant_tool_call_and_tool_result(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "toolu_01.A:B/C",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_01.A:B/C",
                "content": "result",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        assert len(result) == 3
        assert result[1]["tool_calls"][0]["id"] == "toolu_01_A_B_C"
        assert result[2]["tool_call_id"] == "toolu_01_A_B_C"

    def test_sanitizes_function_name_on_assistant_tool_call(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tool_1",
                        "type": "function",
                        "function": {
                            "name": "server.do:thing",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool_1",
                "content": "ok",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        assert result[0]["tool_calls"][0]["function"]["name"] == "server_do_thing"
        # tool_call_id pairing is preserved (not affected by name change).
        assert result[1]["tool_call_id"] == "tool_1"

    def test_sanitizes_overlong_id_on_both_assistant_tool_call_and_tool_result(self):
        long_id = "toolu_" + "a" * 100
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": long_id,
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": long_id,
                "content": "result",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        sanitized_id = result[0]["tool_calls"][0]["id"]
        assert len(sanitized_id) == 64
        assert result[1]["tool_call_id"] == sanitized_id

    def test_sanitizes_overlong_function_name_on_assistant_tool_call(self):
        long_name = "webmcp_https_example_com_tool_" + "a" * 60
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tool_1",
                        "type": "function",
                        "function": {"name": long_name, "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool_1",
                "content": "ok",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        assert len(result[0]["tool_calls"][0]["function"]["name"]) == 64
        # tool_call_id pairing is preserved (not affected by name change).
        assert result[1]["tool_call_id"] == "tool_1"

    def test_orphan_tool_result_dropped_after_id_sanitization_when_no_match(self):
        messages = [
            {
                "role": "assistant",
                "content": "no tools",
            },
            {
                "role": "tool",
                "tool_call_id": "missing.id",
                "content": "orphan",
            },
        ]

        result = filter_tool_call_result_pairing(messages)

        # Orphan tool result dropped regardless of sanitization.
        assert result == [{"role": "assistant", "content": "no tools"}]


class TestFormatToolsForBedrockSanitizesNames:
    """format_tools_for_bedrock must rewrite tool names so they satisfy
    Bedrock's [a-zA-Z0-9_-]+ constraint on toolUse.name."""

    def test_sanitizes_tool_function_name(self):
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="server.do:thing",
                    description="d",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]

        result = format_tools_for_bedrock(tools)

        assert result[0]["function"]["name"] == "server_do_thing"

    def test_leaves_valid_tool_name_unchanged(self):
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="brave_web_search",
                    description="d",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]

        result = format_tools_for_bedrock(tools)

        assert result[0]["function"]["name"] == "brave_web_search"

    def test_truncates_overlong_tool_function_name(self):
        long_name = "brave_search_https_example_com_some_long_path_" + "x" * 40
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name=long_name,
                    description="d",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]

        result = format_tools_for_bedrock(tools)

        assert len(result[0]["function"]["name"]) == 64
