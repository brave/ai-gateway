import pytest

from aichat.prompts.page_text import PageTextContentPart
from aichat.protocol.open_ai_protocol import TextContentPart, UserMessage
from aichat.serve.services.dynamic_leo.user_text import (
    extract_last_n_user_plain_text_turns,
    fuse_turns_for_classifier,
    last_message_and_prior_fused,
)


def test_plain_text_turns_exclude_page_payload_parts():
    messages = [
        UserMessage(
            content=[
                PageTextContentPart(type="brave-page-text", text="whole document"),
                TextContentPart(type="text", text="real question"),
            ]
        ),
    ]
    turns = extract_last_n_user_plain_text_turns(messages, n=5)
    assert turns == ["real question"]


def test_string_user_content_is_kept():
    messages = [UserMessage(content="  hi  ")]
    assert extract_last_n_user_plain_text_turns(messages, n=3) == ["hi"]


def test_latest_user_turn_first_when_traversing_backward():
    messages = [
        UserMessage(content="first"),
        UserMessage(content="second"),
        UserMessage(content="third"),
    ]
    assert extract_last_n_user_plain_text_turns(messages, n=2) == ["second", "third"]


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_n_returns_empty(n):
    messages = [UserMessage(content="x")]
    assert extract_last_n_user_plain_text_turns(messages, n=n) == []


def test_fuse_joins_turns():
    assert fuse_turns_for_classifier(["a", "b"]) == "a\n\nb"


def test_last_message_and_prior_fused_single_turn():
    assert last_message_and_prior_fused(["only"]) == ("only", "")


def test_last_message_and_prior_fused_multiple():
    assert last_message_and_prior_fused(["a", "b", "c"]) == ("c", "a\n\nb")


def test_last_message_and_prior_fused_empty():
    assert last_message_and_prior_fused([]) == ("", "")
