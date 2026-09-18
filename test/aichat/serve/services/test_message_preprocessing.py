from aichat.protocol.open_ai_protocol import (
    File,
    FileContentPart,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    UserMessage,
)
from aichat.serve.message_preprocessing import (
    _deduplicate_attachment_content,
    _strip_file_parts,
    _strip_image_parts,
    preprocess_messages,
)
from aichat.serve.services.models import ModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_part(filename: str, data: str) -> FileContentPart:
    return FileContentPart(file=File(filename=filename, file_data=data))


def _image_part(url: str = "data:image/png;base64,AAAA") -> ImageContentPart:
    return ImageContentPart(type="image_url", image_url=ImageUrl(url=url))


def _user(parts) -> UserMessage:
    if isinstance(parts, list):
        return UserMessage(content=parts)
    return UserMessage(content=parts)


def _make_bedrock_config(**overrides) -> ModelConfig:
    defaults = {
        "model_id": "claude-3-haiku",
        "upstream_model": "anthropic.claude-3-haiku-20240307-v1:0",
        "backend": "bedrock",
        "api_base": None,
        "api_key": None,
        "inference_profile": None,
        "system_prompt_support": True,
        "prompt_caching_support": True,
        "prompt_caching_enabled": True,
        "tool_support": True,
        "image_support": True,
        "audio_support": False,
        "video_support": False,
        "file_support": True,
        "friendly_name": "Claude 3 Haiku",
        "maker": "Anthropic",
        "max_tokens": 4096,
        "max_tokens_premium": 4096,
        "conversation_token_limit": 200000,
        "conversation_token_limit_premium": 200000,
        "free": True,
        "key": None,
        "rate_limit": None,
        "rate_limit_interval_seconds": None,
        "max_pages": None,
        "max_pages_premium": None,
        "extra_body": None,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_non_bedrock_config(**overrides) -> ModelConfig:
    defaults = {
        "model_id": "gpt-4o",
        "upstream_model": "gpt-4o",
        "backend": "litellm",
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "inference_profile": None,
        "system_prompt_support": True,
        "prompt_caching_support": None,
        "prompt_caching_enabled": None,
        "tool_support": True,
        "image_support": True,
        "audio_support": False,
        "video_support": False,
        "file_support": True,
        "friendly_name": "GPT-4o",
        "maker": "OpenAI",
        "max_tokens": 4096,
        "max_tokens_premium": 4096,
        "conversation_token_limit": 128000,
        "conversation_token_limit_premium": 128000,
        "free": False,
        "key": None,
        "rate_limit": None,
        "rate_limit_interval_seconds": None,
        "max_pages": None,
        "max_pages_premium": None,
        "extra_body": None,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests for _deduplicate_attachment_content
# ---------------------------------------------------------------------------


class TestDeduplicateAttachmentContent:
    """Tests for _deduplicate_attachment_content."""

    def test_no_messages_returns_empty(self):
        assert _deduplicate_attachment_content([]) == []

    def test_no_file_parts_unchanged(self):
        messages = [
            _user("Hello"),
            _user([TextContentPart(type="text", text="World")]),
        ]
        result = _deduplicate_attachment_content(messages)
        assert result == messages

    def test_single_file_unchanged(self):
        messages = [
            _user([_file_part("report.pdf", "data:application/pdf;base64,AAAA")])
        ]
        result = _deduplicate_attachment_content(messages)
        assert len(result) == 1
        assert len(result[0].content) == 1

    def test_unique_content_unchanged(self):
        messages = [
            _user([_file_part("a.pdf", "data:application/pdf;base64,AAAA")]),
            _user([_file_part("b.pdf", "data:application/pdf;base64,BBBB")]),
        ]
        result = _deduplicate_attachment_content(messages)
        assert len(result) == 2
        assert len(result[0].content) == 1
        assert len(result[1].content) == 1

    def test_duplicate_content_removes_older_occurrence(self):
        """When the same file_data appears twice, the earlier copy is dropped."""
        pdf_data = "data:application/pdf;base64,SAMECONTENT"
        messages = [
            _user([_file_part("first.pdf", pdf_data)]),  # older — should be dropped
            _user([_file_part("second.pdf", pdf_data)]),  # newer — should be kept
        ]
        result = _deduplicate_attachment_content(messages)

        # The older message's file should be removed.
        assert result[0].content == []
        # The newer message's file should be preserved.
        assert len(result[1].content) == 1
        assert result[1].content[0].file.filename == "second.pdf"

    def test_three_occurrences_keeps_only_latest(self):
        pdf_data = "data:application/pdf;base64,REPEATEDDATA"
        messages = [
            _user([_file_part("v1.pdf", pdf_data)]),
            _user([_file_part("v2.pdf", pdf_data)]),
            _user([_file_part("v3.pdf", pdf_data)]),
        ]
        result = _deduplicate_attachment_content(messages)

        assert result[0].content == []
        assert result[1].content == []
        assert len(result[2].content) == 1
        assert result[2].content[0].file.filename == "v3.pdf"

    def test_same_filename_different_content_both_kept(self):
        """Different content under the same filename must both survive."""
        messages = [
            _user([_file_part("doc.pdf", "data:application/pdf;base64,AAAA")]),
            _user([_file_part("doc.pdf", "data:application/pdf;base64,BBBB")]),
        ]
        result = _deduplicate_attachment_content(messages)
        assert len(result[0].content) == 1
        assert len(result[1].content) == 1

    def test_duplicate_mixed_with_other_parts(self):
        """Non-file parts alongside a duplicate file part are left intact."""
        pdf_data = "data:application/pdf;base64,CONTENT"
        messages = [
            _user(
                [
                    TextContentPart(type="text", text="See attached"),
                    _file_part("old.pdf", pdf_data),
                ]
            ),
            _user(
                [
                    TextContentPart(type="text", text="And again"),
                    _file_part("new.pdf", pdf_data),
                ]
            ),
        ]
        result = _deduplicate_attachment_content(messages)

        # Text parts survive; only the older file part is removed.
        msg0_parts = result[0].content
        assert len(msg0_parts) == 1
        assert isinstance(msg0_parts[0], TextContentPart)

        msg1_parts = result[1].content
        assert len(msg1_parts) == 2
        assert isinstance(msg1_parts[0], TextContentPart)
        assert msg1_parts[1].file.filename == "new.pdf"

    def test_multiple_files_per_message_partial_dedup(self):
        """Only the duplicate file is removed; the unique sibling stays."""
        pdf_data = "data:application/pdf;base64,DUPLICATE"
        messages = [
            _user(
                [
                    _file_part("dup.pdf", pdf_data),
                    _file_part("unique.pdf", "data:application/pdf;base64,UNIQUE"),
                ]
            ),
            _user([_file_part("dup-again.pdf", pdf_data)]),
        ]
        result = _deduplicate_attachment_content(messages)

        # Older dup.pdf removed; unique.pdf stays.
        assert len(result[0].content) == 1
        assert result[0].content[0].file.filename == "unique.pdf"
        # Newer copy kept.
        assert len(result[1].content) == 1

    def test_non_list_content_messages_untouched(self):
        """Messages with string content (not a list) are never modified."""
        pdf_data = "data:application/pdf;base64,SAME"
        messages = [
            UserMessage(content="plain string"),
            _user([_file_part("doc.pdf", pdf_data)]),
        ]
        result = _deduplicate_attachment_content(messages)
        assert result[0].content == "plain string"
        assert len(result[1].content) == 1


# ---------------------------------------------------------------------------
# Integration: preprocess_messages only runs content dedup for Bedrock
# ---------------------------------------------------------------------------


class TestPreprocessMessagesDedupIntegration:
    """Verify that content-dedup runs for Bedrock but not for other backends."""

    def test_bedrock_removes_duplicate_content(self):
        pdf_data = "data:application/pdf;base64,SAMEPDF"
        messages = [
            _user([_file_part("first.pdf", pdf_data)]),
            _user([_file_part("second.pdf", pdf_data)]),
        ]
        result = preprocess_messages(messages, _make_bedrock_config())

        # Older file removed; newer kept.
        assert result[0].content == []
        assert len(result[1].content) == 1
        assert result[1].content[0].file.filename == "second.pdf"

    def test_non_bedrock_keeps_duplicate_content(self):
        """For non-Bedrock backends we do NOT deduplicate by content."""
        pdf_data = "data:application/pdf;base64,SAMEPDF"
        messages = [
            _user([_file_part("first.pdf", pdf_data)]),
            _user([_file_part("second.pdf", pdf_data)]),
        ]
        result = preprocess_messages(messages, _make_non_bedrock_config())

        # Both files remain.
        assert len(result[0].content) == 1
        assert len(result[1].content) == 1


# ---------------------------------------------------------------------------
# Tests for _strip_file_parts
# ---------------------------------------------------------------------------


class TestStripFileParts:
    """Tests for _strip_file_parts."""

    def _make_config(self, **overrides) -> ModelConfig:
        defaults = {
            "model_id": "qwen-14b-instruct",
            "upstream_model": "Qwen/Qwen3-14B",
            "backend": "litellm",
            "api_base": "http://vllm-qwen:8000/v1",
            "api_key": None,
            "inference_profile": None,
            "system_prompt_support": True,
            "prompt_caching_support": None,
            "prompt_caching_enabled": None,
            "tool_support": True,
            "image_support": True,
            "audio_support": False,
            "video_support": False,
            "file_support": False,
            "friendly_name": "Qwen 3",
            "maker": "Alibaba Cloud",
            "max_tokens": 8192,
            "max_tokens_premium": 8192,
            "conversation_token_limit": 81920,
            "conversation_token_limit_premium": 81920,
            "free": True,
            "key": None,
            "rate_limit": None,
            "rate_limit_interval_seconds": None,
            "max_pages": None,
            "max_pages_premium": None,
            "extra_body": None,
        }
        defaults.update(overrides)
        return ModelConfig(**defaults)

    def test_no_messages_returns_empty(self):
        assert _strip_file_parts([], self._make_config()) == []

    def test_no_file_parts_unchanged(self):
        messages = [
            _user("Hello"),
            _user([TextContentPart(type="text", text="World")]),
        ]
        result = _strip_file_parts(messages, self._make_config())
        assert result == messages

    def test_single_file_part_replaced_with_text(self):
        messages = [
            _user([_file_part("report.pdf", "data:application/pdf;base64,AAAA")])
        ]
        result = _strip_file_parts(messages, self._make_config())
        assert len(result[0].content) == 1
        part = result[0].content[0]
        assert isinstance(part, TextContentPart)
        assert "report.pdf" in part.text
        assert "does not support file attachments" in part.text

    def test_file_part_replaced_original_text_preserved(self):
        messages = [
            _user(
                [
                    TextContentPart(type="text", text="See attached"),
                    _file_part("doc.pdf", "data:application/pdf;base64,AAAA"),
                ]
            )
        ]
        result = _strip_file_parts(messages, self._make_config())
        assert len(result[0].content) == 2
        assert result[0].content[0].text == "See attached"
        assert isinstance(result[0].content[1], TextContentPart)
        assert "doc.pdf" in result[0].content[1].text

    def test_multiple_file_parts_all_replaced(self):
        messages = [
            _user([_file_part("a.pdf", "data:application/pdf;base64,AAAA")]),
            _user([_file_part("b.pdf", "data:application/pdf;base64,BBBB")]),
        ]
        result = _strip_file_parts(messages, self._make_config())
        assert len(result[0].content) == 1
        assert isinstance(result[0].content[0], TextContentPart)
        assert "a.pdf" in result[0].content[0].text
        assert len(result[1].content) == 1
        assert isinstance(result[1].content[0], TextContentPart)
        assert "b.pdf" in result[1].content[0].text

    def test_string_content_messages_untouched(self):
        messages = [
            UserMessage(content="plain string"),
            _user([_file_part("doc.pdf", "data:application/pdf;base64,AAAA")]),
        ]
        result = _strip_file_parts(messages, self._make_config())
        assert result[0].content == "plain string"
        assert isinstance(result[1].content[0], TextContentPart)
        assert "doc.pdf" in result[1].content[0].text


# ---------------------------------------------------------------------------
# Integration: preprocess_messages strips files when file_support=False
# ---------------------------------------------------------------------------


class TestPreprocessMessagesFileSupport:
    """Verify file stripping via preprocess_messages when file_support=False."""

    def test_file_support_false_replaces_file_parts_with_text(self):
        messages = [
            _user(
                [
                    TextContentPart(type="text", text="Here is a doc"),
                    _file_part("report.pdf", "data:application/pdf;base64,AAAA"),
                ]
            )
        ]
        config = _make_non_bedrock_config(file_support=False)
        result = preprocess_messages(messages, config)

        assert len(result[0].content) == 2
        assert result[0].content[0].text == "Here is a doc"
        assert isinstance(result[0].content[1], TextContentPart)
        assert "report.pdf" in result[0].content[1].text
        assert "does not support file attachments" in result[0].content[1].text

    def test_file_support_true_keeps_file_parts(self):
        messages = [
            _user([_file_part("report.pdf", "data:application/pdf;base64,AAAA")])
        ]
        config = _make_non_bedrock_config(file_support=True)
        result = preprocess_messages(messages, config)

        assert len(result[0].content) == 1


# ---------------------------------------------------------------------------
# Tests for _strip_image_parts
# ---------------------------------------------------------------------------


class TestStripImageParts:
    """Tests for _strip_image_parts."""

    def test_no_messages_returns_empty(self):
        assert _strip_image_parts([]) == []

    def test_no_image_parts_unchanged(self):
        messages = [
            _user("Hello"),
            _user([TextContentPart(type="text", text="World")]),
        ]
        result = _strip_image_parts(messages)
        assert result == messages

    def test_single_image_part_replaced_with_text(self):
        messages = [_user([_image_part()])]
        result = _strip_image_parts(messages)
        assert len(result[0].content) == 1
        part = result[0].content[0]
        assert isinstance(part, TextContentPart)
        assert "does not support image attachments" in part.text

    def test_image_part_replaced_original_text_preserved(self):
        messages = [
            _user(
                [
                    TextContentPart(type="text", text="See attached"),
                    _image_part(),
                ]
            )
        ]
        result = _strip_image_parts(messages)
        assert len(result[0].content) == 2
        assert result[0].content[0].text == "See attached"
        assert isinstance(result[0].content[1], TextContentPart)
        assert "does not support image attachments" in result[0].content[1].text

    def test_multiple_image_parts_all_replaced(self):
        messages = [
            _user(
                [
                    _image_part("data:image/png;base64,AAAA"),
                    _image_part("data:image/png;base64,BBBB"),
                ]
            ),
            _user([_image_part("data:image/png;base64,CCCC")]),
        ]
        result = _strip_image_parts(messages)
        assert len(result[0].content) == 2
        assert all(isinstance(p, TextContentPart) for p in result[0].content)
        assert len(result[1].content) == 1
        assert isinstance(result[1].content[0], TextContentPart)

    def test_string_content_messages_untouched(self):
        messages = [
            UserMessage(content="plain string"),
            _user([_image_part()]),
        ]
        result = _strip_image_parts(messages)
        assert result[0].content == "plain string"
        assert isinstance(result[1].content[0], TextContentPart)

    def test_non_image_parts_preserved_alongside_images(self):
        messages = [
            _user(
                [
                    TextContentPart(type="text", text="before"),
                    _image_part(),
                    _file_part("doc.pdf", "data:application/pdf;base64,AAAA"),
                    TextContentPart(type="text", text="after"),
                ]
            )
        ]
        result = _strip_image_parts(messages)
        parts = result[0].content
        assert len(parts) == 4
        assert parts[0].text == "before"
        assert isinstance(parts[1], TextContentPart)
        assert "does not support image attachments" in parts[1].text
        assert isinstance(parts[2], FileContentPart)
        assert parts[3].text == "after"


# ---------------------------------------------------------------------------
# Integration: preprocess_messages strips images when image_support=False
# ---------------------------------------------------------------------------


class TestPreprocessMessagesImageSupport:
    """Verify image stripping via preprocess_messages when image_support=False."""

    def test_image_support_false_replaces_image_parts_with_text(self):
        messages = [
            _user(
                [
                    TextContentPart(type="text", text="Look at this"),
                    _image_part(),
                ]
            )
        ]
        config = _make_non_bedrock_config(image_support=False)
        result = preprocess_messages(messages, config)

        assert len(result[0].content) == 2
        assert result[0].content[0].text == "Look at this"
        assert isinstance(result[0].content[1], TextContentPart)
        assert "does not support image attachments" in result[0].content[1].text

    def test_image_support_true_keeps_image_parts(self):
        messages = [_user([_image_part()])]
        config = _make_non_bedrock_config(image_support=True)
        result = preprocess_messages(messages, config)

        assert len(result[0].content) == 1
        assert isinstance(result[0].content[0], ImageContentPart)
