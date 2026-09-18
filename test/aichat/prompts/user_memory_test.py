from aichat.prompts.user_memory import TYPE, user_memory


def test_user_memory_formatting():
    """Test that user memory data is properly formatted with <user_memory> tags"""
    memory_data = {
        "name": "John Doe",
        "hobbies": ["reading", "swimming"],
        "location": "San Francisco",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "memory": memory_data,
                },
            ],
        }
    ]

    actual_messages = user_memory.augment(messages)

    assert len(actual_messages) == 1
    assert actual_messages[0]["role"] == "user"
    assert isinstance(actual_messages[0]["content"], list)
    assert len(actual_messages[0]["content"]) == 1

    content = actual_messages[0]["content"][0]
    assert content["type"] == "text"

    # Check for proper formatting with newlines
    text = content["text"]
    assert text.startswith("About this user:\n<user_memory>\n")
    assert text.endswith("</user_memory>\n\n")
    assert "name: John Doe" in text
    assert "hobbies:" in text
    assert "- reading" in text
    assert "- swimming" in text
    assert "location: San Francisco" in text


def test_user_memory_adds_system_instructions():
    """Test that system message gets augmented with user memory instructions"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "memory": {"name": "Alice"},
                },
            ],
        },
    ]

    actual_messages = user_memory.augment(messages)

    assert len(actual_messages) == 2
    system_message = actual_messages[0]
    assert system_message["role"] == "system"
    assert "You are a helpful assistant." in system_message["content"]
    assert "<user_memory>" in system_message["content"]


def test_user_memory_no_augmentation_without_memory():
    """Test that messages without user memory are not augmented"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello!",
                },
            ],
        },
    ]

    actual_messages = user_memory.augment(messages)

    # System message should not be modified
    assert actual_messages[0] == messages[0]
    assert actual_messages[1] == messages[1]


def test_user_memory_escapes_html():
    """Test that user memory values are properly HTML-escaped"""
    memory_data = {
        "note": "User likes <script>alert('XSS')</script>",
        "tags": ["<tag>", "normal tag"],
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "memory": memory_data,
                },
            ],
        }
    ]

    actual_messages = user_memory.augment(messages)
    content_text = actual_messages[0]["content"][0]["text"]

    # HTML should be escaped
    assert "&lt;script&gt;" in content_text
    assert "<script>" not in content_text
    assert "&lt;tag&gt;" in content_text


def test_user_memory_empty_list_omitted():
    """Test that empty lists are not included in the output"""
    memory_data = {
        "name": "Bob",
        "hobbies": [],
        "skills": ["Python"],
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "memory": memory_data,
                },
            ],
        }
    ]

    actual_messages = user_memory.augment(messages)
    content_text = actual_messages[0]["content"][0]["text"]

    # Empty list should not appear
    assert "hobbies:" not in content_text
    # Non-empty list should appear
    assert "skills:" in content_text
    assert "- Python" in content_text


def test_user_memory_mixed_content():
    """Test that user memory works alongside other content parts"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello!",
                },
                {
                    "type": TYPE,
                    "memory": {"name": "Charlie"},
                },
                {
                    "type": "text",
                    "text": "What's the weather?",
                },
            ],
        }
    ]

    actual_messages = user_memory.augment(messages)

    assert len(actual_messages[0]["content"]) == 3
    assert actual_messages[0]["content"][0]["type"] == "text"
    assert actual_messages[0]["content"][0]["text"] == "Hello!"
    assert actual_messages[0]["content"][1]["type"] == "text"
    assert "<user_memory>" in actual_messages[0]["content"][1]["text"]
    assert actual_messages[0]["content"][2]["type"] == "text"
    assert actual_messages[0]["content"][2]["text"] == "What's the weather?"


def test_user_memory_system_with_list_content():
    """Test that system messages with list content get properly augmented"""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "memory": {"name": "Dave"},
                },
            ],
        },
    ]

    actual_messages = user_memory.augment(messages)

    system_message = actual_messages[0]
    assert system_message["role"] == "system"
    assert isinstance(system_message["content"], list)
    assert len(system_message["content"]) == 2
    assert system_message["content"][0]["type"] == "text"
    assert system_message["content"][0]["text"] == "You are a helpful assistant."
    assert system_message["content"][1]["type"] == "text"
    assert "<user_memory>" in system_message["content"][1]["text"]
