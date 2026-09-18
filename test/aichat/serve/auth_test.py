import os

os.environ.setdefault("ENV", "test")

import json

import blake3

from aichat.serve.auth import create_idempotency_key

_MESSAGES = [{"role": "user", "content": "hello"}]
_MODEL = "claude-sonnet-4"


def test_create_idempotency_key_known_payload():
    payload = json.dumps(
        {"messages": _MESSAGES, "model": _MODEL},
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = blake3.blake3(payload.encode("utf-8")).hexdigest()
    assert create_idempotency_key(_MESSAGES, _MODEL) == expected


def test_create_idempotency_key_is_deterministic():
    key_a = create_idempotency_key(_MESSAGES, _MODEL)
    key_b = create_idempotency_key(_MESSAGES, _MODEL)
    assert key_a == key_b
    assert len(key_a) == 64


def test_create_idempotency_key_changes_with_inputs():
    base = create_idempotency_key(_MESSAGES, _MODEL)
    assert base != create_idempotency_key(
        [{"role": "user", "content": "goodbye"}], _MODEL
    )
    assert base != create_idempotency_key(_MESSAGES, "gpt-4")


def test_create_idempotency_key_none_inputs_random():
    key_a = create_idempotency_key(None, None)
    key_b = create_idempotency_key(None, None)
    assert key_a != key_b
    assert key_a.startswith("rand_")
    assert key_b.startswith("rand_")
    raw_hex = key_a.removeprefix("rand_")
    assert len(raw_hex) == 64
    assert len(bytes.fromhex(raw_hex)) == 32
