from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class ProtocolError(Exception):
    pass


@dataclass
class Request:
    id: int
    op: str
    args: dict[str, Any]

    def encode(self) -> bytes:
        # Guard before json.dumps: a huge string arg (e.g. base64 media) is
        # still fine, but an absurd one would OOM building the payload before
        # the post-dump size check could run.
        for key, value in self.args.items():
            if isinstance(value, str) and len(value) > MAX_MESSAGE_BYTES:
                raise ProtocolError(f"arg '{key}' too large: {len(value)} bytes")
        payload = {"id": self.id, "op": self.op, **self.args}
        data = json.dumps(payload, separators=(",", ":")).encode()
        if len(data) > MAX_MESSAGE_BYTES:
            raise ProtocolError(f"request too large: {len(data)} bytes")
        return data + b"\n"

    @staticmethod
    def decode(line: bytes) -> Request:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"bad request json: {e}") from e
        try:
            return Request(
                id=obj["id"],
                op=obj["op"],
                args={k: v for k, v in obj.items() if k not in ("id", "op")},
            )
        except (KeyError, TypeError) as e:
            raise ProtocolError(f"bad request shape: {e}") from e


@dataclass
class Response:
    id: int
    ok: bool
    value: Any = None
    exc_type: str | None = None
    error: str | None = None

    def encode(self) -> bytes:
        payload: dict[str, Any] = {"id": self.id, "ok": self.ok}
        if self.ok:
            payload["value"] = self.value
        else:
            payload["exc_type"] = self.exc_type
            payload["error"] = self.error
        return json.dumps(payload, separators=(",", ":")).encode() + b"\n"

    @staticmethod
    def decode(line: bytes) -> Response:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"bad response json: {e}") from e
        try:
            return Response(
                id=obj["id"],
                ok=obj["ok"],
                value=obj.get("value"),
                exc_type=obj.get("exc_type"),
                error=obj.get("error"),
            )
        except (KeyError, TypeError) as e:
            raise ProtocolError(f"bad response shape: {e}") from e


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(data: str) -> bytes:
    return base64.b64decode(data)
