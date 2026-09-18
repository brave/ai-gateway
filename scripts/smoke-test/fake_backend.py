#!/usr/bin/env python3
"""
Minimal stdlib-only OpenAI-compatible chat completions server, used by
scripts/smoke-test.sh in place of a real model backend (e.g. Ollama).

Bug H (BLOCKER-HIGH) crashes during model *selection*, before any backend
call is made, so a canned response here is enough to prove the full request
path (auth -> triage -> backend call -> response formatting) works end to
end - no real inference needed.

Usage: fake_backend.py [port]  (defaults to 11434, matching .env.example)
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11434


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(
                200,
                {"object": "list", "data": [{"id": "fake-model", "object": "model"}]},
            )
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body, unused

        self._send_json(
            200,
            {
                "id": "chatcmpl-smoke-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "pong",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    def log_message(self, format, *args):
        sys.stderr.write(f"[fake_backend] {self.address_string()} - {format % args}\n")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[fake_backend] listening on :{PORT}", flush=True)
    server.serve_forever()
