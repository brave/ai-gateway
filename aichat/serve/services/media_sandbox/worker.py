from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import traceback

from aichat.serve.services.media_sandbox import landlock, limits, protocol
from aichat.serve.services.media_sandbox.limits import IN_WORKER_WALL_CLOCK_SECONDS

logger = logging.getLogger(__name__)

# Derived at import time (no literal "/tmp") so the per-run scratch parent is
# created under the platform temp dir; the dir itself is private (0700).
SCRATCH_DIR = os.path.join(tempfile.gettempdir(), "media-sandbox")
ENV_ALLOWLIST = ("PATH", "TIKTOKEN_CACHE_DIR", "LC_ALL", "LANG", "HOME", "TMPDIR")
BLOCKED_AUDIT_EVENTS = (
    "socket.connect",
    "socket.bind",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
    "socket.sethostname",
    "subprocess.Popen",
    "os.exec",
    "os.posix_spawn",
    "os.fork",
    "os.system",
    "ctypes.dlopen",
    "ctypes.dlsym",
    "pty.spawn",
)


def install_audit_hook() -> None:
    def _deny(event, args):
        if event in BLOCKED_AUDIT_EVENTS:
            raise PermissionError(f"blocked in sandbox: {event}")

    sys.addaudithook(_deny)


def scrub_environment() -> None:
    for key in list(os.environ):
        if key not in ENV_ALLOWLIST:
            del os.environ[key]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def init_sandbox() -> int | None:
    limits.apply_rlimits()

    os.makedirs(SCRATCH_DIR, mode=0o700, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="run-", dir=SCRATCH_DIR)
    os.environ["TMPDIR"] = scratch
    tempfile.tempdir = scratch
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/data-gym-cache")

    abi = landlock.probe_abi()
    if abi >= 1:
        exists = lambda paths: [p for p in paths if os.path.exists(p)]
        return landlock.apply_landlock(
            read_paths=exists(
                [
                    "/app",
                    "/usr",
                    "/lib",
                    "/lib64",
                    "/etc/ssl",
                    "/tmp",
                    os.environ.get("TIKTOKEN_CACHE_DIR", ""),
                    os.environ.get("HOME", ""),
                ]
            ),
            rw_paths=exists([scratch]),
            abi=abi,
        )

    if os.uname().sysname == "Linux":
        raise landlock.LandlockUnavailable(
            "Landlock mandatory on Linux but kernel does not support it"
        )
    return 0


def _write_error_response(req_id: int, exc_type: str, error: str) -> None:
    sys.stdout.write(
        json.dumps({"id": req_id, "ok": False, "exc_type": exc_type, "error": error})
        + "\n"
    )
    sys.stdout.flush()


def serve(handlers: dict) -> None:
    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            break
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeDecodeError as e:
            _write_error_response(-1, "ProtocolError", f"non-utf8 request: {e}")
            continue
        if not line:
            continue
        try:
            req = protocol.Request.decode(line.encode())
        except protocol.ProtocolError as e:
            _write_error_response(-1, "ProtocolError", str(e))
            continue

        try:
            handler = handlers.get(req.op)
            if handler is None:
                raise KeyError(f"unknown op: {req.op}")
            try:
                limits.arm_op_limits(wall_clock_seconds=IN_WORKER_WALL_CLOCK_SECONDS)
                value = handler(**req.args)
            finally:
                limits.disarm_op_limits()
            resp = protocol.Response(id=req.id, ok=True, value=value)
        except Exception as e:
            resp = protocol.Response(
                id=req.id,
                ok=False,
                exc_type=type(e).__name__,
                error=str(e),
            )
        try:
            sys.stdout.write(resp.encode().decode())
            sys.stdout.flush()
        except BrokenPipeError:
            return


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/data-gym-cache")
    from aichat.serve.services.media_sandbox.ops_pdf import OP_HANDLERS as PDF_HANDLERS
    from aichat.serve.services.media_sandbox.ops_pdf import preload as preload_pdf
    from aichat.serve.services.media_sandbox.ops_stt import OP_HANDLERS as STT_HANDLERS
    from aichat.serve.services.media_sandbox.ops_stt import preload as preload_stt

    preload_pdf()
    preload_stt()
    enable_probes = os.environ.get("MEDIA_SANDBOX_ENABLE_PROBES", "") == "1"
    handlers = {**PDF_HANDLERS, **STT_HANDLERS}
    if enable_probes:
        from aichat.serve.services.media_sandbox.ops_probe import PROBE_HANDLERS

        handlers.update(PROBE_HANDLERS)
    try:
        abi = init_sandbox()
        logger.info("media sandbox ready landlock_abi=%s", abi)
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)  # noqa: B904
    scrub_environment()
    install_audit_hook()
    serve(handlers)


if __name__ == "__main__":
    main()
