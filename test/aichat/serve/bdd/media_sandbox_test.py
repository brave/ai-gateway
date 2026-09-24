# BDD coverage for aichat/serve/services/media_sandbox/*.
#
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_ping test/aichat/serve/services/media_sandbox_test.py:93
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_pdf_analyze_blank_pages test/aichat/serve/services/media_sandbox_test.py:98
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_unknown_op_is_op_error test/aichat/serve/services/media_sandbox_test.py:105
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_malformed_pdf_raises_op_error test/aichat/serve/services/media_sandbox_test.py:123
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_env_scrubbed test/aichat/serve/services/media_sandbox_test.py:188
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_mono_16k_decode test/aichat/serve/services/media_sandbox_test.py:215
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_stereo_downmix_and_resample test/aichat/serve/services/media_sandbox_test.py:227
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_compressed_wav_rejected test/aichat/serve/services/media_sandbox_test.py:234
# TODO: remove test/aichat/serve/services/media_sandbox_test.py#test_over_duration_rejected test/aichat/serve/services/media_sandbox_test.py:243

import ast
import base64
import io
import json
import os
import resource
import shutil
import signal
import struct
import sys
import tempfile
import types

import numpy as np
import pytest
from pypdf import PdfReader, PdfWriter
from pytest_bdd import given, parsers, scenarios, then, when

import aichat.serve.services.media_sandbox.pool as pool_mod
from aichat.serve.services.media_sandbox import (
    SandboxOpError,
    SandboxWorkerError,
    get_pool,
    landlock,
    limits,
    ops_pdf,
    ops_probe,
    ops_stt,
    protocol,
    reset_pool,
    worker,
)

FEATURE = "features/media_sandbox.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


def _capture_error(fn, *args, **kwargs):
    try:
        return {"result": fn(*args, **kwargs), "error": None}
    except Exception as e:
        return {"result": None, "error": e}


# ---------- protocol ----------


@when(
    parsers.parse(
        'a request id {req_id:d} op "{op_name}" with args {args_json}'
        " is encoded and decoded"
    )
)
def request_roundtrip(ctx, req_id, op_name, args_json):
    req = protocol.Request(id=req_id, op=op_name, args=_literal(args_json))
    ctx["decoded"] = protocol.Request.decode(req.encode())


def _literal(text: str):
    return ast.literal_eval(text)


@then(
    parsers.parse(
        'the decoded request has id {req_id:d} op "{op_name}" and args {args_json}'
    )
)
def decoded_request_assert(ctx, req_id, op_name, args_json):
    req = ctx["decoded"]
    assert req.id == req_id
    assert req.op == op_name
    assert req.args == _literal(args_json)


@when(parsers.parse("a request is encoded with {defect}"))
def request_encode_defect(ctx, defect):
    if defect == "a huge string argument":
        req = protocol.Request(id=1, op="op", args={"k": "x" * (64 * 1024 * 1024 + 1)})
    else:
        req = protocol.Request(id=1, op="op", args={"k": ["x" * (68 * 1024 * 1024)]})
    try:
        req.encode()
        ctx["error"] = None
    except protocol.ProtocolError as e:
        ctx["error"] = e


@then("encoding raises ProtocolError")
def encoding_raises(ctx):
    assert isinstance(ctx["error"], protocol.ProtocolError)


@when(parsers.parse("a request line {line_desc} is decoded"))
def request_decode_line(ctx, line_desc):
    line = b"{not json" if line_desc == "that is invalid json" else b'{"op":"x"}'
    try:
        protocol.Request.decode(line)
        ctx["error"] = None
    except protocol.ProtocolError as e:
        ctx["error"] = e


@then("decoding raises ProtocolError")
def decoding_raises(ctx):
    assert isinstance(ctx["error"], protocol.ProtocolError)


@when("an ok response and an error response are encoded and decoded")
def response_roundtrip(ctx):
    ok = protocol.Response(id=3, ok=True, value={"pong": True})
    err = protocol.Response(id=4, ok=False, exc_type="ValueError", error="boom")
    ctx["ok_decoded"] = protocol.Response.decode(ok.encode())
    ctx["err_decoded"] = protocol.Response.decode(err.encode())


@then("both round trip with their values intact")
def responses_intact(ctx):
    ok, err = ctx["ok_decoded"], ctx["err_decoded"]
    assert (ok.id, ok.ok, ok.value) == (3, True, {"pong": True})
    assert (err.id, err.ok, err.exc_type, err.error) == (4, False, "ValueError", "boom")


@when(parsers.parse("a response line {line_desc} is decoded"))
def response_decode_line(ctx, line_desc):
    line = b"{not json" if line_desc == "that is invalid json" else b'{"ok": true}'
    try:
        protocol.Response.decode(line)
        ctx["error"] = None
    except protocol.ProtocolError as e:
        ctx["error"] = e


@when("bytes are base64 encoded and decoded")
def b64_roundtrip(ctx):
    payload = bytes(range(256))
    ctx["decoded"] = protocol.b64_decode(protocol.b64_encode(payload))
    ctx["payload"] = payload


@then("the bytes match the original")
def b64_matches(ctx):
    assert ctx["decoded"] == ctx["payload"]


# ---------- limits ----------


@when("the SIGALRM handler fires")
def sigalrm_fires(ctx):
    try:
        limits._on_sigalrm(signal.SIGALRM, None)
        ctx["error"] = None
    except Exception as e:
        ctx["error"] = e


@then("the raised error is OpWallClockExceeded")
def raised_op_wall_clock(ctx):
    from aichat.serve.services.media_sandbox.limits import OpWallClockExceeded

    assert isinstance(ctx["error"], OpWallClockExceeded)


@when(parsers.parse("rlimits are applied {limit_desc}"))
def apply_rlimits_desc(ctx, limit_desc, monkeypatch):
    recorded = []
    if limit_desc == "with soft limits below hard limits":
        monkeypatch.setattr(
            limits.resource, "getrlimit", lambda res: (1024, resource.RLIM_INFINITY)
        )
        monkeypatch.setattr(
            limits.resource, "setrlimit", lambda res, pair: recorded.append((res, pair))
        )
        monkeypatch.setattr(
            limits.signal, "signal", lambda *a, **k: recorded.append(("signal", a))
        )
        limits.apply_rlimits()
    else:
        monkeypatch.setattr(
            limits.resource, "getrlimit", lambda res: (1024, resource.RLIM_INFINITY)
        )

        def _boom(res, pair):
            raise ValueError("unsupported")

        monkeypatch.setattr(limits.resource, "setrlimit", _boom)
        # Pin the platform classification so the fallback branch is exercised
        # the same way on linux and non-linux hosts.
        monkeypatch.setattr(limits.sys, "platform", "darwin")
        limits.apply_rlimits()
    ctx["recorded"] = recorded


@then(parsers.parse("the limit application {outcome}"))
def limit_application_outcome(ctx, outcome):
    if outcome == "sets the new soft and hard limits":
        res_names = {pair[0] for pair in ctx["recorded"]}
        assert resource.RLIMIT_AS in res_names
        assert resource.RLIMIT_FSIZE in res_names
        assert resource.RLIMIT_NOFILE in res_names
        assert resource.RLIMIT_CPU in res_names
    else:
        assert ctx["recorded"] == []


@when(parsers.parse("op limits are armed for {cpu:d} cpu seconds"))
def arm_op_limits(ctx, cpu, monkeypatch):
    recorded = {"alarm": [], "signal": []}
    monkeypatch.setattr(
        limits.resource,
        "getrusage",
        lambda who: type("U", (), {"ru_utime": 1.0, "ru_stime": 2.0})(),
    )
    monkeypatch.setattr(
        limits.resource, "getrlimit", lambda res: (1024, resource.RLIM_INFINITY)
    )
    monkeypatch.setattr(
        limits.resource,
        "setrlimit",
        lambda res, pair: recorded.setdefault("rlimit", (res, pair)),
    )
    monkeypatch.setattr(
        limits.signal, "signal", lambda *a: recorded["signal"].append(a)
    )
    monkeypatch.setattr(limits.signal, "alarm", lambda s: recorded["alarm"].append(s))
    limits.arm_op_limits(cpu_seconds=cpu)
    ctx["recorded"] = recorded


@then(parsers.parse("the alarm is set to {seconds:d} seconds"))
def alarm_set(ctx, seconds):
    assert ctx["recorded"]["alarm"] == [seconds]


@then("disarming clears the alarm")
def disarm_clears_alarm(monkeypatch):
    cleared = []
    monkeypatch.setattr(limits.signal, "alarm", lambda s: cleared.append(s))
    limits.disarm_op_limits()
    assert cleared == [0]


# ---------- worker ----------


@when("the worker environment is scrubbed")
def scrub_env(ctx):
    os.environ.setdefault("PATH", "/usr/bin:/bin")
    os.environ["SANDBOX_EVIL_VAR"] = "1"
    saved = dict(os.environ)
    try:
        worker.scrub_environment()
        ctx["env_now"] = dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved)


@then("only allowlisted keys remain and bytecode writing is disabled")
def scrub_env_assert(ctx):
    env = ctx["env_now"]
    assert "SANDBOX_EVIL_VAR" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    allowed = set(worker.ENV_ALLOWLIST) | {"PYTHONDONTWRITEBYTECODE"}
    assert set(env) <= allowed


@when(parsers.parse("an error response is written for request {req_id:d}"))
def write_error_response(ctx, req_id, capsys):
    worker._write_error_response(req_id, "ValueError", "boom")
    ctx["out"] = capsys.readouterr().out


@then(parsers.parse("the output line is the json error payload for request {req_id:d}"))
def error_payload_assert(ctx, req_id):
    assert (
        ctx["out"]
        == json.dumps(
            {"id": req_id, "ok": False, "exc_type": "ValueError", "error": "boom"}
        )
        + "\n"
    )


@given("the sandbox harness")
def sandbox_harness(ctx):
    return ctx


class _FakeStdin:
    def __init__(self, lines):
        self._lines = lines
        self._i = 0
        self.buffer = self

    def readline(self):
        if self._i >= len(self._lines):
            return b""
        line = self._lines[self._i]
        self._i += 1
        return line


@given(parsers.parse('a serve loop with handler for "{op}"'))
def serve_loop(ctx, op):
    ctx["handlers"] = {op: lambda **kwargs: {"pong": True}}


@when(parsers.parse("the input line {line_desc} is processed"))
def serve_process_line(ctx, line_desc, monkeypatch, capsys):
    lines = {
        "is a valid ping request": [b'{"id": 5, "op": "ping"}\n'],
        "names an unknown op": [b'{"id": 6, "op": "nope"}\n'],
        "is invalid json": [b"{bad json\n"],
        "is non-utf8 bytes": [b"\xff\xfe\x00bad\n"],
        "is blank": [b"\n"],
        "reaches end of stream": [],
    }[line_desc]
    monkeypatch.setattr(limits, "arm_op_limits", lambda **k: None)
    monkeypatch.setattr(limits, "disarm_op_limits", lambda: None)
    fake_stdin = _FakeStdin(lines)
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    worker.serve(ctx["handlers"])
    ctx["out"] = capsys.readouterr().out.strip()


@then(parsers.parse("the serve loop {outcome}"))
def serve_outcome(ctx, outcome):
    if outcome == "replies with a pong payload":
        assert json.loads(ctx["out"]) == {"id": 5, "ok": True, "value": {"pong": True}}
    elif outcome == "replies with a KeyError error response":
        payload = json.loads(ctx["out"])
        assert payload["ok"] is False
        assert payload["exc_type"] == "KeyError"
    elif outcome == "replies with a ProtocolError for id -1":
        payload = json.loads(ctx["out"])
        assert payload["id"] == -1
        assert payload["exc_type"] == "ProtocolError"
    elif outcome in ("produces no output and continues", "exits the loop"):
        assert ctx["out"] == ""
    else:
        raise AssertionError(f"unknown serve outcome: {outcome}")


@when("the sandbox initializes on non-linux")
def init_sandbox_non_linux(ctx, monkeypatch):
    monkeypatch.setattr(limits, "apply_rlimits", lambda: None)
    monkeypatch.setattr(worker.landlock, "probe_abi", lambda: 0)
    # Pin the platform so the non-linux guard is exercised on linux hosts too.
    monkeypatch.setattr(
        worker.os, "uname", lambda: types.SimpleNamespace(sysname="Darwin")
    )
    saved_tempdir = tempfile.tempdir
    saved_tmpdir = os.environ.get("TMPDIR")
    monkeypatch.setenv("TMPDIR", saved_tmpdir or "")
    try:
        ctx["abi"] = worker.init_sandbox()
        ctx["scratch"] = os.environ.get("TMPDIR")
    finally:
        scratch = os.environ.get("TMPDIR")
        if scratch and scratch.startswith(worker.SCRATCH_DIR):
            shutil.rmtree(scratch, ignore_errors=True)
        tempfile.tempdir = saved_tempdir


@then("initialization returns 0 without applying landlock")
def init_sandbox_assert(ctx):
    assert ctx["abi"] == 0
    assert ctx["scratch"] and ctx["scratch"].startswith(worker.SCRATCH_DIR)
    shutil.rmtree(worker.SCRATCH_DIR, ignore_errors=True)


# ---------- op_stt_decode ----------


def _wav_bytes(
    fmt_tag=1,
    channels=1,
    rate=16000,
    sampwidth=2,
    bits=16,
    data=b"",
    blockalign=None,
    declared_data_size=None,
):
    if blockalign is None:
        blockalign = channels * sampwidth
    byte_rate = rate * blockalign
    fmt = struct.pack("<HHIIHH", fmt_tag, channels, rate, byte_rate, blockalign, bits)
    data_size = declared_data_size if declared_data_size is not None else len(data)
    chunk = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", data_size)
        + data
    )
    return b"RIFF" + struct.pack("<I", len(chunk)) + chunk


def _decode_wav(payload: bytes, max_duration: float):
    return _guarded_call(
        ops_stt.op_stt_decode,
        audio_b64=base64.b64encode(payload).decode("ascii"),
        max_duration_seconds=max_duration,
    )


def _guarded_call(fn, **kwargs):
    try:
        return {"result": fn(**kwargs), "error": None}
    except Exception as e:
        return {"result": None, "error": e}


@when(parsers.parse("a WAV payload {wav_desc} is decoded"))
def decode_wav(ctx, wav_desc, monkeypatch):
    max_duration = 1.0 if "duration" in wav_desc else 600.0
    if wav_desc == "is 16k mono int16":
        data = (np.zeros(16000, dtype=np.int16) + 100).tobytes()
        payload = _wav_bytes(data=data)
    elif wav_desc == "is stereo 8k":
        frames = np.tile(np.array([100, -100], dtype=np.int16), 8000)
        payload = _wav_bytes(channels=2, rate=8000, bits=16, data=frames.tobytes())
    elif wav_desc == "is 8-bit unsigned":
        data = bytes([0, 64, 128, 192, 255] * 3200)
        payload = _wav_bytes(sampwidth=1, bits=8, data=data)
    elif wav_desc == "is 32-bit signed":
        frames = np.array([0, 2147483647, -2147483648, 1] * 4000, dtype=np.int32)
        payload = _wav_bytes(sampwidth=4, bits=32, data=frames.tobytes())
    elif wav_desc == "is compressed":
        payload = _wav_bytes(sampwidth=1, bits=8, data=b"\x00" * 64)
        # Real ULAW bytes are unreadable by the stdlib wave reader on this
        # Python (audioop removed), so surface the comptype guard directly.
        monkeypatch.setattr(ops_stt.wave.Wave_read, "getcomptype", lambda self: "ULAW")
    elif wav_desc == "exceeds the duration limit":
        data = (np.zeros(32000, dtype=np.int16)).tobytes()
        payload = _wav_bytes(data=data)
    elif wav_desc == "has an invalid framerate":
        payload = _wav_bytes(rate=0, data=b"\x00" * 100)
    elif wav_desc == "has a corrupt stereo frame buffer":
        # Header declares 8 bytes of stereo frames but only 3 are present:
        # the partial read yields a non-channel-aligned sample buffer.
        payload = _wav_bytes(
            channels=2,
            sampwidth=1,
            bits=8,
            data=b"\x00\x80\x80",
            blockalign=2,
            declared_data_size=8,
        )
    elif wav_desc == "has an unsupported sample width":
        payload = _wav_bytes(sampwidth=3, bits=24, data=b"\x00" * 30)
    elif wav_desc == "is empty after decoding":
        payload = _wav_bytes(data=b"")
    else:
        payload = b"RIFFnotawavefile"
    ctx["decode"] = _decode_wav(payload, max_duration)


@then(parsers.parse("the decode {outcome}"))
def decode_outcome(ctx, outcome):
    res, err = ctx["decode"]["result"], ctx["decode"]["error"]
    if outcome == "returns 16000 samples at the target rate":
        assert res["n_samples"] == 16000 and res["sample_rate"] == 16000
    elif outcome == "downmixes and resamples to 16k":
        assert res["n_samples"] == 16000
    elif (
        outcome == "centers the samples around zero"
        or outcome == "normalizes int32 samples"
    ):
        pcm = np.frombuffer(base64.b64decode(res["pcm_b64"]), dtype=np.float32)
        assert res["n_samples"] == 16000
        assert float(np.abs(pcm).max()) <= 1.0
    elif outcome == "rejects compressed WAVE formats":
        assert "compressed WAVE formats" in str(err)
    elif outcome == "rejects with the duration error":
        assert "exceeds maximum" in str(err)
    elif outcome == "rejects with invalid WAV parameters":
        assert "Invalid WAV parameters" in str(err)
    elif outcome == "rejects with corrupt WAV frame buffer":
        assert "Corrupt WAV frame buffer" in str(err)
    elif outcome == "rejects with the sample width error":
        assert "Unsupported WAV sample width" in str(err)
    elif outcome == "rejects with empty audio":
        assert "Empty audio" in str(err)
    elif outcome == "rejects with the PCM WAV support message":
        assert "Only PCM WAV is supported" in str(err)
    else:
        raise AssertionError(f"unknown decode outcome: {outcome}")


# ---------- op_pdf_analyze ----------


class _FakeEncoding:
    def encode(self, text: str):
        return [0] * len(text.split())


@given(parsers.parse("a pdf document {pdf_desc}"))
def make_pdf(ctx, pdf_desc):
    ctx["pdf_desc"] = pdf_desc
    writer = PdfWriter()
    if pdf_desc == "with 2 blank pages":
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
    elif pdf_desc == "with 90 blank pages":
        for _ in range(90):
            writer.add_blank_page(width=612, height=792)
    else:
        for _ in range(3):
            writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    ctx["pdf_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")


@when(parsers.parse("the pdf is analyzed with budget {budget:d}"))
def analyze_sandbox_pdf(ctx, budget, monkeypatch):
    monkeypatch.setattr(ops_pdf, "_get_encoding", lambda name: _FakeEncoding())
    if "native truncation" in ctx.get("pdf_desc", "") or budget == 20:
        monkeypatch.setattr(
            ops_pdf,
            "_safe_extract_page_text",
            lambda reader, page_num: f"page {page_num + 1} content",
        )
        monkeypatch.setattr(ops_pdf, "_tokens_for_page", lambda r, n, e: 10)
    ctx["analysis"] = _guarded_call(
        ops_pdf.op_pdf_analyze,
        pdf_b64=ctx["pdf_b64"],
        encoding_name="o200k_base",
        max_extraction_tokens=budget,
        max_allowed_pages=85,
    )


@then(parsers.parse("the analysis {outcome}"))
def analysis_outcome(ctx, outcome):
    res, err = ctx["analysis"]["result"], ctx["analysis"]["error"]
    assert err is None
    if outcome == "reports 2 pages under the budget":
        assert res["total_pages"] == 2
        assert res["estimated_tokens"] == 0
        assert "native_page_limit" not in res
    elif outcome == "extracts all text over the page limit":
        assert res["total_pages"] == 90
        assert "all_text" in res
    elif outcome == "reports the encrypted passthrough":
        assert res == {"passthrough": "encrypted"}
    elif outcome == "truncates to native pages with overflow text":
        assert res["native_page_limit"] == 2
        truncated = PdfReader(io.BytesIO(base64.b64decode(res["truncated_pdf_b64"])))
        assert len(truncated.pages) == 2
        assert "Page 3" in res["overflow_text"]
    else:
        raise AssertionError(f"unknown analysis outcome: {outcome}")


@when("the pdf analysis is asked about an encrypted document")
def analyze_encrypted(ctx):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)
    ctx["analysis"] = _guarded_call(
        ops_pdf.op_pdf_analyze,
        pdf_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
        encoding_name="o200k_base",
        max_extraction_tokens=100,
        max_allowed_pages=85,
    )


@then("the analysis reports the encrypted passthrough")
def encrypted_passthrough(ctx):
    assert ctx["analysis"]["result"] == {"passthrough": "encrypted"}


# ---------- probes / landlock ----------


@when(parsers.parse("the probe op {op} runs"))
def run_probe(ctx, op, monkeypatch):
    if op == "probe_env":
        ctx["probe"] = ops_probe.op_probe_env()
    elif op == "probe_fs_write":
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        ctx["probe"] = ops_probe.op_probe_fs_write()
    else:
        ctx["probe"] = ops_probe.op_spin_cpu(seconds=0)


@then(parsers.parse("the probe result {outcome}"))
def probe_outcome(ctx, outcome):
    res = ctx["probe"]
    if outcome == "lists the sorted environment keys":
        assert res == {"env_keys": sorted(os.environ.keys())}
    elif outcome == "skips the write check on non-linux":
        assert res == {"write_denied": None, "skipped": "non-linux"}
    else:
        assert res["spun"] == 0.0


@when("the landlock ABI is probed on non-linux")
def probe_landlock_abi(ctx, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    ctx["abi"] = landlock.probe_abi()


@then("the ABI is 0")
def abi_zero(ctx):
    assert ctx["abi"] == 0


@when("landlock is applied on non-linux")
def apply_landlock_non_linux(ctx, monkeypatch):
    # Pin the platform: on a real Linux host this call would apply a
    # process-wide landlock restriction to the pytest process itself.
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    try:
        landlock.apply_landlock([], [])
        ctx["error"] = None
    except Exception as e:
        ctx["error"] = e


@then("it raises LandlockUnavailable")
def landlock_unavailable(ctx):
    from aichat.serve.services.media_sandbox.landlock import LandlockUnavailable

    assert isinstance(ctx["error"], LandlockUnavailable)


# ---------- pool ----------


class _FakePoolWorker:
    def __init__(self, error=None, result=None):
        self.error = error
        self.result = result
        self.calls = 0

    async def call(self, op, args):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@when("the pool singleton is requested twice")
def request_pool_singleton(ctx):
    ctx["pool_a"] = get_pool()
    ctx["pool_b"] = get_pool()


@then("both requests return the same pool instance")
def pool_singleton_assert(ctx):
    try:
        assert ctx["pool_a"] is ctx["pool_b"]
    finally:
        reset_pool()


@when("both pool workers fail hard during a call")
def pool_all_workers_fail(ctx):
    pool = pool_mod.MediaSandboxPool(size=2)
    pool._workers = [
        _FakePoolWorker(error=SandboxWorkerError("dead")),
        _FakePoolWorker(error=SandboxWorkerError("dead")),
    ]
    pool._started = True
    try:
        result = _guarded_async(pool.call, "pdf_analyze", {})
    finally:
        reset_pool()
    ctx["result"] = result


@then(parsers.parse('the call raises SandboxWorkerError "{message}"'))
def call_raises_worker_error(ctx, message):
    assert isinstance(ctx["result"]["error"], SandboxWorkerError)
    assert str(ctx["result"]["error"]) == message


@when("the first pool worker reports an op error")
def pool_op_error(ctx):
    pool = pool_mod.MediaSandboxPool(size=2)
    first = _FakePoolWorker(error=SandboxOpError("op failed in sandbox: ValueError: x"))
    second = _FakePoolWorker(result={})
    pool._workers = [first, second]
    pool._started = True
    ctx["result"] = _guarded_async(pool.call, "pdf_analyze", {})
    ctx["second_calls"] = second.calls


@then("the call raises SandboxOpError")
def call_raises_op_error(ctx):
    assert isinstance(ctx["result"]["error"], SandboxOpError)
    assert ctx["second_calls"] == 0


def _guarded_async(coro_fn, *args, **kwargs):
    import asyncio

    try:
        return {"result": asyncio.run(coro_fn(*args, **kwargs)), "error": None}
    except Exception as e:
        return {"result": None, "error": e}


@when("a pool worker dies and the next call succeeds after restart")
def pool_worker_restart(ctx, monkeypatch):
    w = pool_mod._Worker()
    w.alive = False

    async def fake_start():
        w.alive = True

    async def fake_roundtrip(op, args, timeout):
        return {"restarted": True}

    monkeypatch.setattr(w, "start", fake_start)
    monkeypatch.setattr(w, "_roundtrip", fake_roundtrip)
    ctx["restart_result"] = _guarded_async(w.call, "pdf_analyze", {})

    w2 = pool_mod._Worker()
    w2.alive = False

    async def failing_start():
        raise ConnectionResetError("nope")

    monkeypatch.setattr(w2, "start", failing_start)
    ctx["start_error"] = _guarded_async(w2.call, "pdf_analyze", {})


@then("the call result is the restarted worker payload")
def restart_result_assert(ctx):
    assert ctx["restart_result"] == {"result": {"restarted": True}, "error": None}


@then("a start failure raises SandboxWorkerError")
def start_failure_assert(ctx):
    err = ctx["start_error"]["error"]
    assert isinstance(err, SandboxWorkerError)
    assert "failed to restart sandbox worker" in str(err)


@when("a worker call exceeds its wall clock")
def worker_call_timeout(ctx, monkeypatch):
    w = pool_mod._Worker()
    w.alive = True

    async def slow_roundtrip(op, args, timeout):
        raise TimeoutError()

    monkeypatch.setattr(w, "_roundtrip", slow_roundtrip)
    ctx["timeout_result"] = _guarded_async(w.call, "pdf_analyze", {})
    ctx["timeout_worker"] = w

    w2 = pool_mod._Worker()
    w2.alive = True

    async def dropped_roundtrip(op, args, timeout):
        raise ConnectionResetError("pipe broke")

    monkeypatch.setattr(w2, "_roundtrip", dropped_roundtrip)
    ctx["drop_result"] = _guarded_async(w2.call, "pdf_analyze", {})
    ctx["drop_worker"] = w2


@then("the worker is killed and raises SandboxWorkerError")
def worker_timeout_assert(ctx):
    err = ctx["timeout_result"]["error"]
    assert isinstance(err, SandboxWorkerError)
    assert "wall clock" in str(err)
    assert ctx["timeout_worker"].alive is False


@then("a transport drop marks the worker dead and raises SandboxWorkerError")
def worker_transport_drop_assert(ctx):
    err = ctx["drop_result"]["error"]
    assert isinstance(err, SandboxWorkerError)
    assert "worker died during op" in str(err)
    assert ctx["drop_worker"].alive is False


@when("the pool is shut down and reset")
def pool_shutdown_and_reset(ctx):
    pool = pool_mod.MediaSandboxPool()
    killed = []

    class _Killable:
        async def kill(self):
            killed.append(True)

    pool._workers = [_Killable()]
    pool._started = True
    pool_mod._pool = pool
    ctx["killed_before"] = killed
    _guarded_async(pool.shutdown)
    reset_pool()


@then("the pool is empty and the singleton is cleared")
def pool_shutdown_assert(ctx):
    assert pool_mod._pool is None
    assert ctx["killed_before"] == [True]
