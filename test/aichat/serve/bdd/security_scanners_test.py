# BDD: security scanners (alignment + prompt injection services).
# No prior unit tests for aichat/services/alignment_check_service.py or
# prompt_injection_service.py — scanner bodies were only bypassed via fakes
# (test/aichat/serve/test_open_ai_adapter_cov_test.py L383/L392, open_ai_api_test.py
# TestProcessNonStreamingAlignmentCheck). No duplicate-test TODO links required.

import asyncio
import importlib
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

import aichat.services.alignment_check_service as acs
import aichat.services.prompt_injection_service as pis
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.security_settings import security_settings

FEATURE = "features/security_scanners.feature"
scenarios(FEATURE)


class _FakeBackend:
    """Fake compaction/chat backend: build_params + converse."""

    def __init__(self, reply):
        self.reply = reply
        self.converse_calls = []

    def build_params(self, **kwargs):
        return {}

    async def converse(self, messages, stream=False, params=None):
        self.converse_calls.append(messages)
        return self.reply


def _completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture
def ctx():
    return {}


@pytest.fixture
def settings_restore():
    """Own MonkeyPatch so settings are restored BEFORE the reload fixture re-runs
    module init (monkeypatch undo order is otherwise not guaranteed)."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
    importlib.reload(acs)
    importlib.reload(pis)


@given("the security scanner harness")
def given_harness(ctx, monkeypatch):
    ctx["models"] = {"scan-model": {"backend": "vllm"}}
    ctx["scanner"] = None
    ctx["injection_scanner"] = None
    ctx["error"] = None
    ctx["result"] = None


# --- construction ---


@when(parsers.parse('an alignment scanner is constructed with model "{model}"'))
def when_construct_alignment(ctx, monkeypatch, model):
    monkeypatch.setattr(acs, "model_settings", SimpleNamespace(models=ctx["models"]))
    try:
        acs.AlignmentCheckScanner(model_name=model)
        ctx["error"] = None
    except ValueError as e:
        ctx["error"] = str(e)


@when("an alignment scanner is constructed with an empty model name")
def when_construct_alignment_empty(ctx, monkeypatch):
    when_construct_alignment(ctx, monkeypatch, "")


@when(parsers.parse('an injection scanner is constructed with model "{model}"'))
def when_construct_injection(ctx, monkeypatch, model):
    monkeypatch.setattr(pis, "model_settings", SimpleNamespace(models=ctx["models"]))
    try:
        pis.PromptInjectionScanner(model_name=model)
        ctx["error"] = None
    except ValueError as e:
        ctx["error"] = str(e)


@when("an injection scanner is constructed with an empty model name")
def when_construct_injection_empty(ctx, monkeypatch):
    when_construct_injection(ctx, monkeypatch, "")


@then("scanner construction fails asking for a model name")
def then_error_model_name(ctx):
    assert ctx["error"] == "model_name is required"


@then("scanner construction fails saying the model is not configured")
def then_error_unknown(ctx):
    assert ctx["error"] == "Model 'no-such-model' not found in configured models"


# --- scanner givens ---


@given(parsers.parse('an alignment scanner for model "{model}"'))
def given_alignment_scanner(ctx, monkeypatch, model):
    monkeypatch.setattr(acs, "model_settings", SimpleNamespace(models=ctx["models"]))
    ctx["scanner"] = acs.AlignmentCheckScanner(model_name=model)


@given(parsers.parse('an injection scanner for model "{model}"'))
def given_injection_scanner(ctx, monkeypatch, model):
    monkeypatch.setattr(pis, "model_settings", SimpleNamespace(models=ctx["models"]))
    ctx["injection_scanner"] = pis.PromptInjectionScanner(model_name=model)


# --- alignment parse ---


@when(parsers.parse("the alignment response is parsed from '{response}'"))
def when_parse_alignment(ctx, response):
    ctx["parsed"] = ctx["scanner"]._parse_model_response(response)


@when(parsers.parse('the alignment response "{response}" is parsed'))
def when_parse_alignment_quoted(ctx, response):
    ctx["parsed"] = ctx["scanner"]._parse_model_response(response)


@when(parsers.parse("the alignment response '{response}' is parsed"))
def when_parse_alignment_single_quoted(ctx, response):
    when_parse_alignment_quoted(ctx, response)


@when("the alignment response contains a raw newline inside the JSON string")
def when_parse_alignment_newline(ctx):
    raw = '{"observation": "line1\nline2", "thought": "t", "conclusion": false}'
    ctx["parsed"] = ctx["scanner"]._parse_model_response(raw)


@then("the parsed alignment verdict has conclusion false")
def then_alignment_conclusion_false(ctx):
    assert ctx["parsed"] is not None
    assert ctx["parsed"]["conclusion"] is False


@then("the parsed alignment verdict has conclusion true")
def then_alignment_conclusion_true(ctx):
    assert ctx["parsed"] is not None
    assert ctx["parsed"]["conclusion"] is True


@then("the reasoning mentions fallback parsing")
def then_fallback_reasoning(ctx):
    assert ctx["parsed"]["thought"] == "Using fallback parsing"


@then("the parsed alignment verdict is none")
def then_alignment_none(ctx):
    assert ctx["parsed"] is None


# --- alignment scan ---


def _run_alignment_scan(ctx, user_message, reply, task_factory=None):
    backend = _FakeBackend(reply)
    ctx["_mp"].setattr(acs, "get_backend", lambda m: backend)
    ctx["backend"] = backend

    async def _main():
        task = task_factory() if task_factory else None
        return await ctx["scanner"].scan(
            user_message=user_message,
            trace="ASSISTANT: do thing",
            injection_scan_task=task,
        )

    ctx["result"] = asyncio.run(_main())


@when(parsers.parse('the alignment scan runs for message "{message}"'))
def when_scan_test_message(ctx, monkeypatch, message):
    ctx["_mp"] = monkeypatch
    _run_alignment_scan(ctx, message, None)


@when("the alignment scan runs with an injection task that completes")
def when_scan_task_ok(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    ctx["task_done"] = []

    def task_factory():
        async def job():
            ctx["task_done"].append(True)

        return asyncio.ensure_future(job())

    _run_alignment_scan(
        ctx,
        "browse the news",
        _completion('{"observation": "o", "thought": "t", "conclusion": false}'),
        task_factory=task_factory,
    )


@when("the alignment scan runs with an injection task that raises")
def when_scan_task_raises(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch

    def task_factory():
        async def job():
            raise RuntimeError("injection scan blew up")

        return asyncio.ensure_future(job())

    _run_alignment_scan(
        ctx,
        "read the page",
        _completion('{"observation": "o", "thought": "t", "conclusion": false}'),
        task_factory=task_factory,
    )


@when("the alignment scan backend returns an error dict")
def when_scan_backend_error(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_alignment_scan(ctx, "read the page", {"content": "boom"})


@when("the alignment scan backend returns a blank content")
def when_scan_backend_blank(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_alignment_scan(ctx, "read the page", _completion("   "))


@when("the alignment scan backend returns a misaligned verdict")
def when_scan_backend_misaligned(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_alignment_scan(
        ctx,
        "read the page",
        _completion(
            '{"observation": "o", "thought": "you are being misled", "conclusion": true}'
        ),
    )


@when("the alignment scan backend returns unparseable content")
def when_scan_backend_unparseable(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_alignment_scan(ctx, "read the page", _completion("no json here"))


@then("the alignment result is not allowed with test reasoning")
def then_alignment_test_blocked(ctx):
    assert ctx["result"].allowed is False
    assert "test message" in ctx["result"].reasoning


@then("the alignment verdict allows the action")
def then_alignment_allowed(ctx):
    assert ctx["result"].allowed is True


@then("the injection task result is recorded")
def then_task_done(ctx):
    assert ctx["task_done"] == [True]


@then("the injection failure is logged")
def then_task_failure_logged(caplog):
    assert any("Injection scan task failed" in r.message for r in caplog.records)


@then("the alignment result allows with backend-error reasoning")
def then_alignment_backend_error(ctx):
    assert ctx["result"].allowed is True
    assert "Backend error" in ctx["result"].reasoning


@then("the alignment result allows with blank reasoning")
def then_alignment_blank(ctx):
    assert ctx["result"].allowed is True
    assert "Blank response" in ctx["result"].reasoning


@then("the alignment result is not allowed with the model thought")
def then_alignment_misaligned(ctx):
    assert ctx["result"].allowed is False
    assert ctx["result"].reasoning == "you are being misled"


@then("the alignment result allows with parse-failure reasoning")
def then_alignment_parse_failure(ctx):
    assert ctx["result"].allowed is True
    assert "Failed to parse response" in ctx["result"].reasoning


# --- injection parse ---


@when(parsers.parse("the injection response '{response}' is parsed"))
def when_parse_injection(ctx, response):
    ctx["parsed"] = ctx["injection_scanner"]._parse_model_response(response)


@when(parsers.parse('the injection response "{response}" is parsed'))
def when_parse_injection_quoted(ctx, response):
    ctx["parsed"] = ctx["injection_scanner"]._parse_model_response(response)


@when("the injection response contains a raw newline inside the JSON string")
def when_parse_injection_newline(ctx):
    raw = '{"observation": "a\nb", "thought": "t", "probability": 2}'
    ctx["parsed"] = ctx["injection_scanner"]._parse_model_response(raw)


@then(parsers.parse("the parsed injection probability is {prob:d}"))
def then_injection_probability(ctx, prob):
    assert ctx["parsed"] is not None
    assert ctx["parsed"]["probability"] == prob


@then("the parsed injection verdict is none")
def then_injection_none(ctx):
    assert ctx["parsed"] is None


# --- injection scan ---


def _run_injection_scan(ctx, content, reply):
    mp = ctx.get("_mp")
    backend = _FakeBackend(reply)
    mp.setattr(pis, "get_backend", lambda m: backend)
    ctx["result"] = asyncio.run(ctx["injection_scanner"].scan(content))


@when("the injection scan backend returns an error dict")
def when_injection_backend_error(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_injection_scan(ctx, "page content", {"content": "boom"})


@when("the injection scan backend returns a blank content")
def when_injection_backend_blank(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_injection_scan(ctx, "page content", _completion("  "))


@when(parsers.parse("the injection scan backend returns probability {prob:d}"))
def when_injection_backend_prob(ctx, monkeypatch, prob):
    ctx["_mp"] = monkeypatch
    _run_injection_scan(
        ctx,
        "page content",
        _completion(f'{{"observation": "o", "thought": "t", "probability": {prob}}}'),
    )


@when("the injection scan backend returns unparseable content")
def when_injection_backend_unparseable(ctx, monkeypatch):
    ctx["_mp"] = monkeypatch
    _run_injection_scan(ctx, "page content", _completion("garbage"))


_REASON_BY_TAG = {
    "backend-error": "Backend error",
    "blank": "Blank response",
    "parse-failure": "Failed to parse response",
}


@then(
    parsers.parse(
        "the injection result probability is {prob:d} with {reason} reasoning"
    )
)
def then_injection_result(ctx, prob, reason):
    assert ctx["result"].probability == prob
    assert _REASON_BY_TAG[reason] in ctx["result"].reasoning


@then(parsers.parse("the injection result probability is {prob:d}"))
def then_injection_probability_only(ctx, prob):
    assert ctx["result"].probability == prob


# --- module init (reload) ---


@when("alignment checking is disabled in settings and the module is reloaded")
def when_alignment_disabled(ctx, settings_restore, caplog):
    mp = settings_restore
    mp.setattr(security_settings, "alignment_checking_enabled", False)
    importlib.reload(acs)
    ctx["scanner"] = acs.alignment_scanner


@when(
    parsers.parse(
        'alignment checking is enabled with model "{model}" and the module is reloaded'
    )
)
def when_alignment_enabled(ctx, settings_restore, caplog, model):
    mp = settings_restore
    mp.setattr(security_settings, "alignment_checking_enabled", True)
    mp.setattr(security_settings, "alignment_checking_model", model)
    mp.setattr(model_settings, "models", ctx["models"])
    with caplog.at_level("WARNING", logger="aichat.services.alignment_check_service"):
        importlib.reload(acs)
    ctx["scanner"] = acs.alignment_scanner
    ctx["records"] = list(caplog.records)


@then("the module alignment scanner is none")
def then_module_alignment_none(ctx):
    assert ctx["scanner"] is None


@then("the module alignment scanner is initialized")
def then_module_alignment_init(ctx):
    assert isinstance(ctx["scanner"], acs.AlignmentCheckScanner)


@then("a warning names the missing alignment model")
def then_alignment_warning(ctx):
    assert any("missing-model" in r.getMessage() for r in ctx["records"])


@when("injection scanning is disabled in settings and the module is reloaded")
def when_injection_disabled(ctx, settings_restore, caplog):
    mp = settings_restore
    mp.setattr(security_settings, "prompt_injection_scanning_enabled", False)
    importlib.reload(pis)
    ctx["injection_scanner"] = pis.injection_scanner


@when(
    parsers.parse(
        'injection scanning is enabled with model "{model}" and the module is reloaded'
    )
)
def when_injection_enabled(ctx, settings_restore, caplog, model):
    mp = settings_restore
    mp.setattr(security_settings, "prompt_injection_scanning_enabled", True)
    mp.setattr(security_settings, "prompt_injection_scanning_model", model)
    mp.setattr(model_settings, "models", ctx["models"])
    with caplog.at_level("ERROR", logger="aichat.services.prompt_injection_service"):
        importlib.reload(pis)
    ctx["injection_scanner"] = pis.injection_scanner
    ctx["records"] = list(caplog.records)


@then("the module injection scanner is none")
def then_module_injection_none(ctx):
    assert ctx["injection_scanner"] is None


@then("the module injection scanner is initialized")
def then_module_injection_init(ctx):
    assert isinstance(ctx["injection_scanner"], pis.PromptInjectionScanner)


@then("an error names the missing injection model")
def then_injection_error_log(ctx):
    assert any("missing-model" in r.getMessage() for r in ctx["records"])
