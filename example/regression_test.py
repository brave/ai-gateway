import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from conversation_protocol_prompts import prompts as conversation_prompts
from mcp_protocol_prompts import prompts as mcp_prompts
from openai_protocol_prompts import prompts as openai_prompts

SERVER_URL = "http://localhost:8000"  # overridden by --url
MODELS_ENDPOINT = "/v1/models"
OPENAI_ENDPOINT = "/v1/chat/completions"
CONVERSATION_ENDPOINT = "/v1/conversation"

EXPECTED_HTTP_STATUS = 200


def _model_has_capability(model, capability):
    return capability in model.get("capabilities", [])


def test_model_retrieval() -> list[str]:
    print("Retrieving models")

    try:
        response = requests.get(f"{SERVER_URL}{MODELS_ENDPOINT}?lang=en", timeout=10)

        if response.status_code != EXPECTED_HTTP_STATUS:
            print(f"FAIL: Unexpected status code {response.status_code}")
            return []

        models = response.json()
        keys = [model for model in models if "chat" in model.get("capabilities", [])]
        return keys
    except Exception as e:
        print(f"FAIL: Error during model retrival test - {e!s}")
        return []


def _prompt_label(prompt_entry):
    prompt = prompt_entry.get("prompt", {})

    # Conversation protocol: derive label from unique event types
    events = prompt.get("events", [])
    if events:
        seen = set()
        types = []
        for e in events:
            t = e.get("type", "")
            if t and t not in seen:
                seen.add(t)
                types.append(t)
        distinctive = [t for t in types if t != "chatMessage"]
        return "+".join(distinctive) if distinctive else "text"

    # OpenAI/MCP protocol: derive label from first message content type
    content = prompt.get("messages", [{}])[0].get("content")
    if isinstance(content, str):
        return "text"
    if isinstance(content, list) and content:
        return content[0].get("type", "unknown")
    return "unknown"


def _run_single(model, prompt_entry, timeout=120):
    key = model.get("key")
    requirements = prompt_entry.get("requirements", {})

    for req_key in (
        "vision",
        "tools",
        "audio",
        "video",
    ):
        if requirements.get(req_key, False) and not _model_has_capability(
            model, req_key
        ):
            return None

    label = _prompt_label(prompt_entry)
    prompt = {**prompt_entry.get("prompt"), "model": key}

    try:
        response = requests.post(
            f"{SERVER_URL}{OPENAI_ENDPOINT}", json=prompt, timeout=timeout
        )

        if response.status_code != EXPECTED_HTTP_STATUS:
            return (key, label, f"HTTP status code: {response.status_code}")

        if (
            key.startswith("near-")
            and response.headers.get("brave-near-verified", False) is False
        ):
            return (key, label, "NEAR verification failure")

    except requests.exceptions.Timeout:
        return (key, label, "Timeout")
    except Exception as e:
        print(prompt)
        return (key, label, str(e))

    return None


def _run_single_streaming(model, prompt_entry, timeout=120):
    key = model.get("key")
    requirements = prompt_entry.get("requirements", {})

    for req_key in (
        "vision",
        "tools",
        "audio",
        "video",
    ):
        if requirements.get(req_key, False) and not _model_has_capability(
            model, req_key
        ):
            return None

    label = _prompt_label(prompt_entry)
    prompt = {**prompt_entry.get("prompt"), "model": key, "stream": True}

    try:
        with requests.post(
            f"{SERVER_URL}{OPENAI_ENDPOINT}",
            json=prompt,
            stream=True,
            timeout=timeout,
        ) as response:
            if response.status_code != EXPECTED_HTTP_STATUS:
                return (key, label, f"HTTP status code: {response.status_code}")

            got_done = False
            got_content = False

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    got_done = True
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "compaction_failed":
                    return (key, label, f"compaction_failed: {event.get('error', '')}")
                if event.get("object") == "chat.completion.chunk":
                    content = (
                        (event.get("choices") or [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if content:
                        got_content = True

            if not got_done:
                return (key, label, "Stream closed without [DONE]")
            if not got_content:
                return (key, label, "No content received")

    except requests.exceptions.Timeout:
        return (key, label, "Timeout")
    except Exception as e:
        return (key, label, str(e))

    return None


def _run_single_conversation(model, prompt_entry, timeout=120):
    key = model.get("key")
    requirements = prompt_entry.get("requirements", {})

    for req_key in (
        "vision",
        "tools",
        "audio",
        "video",
    ):
        if requirements.get(req_key, False) and not _model_has_capability(
            model, req_key
        ):
            return None
    if (
        requirements.get("anthropic_only", False)
        and model.get("options", {}).get("display_maker") != "Anthropic"
    ):
        return None

    label = _prompt_label(prompt_entry)
    prompt = {**prompt_entry.get("prompt"), "model": key}

    try:
        response = requests.post(
            f"{SERVER_URL}{CONVERSATION_ENDPOINT}", json=prompt, timeout=timeout
        )

        if response.status_code != EXPECTED_HTTP_STATUS:
            return (key, label, f"HTTP status code: {response.status_code}")

    except requests.exceptions.Timeout:
        return (key, label, "Timeout")
    except Exception as e:
        return (key, label, str(e))

    return None


def _run_single_conversation_streaming(model, prompt_entry, timeout=120):
    key = model.get("key")
    requirements = prompt_entry.get("requirements", {})

    for req_key in (
        "vision",
        "tools",
        "audio",
        "video",
    ):
        if requirements.get(req_key, False) and not _model_has_capability(
            model, req_key
        ):
            return None
    if (
        requirements.get("anthropic_only", False)
        and model.get("options", {}).get("display_maker") != "Anthropic"
    ):
        return None

    label = _prompt_label(prompt_entry)
    prompt = {**prompt_entry.get("prompt"), "model": key, "stream": True}

    try:
        with requests.post(
            f"{SERVER_URL}{CONVERSATION_ENDPOINT}",
            json=prompt,
            stream=True,
            timeout=timeout,
        ) as response:
            if response.status_code != EXPECTED_HTTP_STATUS:
                return (key, label, f"HTTP status code: {response.status_code}")

            got_done = False
            got_content = False

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    got_done = True
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "compaction_failed":
                    return (key, label, f"compaction_failed: {event.get('error', '')}")
                if event.get("type") == "completion" and event.get("completion"):
                    got_content = True

            if not got_done:
                return (key, label, "Stream closed without [DONE]")
            if not got_content:
                return (key, label, "No content received")

    except requests.exceptions.Timeout:
        return (key, label, "Timeout")
    except Exception as e:
        return (key, label, str(e))

    return None


def _select_minimal_models(models, skip=()):
    """Pick the smallest set of models that covers all prompt capability combinations."""
    cap_combos = {
        frozenset(p.get("requirements", {}).items())
        for p in openai_prompts + mcp_prompts + conversation_prompts
    }
    selected = []
    for caps in cap_combos:
        reqs = dict(caps)
        for model in models:
            if model.get("key") in skip:
                continue
            if any(
                reqs.get(req_key, False) and not _model_has_capability(model, req_key)
                for req_key in (
                    "vision",
                    "tools",
                    "audio",
                    "video",
                )
            ):
                continue
            if (
                reqs.get("anthropic_only", False)
                and model.get("options", {}).get("display_maker") != "Anthropic"
            ):
                continue
            selected.append(model)
            break
    seen = set()
    result = []
    for m in selected:
        k = m.get("key")
        if k not in seen:
            seen.add(k)
            result.append(m)

    if not any(m.get("key", "").startswith("near-") for m in result):
        near_model = next(
            (
                m
                for m in models
                if m.get("key", "").startswith("near-") and m.get("key") not in skip
            ),
            None,
        )
        if near_model:
            result.append(near_model)
        else:
            print("WARNING: No NEAR model available; NEAR coverage skipped.")

    return result


def _run_test(label, runner, prompts_list, models, max_workers=20, timeout=120):
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(runner, model, prompt_entry, timeout): model.get("key")
            for model in models
            for prompt_entry in prompts_list
        }

        passed = 0
        failed = 0
        total = len(futures)
        milestones = {25, 50, 75, 100}
        printed_milestones = set()
        print(f"{label} ({total} requests)")
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is None:
                passed += 1
            else:
                failed += 1
                failures.append(result)
            pct = completed * 100 // total
            for m in sorted(
                m for m in milestones if m <= pct and m not in printed_milestones
            ):
                printed_milestones.add(m)
                print(f"  {m}% ({completed}/{total})")

    return total, passed, failed, failures


# Main function to run all tests
def run_regression_tests(
    minimal=False, timeout=120, skip=(), stream_mode=None, protocol=None, url=None
):
    global SERVER_URL
    if url:
        SERVER_URL = url.rstrip("/")
    print("Starting regression tests...")
    models = test_model_retrieval()

    if len(models) == 0:
        print("FAIL: No models retrieved.")
        return

    if skip:
        print(f"Skipping: {', '.join(skip)}")

    if minimal:
        models = _select_minimal_models(models, skip=skip)
        print(
            f"Minimal mode: selected {len(models)} model(s): {', '.join(m.get('key') for m in models)}"
        )
    elif skip:
        models = [m for m in models if m.get("key") not in skip]

    test_cases = [
        (label, runner, prompts_list, tag)
        for label, runner, prompts_list, tag, proto, is_streaming in [
            (
                "Testing non-streaming model responses...",
                _run_single,
                openai_prompts,
                "[non-stream]",
                "openai",
                False,
            ),
            (
                "Testing streaming model responses...",
                _run_single_streaming,
                openai_prompts,
                "[stream]",
                "openai",
                True,
            ),
            (
                "Testing non-streaming MCP responses...",
                _run_single,
                mcp_prompts,
                "[mcp-non-stream]",
                "mcp",
                False,
            ),
            (
                "Testing streaming MCP responses...",
                _run_single_streaming,
                mcp_prompts,
                "[mcp-stream]",
                "mcp",
                True,
            ),
            (
                "Testing non-streaming conversation responses...",
                _run_single_conversation,
                conversation_prompts,
                "[conv-non-stream]",
                "conversation",
                False,
            ),
            (
                "Testing streaming conversation responses...",
                _run_single_conversation_streaming,
                conversation_prompts,
                "[conv-stream]",
                "conversation",
                True,
            ),
        ]
        if (stream_mode is None or (stream_mode == "streaming") == is_streaming)
        if (protocol is None or proto == protocol)
    ]

    all_failures = []
    grand_total = grand_passed = grand_failed = 0

    for label, runner, prompts_list, tag in test_cases:
        start = time.time()
        total, passed, failed, failures = _run_test(
            label, runner, prompts_list, models, timeout=timeout
        )
        elapsed = time.time() - start
        print(
            f"\n{total} requests: {passed} passed, {failed} failed ({elapsed:.1f}s)\n"
        )
        grand_total += total
        grand_passed += passed
        grand_failed += failed
        all_failures.extend((tag, key, lbl, exc) for key, lbl, exc in failures)

    print(
        f"Grand total — {grand_total} requests: {grand_passed} passed, {grand_failed} failed"
    )

    if all_failures:
        print("\nFailures:")
        for mode, key, lbl, exc in all_failures:
            print(f"  {mode:<20} {key:<40} {lbl:<22} {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run regression tests against the ai-gateway server."
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Test only a minimal set of models covering all capability combinations (faster).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="MODEL_KEY",
        default=[],
        help="Skip one or more models by key (e.g. --skip chat-basic chat-claude-sonnet).",
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "non-streaming"],
        default=None,
        help="Only run streaming or non-streaming tests.",
    )
    parser.add_argument(
        "--protocol",
        choices=["openai", "mcp", "conversation"],
        default=None,
        help="Only run tests for the specified protocol.",
    )
    parser.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help=f"Server base URL (default: {SERVER_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Per-request timeout in seconds (default: 120).",
    )
    args = parser.parse_args()
    run_regression_tests(
        minimal=args.minimal,
        timeout=args.timeout,
        skip=args.skip,
        stream_mode=args.mode,
        protocol=args.protocol,
        url=args.url,
    )
