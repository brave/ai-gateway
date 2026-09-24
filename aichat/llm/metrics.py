from prometheus_client import REGISTRY, Counter, Histogram

PROMPT_LENGTH_BUCKETS = (
    0,
    100,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    10000000,
)

COMPACTION_TOTAL = Counter(
    name="compaction_total",
    documentation="Total number of conversation compactions performed",
    labelnames=("model",),
    registry=REGISTRY,
)

COMPACTION_STARTING_TOKEN_COUNT = Histogram(
    name="compaction_starting_token_count",
    documentation="The number of tokens in the starting conversation before compaction",
    labelnames=("model",),
    registry=REGISTRY,
    buckets=PROMPT_LENGTH_BUCKETS,
)

COMPACTION_ENDING_TOKEN_COUNT = Histogram(
    name="compaction_ending_token_count",
    documentation="The number of tokens in the ending conversation after compaction",
    labelnames=("model",),
    registry=REGISTRY,
    buckets=PROMPT_LENGTH_BUCKETS,
)

PREMIUM_MODEL_DOWNGRADE_TOTAL = Counter(
    name="premium_model_downgrade_total",
    documentation="Total number of requests downgraded from a premium model to automatic on the non-premium endpoint",
    labelnames=("model",),
    registry=REGISTRY,
)

# Count requests per service key (SKv2)
AUTH_REQUESTS_BY_KEY = Counter(
    name="auth_requests_by_key_total",
    documentation="Total number of requests by service key ID (SKv2), model, and premium status",
    labelnames=("key_id", "model", "is_premium", "api_version"),
    registry=REGISTRY,
)

MODEL_OVERRIDE_TOTAL = Counter(
    name="model_override_total",
    documentation="Requests whose requested model was overridden server-side, by key_id",
    labelnames=("key_id", "requested_model"),
    registry=REGISTRY,
)


ALIGNMENT_CHECK_TOTAL = Counter(
    name="alignment_check_total",
    documentation="Total number of alignment checks performed",
    labelnames=("allowed", "tool", "bypassed"),
    registry=REGISTRY,
)

PROMPT_INJECTION_SCAN_TOTAL = Counter(
    name="prompt_injection_scan_total",
    documentation="Total number of prompt injection scans performed",
    labelnames=("probability",),
    registry=REGISTRY,
)

TOKEN_GEN_LATENCY = Histogram(
    name="token_generation_latency_seconds",
    documentation="Latency for generating first token in seconds",
    labelnames=("model",),
    registry=REGISTRY,
)

TIME_TO_TOOLS_USE = Histogram(
    name="time_to_tools_use",
    documentation="Time taken to determine the appropriate tool for the request",
    labelnames=("model",),
    registry=REGISTRY,
)

TIME_TO_FIRST_RESPONSE_TOKEN = Histogram(
    name="time_to_first_response_token",
    documentation="Time taken to generate the first response token in the conversation",
    labelnames=("model", "uses_search"),
    registry=REGISTRY,
)

PROMPT_LENGTH = Histogram(
    name="prompt_length_characters",
    documentation="The length of the prompt in characters",
    labelnames=("model",),
    registry=REGISTRY,
    buckets=PROMPT_LENGTH_BUCKETS,
)

GENERATED_LENGTH = Histogram(
    name="generated_length_characters",
    documentation="The length of the model generated ouput in characters",
    labelnames=("model",),
    registry=REGISTRY,
    buckets=PROMPT_LENGTH_BUCKETS,
)

FUNCTION_CALL_COUNTER = Counter(
    name="function_calls_counter",
    documentation="Counts the number of function calls",
    labelnames=("function_name",),
    registry=REGISTRY,
)

TOKEN_TRIMMING_TOTAL = Counter(
    name="token_trimming_total",
    documentation="Total number of requests where token trimming occurred",
    labelnames=("model",),
    registry=REGISTRY,
)

TOKEN_TRIMMING_AMOUNT = Histogram(
    name="token_trimming_amount",
    documentation="The number of tokens trimmed from requests",
    labelnames=("model",),
    registry=REGISTRY,
    buckets=PROMPT_LENGTH_BUCKETS,
)

DEEP_RESEARCH_CAPABILITY_TOTAL = Counter(
    name="deep_research_capability_total",
    documentation="Requests with deep_research capability, tracking whether the tool was triggered",
    labelnames=("model", "triggered", "message_count_before_deep_research"),
    registry=REGISTRY,
)

EMPTY_STREAMING_RESPONSE_TOTAL = Counter(
    name="empty_streaming_response_total",
    documentation="Streaming responses that yielded no content and no tool calls, excluding ContentReceipt",
    labelnames=("model", "reason"),
    registry=REGISTRY,
)

MID_STREAM_ERROR_TOTAL = Counter(
    name="mid_stream_error_total",
    documentation="Streaming responses where an exception was raised mid-stream and surfaced to the client as an error chunk",
    labelnames=("model",),
    registry=REGISTRY,
)

TRUNCATION_RECOVERY_TOTAL = Counter(
    name="truncation_recovery_total",
    documentation="Recovery actions for assistant replies that stopped early (finish_reason=stop)",
    labelnames=("model", "action"),
    registry=REGISTRY,
)

DUPLICATE_TOOL_CALL_ID_TOTAL = Counter(
    name="duplicate_tool_call_id_total",
    documentation="Total number of tool call responses with duplicate IDs detected before sending to client, by model and tool name",
    labelnames=("model", "tool"),
    registry=REGISTRY,
)

CONVERSATION_TITLE_INVALID_REQUEST_TOTAL = Counter(
    name="conversation_title_invalid_request_total",
    documentation="Conversation title requests where brave-conversation-title had no usable text",
    registry=REGISTRY,
)

INTERNAL_REQUEST_RETRY_TOTAL = Counter(
    name="internal_request_retry_total",
    documentation="Total number of aichat-internal requests retried after a RemoteProtocolError (stale keep-alive connection), by path and outcome",
    labelnames=("path", "outcome"),
    registry=REGISTRY,
)
