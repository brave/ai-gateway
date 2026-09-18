from pydantic_settings import BaseSettings


class CompactionSettings(BaseSettings):
    compaction_model: str = "jaguar"
    compaction_fallback_model: str = "claude-3-haiku"
    compaction_threshold: float = 0.60
    compaction_min_preserved_turns: int = 4
    compaction_summary_temperature: float = 0.3
    compaction_summary_max_tokens: int = 1000
    compaction_fallback_max_tokens: int = 8000
    compaction_fallback_output_tokens: int = 1000
    compaction_input_margin: float = 0.9
    compaction_max_resummarize_attempts: int = 4
    trimming_input_margin: float = 0.9
    compaction_model_context_window: int = 80_000
    compaction_fallback_model_context_window: int = 200_000
    # Per-part cap for plain text content parts (type: "text") during trimming.
    # Parts exceeding this limit are truncated when the conversation is still over
    # budget after the primary Brave-type and tool-message passes.
    max_text_part_tokens: int = 50_000
    # Hard ceiling for any request after trimming. Independent of compaction /
    # target-model context limits.
    absolute_max_tokens: int = 500_000
    # Max tokens per chunk when splitting messages for the compaction model.
    # Note this is a hard upper limit, in the compaction chunking a safety margin is applied.
    # This means we take the min of this and compaction_context_safety_factor * context window.
    compaction_chunk_max_tokens: int = 75_000
    compaction_context_safety_factor: float = 0.85
    # Max tokens per chunk when splitting for Haiku fallback (separate from Jaguar).
    compaction_fallback_chunk_max_tokens: int = 195_000
    # Max concurrent Jaguar compaction requests when summarizing multiple chunks.
    # Don't make too large as compaction requests will be large
    compaction_max_parallel_requests: int = 2
    # Safety cap on images sent to the compaction model (upstream preprocessing
    # also limits images for the target chat model).
    compaction_max_images: int = 20

    # Jaguar Compaction Settings
    jaguar_style: str = (
        "markdown"  # Select style from ['json', 'markdown', 'plain', 'xml', 'yaml']
    )
    jaguar_preserve: str = (
        "all"  # Select preserve from ['all', 'code', 'none', 'quotes']
    )
    jaguar_focus: str = (
        "general"  # Select focus from [''coding', 'creative', 'debugging', 'general', 'planning', 'research', 'tutoring']
    )
    jaguar_recency: str = "uniform"  # Select recency from ['tail-verbatim', 'uniform']
    jaguar_language: str = (
        "source"  # source = same language as conversation otherwise choose 'English', 'Chinese', 'French' etc
    )


compaction_settings = CompactionSettings()
