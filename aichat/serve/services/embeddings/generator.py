from aichat.serve.backend.litellm import get_global_router


def _transform_openai_to_triton(model: str, input_data: str | list[str]) -> dict:
    """Transform OpenAI-style embeddings request to Triton format."""
    # Normalize input to list
    if isinstance(input_data, str):
        inputs_list = [input_data]
    else:
        inputs_list = input_data

    return {
        "model": model,
        "json": {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [len(inputs_list), 1],
                    "datatype": "BYTES",
                    "data": [inputs_list],
                }
            ],
        },
    }


def _transform_triton_to_openai(
    model: str, triton_response: dict, input_count: int, input_texts: list[str]
) -> dict:
    """Transform Triton response to OpenAI-style embeddings response."""
    outputs = triton_response.get("outputs", [])
    if not outputs:
        raise ValueError("No outputs in Triton response")

    # Extract embeddings data
    embeddings_data = outputs[0].get("data", [])
    embedding_dim = len(embeddings_data) // input_count if input_count > 0 else 0

    # Split embeddings into individual vectors
    embeddings_list = []
    for i in range(input_count):
        start_idx = i * embedding_dim
        end_idx = start_idx + embedding_dim
        embedding_vector = embeddings_data[start_idx:end_idx]
        embeddings_list.append(
            {
                "object": "embedding",
                "embedding": embedding_vector,
                "index": i,
            }
        )

    # Calculate token usage (approximate)
    total_tokens = sum(len(text.split()) for text in input_texts)

    return {
        "object": "list",
        "data": embeddings_list,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


async def generate_embeddings(model: str, input: str | list[str], **kwargs):
    """
    Generate embeddings using OpenAI-style API that transforms to Triton format.

    Args:
        model: Model name
        input: Input text(s) - can be a string or list of strings
        **kwargs: Additional parameters (ignored for now)

    Returns:
        OpenAI-style embeddings response
    """
    router = get_global_router()

    # Normalize input to list for counting
    if isinstance(input, str):
        inputs_list = [input]
    else:
        inputs_list = input

    input_count = len(inputs_list)

    # Transform to Triton format
    triton_request = _transform_openai_to_triton(model, input)

    # Call passthrough route
    triton_response = await router.allm_passthrough_route(**triton_request)

    # Handle response - it might be a response object or dict
    if isinstance(triton_response, dict):
        triton_data = triton_response
    elif hasattr(triton_response, "json"):
        # It's likely an httpx Response object
        if callable(getattr(triton_response, "json", None)):
            triton_data = triton_response.json()
        else:
            triton_data = triton_response
    else:
        # Fallback: try to convert to dict or use as-is
        try:
            triton_data = (
                dict(triton_response)
                if hasattr(triton_response, "__dict__")
                else triton_response
            )
        except Exception as e:
            raise ValueError(
                f"Unexpected response type: {type(triton_response)}"
            ) from e

    # Transform back to OpenAI format
    return _transform_triton_to_openai(model, triton_data, input_count, inputs_list)
