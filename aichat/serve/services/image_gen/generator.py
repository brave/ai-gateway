from aichat.serve.backend.litellm import get_global_router


async def generate_image(**kwargs):
    """Passthrough to litellm router's aimage_generation method."""
    router = get_global_router()
    return await router.aimage_generation(**kwargs)
