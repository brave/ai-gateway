from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aichat.serve.services.embeddings import generator


def test_transform_openai_to_triton_single_string():
    req = generator._transform_openai_to_triton("m", "hello")
    assert req["model"] == "m"
    assert req["json"]["inputs"][0]["shape"] == [1, 1]
    assert req["json"]["inputs"][0]["data"] == [["hello"]]
    assert req["json"]["inputs"][0]["datatype"] == "BYTES"


def test_transform_openai_to_triton_list():
    req = generator._transform_openai_to_triton("m", ["a", "b"])
    assert req["json"]["inputs"][0]["shape"] == [2, 1]
    assert req["json"]["inputs"][0]["data"] == [["a", "b"]]


def test_transform_triton_to_openai():
    triton = {
        "outputs": [
            {
                "name": "embedding",
                "datatype": "FP32",
                "shape": [2, 2],
                "data": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    }
    resp = generator._transform_triton_to_openai("m", triton, 2, ["a", "b"])
    assert resp["object"] == "list"
    assert resp["model"] == "m"
    assert len(resp["data"]) == 2
    assert resp["data"][0]["embedding"] == [0.1, 0.2]
    assert resp["data"][1]["index"] == 1
    assert resp["usage"]["total_tokens"] == 2


def test_transform_triton_to_openai_no_outputs():
    with pytest.raises(ValueError):
        generator._transform_triton_to_openai("m", {}, 1, ["a"])


@pytest.mark.asyncio
async def test_generate_embeddings_dict_response():
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value={"outputs": [{"data": [0.1, 0.2]}]}
    )
    with patch(
        "aichat.serve.services.embeddings.generator.get_global_router",
        return_value=router,
    ):
        resp = await generator.generate_embeddings("m", "hello")
    assert resp["object"] == "list"
    assert resp["data"][0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_generate_embeddings_response_object_with_json():
    router = MagicMock()
    response_mock = MagicMock()
    response_mock.json = MagicMock(return_value={"outputs": [{"data": [0.1]}]})
    router.allm_passthrough_route = AsyncMock(return_value=response_mock)
    with patch(
        "aichat.serve.services.embeddings.generator.get_global_router",
        return_value=router,
    ):
        resp = await generator.generate_embeddings("m", ["a", "b"])
    assert resp["data"][1]["index"] == 1


class _ObjWithDict:
    """Non-dict object with __dict__ that dict() can consume."""

    def __init__(self, d):
        self.__dict__ = d

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, k):
        return self.__dict__[k]


class _ObjWithNonKeyableDict:
    def __init__(self):
        self.__dict__ = {"outputs": [{"data": [0.1]}]}


@pytest.mark.asyncio
async def test_generate_embeddings_obj_with_dict_fallback():
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value=_ObjWithDict({"outputs": [{"data": [0.1, 0.2]}]})
    )
    with patch(
        "aichat.serve.services.embeddings.generator.get_global_router",
        return_value=router,
    ):
        resp = await generator.generate_embeddings("m", "hello")
    assert resp["data"][0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_generate_embeddings_unexpected_response_type():
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(return_value=_ObjWithNonKeyableDict())
    with patch(
        "aichat.serve.services.embeddings.generator.get_global_router",
        return_value=router,
    ):
        with pytest.raises(Exception) as exc:
            await generator.generate_embeddings("m", "hello")
        assert "Unexpected response type" in str(exc.value)
