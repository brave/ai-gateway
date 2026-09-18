"""Tests for inline search service"""

from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from aichat.responses import InlineSearch
from aichat.serve.services.search import InlineSearchHelper
from aichat.serve.services.search_settings import search_settings


class TestInlineSearchHelperHandleReceivedCompletion:
    """Tests for InlineSearchHelper.handle_received_completion"""

    @pytest.mark.asyncio
    async def test_detects_single_query(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion("::search[python tutorial]{type=web}\n")
            assert len(helper.inline_search_promises) == 1

    @pytest.mark.asyncio
    async def test_detects_multiple_queries_across_chunks(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion("::search[first query]{type=web}\n")
            helper.handle_received_completion(
                "::search[first query]{type=web}\n::search[second query]{type=images}\n"
            )
            assert len(helper.inline_search_promises) == 2

    @pytest.mark.asyncio
    async def test_deduplicates_same_query(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion(
                "::search[same query]{type=web}\n::search[same query]{type=web}\n"
            )
            assert len(helper.inline_search_promises) == 1

    @pytest.mark.asyncio
    async def test_no_queries_in_text(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion(
                "Just some regular text with no search queries"
            )
            assert len(helper.inline_search_promises) == 0

    @pytest.mark.asyncio
    async def test_malformed_queries_ignored(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion(
                "::search[missing closing brace]{type=web\n"
                "::search[valid query]{type=web}\n"
                "search[missing double colon]{type=web}\n"
            )
            assert len(helper.inline_search_promises) == 1

    @pytest.mark.asyncio
    async def test_respects_max_inline_searches(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            with patch("aichat.serve.services.search.search_settings") as mock_settings:
                mock_settings.max_inline_searches = 2
                helper.handle_received_completion(
                    "::search[query 0]{type=web}\n"
                    "::search[query 1]{type=web}\n"
                    "::search[query 2]{type=web}\n"
                )
            assert len(helper.inline_search_promises) == 2

    @pytest.mark.asyncio
    async def test_different_search_types(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion(
                "::search[web query]{type=web}\n"
                "::search[image query]{type=images}\n"
                "::search[news query]{type=news}\n"
                "::search[video query]{type=videos}\n"
            )
            assert len(helper.inline_search_promises) == 4

    @pytest.mark.asyncio
    async def test_queries_with_special_characters(self):
        async with httpx.AsyncClient() as client:
            helper = InlineSearchHelper(client)
            helper.handle_received_completion(
                "::search[heart-rate monitors]{type=web}\n"
                "::search[C++ programming]{type=web}\n"
                "::search[best 5-star hotels in new york]{type=web}\n"
            )
            assert len(helper.inline_search_promises) == 3


class TestInlineSearchHelperGetChunks:
    """Tests for InlineSearchHelper.get_inline_search_chunks"""

    @pytest_asyncio.fixture
    async def httpx_client(self):
        async with httpx.AsyncClient() as client:
            yield client

    @pytest.mark.asyncio
    async def test_returns_empty_with_no_queries(self, httpx_client):
        helper = InlineSearchHelper(httpx_client)
        results = await helper.get_inline_search_chunks()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_web_search(self, httpx_client, httpx_mock):
        query = "python programming"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={query}&type=web",
            json={
                "query": {"original": query},
                "web": {
                    "results": [
                        {
                            "type": "search_result",
                            "title": "Learn Python",
                            "url": "https://example.com/python",
                            "description": "A guide to Python",
                            "meta_url": {
                                "netloc": "example.com",
                                "path": "/python",
                                "favicon": "",
                            },
                            "age": "2 days ago",
                        }
                    ]
                },
            },
            status_code=200,
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=web}}\n")
        results = await helper.get_inline_search_chunks()

        assert len(results) == 1
        assert isinstance(results[0], InlineSearch)
        assert results[0].query == query
        assert results[0].results[0].title == "Learn Python"
        assert results[0].results[0].age == "2 days ago"

    @pytest.mark.asyncio
    async def test_images_search_uses_images_endpoint(self, httpx_client, httpx_mock):
        query = "cute cats"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/images/search?q={query}&type=images",
            json={
                "query": {"original": query},
                "images": {
                    "results": [
                        {
                            "type": "image",
                            "title": "Cute Cat",
                            "url": "https://example.com/cat.jpg",
                            "description": "A cute cat",
                            "meta_url": {
                                "netloc": "example.com",
                                "path": "/cat.jpg",
                                "favicon": "",
                            },
                            "thumbnail": {"src": "https://example.com/thumb.jpg"},
                        }
                    ]
                },
            },
            status_code=200,
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=images}}\n")
        results = await helper.get_inline_search_chunks()

        assert len(results) == 1
        assert results[0].results[0].thumbnail.src == "https://example.com/thumb.jpg"

    @pytest.mark.asyncio
    async def test_news_search_uses_web_endpoint(self, httpx_client, httpx_mock):
        query = "breaking news"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={query}&type=news",
            json={
                "query": {"original": query},
                "web": {
                    "results": [
                        {
                            "type": "news_result",
                            "title": "Latest News",
                            "url": "https://news.example.com/latest",
                            "description": "Today's stories",
                            "meta_url": {
                                "netloc": "news.example.com",
                                "path": "/latest",
                                "favicon": "",
                            },
                            "age": "1 hour ago",
                        }
                    ]
                },
            },
            status_code=200,
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=news}}\n")
        results = await helper.get_inline_search_chunks()

        assert len(results) == 1
        assert results[0].results[0].type == "news_result"

    @pytest.mark.asyncio
    async def test_multiple_results(self, httpx_client, httpx_mock):
        query = "tech news"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={query}&type=web",
            json={
                "query": {"original": query},
                "web": {
                    "results": [
                        {
                            "type": "search_result",
                            "title": f"Result {i}",
                            "url": f"https://example.com/{i}",
                            "description": f"Result {i} description",
                            "meta_url": {
                                "netloc": "example.com",
                                "path": f"/{i}",
                                "favicon": "",
                            },
                        }
                        for i in range(3)
                    ]
                },
            },
            status_code=200,
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=web}}\n")
        results = await helper.get_inline_search_chunks()

        assert len(results[0].results) == 3

    @pytest.mark.asyncio
    async def test_no_results(self, httpx_client, httpx_mock):
        query = "nonexistent query"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={query}&type=web",
            json={"query": {"original": query}, "web": {"results": []}},
            status_code=200,
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=web}}\n")
        results = await helper.get_inline_search_chunks()

        assert results[0].query == query
        assert len(results[0].results) == 0

    @pytest.mark.asyncio
    async def test_api_error_is_skipped(self, httpx_client, httpx_mock):
        query = "test query"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={query}&type=web",
            status_code=500,
            text="Internal Server Error",
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=web}}\n")

        # A failed search is swallowed and filtered out so it can't abort the
        # stream; the failing query simply produces no chunk.
        results = await helper.get_inline_search_chunks()
        assert results == []

    @pytest.mark.asyncio
    async def test_failed_search_does_not_drop_successful_ones(
        self, httpx_client, httpx_mock
    ):
        good_query = "good query"
        bad_query = "bad query"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={good_query}&type=web",
            json={
                "query": {"original": good_query},
                "web": {"results": []},
            },
            status_code=200,
        )
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={bad_query}&type=web",
            status_code=500,
            text="Internal Server Error",
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(
            f"::search[{good_query}]{{type=web}}\n::search[{bad_query}]{{type=web}}\n"
        )

        results = await helper.get_inline_search_chunks()
        assert len(results) == 1
        assert results[0].query == good_query

    @pytest.mark.asyncio
    async def test_missing_thumbnail_is_none(self, httpx_client, httpx_mock):
        query = "test query"
        httpx_mock.add_response(
            method="GET",
            url=f"{search_settings.brave_search_api_url}/res/v1/web/search?q={query}&type=web",
            json={
                "query": {"original": query},
                "web": {
                    "results": [
                        {
                            "type": "search_result",
                            "title": "No Thumbnail",
                            "url": "https://example.com/no-thumb",
                            "description": "No thumbnail here",
                            "meta_url": {
                                "netloc": "example.com",
                                "path": "/no-thumb",
                                "favicon": "",
                            },
                        }
                    ]
                },
            },
            status_code=200,
        )

        helper = InlineSearchHelper(httpx_client)
        helper.handle_received_completion(f"::search[{query}]{{type=web}}\n")
        results = await helper.get_inline_search_chunks()

        assert results[0].results[0].thumbnail is None
        assert results[0].results[0].age is None
