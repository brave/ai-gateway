import asyncio
import logging
import re

import httpx

from aichat.responses import (
    InlineSearch,
    InlineSearchResult,
    InlineSearchResultMeta,
    InlineSearchResultThumbnail,
)
from aichat.serve.services.search_settings import search_settings

logger = logging.getLogger(__name__)

_headers = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
}


def _get_new_inline_searches(seen: set[str], completion: str) -> list[str]:
    """Get new inline queries from the completion"""
    regex = r"^::search\[(.+?)\]{type=(\w+)}$"
    for match in re.findall(regex, completion, re.MULTILINE):
        query = match[0]
        query_type = match[1]

        if query not in seen:
            seen.add(query)
            yield query, query_type


async def _do_inline_search(httpx_client, query: str, type: str):
    """Does a query for generating an inline search event, so the client can display a rich widget"""
    assert type in ["web", "images", "videos", "news"]

    # The web endpoint supports images, news & videos but not images, so we need to hit
    # that API directly.
    search_endpoint = "web"
    if type == "images":
        search_endpoint = "images"

    headers = {**_headers, "X-Subscription-Token": search_settings.brave_search_api_key}
    url = f"{search_settings.brave_search_api_url}/res/v1/{search_endpoint}/search?q={query}&type={type}"
    response = await httpx_client.get(url, headers=headers)

    # Raise an exception if we didn't get a success status code.
    if response.status_code != 200:
        logger.warning(
            f"Failed to fetch inline search results for query '{query}': {response.status_code} {response.text}"
        )
        raise RuntimeError(
            f"Failed to fetch inline search results for query '{query}': {response.status_code} {response.text}"
        )

    data = response.json()

    result = InlineSearch(query=data["query"]["original"], results=[])

    # Results can be in a number of places:
    potential_results_containers = [
        data,  # directly on the element
        data.get("web"),  # in a web child element
        data.get("videos"),  # in a video child element
        data.get("images"),  # in an image child element
        data.get("news"),  # in a news child element
    ]

    for result_container in potential_results_containers:
        if not result_container or not "results" in result_container:
            continue

        for r in result_container["results"]:
            meta = r.get("meta_url")
            search_result = InlineSearchResult(
                type=r.get("type", ""),
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
                meta_url=InlineSearchResultMeta(
                    netloc=meta.get("netloc", ""),
                    path=meta.get("path", ""),
                    favicon=meta.get("favicon", ""),
                ),
                age=r.get("age"),
            )

            tn = r.get("thumbnail")
            if tn:
                search_result.thumbnail = InlineSearchResultThumbnail(
                    src=tn.get("src", "")
                )
            result.results.append(search_result)

    return result


class InlineSearchHelper:
    def __init__(self, httpx_client: httpx.AsyncClient):
        self.httpx_client = httpx_client
        self.seen = set[str]()
        self.inline_search_promises = []

    def handle_received_completion(self, accumulated_completion: str) -> list[str]:
        """Get new inline queries from the completion"""
        new_inline_searches = list(
            _get_new_inline_searches(self.seen, accumulated_completion)
        )

        for query, query_type in new_inline_searches:
            if len(self.inline_search_promises) >= search_settings.max_inline_searches:
                break

            # Immediately start running the function, rather than waiting
            # until the asyncio.gather call to get invoked when generation
            # completes.
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                _do_inline_search(self.httpx_client, query, query_type)
            )
            self.inline_search_promises.append(task)

    def pop_completed_searches(self) -> list:
        """Return results of any completed search tasks and remove them from the pending list."""
        completed = []
        remaining = []
        for task in self.inline_search_promises:
            if task.done():
                try:
                    completed.append(task.result())
                except Exception as e:
                    logger.warning(f"Inline search task failed: {e}")
            else:
                remaining.append(task)
        self.inline_search_promises = remaining
        return completed

    async def get_inline_search_chunks(self):
        results = await asyncio.gather(
            *self.inline_search_promises, return_exceptions=True
        )
        chunks = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Inline search task failed")
                continue
            chunks.append(result)
        return chunks
