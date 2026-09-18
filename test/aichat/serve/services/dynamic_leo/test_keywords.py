import pytest

from aichat.serve.services.dynamic_leo import keywords
from aichat.serve.services.dynamic_leo.config import DynamicLeoCategoryConfig
from aichat.serve.services.dynamic_leo.keywords import (
    automaton_for_categories,
    keyword_match_reasons_for_text,
)


@pytest.fixture(autouse=True)
def clear_automaton_cache():
    keywords._automaton_cache_sig = None
    keywords._automaton_cache = None
    yield
    keywords._automaton_cache_sig = None
    keywords._automaton_cache = None


@pytest.fixture
def automaton():
    return automaton_for_categories(
        {
            "coding": DynamicLeoCategoryConfig(
                keywords=["python", "function", "import"],
            ),
            "choice_comparison": DynamicLeoCategoryConfig(
                keywords=["pick between", "vs"],
            ),
            "brave_products": DynamicLeoCategoryConfig(
                keywords=["bat", "brave"],
            ),
        }
    )


def test_keyword_match(automaton):
    result = keyword_match_reasons_for_text(
        "write a Python function",
        automaton,
        reason_tag="keywords_last",
    )
    assert result == {"coding": frozenset({"keywords_last"})}


def test_multi_word_keyword(automaton):
    result = keyword_match_reasons_for_text(
        "pick between these two options",
        automaton,
        reason_tag="keywords_last",
    )
    assert result == {"choice_comparison": frozenset({"keywords_last"})}


def test_word_boundaries_reject_substrings(automaton):
    for text in (
        "this is a debate about politics",
        "this is important information",
    ):
        assert (
            keyword_match_reasons_for_text(text, automaton, reason_tag="keywords_last")
            == {}
        )


def test_automaton_cache_reuses_instance():
    cats = {
        "coding": DynamicLeoCategoryConfig(keywords=["python"]),
    }
    assert automaton_for_categories(cats) is automaton_for_categories(cats)
