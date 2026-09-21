import pytest

from simple_topic_modeling.config import LANGUAGE_LABELS, StopWordConfig
from simple_topic_modeling.stopwords import base_stopwords, effective_stopwords, parse_word_input


@pytest.mark.parametrize("language", sorted(LANGUAGE_LABELS))
def test_every_language_ships_a_non_empty_lowercase_list(language):
    words = base_stopwords(language)
    assert len(words) > 100
    assert all(word == word.lower() for word in words)
    assert all(word.strip() for word in words)


@pytest.mark.parametrize(
    ("language", "word"),
    [("en", "the"), ("de", "und"), ("fr", "le"), ("it", "di"), ("es", "el")],
)
def test_each_list_holds_the_word_named_in_the_spec(language, word):
    assert word in base_stopwords(language)


def test_separators_are_interchangeable():
    assert parse_word_input("a,b") == parse_word_input("a b") == parse_word_input("a\nb")


def test_added_words_join_the_base_list():
    words = effective_stopwords("en", StopWordConfig(added="widget"))
    assert "widget" in words
    assert "the" in words


def test_kept_words_leave_the_effective_set():
    words = effective_stopwords("en", StopWordConfig(always_keep="the, and"))
    assert "the" not in words
    assert "and" not in words


def test_always_keep_wins_over_added():
    words = effective_stopwords("en", StopWordConfig(added="widget", always_keep="widget"))
    assert "widget" not in words


def test_no_base_list_leaves_only_the_added_words():
    config = StopWordConfig(use_base_list=False, added="alpha beta")
    assert effective_stopwords("en", config) == frozenset({"alpha", "beta"})


def test_no_base_list_and_no_additions_is_empty():
    assert effective_stopwords("en", StopWordConfig(use_base_list=False)) == frozenset()
