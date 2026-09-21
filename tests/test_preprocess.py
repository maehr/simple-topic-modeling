from simple_topic_modeling.config import PreprocessConfig
from simple_topic_modeling.preprocess import clean_text, clean_texts, frequent_terms, token_pattern


def test_each_cleaning_switch_can_be_turned_off():
    text = "Visit https://a.co mail me@a.co 1,250 Café"
    off = PreprocessConfig(
        lowercase=False, strip_urls=False, strip_emails=False, normalize_accents=False
    )
    assert clean_text(text, off) == text


def test_urls_are_stripped_before_lowercasing():
    config = PreprocessConfig()
    assert clean_text("See HTTPS://Example.COM/Path now", config) == "see now"


def test_www_urls_are_stripped():
    assert clean_text("see www.example.com now", PreprocessConfig()) == "see now"


def test_numbers_are_kept_by_default():
    assert clean_text("Total 1250 items", PreprocessConfig()) == "total 1250 items"


def test_accents_are_kept_by_default():
    assert clean_text("Café", PreprocessConfig()) == "café"


def test_whitespace_collapses():
    assert clean_text("a \n\t  b", PreprocessConfig()) == "a b"


def test_clean_texts_maps_over_the_corpus():
    assert clean_texts(["A  B", "C"], PreprocessConfig()) == ["a b", "c"]


def test_token_pattern_enforces_the_minimum_length():
    import re

    assert re.findall(token_pattern(2), "a bb ccc") == ["bb", "ccc"]


def test_frequent_terms_ranks_ties_alphabetically():
    assert frequent_terms(["beta alpha"], frozenset()) == [("alpha", 1), ("beta", 1)]


def test_frequent_terms_respects_the_minimum_token_length():
    terms = frequent_terms(["a bb ccc"], frozenset(), min_token_length=3)
    assert terms == [("ccc", 1)]


def test_frequent_terms_returns_empty_when_every_term_is_a_stop_word():
    assert frequent_terms(["cat dog"], frozenset({"cat", "dog"})) == []


def test_frequent_terms_returns_empty_for_an_empty_corpus():
    assert frequent_terms([""], frozenset()) == []
