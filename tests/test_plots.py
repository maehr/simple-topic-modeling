import numpy as np
import pandas as pd
import pytest

from browser_topics.config import AppConfig, ModelConfig
from browser_topics.io import build_corpus
from browser_topics.modeling import fit_topic_model
from browser_topics.plots import (
    SNIPPET_LENGTH,
    representative_documents,
    snippet,
    top_term_frame,
    topic_cards,
)
from browser_topics.result import _example_result, rename_topic


@pytest.fixture
def result(corpus, config):
    return fit_topic_model(corpus, config)


def test_snippet_collapses_whitespace():
    assert snippet("a\n\n  b\tc") == "a b c"


def test_snippet_truncates_at_the_documented_length():
    long_text = "x" * (SNIPPET_LENGTH + 50)
    out = snippet(long_text)
    assert len(out) == SNIPPET_LENGTH + 1
    assert out.endswith("…")


def test_cards_cover_every_topic(result):
    cards = topic_cards(result)
    assert len(cards) == result.n_topics
    assert cards["documents"].sum() == result.n_documents


def test_cards_count_a_topic_that_dominates_nothing():
    built = _example_result()
    object.__setattr__(built, "dominant_topic", np.zeros(built.n_documents, dtype=int))
    cards = topic_cards(built)
    assert cards["documents"].tolist() == [built.n_documents, 0]


def test_cards_show_the_current_topic_names(result):
    renamed = rename_topic(result, 0, "Economy")
    assert topic_cards(renamed)["topic"].tolist()[0] == "Economy"


def test_term_frame_is_ranked_and_bounded(result):
    frame = top_term_frame(result, 0, term_count=4)
    assert len(frame) == 4
    assert frame["weight"].is_monotonic_decreasing


def test_term_frame_asks_for_more_terms_than_the_vocabulary_holds(result):
    frame = top_term_frame(result, 0, term_count=10_000)
    assert len(frame) == len(result.feature_names)


def test_representative_documents_are_ranked_by_score(result):
    frame = representative_documents(result, 0, count=5)
    assert frame["score"].is_monotonic_decreasing
    assert len(frame) == 5


def test_representative_documents_carry_metadata():
    texts = [
        "cat dog runs fast",
        "cat sleeps warm couch",
        "dog barks postman loudly",
        "bird sings morning song",
        "bird flies above trees",
        "fish swims cold water",
    ]
    metadata = pd.DataFrame({"group": list("aabbcc")})
    corpus, _ = build_corpus(texts, [str(i) for i in range(6)], metadata)
    built = fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))
    assert "group" in representative_documents(built, 0).columns


def test_representative_documents_show_a_snippet_not_the_full_text():
    texts = [f"{word} " * 400 for word in ("cat", "dog", "bird", "fish", "tree", "stone")]
    corpus, _ = build_corpus(texts, [str(index) for index in range(6)])
    built = fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))
    frame = representative_documents(built, 0)
    assert all(len(value) <= SNIPPET_LENGTH + 1 for value in frame["snippet"])
    assert all(value.endswith("…") for value in frame["snippet"])
