import numpy as np
import pandas as pd
import pytest

from browser_topics.config import AppConfig, ModelConfig
from browser_topics.io import build_corpus
from browser_topics.modeling import fit_topic_model
from browser_topics.plots import (
    SNIPPET_LENGTH,
    document_frame,
    document_scatter,
    prevalence_bars,
    representative_documents,
    similarity_heatmap,
    similarity_long_frame,
    snippet,
    top_term_bars,
    top_term_frame,
    topic_cards,
    topic_map,
    word_cloud_png,
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


def _spec(chart):
    return chart.to_dict()


def test_topic_map_marks_size_by_prevalence(result):
    spec = _spec(topic_map(result))
    assert spec["mark"]["type"] == "circle"
    assert spec["encoding"]["size"]["field"] == "prevalence"


def test_projection_axes_are_de_emphasized(result):
    spec = _spec(topic_map(result))
    for channel in ("x", "y"):
        axis = spec["encoding"][channel]["axis"]
        assert axis["labels"] is False
        assert axis["ticks"] is False
        assert axis["grid"] is False


def test_prevalence_bars_are_sorted_by_value(result):
    spec = _spec(prevalence_bars(result))
    assert spec["encoding"]["y"]["sort"] == "-x"


def test_similarity_heatmap_covers_the_full_matrix(result):
    frame = similarity_long_frame(result)
    assert len(frame) == result.n_topics**2
    assert _spec(similarity_heatmap(result))["mark"]["type"] == "rect"


def test_similarity_scale_is_fixed_so_runs_compare(result):
    spec = _spec(similarity_heatmap(result))
    assert spec["encoding"]["color"]["scale"]["domain"] == [0, 1]


def test_every_chart_carries_a_title(result):
    charts = [
        topic_map(result),
        prevalence_bars(result),
        similarity_heatmap(result),
        top_term_bars(result, 0),
        document_scatter(document_frame(result)),
    ]
    for chart in charts:
        assert _spec(chart)["title"]


def test_every_chart_carries_a_tooltip(result):
    charts = [
        topic_map(result),
        prevalence_bars(result),
        similarity_heatmap(result),
        top_term_bars(result, 0),
        document_scatter(document_frame(result)),
    ]
    for chart in charts:
        assert _spec(chart)["encoding"]["tooltip"]


def test_document_frame_has_one_row_per_document(result):
    frame = document_frame(result)
    assert len(frame) == result.n_documents
    assert frame["topic"].isin(result.topic_names).all()


def test_document_frame_carries_metadata():
    texts = [
        "cat dog runs fast",
        "cat sleeps warm couch",
        "dog barks postman loudly",
        "bird sings morning song",
        "bird flies above trees",
        "fish swims cold water",
    ]
    metadata = pd.DataFrame({"group": list("aabbcc")})
    corpus, _ = build_corpus(texts, [str(index) for index in range(6)], metadata)
    built = fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))
    assert "group" in document_frame(built).columns


def test_scatter_sampling_is_reproducible(result):
    big = pd.concat([document_frame(result)] * 40, ignore_index=True)
    first = document_scatter(big, sample_limit=50).data
    second = document_scatter(big, sample_limit=50).data
    assert first["document_id"].tolist() == second["document_id"].tolist()


def test_scatter_keeps_every_point_below_the_limit(result):
    frame = document_frame(result)
    assert len(document_scatter(frame, sample_limit=10_000).data) == len(frame)


def test_word_cloud_is_a_png(result):
    data = word_cloud_png(result, 0)
    assert data.startswith(b"\x89PNG")
    assert len(data) > 1000


def test_word_cloud_is_deterministic(result):
    assert word_cloud_png(result, 0) == word_cloud_png(result, 0)


def test_word_cloud_uses_the_renamed_topic_terms(result):
    assert word_cloud_png(result, 0) != word_cloud_png(result, 1)
