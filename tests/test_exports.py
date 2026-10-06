import io
import json
import zipfile

import altair as alt
import numpy as np
import pandas as pd
import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig, load_app_config
from simple_topic_modeling.exports import (
    EXPORT_LABELS,
    config_json,
    documents_topics_frame,
    figure_files,
    project_zip,
    to_csv_bytes,
    topic_similarity_frame,
    topic_terms_frame,
    topics_frame,
)
from simple_topic_modeling.io import build_corpus, split_long_document
from simple_topic_modeling.modeling import fit_topic_model
from simple_topic_modeling.result import rename_topic


@pytest.fixture
def result(corpus, config):
    return fit_topic_model(corpus, config)


def test_document_rows_match_the_corpus(result):
    frame = documents_topics_frame(result)
    assert len(frame) == result.n_documents
    assert frame["document_id"].tolist() == result.document_ids


def test_topic_score_columns_follow_the_documented_order(result):
    columns = documents_topics_frame(result).columns.tolist()
    expected = [f"topic_{index}_score" for index in range(result.n_topics)]
    start = columns.index("topic_0_score")
    assert columns[start : start + result.n_topics] == expected
    assert columns[-2:] == ["projection_x", "projection_y"]


def test_document_scores_sum_to_one_in_the_export(result):
    frame = documents_topics_frame(result)
    scores = frame[[f"topic_{i}_score" for i in range(result.n_topics)]]
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_metadata_columns_are_carried_through():
    texts = [
        "cat dog runs fast",
        "cat sleeps warm couch",
        "dog barks postman loudly",
        "bird sings morning song",
        "bird flies above trees",
        "fish swims cold water",
    ]
    metadata = pd.DataFrame({"group": list("aabbcc"), "date": ["2024-01-01"] * 6})
    corpus, _ = build_corpus(texts, [str(i) for i in range(6)], metadata)
    result = fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))
    frame = documents_topics_frame(result)
    assert "group" in frame.columns
    assert "date" in frame.columns


def test_text_column_is_opt_in(result):
    assert "text" not in documents_topics_frame(result).columns
    with_text = documents_topics_frame(result, include_text=True)
    assert with_text["text"].tolist() == result.documents


def test_renaming_reaches_every_export(result):
    renamed = rename_topic(result, 0, "Economy")
    assert "Economy" in topics_frame(renamed)["topic_name"].tolist()
    assert "Economy" in topic_terms_frame(renamed)["topic_name"].tolist()
    names = topic_similarity_frame(renamed)
    assert "Economy" in set(names["topic_a_name"]) | set(names["topic_b_name"])


def test_topics_frame_has_one_row_per_topic(result):
    frame = topics_frame(result)
    assert len(frame) == result.n_topics
    assert np.isclose(frame["prevalence"].sum(), 1.0)


def test_topic_terms_are_ranked_by_descending_weight(result):
    frame = topic_terms_frame(result, top_n=5)
    for _, group in frame.groupby("topic_id"):
        assert group["rank"].tolist() == [1, 2, 3, 4, 5]
        assert group["weight_normalized"].is_monotonic_decreasing


def test_topic_terms_asks_for_more_terms_than_the_vocabulary_holds(result):
    frame = topic_terms_frame(result, top_n=10_000)
    assert len(frame) == result.n_topics * len(result.feature_names)


def test_similarity_frame_lists_each_pair_once(result):
    frame = topic_similarity_frame(result)
    expected = result.n_topics * (result.n_topics - 1) // 2
    assert len(frame) == expected
    assert (frame["topic_a_id"] < frame["topic_b_id"]).all()


def test_config_json_round_trips_through_the_loader():
    original = AppConfig(language="de", model=ModelConfig(n_topics=7))
    restored = load_app_config(json.loads(config_json(original, ["Economy"])))
    assert restored == original


def test_an_exported_config_with_the_sp_alias_imports_as_es():
    payload = json.loads(config_json(AppConfig(language="es")))
    payload["language"] = "sp"
    assert load_app_config(payload).language == "es"


def test_config_json_records_the_topic_names():
    payload = json.loads(config_json(AppConfig(), ["Economy", "Sport"]))
    assert payload["topic_names"] == ["Economy", "Sport"]


def test_config_json_keeps_non_ascii_readable():
    payload = json.loads(config_json(AppConfig(), ["Zürich"]))
    assert payload["topic_names"] == ["Zürich"]


def test_csv_bytes_are_utf8():
    frame = pd.DataFrame({"term": ["Zürich"]})
    assert "Zürich" in to_csv_bytes(frame).decode("utf-8")


def test_zip_holds_the_six_documented_entries(result):
    archive = zipfile.ZipFile(io.BytesIO(project_zip(result, AppConfig())))
    names = [name for name in archive.namelist() if not name.startswith("figures/")]
    assert names == [
        "documents_topics.csv",
        "topics.csv",
        "topic_terms.csv",
        "topic_similarity.csv",
        "config.json",
        "README.txt",
    ]


def test_zip_entries_are_readable(result):
    archive = zipfile.ZipFile(io.BytesIO(project_zip(result, AppConfig())))
    frame = pd.read_csv(io.BytesIO(archive.read("documents_topics.csv")))
    assert len(frame) == result.n_documents
    assert json.loads(archive.read("config.json"))["language"] == "en"
    assert "Topic scores are shares" in archive.read("README.txt").decode("utf-8")


def test_zip_carries_the_text_when_asked(result):
    archive = zipfile.ZipFile(io.BytesIO(project_zip(result, AppConfig(), include_text=True)))
    frame = pd.read_csv(io.BytesIO(archive.read("documents_topics.csv")))
    assert "text" in frame.columns


def test_every_single_file_has_a_task_label(result):
    archive = zipfile.ZipFile(io.BytesIO(project_zip(result, AppConfig())))
    files = {name for name in archive.namelist() if not name.startswith("figures/")}
    assert set(EXPORT_LABELS) == files - {"README.txt"}


def test_zip_holds_every_figure_of_a_plain_corpus(result):
    archive = zipfile.ZipFile(io.BytesIO(project_zip(result, AppConfig())))
    figures = {name for name in archive.namelist() if name.startswith("figures/")}
    per_topic = {
        f"figures/topic_{topic:02d}_{kind}"
        for topic in range(1, result.n_topics + 1)
        for kind in ("top_terms.html", "wordcloud.png")
    }
    base = {
        "figures/topic_map.html",
        "figures/topic_prevalence.html",
        "figures/topic_similarity.html",
        "figures/document_map.html",
        "figures/dominant_topic_score_distribution.html",
    }
    assert figures == base | per_topic


def test_an_html_figure_carries_its_data(result):
    page = figure_files(result)["topic_map.html"].decode("utf-8")
    assert "vega-embed" in page
    assert '"datasets"' in page
    assert '"number"' in page


def test_the_document_map_holds_text_only_when_asked(result):
    assert b'"snippet"' not in figure_files(result)["document_map.html"]
    assert b'"snippet"' in figure_files(result, include_text=True)["document_map.html"]


def _fit_with_metadata(dates):
    texts = [
        "cat dog runs fast",
        "cat sleeps warm couch",
        "dog barks postman loudly",
        "bird sings morning song",
        "bird flies above trees",
        "fish swims cold water",
    ]
    metadata = pd.DataFrame({"group": ["a", "a", "b", "b", "c", "c"], "date": dates})
    corpus, _ = build_corpus(texts, [f"d{index}" for index in range(len(texts))], metadata)
    return fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))


def test_metadata_figures_appear_when_their_data_exists():
    dates = ["2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20", "2024-05-25", "2024-06-30"]
    names = set(figure_files(_fit_with_metadata(dates)))
    assert {"group_shares.html", "topic_shares_over_time.html"} <= names


def test_unreadable_dates_give_no_time_figure():
    names = set(figure_files(_fit_with_metadata(["soon"] * 6)))
    assert "group_shares.html" in names
    assert "topic_shares_over_time.html" not in names


def test_a_long_text_adds_the_position_figures():
    text = "\n\n".join(
        ["cat dog runs fast", "cat sleeps warm couch", "bird sings morning song"] * 4
    )
    segments, identifiers, metadata = split_long_document(text, "book.txt")
    corpus, _ = build_corpus(segments, identifiers, metadata)
    config = AppConfig(model=ModelConfig(n_topics=2, min_df=1), analyse_as="long_document")
    names = set(figure_files(fit_topic_model(corpus, config)))
    assert {"topic_positions.html", "topic_01_positions.html", "topic_02_positions.html"} <= names


def test_figures_ignore_the_row_limit(result, monkeypatch):
    # Altair refuses more than 5,000 rows by default. A long text exceeds that limit.
    big = alt.Chart(pd.DataFrame({"x": range(5001)})).mark_point()
    monkeypatch.setattr("simple_topic_modeling.plots.score_histogram", lambda frame: big)
    page = figure_files(result)["dominant_topic_score_distribution.html"]
    assert b'"datasets"' in page


def test_a_figure_escapes_a_name_that_closes_the_script(result):
    attack = "</script><img src=x onerror=alert(1)>"
    files = figure_files(rename_topic(result, 0, attack))
    for name, data in files.items():
        if name.endswith(".html"):
            assert attack.encode("utf-8") not in data
            assert data.count(b"</script>") == data.count(b"<script")
    assert b"\\u003c/script\\u003e" in files["topic_map.html"]
