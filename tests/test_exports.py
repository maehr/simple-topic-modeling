import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig, load_app_config
from simple_topic_modeling.exports import (
    config_json,
    documents_topics_frame,
    project_zip,
    to_csv_bytes,
    topic_similarity_frame,
    topic_terms_frame,
    topics_frame,
)
from simple_topic_modeling.io import build_corpus
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
    assert archive.namelist() == [
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
