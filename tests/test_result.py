import numpy as np
import pytest
from scipy.sparse import csr_matrix

from simple_topic_modeling.result import (
    _example_result,
    auto_labels,
    dominant_topics,
    normalize_rows,
    project_documents,
    rename_topic,
    top_terms,
    topic_centroids,
)


def test_normalize_rows_leaves_an_all_zero_row_alone():
    out = normalize_rows(np.array([[0.0, 0.0], [2.0, 2.0]]))
    assert out.tolist() == [[0.0, 0.0], [0.5, 0.5]]


def test_top_terms_asks_for_more_terms_than_exist():
    weights = np.array([[0.6, 0.4]])
    assert top_terms(weights, ["a", "b"], 5) == [["a", "b"]]


def test_auto_label_is_one_indexed_and_uses_three_terms():
    weights = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    labels = auto_labels(weights, ["a", "b", "c", "d"])
    assert labels == ["Topic 1 · a, b, c", "Topic 2 · d, c, b"]


def test_projection_is_deterministic():
    matrix = csr_matrix(np.random.default_rng(0).random((10, 6)))
    assert np.allclose(project_documents(matrix), project_documents(matrix))


def test_projection_pads_a_single_column_matrix():
    out = project_documents(csr_matrix(np.array([[1.0], [2.0], [3.0]])))
    assert out.shape == (3, 2)
    assert out[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_projection_handles_a_two_column_matrix():
    out = project_documents(csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]])))
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_a_topic_with_no_weight_sits_at_the_origin():
    xy = np.array([[1.0, 1.0], [2.0, 2.0]])
    weights = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert topic_centroids(xy, weights)[1].tolist() == [0.0, 0.0]


def test_dominant_topic_breaks_a_tie_with_the_lowest_index():
    index, _ = dominant_topics(np.array([[0.5, 0.5]]))
    assert index.tolist() == [0]


def test_renaming_does_not_change_the_original_result():
    original = _example_result()
    renamed = rename_topic(original, 0, "Economy")
    assert original.topic_names[0].startswith("Topic 1 ·")
    assert renamed.topic_names[0] == "Economy"


def test_renaming_keeps_every_other_field_identical():
    original = _example_result()
    renamed = rename_topic(original, 0, "Economy")
    assert np.array_equal(original.document_topic, renamed.document_topic)
    assert original.topic_auto_labels == renamed.topic_auto_labels


def test_result_is_frozen():
    with pytest.raises(AttributeError):
        _example_result().model_type = "lda"  # ty: ignore[invalid-assignment]


def test_to_dict_exposes_every_documented_field():
    keys = set(_example_result().to_dict())
    assert {"model_type", "config", "documents", "document_topic", "metrics"} <= keys
