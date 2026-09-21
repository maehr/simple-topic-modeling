import numpy as np

from simple_topic_modeling.config import AppConfig, ModelConfig
from simple_topic_modeling.metrics import (
    diagnostics,
    mean_pairwise_similarity,
    topic_diversity,
    topic_similarity,
)
from simple_topic_modeling.modeling import fit_topic_model
from simple_topic_modeling.result import _example_result


def test_similarity_is_symmetric_with_a_unit_diagonal():
    weights = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
    similarity = topic_similarity(weights)
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)


def test_similarity_stays_inside_the_valid_range():
    weights = np.random.default_rng(0).random((5, 8))
    similarity = topic_similarity(weights)
    assert similarity.min() >= -1.0
    assert similarity.max() <= 1.0


def test_an_all_zero_topic_has_zero_similarity_to_the_others():
    weights = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert topic_similarity(weights)[1].tolist() == [0.0, 0.0]


def test_mean_pairwise_similarity_ignores_the_diagonal():
    similarity = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert np.isclose(mean_pairwise_similarity(similarity), 0.4)


def test_diversity_handles_a_vocabulary_smaller_than_the_term_count():
    assert topic_diversity(np.array([[1.0, 0.5]]), ["a", "b"], count=10) == 1.0


def test_diversity_is_zero_without_a_vocabulary():
    assert topic_diversity(np.zeros((2, 0)), [], count=5) == 0.0


def test_similar_topics_raise_a_notice():
    result = _example_result()
    details = [notice.detail for notice in diagnostics(result)["notices"]]
    assert "Several topics use very similar terms." in details


def test_a_vocabulary_at_the_limit_raises_a_notice(corpus):
    config = AppConfig(model=ModelConfig(n_topics=3, min_df=1, max_features=5))
    notices = diagnostics(fit_topic_model(corpus, config))["notices"]
    details = [notice.detail for notice in notices]
    assert "The vocabulary reached the configured maximum." in details


def test_a_convergence_notice_reaches_the_diagnostics(corpus):
    config = AppConfig(model=ModelConfig(n_topics=3, min_df=1, max_iter=1))
    notices = diagnostics(fit_topic_model(corpus, config))["notices"]
    details = [notice.detail for notice in notices]
    assert "The model stopped before it converged." in details


def test_weak_dominant_scores_raise_a_notice():
    result = _example_result()
    weak = np.full(result.dominant_topic_score.shape, 0.1)
    object.__setattr__(result, "dominant_topic_score", weak)
    details = [notice.detail for notice in diagnostics(result)["notices"]]
    assert "Many documents have weak dominant-topic scores." in details


def test_a_clean_run_reports_the_numbers_without_a_quality_score(corpus, config):
    report = diagnostics(fit_topic_model(corpus, config))
    assert report["documents_used"] == len(corpus)
    assert report["topic_count"] == 3
    assert 0.0 <= report["topic_diversity"] <= 1.0
    assert report["perplexity"] is None
    assert "score" not in report
    assert "quality" not in report
