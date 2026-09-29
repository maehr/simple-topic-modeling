from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig
from simple_topic_modeling.metrics import (
    DIAGNOSTIC_HELP,
    RunSummary,
    compare_runs,
    corpus_fingerprint,
    diagnostic_table,
    diagnostics,
    mean_pairwise_similarity,
    run_summary,
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


def test_every_measure_has_a_help_sentence_without_a_threshold():
    assert len(DIAGNOSTIC_HELP) == 8
    for sentence in DIAGNOSTIC_HELP.values():
        assert sentence.endswith(".")
        assert not any(mark in sentence for mark in (". ", "? ", "! "))
        assert "good" not in sentence.lower()


def test_the_topic_help_says_that_the_reader_sets_the_count():
    assert "asked for" in DIAGNOSTIC_HELP["Topics"]
    assert "found" not in DIAGNOSTIC_HELP["Topics"]


def test_the_nmf_help_names_the_tfidf_weights():
    assert "TF-IDF" in DIAGNOSTIC_HELP["Reconstruction error"]


def test_the_fingerprint_is_short_and_stable():
    first = corpus_fingerprint(["alpha", "beta"], ["a", "b"])
    assert len(first) == 12
    assert first == corpus_fingerprint(["alpha", "beta"], ["a", "b"])


def test_the_fingerprint_changes_with_a_text_an_id_or_the_order():
    base = corpus_fingerprint(["alpha", "beta"], ["a", "b"])
    assert corpus_fingerprint(["alpha", "gamma"], ["a", "b"]) != base
    assert corpus_fingerprint(["alpha", "beta"], ["a", "c"]) != base
    assert corpus_fingerprint(["beta", "alpha"], ["b", "a"]) != base


def test_the_fingerprint_keeps_a_shifted_boundary_apart():
    assert corpus_fingerprint(["ab", "c"], ["1", "2"]) != corpus_fingerprint(
        ["a", "bc"], ["1", "2"]
    )


def test_run_summary_keeps_the_scalars_and_a_label_with_the_seed(corpus, config):
    summary = run_summary(fit_topic_model(corpus, config))
    assert isinstance(summary, RunSummary)
    assert summary.label == "NMF · 3 topics · seed 42"
    assert summary.model_type == "nmf"
    assert summary.topic_count == 3
    assert summary.corpus_fingerprint == corpus_fingerprint(corpus.documents, corpus.document_ids)
    assert summary.perplexity is None
    assert summary.reconstruction_error is not None


def test_run_summary_reuses_a_report_that_the_caller_holds(corpus, config):
    result = fit_topic_model(corpus, config)
    assert run_summary(result, diagnostics(result)) == run_summary(result)


def test_run_summary_is_frozen(corpus, config):
    summary = run_summary(fit_topic_model(corpus, config))
    with pytest.raises(FrozenInstanceError):
        summary.label = "other"  # ty: ignore[invalid-assignment]


def test_run_summary_drops_the_seed_when_the_config_has_none():
    assert run_summary(_example_result()).label == "NMF · 2 topics"


def test_diagnostic_table_lists_every_measure_with_its_help(corpus, config):
    table = diagnostic_table(run_summary(fit_topic_model(corpus, config)))
    assert table["Measure"].tolist() == list(DIAGNOSTIC_HELP)
    assert table["What it tells you"].tolist() == list(DIAGNOSTIC_HELP.values())
    values = table.set_index("Measure")["This run"]
    assert values["Documents used"] == f"{len(corpus):,}"
    assert values["Perplexity"] == "—"
    assert values["Reconstruction error"] != "—"


def test_compare_runs_orders_the_runs_newest_first_with_unique_names(corpus, config):
    summary = run_summary(fit_topic_model(corpus, config))
    table = compare_runs([summary] * 4)
    assert table.columns.tolist() == [
        "Measure",
        "This run",
        "Run before",
        "2 runs before",
        "3 runs before",
    ]
    assert table["Measure"].tolist()[0] == "Settings"


def test_compare_runs_shows_a_dash_for_the_other_model_type(corpus, config):
    nmf = run_summary(fit_topic_model(corpus, config))
    lda_config = AppConfig(model=ModelConfig(model_type="lda", n_topics=3, min_df=1))
    lda = run_summary(fit_topic_model(corpus, lda_config))
    table = compare_runs([lda, nmf]).set_index("Measure")
    assert table.loc["Settings", "This run"].startswith("LDA")
    assert table.loc["Reconstruction error", "This run"] == "—"
    assert table.loc["Perplexity", "This run"] != "—"
    assert table.loc["Reconstruction error", "Run before"] != "—"
    assert table.loc["Perplexity", "Run before"] == "—"


def test_compare_runs_of_no_runs_has_only_the_measure_column():
    assert compare_runs([]).columns.tolist() == ["Measure"]
