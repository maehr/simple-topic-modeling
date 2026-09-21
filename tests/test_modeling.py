import numpy as np
import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig, PreprocessConfig, StopWordConfig
from simple_topic_modeling.errors import (
    EmptyVocabularyError,
    TooFewDocumentsError,
    TooManyTopicsError,
)
from simple_topic_modeling.io import build_corpus
from simple_topic_modeling.modeling import build_vectorizer, fit_topic_model, vectorize


def test_nmf_uses_tfidf_with_sublinear_scaling():
    vectorizer = build_vectorizer(AppConfig())
    assert type(vectorizer).__name__ == "TfidfVectorizer"
    assert vectorizer.sublinear_tf is True  # ty: ignore[unresolved-attribute]
    assert vectorizer.lowercase is False


def test_lda_uses_raw_counts():
    config = AppConfig(model=ModelConfig(model_type="lda"))
    assert type(build_vectorizer(config)).__name__ == "CountVectorizer"


def test_vectorizer_carries_the_documented_parameters():
    model = ModelConfig(min_df=3, max_df=0.8, max_features=123, ngrams="1-2")
    vectorizer = build_vectorizer(AppConfig(model=model))
    assert vectorizer.min_df == 3
    assert vectorizer.max_df == 0.8
    assert vectorizer.max_features == 123
    assert vectorizer.ngram_range == (1, 2)


def test_no_base_list_leaves_the_vectorizer_without_stop_words():
    config = AppConfig(stop_words=StopWordConfig(use_base_list=False))
    assert build_vectorizer(config).stop_words is None


def test_document_term_matrix_stays_sparse(corpus, config):
    matrix, _ = vectorize(corpus.documents, config)
    assert hasattr(matrix, "toarray")
    assert matrix.shape[0] == len(corpus)


def test_empty_vocabulary_is_reported(corpus):
    config = AppConfig(model=ModelConfig(min_df=99))
    with pytest.raises(EmptyVocabularyError):
        vectorize(corpus.documents, config)


def test_a_corpus_of_only_stop_words_reports_an_empty_vocabulary():
    with pytest.raises(EmptyVocabularyError):
        vectorize(["the and of", "the and of"], AppConfig(model=ModelConfig(min_df=1)))


def test_too_few_documents_is_reported():
    small, _ = build_corpus(["alpha beta", "gamma delta"], ["a", "b"])
    with pytest.raises(TooFewDocumentsError):
        fit_topic_model(small, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))


def test_too_many_topics_is_reported(corpus):
    config = AppConfig(model=ModelConfig(n_topics=30, min_df=1))
    with pytest.raises(TooManyTopicsError):
        fit_topic_model(corpus, config)


def test_fit_is_reproducible_under_the_fixed_seed(corpus, config):
    first = fit_topic_model(corpus, config)
    second = fit_topic_model(corpus, config)
    assert np.allclose(first.document_topic, second.document_topic)
    assert np.allclose(first.document_xy, second.document_xy)


def test_document_topic_rows_are_shares(corpus, config):
    result = fit_topic_model(corpus, config)
    assert np.allclose(result.document_topic.sum(axis=1), 1.0)
    assert result.document_topic.min() >= 0.0


def test_topic_term_rows_are_shares(corpus, config):
    result = fit_topic_model(corpus, config)
    assert np.allclose(result.topic_term.sum(axis=1), 1.0)


def test_prevalence_sums_to_one(corpus, config):
    result = fit_topic_model(corpus, config)
    assert np.isclose(result.topic_prevalence.sum(), 1.0)


def test_result_shapes_agree(corpus, config):
    result = fit_topic_model(corpus, config)
    topics, documents = config.model.n_topics, len(corpus)
    assert result.document_topic.shape == (documents, topics)
    assert result.topic_term.shape == (topics, len(result.feature_names))
    assert result.document_xy.shape == (documents, 2)
    assert result.topic_xy.shape == (topics, 2)
    assert result.dominant_topic.shape == (documents,)
    assert len(result.topic_names) == topics


def test_raw_matrices_are_kept_next_to_the_normalized_ones(corpus, config):
    result = fit_topic_model(corpus, config)
    assert result.document_topic_raw.shape == result.document_topic.shape
    assert not np.allclose(result.document_topic_raw.sum(axis=1), 1.0)


def test_metrics_report_the_nmf_reconstruction_error(corpus, config):
    result = fit_topic_model(corpus, config)
    assert result.metrics["reconstruction_error"] > 0
    assert result.metrics["converged"] is True
    assert result.metrics["vocabulary_at_limit"] is False


def test_lda_reports_perplexity(corpus):
    config = AppConfig(model=ModelConfig(model_type="lda", n_topics=3, min_df=1))
    result = fit_topic_model(corpus, config)
    assert result.metrics["perplexity"] > 0
    assert result.model_type == "lda"


def test_a_low_iteration_limit_reports_a_convergence_notice(corpus):
    config = AppConfig(model=ModelConfig(n_topics=3, min_df=1, max_iter=1))
    result = fit_topic_model(corpus, config)
    assert result.metrics["converged"] is False
    assert "Raise the iteration limit above 1" in result.metrics["convergence_notice"].recovery


def test_vocabulary_at_limit_is_flagged(corpus):
    config = AppConfig(model=ModelConfig(n_topics=3, min_df=1, max_features=5))
    result = fit_topic_model(corpus, config)
    assert result.metrics["vocabulary_at_limit"] is True


def test_config_travels_with_the_result(corpus, config):
    result = fit_topic_model(corpus, config)
    assert result.config["language"] == "en"
    assert result.config["model"]["n_topics"] == 3


def test_cleaning_options_reach_the_fit(corpus):
    config = AppConfig(
        preprocess=PreprocessConfig(min_token_length=6),
        model=ModelConfig(n_topics=3, min_df=1),
    )
    result = fit_topic_model(corpus, config)
    assert all(len(term) >= 6 for term in result.feature_names)
