"""Vectorize a corpus and fit a topic model.

`SPECS.md` section 4 fixes the vectorizer and the estimator settings. The document-term matrix stays
sparse, as `SPECS.md` section 8 requires.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from simple_topic_modeling.config import AppConfig, ModelConfig
from simple_topic_modeling.errors import (
    MIN_DOCUMENTS,
    EmptyVocabularyError,
    TooFewDocumentsError,
    TooManyTopicsError,
    model_did_not_converge,
)
from simple_topic_modeling.preprocess import clean_texts, token_pattern
from simple_topic_modeling.result import (
    TopicModelResult,
    auto_labels,
    dominant_topics,
    normalize_rows,
    project_documents,
    topic_centroids,
)
from simple_topic_modeling.stopwords import effective_stopwords, vectorizer_stopwords

if TYPE_CHECKING:
    from simple_topic_modeling.io import Corpus

__all__ = ["build_vectorizer", "fit_topic_model", "vectorize"]


def build_vectorizer(config: AppConfig) -> TfidfVectorizer | CountVectorizer:
    """Build the vectorizer that `SPECS.md` section 4 specifies for the chosen model.

    NMF uses TF-IDF. LDA uses raw counts. Both run with `lowercase=False`, because
    `simple_topic_modeling.preprocess` already lowercased the text.

    >>> build_vectorizer(AppConfig()).sublinear_tf
    True
    >>> type(build_vectorizer(AppConfig(model=ModelConfig(model_type="lda")))).__name__
    'CountVectorizer'
    """
    model = config.model
    shared: dict[str, Any] = {
        "stop_words": vectorizer_stopwords(
            effective_stopwords(config.language, config.stop_words),
            config.preprocess.min_token_length,
        ),
        "min_df": model.min_df,
        "max_df": model.max_df,
        "max_features": model.max_features,
        "ngram_range": model.ngram_range,
        "lowercase": False,
        "token_pattern": token_pattern(config.preprocess.min_token_length),
    }
    if model.model_type == "nmf":
        return TfidfVectorizer(sublinear_tf=True, **shared)
    return CountVectorizer(**shared)


def vectorize(documents: Sequence[str], config: AppConfig) -> tuple[spmatrix, list[str]]:
    """Clean the documents and build the sparse document-term matrix.

    >>> corpus = ["cat dog runs", "cat sleeps", "dog barks", "cat dog plays"]
    >>> matrix, terms = vectorize(corpus, AppConfig())
    >>> matrix.shape[0]
    4
    >>> "cat" in terms
    True
    """
    cleaned = clean_texts(documents, config.preprocess)
    vectorizer = build_vectorizer(config)
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError as error:
        raise EmptyVocabularyError from error
    return matrix, list(vectorizer.get_feature_names_out())


def _fit_estimator(
    matrix: spmatrix, model: ModelConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool]:
    """Fit NMF or LDA and report the document-topic and topic-term matrices."""
    max_iter = model.resolved_max_iter
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if model.model_type == "nmf":
            estimator = NMF(
                n_components=model.n_topics,
                init="nndsvda",
                max_iter=max_iter,
                random_state=model.random_seed,
            )
            document_topic = estimator.fit_transform(matrix)
            extra: dict[str, Any] = {"reconstruction_error": float(estimator.reconstruction_err_)}
        else:
            estimator = LatentDirichletAllocation(
                n_components=model.n_topics,
                max_iter=max_iter,
                random_state=model.random_seed,
                learning_method="batch",
            )
            document_topic = estimator.fit_transform(matrix)
            extra = {"perplexity": float(estimator.perplexity(matrix))}
        converged = not any(issubclass(entry.category, ConvergenceWarning) for entry in caught)
    extra["n_iter"] = int(getattr(estimator, "n_iter_", max_iter))
    return np.asarray(document_topic), np.asarray(estimator.components_), extra, converged


def fit_topic_model(corpus: Corpus, config: AppConfig) -> TopicModelResult:
    """Fit the model and build the full result object.

    This is the one expensive step. `SPECS.md` section 4 runs it only on an explicit action.

    >>> from simple_topic_modeling.io import build_corpus
    >>> texts = ["cat dog runs fast", "cat sleeps often", "dog barks loudly",
    ...          "cat dog play together", "bird sings song", "bird flies high"]
    >>> corpus, _ = build_corpus(texts, [f"d{i}" for i in range(6)])
    >>> config = AppConfig(model=ModelConfig(n_topics=2, min_df=1))
    >>> result = fit_topic_model(corpus, config)
    >>> result.document_topic.shape
    (6, 2)
    >>> bool(np.allclose(result.document_topic.sum(axis=1), 1.0))
    True
    >>> bool(np.isclose(result.topic_prevalence.sum(), 1.0))
    True
    >>> result.topic_xy.shape
    (2, 2)
    >>> result.metrics["converged"]
    True
    """
    if len(corpus) < MIN_DOCUMENTS:
        raise TooFewDocumentsError(len(corpus))
    matrix, feature_names = vectorize(corpus.documents, config)
    limit = min(matrix.shape)
    if config.model.n_topics > limit:
        raise TooManyTopicsError(config.model.n_topics, limit)

    document_topic_raw, topic_term_raw, extra, converged = _fit_estimator(matrix, config.model)
    document_topic = normalize_rows(document_topic_raw)
    topic_term = normalize_rows(topic_term_raw)
    document_xy = project_documents(matrix)
    labels = auto_labels(topic_term, feature_names)
    index, score = dominant_topics(document_topic)

    metrics: dict[str, Any] = {
        "document_count": len(corpus),
        "vocabulary_size": len(feature_names),
        "vocabulary_at_limit": len(feature_names) >= config.model.max_features,
        "converged": converged,
        **extra,
    }
    if not converged:
        metrics["convergence_notice"] = model_did_not_converge(config.model.resolved_max_iter)

    return TopicModelResult(
        model_type=config.model.model_type,
        config=config.model_dump(),
        documents=list(corpus.documents),
        document_ids=list(corpus.document_ids),
        metadata=corpus.metadata,
        feature_names=feature_names,
        document_topic_raw=document_topic_raw,
        document_topic=document_topic,
        topic_term_raw=topic_term_raw,
        topic_term=topic_term,
        document_xy=document_xy,
        topic_xy=topic_centroids(document_xy, document_topic),
        topic_names=list(labels),
        topic_auto_labels=labels,
        topic_prevalence=document_topic.mean(axis=0),
        dominant_topic=index,
        dominant_topic_score=score,
        metrics=metrics,
    )
