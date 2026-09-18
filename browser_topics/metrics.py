"""Descriptive diagnostics.

`SPECS.md` section 6 asks for descriptive aids and friendly warnings. It forbids an automatic
quality score, so this module returns numbers and notices only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from browser_topics.errors import FriendlyMessage
from browser_topics.result import top_terms

if TYPE_CHECKING:
    from browser_topics.result import TopicModelResult

__all__ = [
    "DIVERSITY_TERM_COUNT",
    "SIMILAR_TOPIC_THRESHOLD",
    "WEAK_SCORE_SHARE",
    "WEAK_SCORE_THRESHOLD",
    "diagnostics",
    "mean_pairwise_similarity",
    "topic_diversity",
    "topic_similarity",
]

DIVERSITY_TERM_COUNT = 10
"""Terms per topic that the diversity measure compares."""

SIMILAR_TOPIC_THRESHOLD = 0.35
"""Mean pairwise similarity above which the app suggests fewer topics."""

WEAK_SCORE_THRESHOLD = 0.4
"""A dominant-topic score below this counts as weak."""

WEAK_SCORE_SHARE = 0.4
"""Share of weak documents above which the app mentions mixed themes."""


def topic_similarity(topic_term: np.ndarray) -> np.ndarray:
    """Return the cosine similarity between the topic-term vectors.

    A topic with no weight anywhere gets a similarity of 0 against every topic.

    >>> weights = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> topic_similarity(weights).round(3).tolist()
    [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    norms = np.linalg.norm(topic_term, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    unit = topic_term / safe
    return np.asarray(np.clip(unit @ unit.T, -1.0, 1.0), dtype=float)


def mean_pairwise_similarity(similarity: np.ndarray) -> float:
    """Average the similarity of every distinct topic pair.

    A single topic has no pair, so the result is 0.

    >>> mean_pairwise_similarity(np.array([[1.0, 0.5], [0.5, 1.0]]))
    0.5
    >>> mean_pairwise_similarity(np.array([[1.0]]))
    0.0
    """
    count = similarity.shape[0]
    if count < 2:
        return 0.0
    rows, columns = np.triu_indices(count, k=1)
    return float(similarity[rows, columns].mean())


def topic_diversity(
    topic_term: np.ndarray, feature_names: list[str], count: int = DIVERSITY_TERM_COUNT
) -> float:
    """Return the share of distinct terms across the top terms of every topic.

    1.0 means no topic repeats another topic's term. A low value means the topics overlap.

    >>> names = ["a", "b", "c", "d"]
    >>> identical = np.array([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    >>> topic_diversity(identical, names, count=2)
    0.5
    >>> distinct = np.array([[4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 3.0]])
    >>> topic_diversity(distinct, names, count=2)
    1.0
    """
    limit = min(count, len(feature_names))
    terms = top_terms(topic_term, feature_names, limit)
    total = sum(len(row) for row in terms)
    if total == 0:
        return 0.0
    distinct = {term for row in terms for term in row}
    return len(distinct) / total


def diagnostics(result: TopicModelResult) -> dict[str, Any]:
    """Collect the descriptive numbers and the friendly notices of `SPECS.md` section 6.

    >>> from browser_topics.result import _example_result
    >>> report = diagnostics(_example_result())
    >>> report["topic_count"], report["documents_used"]
    (2, 3)
    >>> sorted(report)[:3]
    ['documents_used', 'mean_pairwise_similarity', 'notices']
    >>> all(isinstance(notice, FriendlyMessage) for notice in report["notices"])
    True
    """
    similarity = topic_similarity(result.topic_term)
    mean_similarity = mean_pairwise_similarity(similarity)
    diversity = topic_diversity(result.topic_term, result.feature_names)
    weak = float((result.dominant_topic_score < WEAK_SCORE_THRESHOLD).mean())

    notices: list[FriendlyMessage] = []
    if mean_similarity > SIMILAR_TOPIC_THRESHOLD:
        notices.append(
            FriendlyMessage(
                "Several topics use very similar terms.",
                "Consider fewer topics.",
            )
        )
    if weak > WEAK_SCORE_SHARE:
        notices.append(
            FriendlyMessage(
                "Many documents have weak dominant-topic scores.",
                "Inspect whether the corpus contains mixed themes.",
            )
        )
    if result.metrics.get("vocabulary_at_limit"):
        notices.append(
            FriendlyMessage(
                "The vocabulary reached the configured maximum.",
                "Increase it if important terms appear to be missing.",
            )
        )
    if "convergence_notice" in result.metrics:
        notices.append(result.metrics["convergence_notice"])

    return {
        "documents_used": result.n_documents,
        "vocabulary_size": len(result.feature_names),
        "topic_count": result.n_topics,
        "topic_diversity": diversity,
        "mean_pairwise_similarity": mean_similarity,
        "weak_dominant_share": weak,
        "similarity_matrix": similarity,
        "reconstruction_error": result.metrics.get("reconstruction_error"),
        "perplexity": result.metrics.get("perplexity"),
        "notices": notices,
    }
