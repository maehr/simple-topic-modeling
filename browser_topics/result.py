"""The model-agnostic result object.

Charts and exports read this one object, as `SPECS.md` section 5 requires. The fit computes every
derived value once, so exploring the result stays cheap.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.decomposition import TruncatedSVD

__all__ = [
    "TopicModelResult",
    "auto_labels",
    "dominant_topics",
    "normalize_rows",
    "project_documents",
    "rename_topic",
    "top_terms",
    "topic_centroids",
]

PROJECTION_SEED = 42
"""`SPECS.md` section 5 fixes the projection seed."""

LABEL_TERM_COUNT = 3
"""`SPECS.md` section 5 builds the initial label from the three strongest terms."""


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to sum to 1, and leave an all-zero row untouched.

    >>> normalize_rows(np.array([[1.0, 3.0], [0.0, 0.0]])).tolist()
    [[0.25, 0.75], [0.0, 0.0]]
    """
    totals = matrix.sum(axis=1, keepdims=True)
    safe = np.where(totals == 0, 1.0, totals)
    return np.asarray(matrix / safe, dtype=float)


def top_terms(topic_term: np.ndarray, feature_names: list[str], count: int) -> list[list[str]]:
    """Return the strongest terms of each topic, strongest first.

    >>> weights = np.array([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]])
    >>> top_terms(weights, ["alpha", "beta", "gamma"], 2)
    [['beta', 'gamma'], ['alpha', 'gamma']]
    """
    order = np.argsort(-topic_term, axis=1)[:, :count]
    return [[feature_names[index] for index in row] for row in order]


def auto_labels(topic_term: np.ndarray, feature_names: list[str]) -> list[str]:
    """Build the initial topic labels of `SPECS.md` section 5.

    >>> weights = np.array([[0.5, 0.3, 0.2]])
    >>> auto_labels(weights, ["economy", "market", "growth"])
    ['Topic 1 · economy, market, growth']
    """
    terms = top_terms(topic_term, feature_names, LABEL_TERM_COUNT)
    return [f"Topic {index + 1} · {', '.join(row)}" for index, row in enumerate(terms)]


def project_documents(document_term: spmatrix) -> np.ndarray:
    """Project the sparse document-term matrix onto two dimensions.

    The axes carry no meaning. `SPECS.md` section 5 uses them for visual separation only.
    A matrix with fewer than three columns cannot feed `TruncatedSVD`, so this pads with zeros.

    >>> from scipy.sparse import csr_matrix
    >>> coordinates = project_documents(csr_matrix(np.eye(4)))
    >>> coordinates.shape
    (4, 2)
    >>> project_documents(csr_matrix(np.ones((3, 1)))).shape
    (3, 2)
    """
    rows, columns = document_term.shape
    limit = min(rows, columns)
    if limit < 3:
        dense = np.asarray(document_term.todense(), dtype=float)
        padded = np.zeros((rows, 2), dtype=float)
        padded[:, : min(2, columns)] = dense[:, : min(2, columns)]
        return padded
    svd = TruncatedSVD(n_components=2, random_state=PROJECTION_SEED)
    return np.asarray(svd.fit_transform(document_term), dtype=float)


def topic_centroids(document_xy: np.ndarray, document_topic: np.ndarray) -> np.ndarray:
    """Place each topic at the weighted centre of its documents.

    A topic with no weight anywhere lands at the origin.

    >>> xy = np.array([[0.0, 0.0], [2.0, 4.0]])
    >>> weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> topic_centroids(xy, weights).tolist()
    [[0.0, 0.0], [2.0, 4.0]]
    """
    totals = document_topic.sum(axis=0)
    safe = np.where(totals == 0, 1.0, totals)
    return np.asarray((document_topic.T @ document_xy) / safe[:, None], dtype=float)


def dominant_topics(document_topic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest topic of each document and its score.

    >>> index, score = dominant_topics(np.array([[0.2, 0.8], [0.6, 0.4]]))
    >>> index.tolist()
    [1, 0]
    >>> score.tolist()
    [0.8, 0.6]
    """
    index = np.asarray(document_topic.argmax(axis=1), dtype=int)
    score = np.asarray(document_topic.max(axis=1), dtype=float)
    return index, score


@dataclass(frozen=True, slots=True)
class TopicModelResult:
    """Everything the charts and the exports need.

    `SPECS.md` section 5 defines the fields. `document_topic` rows sum to 1, so the app can call
    the value a topic share.

    >>> result = _example_result()
    >>> result.n_topics, result.n_documents
    (2, 3)
    >>> result.topic_names[0]
    'Topic 1 · alpha, beta, gamma'
    """

    model_type: str
    config: dict[str, Any]
    documents: list[str]
    document_ids: list[str]
    metadata: pd.DataFrame
    feature_names: list[str]
    document_topic_raw: np.ndarray
    document_topic: np.ndarray
    topic_term_raw: np.ndarray
    topic_term: np.ndarray
    document_xy: np.ndarray
    topic_xy: np.ndarray
    topic_names: list[str]
    topic_auto_labels: list[str]
    topic_prevalence: np.ndarray
    dominant_topic: np.ndarray
    dominant_topic_score: np.ndarray
    metrics: dict[str, Any]

    @property
    def n_topics(self) -> int:
        """Return the topic count.

        >>> _example_result().n_topics
        2
        """
        return len(self.topic_names)

    @property
    def n_documents(self) -> int:
        """Return the modelled document count.

        >>> _example_result().n_documents
        3
        """
        return len(self.documents)

    def top_terms(self, count: int = 5) -> list[list[str]]:
        """Return the strongest terms of every topic.

        >>> _example_result().top_terms(2)[0]
        ['alpha', 'beta']
        """
        return top_terms(self.topic_term, self.feature_names, count)

    def to_dict(self) -> dict[str, Any]:
        """Return the dict shape that `SPECS.md` section 5 documents.

        >>> sorted(_example_result().to_dict())[:3]
        ['config', 'document_ids', 'document_topic']
        """
        return dataclasses.asdict(self)


def rename_topic(result: TopicModelResult, index: int, name: str) -> TopicModelResult:
    """Return a copy with one topic renamed. This never refits the model.

    An empty name restores the automatic label.

    >>> renamed = rename_topic(_example_result(), 0, "Economy")
    >>> renamed.topic_names[0]
    'Economy'
    >>> rename_topic(renamed, 0, "  ").topic_names[0]
    'Topic 1 · alpha, beta, gamma'
    """
    names = list(result.topic_names)
    names[index] = name.strip() or result.topic_auto_labels[index]
    return dataclasses.replace(result, topic_names=names)


def _example_result() -> TopicModelResult:
    """Build a small result for the doctests in this module.

    >>> _example_result().model_type
    'nmf'
    """
    feature_names = ["alpha", "beta", "gamma"]
    topic_term_raw = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    document_topic_raw = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    topic_term = normalize_rows(topic_term_raw)
    document_topic = normalize_rows(document_topic_raw)
    document_xy = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    labels = auto_labels(topic_term, feature_names)
    index, score = dominant_topics(document_topic)
    return TopicModelResult(
        model_type="nmf",
        config={},
        documents=["a", "b", "c"],
        document_ids=["1", "2", "3"],
        metadata=pd.DataFrame(index=range(3)),
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
        metrics={},
    )
