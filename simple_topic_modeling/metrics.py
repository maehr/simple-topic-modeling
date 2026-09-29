"""Descriptive diagnostics.

`SPECS.md` section 6 asks for descriptive aids and friendly warnings. It forbids an automatic
quality score, so this module returns numbers and notices only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from simple_topic_modeling.errors import FriendlyMessage
from simple_topic_modeling.result import top_terms

if TYPE_CHECKING:
    from simple_topic_modeling.result import TopicModelResult

__all__ = [
    "DIAGNOSTIC_HELP",
    "DIVERSITY_TERM_COUNT",
    "SIMILAR_TOPIC_THRESHOLD",
    "WEAK_SCORE_SHARE",
    "WEAK_SCORE_THRESHOLD",
    "compare_runs",
    "diagnostic_table",
    "diagnostics",
    "mean_pairwise_similarity",
    "run_summary",
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

    >>> from simple_topic_modeling.result import _example_result
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


DIAGNOSTIC_HELP: dict[str, str] = {
    "Documents used": (
        "The number of documents the model read after cleaning. "
        "A higher value means more text stands behind the topics."
    ),
    "Vocabulary": (
        "The number of distinct terms the model kept. "
        "A higher value means the model works with more terms."
    ),
    "Topics": (
        "The number of topics the model found. A higher value splits the corpus into finer groups."
    ),
    "Topic diversity": (
        "The share of top terms that belong to one topic only. "
        "A higher value means the topics repeat each other less."
    ),
    "Mean pairwise similarity": (
        "How much the term lists of two topics overlap, averaged over all pairs. "
        "A higher value means the topics resemble each other more."
    ),
    "Weak dominant scores": (
        "The share of documents whose strongest topic covers only a small part of them. "
        "A higher value means that more documents fit no single topic well."
    ),
    "Reconstruction error": (
        "NMF only. How far the topics fall short of rebuilding the word counts. "
        "A higher value means a looser fit. Compare it only between NMF runs."
    ),
    "Perplexity": (
        "LDA only. How surprised the model is by the words. "
        "A higher value means a looser fit. Compare it only between LDA runs."
    ),
}
"""One plain sentence per measure. It says what a higher value means and never sets a threshold."""

_MISSING = "—"

_FORMATS: list[tuple[str, str, str]] = [
    ("Documents used", "documents_used", "{:,}"),
    ("Vocabulary", "vocabulary_size", "{:,}"),
    ("Topics", "topic_count", "{}"),
    ("Topic diversity", "topic_diversity", "{:.2f}"),
    ("Mean pairwise similarity", "mean_pairwise_similarity", "{:.2f}"),
    ("Weak dominant scores", "weak_dominant_share", "{:.0%}"),
    ("Reconstruction error", "reconstruction_error", "{:.3f}"),
    ("Perplexity", "perplexity", "{:.1f}"),
]


def _format_measures(summary: dict[str, Any]) -> dict[str, str]:
    """Format each measure for display. A measure that the run lacks shows a dash."""
    return {
        label: _MISSING if summary[key] is None else template.format(summary[key])
        for label, key, template in _FORMATS
    }


def run_summary(result: TopicModelResult) -> dict[str, Any]:
    """Keep the scalar diagnostics of one run, its model type, and a short label.

    The label reads the seed from `result.config` when the config holds one.

    >>> from simple_topic_modeling.result import _example_result
    >>> summary = run_summary(_example_result())
    >>> summary["label"], summary["model_type"]
    ('NMF · 2 topics', 'nmf')
    >>> "notices" in summary, "similarity_matrix" in summary
    (False, False)
    """
    report = diagnostics(result)
    scalars = {
        key: value for key, value in report.items() if key not in {"notices", "similarity_matrix"}
    }
    label = f"{result.model_type.upper()} · {result.n_topics} topics"
    seed = result.config.get("model", {}).get("random_seed")
    if seed is not None:
        label += f" · seed {seed}"
    return {"label": label, "model_type": result.model_type, **scalars}


def diagnostic_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Lay out one run as a long table: the measure, its value, and what it tells you.

    A measure that the model type does not produce shows a dash.

    >>> from simple_topic_modeling.result import _example_result
    >>> table = diagnostic_table(run_summary(_example_result()))
    >>> table.columns.tolist()
    ['Measure', 'This run', 'What it tells you']
    >>> table.set_index("Measure")["This run"]["Perplexity"]
    '—'
    """
    values = _format_measures(summary)
    return pd.DataFrame(
        {
            "Measure": list(DIAGNOSTIC_HELP),
            "This run": [values[label] for label in DIAGNOSTIC_HELP],
            "What it tells you": list(DIAGNOSTIC_HELP.values()),
        }
    )


def _run_name(position: int) -> str:
    """Name a run by its distance from the newest run."""
    if position == 0:
        return "This run"
    if position == 1:
        return "Run before"
    return f"{position} runs before"


def compare_runs(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Put the runs side by side: one row per measure, one column per run, newest first.

    The first row names the settings of each run. Reconstruction error belongs to NMF and
    perplexity to LDA. The two do not compare across model types, so a run of the other type
    shows a dash.

    >>> from simple_topic_modeling.result import _example_result
    >>> summary = run_summary(_example_result())
    >>> table = compare_runs([summary, summary])
    >>> table.columns.tolist()
    ['Measure', 'This run', 'Run before']
    >>> table["This run"].tolist()[:2]
    ['NMF · 2 topics', '3']
    >>> table["Measure"].tolist()[-1]
    'Perplexity'
    """
    columns: dict[str, list[str]] = {
        "Measure": ["Settings", *DIAGNOSTIC_HELP],
    }
    for position, summary in enumerate(runs):
        values = _format_measures(summary)
        columns[_run_name(position)] = [
            summary["label"],
            *(values[label] for label in DIAGNOSTIC_HELP),
        ]
    return pd.DataFrame(columns)
