"""Chart data and chart specifications.

`SPECS.md` section 6 keeps the plotting layer separate, so the app can change the chart library
later. Each function here is pure: it takes a result and returns a frame or an Altair chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from browser_topics.result import TopicModelResult

__all__ = [
    "SNIPPET_LENGTH",
    "representative_documents",
    "snippet",
    "top_term_frame",
    "topic_cards",
]

SNIPPET_LENGTH = 220
"""`SPECS.md` section 8 shows a snippet in a table, never the full text."""


def snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    """Shorten a document for a table cell.

    >>> snippet("a short document")
    'a short document'
    >>> snippet("abcdefghij", length=5)
    'abcde…'
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= length:
        return collapsed
    return f"{collapsed[:length]}…"


def topic_cards(result: TopicModelResult, term_count: int = 5) -> pd.DataFrame:
    """Build the topic cards of `SPECS.md` section 6.

    >>> from browser_topics.result import _example_result
    >>> cards = topic_cards(_example_result(), term_count=2)
    >>> cards.columns.tolist()
    ['topic_id', 'topic', 'top_terms', 'prevalence', 'documents']
    >>> cards["top_terms"].tolist()
    ['alpha, beta', 'gamma, beta']
    >>> cards["documents"].tolist()
    [2, 1]
    """
    terms = result.top_terms(term_count)
    counts = np.bincount(result.dominant_topic, minlength=result.n_topics)
    return pd.DataFrame(
        {
            "topic_id": range(result.n_topics),
            "topic": result.topic_names,
            "top_terms": [", ".join(row) for row in terms],
            "prevalence": [f"{value:.1%}" for value in result.topic_prevalence],
            "documents": counts,
        }
    )


def top_term_frame(result: TopicModelResult, topic: int, term_count: int = 15) -> pd.DataFrame:
    """Build the bar-chart data for one topic, strongest term first.

    >>> from browser_topics.result import _example_result
    >>> frame = top_term_frame(_example_result(), 0, term_count=2)
    >>> frame["term"].tolist()
    ['alpha', 'beta']
    >>> bool(frame["weight"].is_monotonic_decreasing)
    True
    """
    limit = min(term_count, len(result.feature_names))
    order = np.argsort(-result.topic_term[topic])[:limit]
    return pd.DataFrame(
        {
            "term": [result.feature_names[index] for index in order],
            "weight": [float(result.topic_term[topic, index]) for index in order],
        }
    )


def representative_documents(result: TopicModelResult, topic: int, count: int = 10) -> pd.DataFrame:
    """Rank the documents of one topic by their score, as `SPECS.md` section 6 requires.

    >>> from browser_topics.result import _example_result
    >>> frame = representative_documents(_example_result(), 0, count=2)
    >>> frame.columns.tolist()[:3]
    ['document_id', 'score', 'snippet']
    >>> frame["document_id"].tolist()
    ['1', '3']
    """
    scores = result.document_topic[:, topic]
    order = np.argsort(-scores)[:count]
    columns = {
        "document_id": [result.document_ids[index] for index in order],
        "score": [round(float(scores[index]), 4) for index in order],
        "snippet": [snippet(result.documents[index]) for index in order],
    }
    for name in result.metadata.columns:
        columns[str(name)] = [result.metadata[name].iloc[index] for index in order]
    return pd.DataFrame(columns)
