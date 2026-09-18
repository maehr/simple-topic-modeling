"""Chart data and chart specifications.

`SPECS.md` section 6 keeps the plotting layer separate, so the app can change the chart library
later. Each function here is pure: it takes a result and returns a frame or an Altair chart.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd

from browser_topics.metrics import topic_similarity

if TYPE_CHECKING:
    from browser_topics.result import TopicModelResult

__all__ = [
    "SNIPPET_LENGTH",
    "document_frame",
    "document_scatter",
    "prevalence_bars",
    "representative_documents",
    "similarity_heatmap",
    "similarity_long_frame",
    "snippet",
    "top_term_bars",
    "top_term_frame",
    "topic_cards",
    "topic_map",
    "topic_map_frame",
    "word_cloud_png",
]

AXIS_NOTE = "The axes carry no meaning. They separate the points only."
"""`SPECS.md` section 5 de-emphasizes the projection axes."""

EMPTY_STATE = "No data to show yet."
"""`SPECS.md` section 6 asks every chart for an empty state."""

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


def _blank_axis() -> alt.Axis:
    """Build the de-emphasized axis that the projection charts use."""
    return alt.Axis(labels=False, ticks=False, grid=False, domain=False, title=None)


def topic_map_frame(result: TopicModelResult, term_count: int = 5) -> pd.DataFrame:
    """Build the bubble data for the topic map.

    >>> from browser_topics.result import _example_result
    >>> frame = topic_map_frame(_example_result(), term_count=2)
    >>> frame.columns.tolist()
    ['topic_id', 'topic', 'x', 'y', 'prevalence', 'top_terms']
    >>> len(frame)
    2
    """
    terms = result.top_terms(term_count)
    return pd.DataFrame(
        {
            "topic_id": range(result.n_topics),
            "topic": result.topic_names,
            "x": result.topic_xy[:, 0],
            "y": result.topic_xy[:, 1],
            "prevalence": result.topic_prevalence,
            "top_terms": [", ".join(row) for row in terms],
        }
    )


def topic_map(result: TopicModelResult) -> alt.Chart:
    """Place each topic on the 2-D map. Bubble size is the prevalence.

    >>> from browser_topics.result import _example_result
    >>> topic_map(_example_result()).to_dict()["mark"]["type"]
    'circle'
    """
    frame = topic_map_frame(result)
    return (
        alt.Chart(frame, title="Topic map")
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X("x:Q", axis=_blank_axis()),
            y=alt.Y("y:Q", axis=_blank_axis()),
            size=alt.Size("prevalence:Q", title="Prevalence", scale=alt.Scale(range=[100, 2000])),
            color=alt.Color("topic:N", legend=None),
            tooltip=[
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("top_terms:N", title="Top terms"),
                alt.Tooltip("prevalence:Q", title="Prevalence", format=".1%"),
            ],
        )
        .properties(height=380)
    )


def prevalence_bars(result: TopicModelResult) -> alt.Chart:
    """Rank the topics by how much of the corpus they cover.

    >>> from browser_topics.result import _example_result
    >>> prevalence_bars(_example_result()).to_dict()["mark"]["type"]
    'bar'
    """
    frame = topic_map_frame(result)
    return (
        alt.Chart(frame, title="Topic prevalence")
        .mark_bar()
        .encode(
            x=alt.X("prevalence:Q", title="Share of the corpus", axis=alt.Axis(format="%")),
            y=alt.Y("topic:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("top_terms:N", title="Top terms"),
                alt.Tooltip("prevalence:Q", title="Prevalence", format=".1%"),
            ],
        )
        .properties(height=alt.Step(22))
    )


def top_term_bars(result: TopicModelResult, topic: int, term_count: int = 15) -> alt.Chart:
    """Draw the top terms of one topic as horizontal bars.

    >>> from browser_topics.result import _example_result
    >>> top_term_bars(_example_result(), 0).to_dict()["mark"]["type"]
    'bar'
    """
    frame = top_term_frame(result, topic, term_count)
    return (
        alt.Chart(frame, title=f"Top terms · {result.topic_names[topic]}")
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="Share of the topic"),
            y=alt.Y("term:N", sort="-x", title=None),
            tooltip=["term:N", alt.Tooltip("weight:Q", format=".3f")],
        )
        .properties(height=alt.Step(20))
    )


def similarity_long_frame(result: TopicModelResult) -> pd.DataFrame:
    """Build the full similarity matrix in long form, for the heatmap.

    >>> from browser_topics.result import _example_result
    >>> frame = similarity_long_frame(_example_result())
    >>> len(frame)
    4
    >>> frame.columns.tolist()
    ['topic_a', 'topic_b', 'similarity']
    """
    matrix = topic_similarity(result.topic_term)
    rows = []
    for first in range(result.n_topics):
        for second in range(result.n_topics):
            rows.append(
                {
                    "topic_a": result.topic_names[first],
                    "topic_b": result.topic_names[second],
                    "similarity": float(matrix[first, second]),
                }
            )
    return pd.DataFrame(rows)


def similarity_heatmap(result: TopicModelResult) -> alt.Chart:
    """Show the cosine similarity between every pair of topics.

    >>> from browser_topics.result import _example_result
    >>> similarity_heatmap(_example_result()).to_dict()["mark"]["type"]
    'rect'
    """
    frame = similarity_long_frame(result)
    return (
        alt.Chart(frame, title="Topic similarity")
        .mark_rect()
        .encode(
            x=alt.X("topic_a:N", title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("topic_b:N", title=None),
            color=alt.Color(
                "similarity:Q",
                title="Cosine similarity",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("topic_a:N", title="Topic"),
                alt.Tooltip("topic_b:N", title="Compared with"),
                alt.Tooltip("similarity:Q", format=".3f"),
            ],
        )
        .properties(height=alt.Step(20))
    )


def document_frame(result: TopicModelResult) -> pd.DataFrame:
    """Build one row per document for the scatter plot and the linked table.

    >>> from browser_topics.result import _example_result
    >>> frame = document_frame(_example_result())
    >>> frame.columns.tolist()[:5]
    ['document_id', 'topic', 'score', 'snippet', 'x']
    >>> len(frame)
    3
    """
    columns = {
        "document_id": result.document_ids,
        "topic": [result.topic_names[index] for index in result.dominant_topic],
        "score": [round(float(value), 4) for value in result.dominant_topic_score],
        "snippet": [snippet(text) for text in result.documents],
        "x": result.document_xy[:, 0],
        "y": result.document_xy[:, 1],
    }
    for name in result.metadata.columns:
        columns[str(name)] = result.metadata[name].tolist()
    return pd.DataFrame(columns)


def document_scatter(frame: pd.DataFrame, sample_limit: int = 4000) -> alt.Chart:
    """Place every document on the 2-D map, coloured by its dominant topic.

    Above `sample_limit` rows the chart draws a reproducible sample. `SPECS.md` section 8 allows
    this, because sampling changes the picture only, never the model.

    >>> from browser_topics.result import _example_result
    >>> document_scatter(document_frame(_example_result())).to_dict()["mark"]["type"]
    'circle'
    >>> import pandas as pd
    >>> big = pd.concat([document_frame(_example_result())] * 5, ignore_index=True)
    >>> len(document_scatter(big, sample_limit=4).data)
    4
    """
    if len(frame) > sample_limit:
        frame = frame.sample(sample_limit, random_state=42).reset_index(drop=True)
    tooltip = [
        alt.Tooltip("document_id:N", title="Document"),
        alt.Tooltip("topic:N", title="Topic"),
        alt.Tooltip("score:Q", title="Score", format=".3f"),
        alt.Tooltip("snippet:N", title="Text"),
    ]
    return (
        alt.Chart(frame, title="Document map")
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("x:Q", axis=_blank_axis()),
            y=alt.Y("y:Q", axis=_blank_axis()),
            color=alt.Color("topic:N", title="Dominant topic"),
            tooltip=tooltip,
        )
        .properties(height=440)
    )


def word_cloud_png(result: TopicModelResult, topic: int, term_count: int = 60) -> bytes:
    r"""Render one topic as a word cloud and return PNG bytes.

    The seed is fixed, so the same topic always gives the same picture.

    >>> from browser_topics.result import _example_result
    >>> word_cloud_png(_example_result(), 0)[:4]
    b'\x89PNG'
    """
    from wordcloud import WordCloud

    frame = top_term_frame(result, topic, term_count)
    weights = {
        str(term): float(weight)
        for term, weight in zip(frame["term"], frame["weight"], strict=True)
        if weight > 0
    }
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        prefer_horizontal=0.9,
        random_state=42,
    ).generate_from_frequencies(weights)
    buffer = BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return buffer.getvalue()
