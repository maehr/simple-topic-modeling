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

from simple_topic_modeling.metrics import topic_similarity

if TYPE_CHECKING:
    from simple_topic_modeling.result import TopicModelResult

__all__ = [
    "SNIPPET_LENGTH",
    "choose_date_bin",
    "document_frame",
    "document_scatter",
    "group_share_frame",
    "group_stacked_bars",
    "parse_dates",
    "prevalence_bars",
    "representative_documents",
    "score_histogram",
    "similarity_heatmap",
    "similarity_long_frame",
    "snippet",
    "time_line_chart",
    "time_share_frame",
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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

    >>> from simple_topic_modeling.result import _example_result
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


def group_share_frame(result: TopicModelResult, column: str) -> pd.DataFrame:
    """Average the topic shares inside each group.

    Every group's shares sum to 1, so the stacked bars compare groups of any size.

    >>> from simple_topic_modeling.result import _example_result
    >>> import pandas as pd
    >>> example = _example_result()
    >>> object.__setattr__(example, "metadata", pd.DataFrame({"group": ["a", "a", "b"]}))
    >>> frame = group_share_frame(example, "group")
    >>> frame.columns.tolist()
    ['group', 'topic', 'share', 'documents']
    >>> bool(frame.groupby("group")["share"].sum().round(6).eq(1.0).all())
    True
    """
    labels = result.metadata[column].fillna("(missing)").astype(str)
    rows = []
    for name, index in labels.groupby(labels).groups.items():
        positions = [labels.index.get_loc(item) for item in index]
        means = result.document_topic[positions].mean(axis=0)
        total = means.sum() or 1.0
        for topic_id, value in enumerate(means):
            rows.append(
                {
                    "group": str(name),
                    "topic": result.topic_names[topic_id],
                    "share": float(value / total),
                    "documents": len(positions),
                }
            )
    return pd.DataFrame(rows)


def group_stacked_bars(frame: pd.DataFrame) -> alt.Chart:
    """Compare the topic mix of each group as a normalized stacked bar chart.

    >>> from simple_topic_modeling.result import _example_result
    >>> import pandas as pd
    >>> example = _example_result()
    >>> object.__setattr__(example, "metadata", pd.DataFrame({"group": ["a", "a", "b"]}))
    >>> group_stacked_bars(group_share_frame(example, "group")).to_dict()["mark"]["type"]
    'bar'
    """
    return (
        alt.Chart(frame, title="Topic mix by group")
        .mark_bar()
        .encode(
            x=alt.X(
                "share:Q", title="Share of the group", stack="normalize", axis=alt.Axis(format="%")
            ),
            y=alt.Y("group:N", title=None),
            color=alt.Color("topic:N", title="Topic"),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
                alt.Tooltip("documents:Q", title="Documents"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def parse_dates(values: pd.Series) -> tuple[pd.Series, int]:
    """Read a date column and report how many values the app could not read.

    >>> import pandas as pd
    >>> parsed, unparsed = parse_dates(pd.Series(["2025-01-01", "not a date"]))
    >>> unparsed
    1
    >>> parsed.notna().sum()
    np.int64(1)
    """
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed, int(parsed.isna().sum())


def choose_date_bin(parsed: pd.Series) -> str:
    """Pick a day, month, or year bin from the span of the dates.

    >>> import pandas as pd
    >>> choose_date_bin(pd.to_datetime(pd.Series(["2025-01-01", "2025-01-20"])))
    'day'
    >>> choose_date_bin(pd.to_datetime(pd.Series(["2025-01-01", "2025-11-01"])))
    'month'
    >>> choose_date_bin(pd.to_datetime(pd.Series(["2005-01-01", "2025-01-01"])))
    'year'
    >>> choose_date_bin(pd.Series([pd.NaT]))
    'day'
    """
    usable = parsed.dropna()
    if usable.empty:
        return "day"
    span = (usable.max() - usable.min()).days
    if span <= 92:
        return "day"
    if span <= 1460:
        return "month"
    return "year"


def time_share_frame(result: TopicModelResult, parsed: pd.Series, bin_by: str) -> pd.DataFrame:
    """Average the topic shares inside each time bin.

    >>> from simple_topic_modeling.result import _example_result
    >>> import pandas as pd
    >>> dates = pd.to_datetime(pd.Series(["2025-01-01", "2025-01-01", "2025-02-01"]))
    >>> frame = time_share_frame(_example_result(), dates, "month")
    >>> frame.columns.tolist()
    ['period', 'topic', 'share']
    >>> len(frame)
    4
    """
    freq = {"day": "D", "month": "M", "year": "Y"}[bin_by]
    keep = parsed.notna().to_numpy()
    periods = parsed[keep].dt.to_period(freq).dt.to_timestamp().reset_index(drop=True)
    scores = result.document_topic[keep]
    rows = []
    for period, positions in periods.groupby(periods).groups.items():
        means = scores[list(positions)].mean(axis=0)
        total = means.sum() or 1.0
        for topic_id, value in enumerate(means):
            rows.append(
                {
                    "period": period,
                    "topic": result.topic_names[topic_id],
                    "share": float(value / total),
                }
            )
    return pd.DataFrame(rows)


def time_line_chart(frame: pd.DataFrame) -> alt.Chart:
    """Draw the topic share over time as one line per topic.

    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {"period": pd.to_datetime(["2025-01-01"]), "topic": ["Topic 1"], "share": [1.0]}
    ... )
    >>> time_line_chart(frame).to_dict()["mark"]["type"]
    'line'
    """
    return (
        alt.Chart(frame, title="Topic share over time")
        .mark_line(point=True)
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("share:Q", title="Share of the period", axis=alt.Axis(format="%")),
            color=alt.Color("topic:N", title="Topic"),
            tooltip=[
                alt.Tooltip("period:T", title="Period"),
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=380)
    )


def score_histogram(frame: pd.DataFrame) -> alt.Chart:
    """Show how the dominant-topic scores spread across the corpus.

    >>> import pandas as pd
    >>> score_histogram(pd.DataFrame({"score": [0.1, 0.9]})).to_dict()["mark"]["type"]
    'bar'
    """
    return (
        alt.Chart(frame, title="Dominant-topic score distribution")
        .mark_bar()
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Dominant-topic score"),
            y=alt.Y("count()", title="Documents"),
            tooltip=[alt.Tooltip("count()", title="Documents")],
        )
        .properties(height=260)
    )
