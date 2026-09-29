"""Pure helpers for the one topic that every view shares.

`SPECS.md` section 2 shares the topic selection across tabs. The notebook keeps one state value.
A chart click, a card button, and the dropdown all write to it. These helpers hold the logic that
needs no notebook, so a test can reach every branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["resolve_topic", "selected_topic_from_chart"]


def selected_topic_from_chart(
    value: pd.DataFrame | None, column: str = "topic_id", n_topics: int | None = None
) -> int | None:
    """Read the topic that a chart selection holds, or None when no single topic is selected.

    An empty selection, a missing column, and a selection of several topics all give None. The
    notebook then leaves the shared selection unchanged.

    >>> import pandas as pd
    >>> selected_topic_from_chart(pd.DataFrame({"topic_id": [1]}))
    1
    >>> selected_topic_from_chart(pd.DataFrame({"topic_id": []})) is None
    True
    >>> selected_topic_from_chart(pd.DataFrame({"topic_id": [0, 1]})) is None
    True
    >>> selected_topic_from_chart(pd.DataFrame({"topic_id": [5]}), n_topics=3) is None
    True
    """
    if value is None or column not in value.columns:
        return None
    found = value[column].dropna().unique()
    if len(found) != 1:
        return None
    topic = int(found[0])
    if n_topics is not None and not 0 <= topic < n_topics:
        return None
    return topic


def resolve_topic(selected: int | None, n_topics: int) -> int | None:
    """Turn the stored selection into a valid topic index, or None when there are no topics.

    A new fit can have fewer topics than the last one, so a stale index falls back to topic 0.

    >>> resolve_topic(None, 3)
    0
    >>> resolve_topic(2, 3)
    2
    >>> resolve_topic(7, 3)
    0
    >>> resolve_topic(1, 0) is None
    True
    """
    if n_topics <= 0:
        return None
    if selected is None or not 0 <= selected < n_topics:
        return 0
    return selected
