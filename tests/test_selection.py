import numpy as np
import pandas as pd

from simple_topic_modeling.selection import resolve_topic, selected_topic_from_chart


def test_single_topic_is_returned():
    assert selected_topic_from_chart(pd.DataFrame({"topic_id": np.array([2, 2])})) == 2


def test_no_selection_gives_none():
    assert selected_topic_from_chart(None) is None
    assert selected_topic_from_chart(pd.DataFrame({"topic_id": []})) is None
    assert selected_topic_from_chart(pd.DataFrame({"other": [1]})) is None


def test_several_topics_give_none():
    assert selected_topic_from_chart(pd.DataFrame({"topic_id": [0, 1]})) is None


def test_out_of_range_gives_none():
    frame = pd.DataFrame({"topic_id": [3]})
    assert selected_topic_from_chart(frame, n_topics=3) is None
    assert selected_topic_from_chart(frame, n_topics=4) == 3
    assert selected_topic_from_chart(pd.DataFrame({"topic_id": [-1]}), n_topics=4) is None


def test_resolve_topic_clamps():
    assert resolve_topic(None, 2) == 0
    assert resolve_topic(1, 2) == 1
    assert resolve_topic(2, 2) == 0
    assert resolve_topic(-1, 2) == 0
    assert resolve_topic(0, 0) is None
