import numpy as np
import pandas as pd
import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig
from simple_topic_modeling.io import build_corpus
from simple_topic_modeling.modeling import fit_topic_model
from simple_topic_modeling.plots import (
    choose_date_bin,
    group_share_frame,
    group_stacked_bars,
    parse_dates,
    time_line_chart,
    time_share_frame,
)

TEXTS = [
    "cat dog runs fast",
    "cat sleeps warm couch",
    "dog barks postman loudly",
    "bird sings morning song",
    "bird flies above trees",
    "fish swims cold water",
]


@pytest.fixture
def result():
    metadata = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "date": [
                "2024-01-05",
                "2024-02-10",
                "2024-06-01",
                "2025-01-01",
                "2025-06-01",
                "not a date",
            ],
        }
    )
    corpus, _ = build_corpus(TEXTS, [str(index) for index in range(6)], metadata)
    return fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))


def test_group_shares_sum_to_one_per_group(result):
    frame = group_share_frame(result, "group")
    totals = frame.groupby("group")["share"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_group_frame_covers_every_group_and_topic(result):
    frame = group_share_frame(result, "group")
    assert set(frame["group"]) == {"a", "b", "c"}
    assert len(frame) == 3 * result.n_topics


def test_group_document_counts_are_reported(result):
    frame = group_share_frame(result, "group")
    assert frame.groupby("group")["documents"].first().tolist() == [2, 2, 2]


def test_missing_group_values_get_their_own_label():
    metadata = pd.DataFrame({"group": ["a", None, "a", "b", "b", None]})
    corpus, _ = build_corpus(TEXTS, [str(index) for index in range(6)], metadata)
    built = fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))
    assert "(missing)" in set(group_share_frame(built, "group")["group"])


def test_stacked_bars_normalize(result):
    spec = group_stacked_bars(group_share_frame(result, "group")).to_dict()
    assert spec["encoding"]["x"]["stack"] == "normalize"
    assert spec["encoding"]["tooltip"]
    assert spec["title"]


def test_unparsed_dates_are_counted(result):
    parsed, unparsed = parse_dates(result.metadata["date"])
    assert unparsed == 1
    assert parsed.notna().sum() == 5


def test_a_column_with_no_readable_date_reports_every_row():
    parsed, unparsed = parse_dates(pd.Series(["x", "y"]))
    assert unparsed == 2
    assert choose_date_bin(parsed) == "day"


@pytest.mark.parametrize(
    ("last", "expected"),
    [("2024-03-01", "day"), ("2024-10-01", "month"), ("2030-01-01", "year")],
)
def test_bin_follows_the_span(last, expected):
    parsed, _ = parse_dates(pd.Series(["2024-01-01", last]))
    assert choose_date_bin(parsed) == expected


def test_time_shares_sum_to_one_per_period(result):
    parsed, _ = parse_dates(result.metadata["date"])
    frame = time_share_frame(result, parsed, "month")
    totals = frame.groupby("period")["share"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_time_frame_drops_the_unparsed_rows(result):
    parsed, _ = parse_dates(result.metadata["date"])
    frame = time_share_frame(result, parsed, "year")
    assert set(frame["period"].dt.year) == {2024, 2025}


def test_time_chart_is_a_line_with_a_title(result):
    parsed, _ = parse_dates(result.metadata["date"])
    spec = time_line_chart(time_share_frame(result, parsed, "month")).to_dict()
    assert spec["mark"]["type"] == "line"
    assert spec["title"]
    assert spec["encoding"]["tooltip"]
