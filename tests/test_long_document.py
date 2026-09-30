import json

import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import ANIMAL_TEXTS
from simple_topic_modeling import plots
from simple_topic_modeling.config import AppConfig, ModelConfig, config_from_upload
from simple_topic_modeling.exports import config_json, documents_topics_frame
from simple_topic_modeling.io import (
    DEMO_TEXT_NAME,
    UploadedFile,
    build_corpus,
    decode_text,
    demo_text,
    sample_corpus,
    split_long_document,
    strip_markup,
)
from simple_topic_modeling.modeling import fit_topic_model

BOOK = "\n\n".join(ANIMAL_TEXTS)


@pytest.fixture
def long_result():
    segments, identifiers, metadata = split_long_document(BOOK, "book.txt")
    corpus, _ = build_corpus(segments, identifiers, metadata)
    config = AppConfig(
        model=ModelConfig(n_topics=3, min_df=1),
        analyse_as="long_document",
        split_mode="blank_lines",
    )
    return fit_topic_model(corpus, config)


def test_segment_ids_are_deterministic_and_ordered():
    first = split_long_document(BOOK, "book.txt")
    second = split_long_document(BOOK, "book.txt")
    assert first[1] == second[1]
    assert first[1] == [f"book.txt#{number}" for number in range(1, len(ANIMAL_TEXTS) + 1)]
    assert first[0] == ANIMAL_TEXTS


def test_empty_paragraphs_do_not_consume_a_segment_number():
    segments, identifiers, metadata = split_long_document("a\n\n \n\n\n\nb\n\n", "x.txt")
    assert segments == ["a", "b"]
    assert identifiers == ["x.txt#1", "x.txt#2"]
    assert metadata["segment_number"].tolist() == [1, 2]


def test_every_segment_keeps_the_parent_document():
    _, _, metadata = split_long_document(BOOK, "book.txt")
    assert set(metadata["parent_document_id"]) == {"book.txt"}


def test_blank_input_gives_no_segments():
    segments, identifiers, metadata = split_long_document("  \n\n ", "empty.txt")
    assert segments == []
    assert identifiers == []
    assert metadata.empty


def test_the_segment_index_survives_the_empty_document_filter():
    segments, identifiers, metadata = split_long_document("a\n\nb\n\nc", "x.txt")
    segments[1] = "   "
    corpus, stats = build_corpus(segments, identifiers, metadata)
    assert stats.empty == 1
    assert corpus.document_ids == ["x.txt#1", "x.txt#3"]
    assert corpus.metadata["segment_index"].tolist() == [0, 2]
    assert corpus.metadata["segment_number"].tolist() == [1, 3]


def test_a_sample_keeps_the_segment_positions():
    text = "\n\n".join(f"paragraph {number}" for number in range(40))
    corpus, _ = build_corpus(*split_long_document(text, "x.txt"))
    sampled = sample_corpus(corpus, 10)
    numbers = sampled.metadata["segment_number"].tolist()
    assert numbers == sorted(numbers)
    assert [f"x.txt#{number}" for number in numbers] == sampled.document_ids


def test_an_uploaded_text_file_keeps_its_paragraphs():
    file = UploadedFile("book.txt", b"First  paragraph\nstill first.\n\nSecond paragraph.\r\n")
    text = strip_markup(decode_text(file), file.name)
    segments, _, _ = split_long_document(text, file.name)
    assert segments == ["First paragraph\nstill first.", "Second paragraph."]


def test_an_uploaded_markdown_file_keeps_its_paragraphs():
    text = strip_markup("# Chapter\n\nSome **bold** text.\n\n- a list", "book.md")
    assert split_long_document(text, "book.md")[0] == ["Chapter", "Some bold text.", "a list"]


def test_an_uploaded_html_file_keeps_its_paragraphs():
    page = (
        "<head><title>Book</title></head><h1>Chapter</h1>"
        "<p>One <b>bold</b> line.<br/>Same paragraph.</p>\n<p>Two.</p>"
    )
    text = strip_markup(page, "book.html")
    assert split_long_document(text, "book.html")[0] == [
        "Chapter",
        "One bold line.\nSame paragraph.",
        "Two.",
    ]


def test_an_uploaded_tei_file_keeps_its_paragraphs():
    text = strip_markup("<TEI><text><head>Title</head><p>a</p><p>b</p></text></TEI>", "ed.xml")
    assert split_long_document(text, "ed.xml")[0] == ["Title", "a", "b"]


def test_topic_modelling_treats_each_segment_as_a_document(long_result):
    assert long_result.n_documents == len(ANIMAL_TEXTS)
    assert long_result.config["analyse_as"] == "long_document"
    assert long_result.config["split_mode"] == "blank_lines"


def test_the_export_leads_with_the_segment_provenance(long_result):
    columns = documents_topics_frame(long_result).columns.tolist()
    assert columns[:7] == [
        "document_id",
        "parent_document_id",
        "segment_index",
        "segment_number",
        "dominant_topic_id",
        "dominant_topic_name",
        "dominant_topic_score",
    ]
    assert columns[7] == "topic_0_score"


def test_the_export_keeps_the_source_order(long_result):
    frame = documents_topics_frame(long_result, include_text=True)
    assert frame["segment_index"].tolist() == list(range(len(ANIMAL_TEXTS)))
    assert frame["text"].tolist() == ANIMAL_TEXTS
    assert frame.columns.tolist()[:3] == ["document_id", "text", "parent_document_id"]


def test_the_config_export_records_the_split(long_result):
    config = AppConfig(analyse_as="long_document", split_mode="lines")
    payload = json.loads(config_json(config))
    assert payload["analyse_as"] == "long_document"
    assert payload["split_mode"] == "lines"
    restored = config_from_upload(config_json(config))
    assert restored == config


def test_an_old_config_without_the_split_fields_still_loads():
    old = {"app_version": "2.0.0a0", "language": "fr", "model": {"n_topics": 6}}
    restored = config_from_upload(json.dumps(old).encode("utf-8"))
    assert restored.analyse_as == "corpus"
    assert restored.split_mode is None


def test_an_unknown_split_mode_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"split_mode": "sentences"})


def test_the_position_frame_follows_the_segment_numbers(long_result):
    frame = plots.position_frame(long_result)
    assert len(frame) == long_result.n_documents * long_result.n_topics
    assert sorted(frame["segment_start"].unique().tolist()) == list(range(1, len(ANIMAL_TEXTS) + 1))
    shares = frame.groupby("segment_start")["share"].sum().round(6)
    assert bool(shares.eq(1.0).all())


def test_the_position_frame_bins_a_long_book():
    segments = [f"word{number % 7} other{number % 3} filler" for number in range(900)]
    corpus, _ = build_corpus(*split_long_document("\n\n".join(segments), "big.txt"))
    result = fit_topic_model(corpus, AppConfig(model=ModelConfig(n_topics=2, min_df=1)))
    frame = plots.position_frame(result, max_columns=300)
    assert frame["segment_start"].nunique() == 300
    assert frame["segment_start"].min() == 1
    assert frame["segment_end"].max() == 900
    ticks = plots.position_heatmap(frame, result.topic_names).to_dict()["encoding"]["x"]["axis"]
    assert ticks["values"][0] == 1
    assert len(ticks["values"]) <= plots.POSITION_TICKS + 1


def test_the_heatmap_lists_the_topics_in_topic_order(long_result):
    chart = plots.position_heatmap(plots.position_frame(long_result), long_result.topic_names)
    assert chart.to_dict()["encoding"]["y"]["sort"] == long_result.topic_names


def test_representative_passages_name_their_position(long_result):
    passages = plots.representative_passages(long_result, 0, count=3)
    documents = plots.representative_documents(long_result, 0, count=3)
    expected = [int(value.split("#")[1]) for value in documents["document_id"]]
    assert passages["segment_number"].tolist() == expected
    assert passages["score"].tolist() == documents["score"].tolist()
    assert set(passages["dominant_topic"]) <= set(long_result.topic_names)


def test_the_document_table_keeps_the_source_order(long_result):
    frame = plots.document_frame(long_result)
    assert frame["segment_number"].tolist() == list(range(1, len(ANIMAL_TEXTS) + 1))
    assert isinstance(frame, pd.DataFrame)


def test_mode_labels_round_trip_and_keep_stored_values():
    from simple_topic_modeling.config import MODE_LABELS, mode_from_label

    assert set(MODE_LABELS) == {"corpus", "long_document"}
    assert MODE_LABELS["corpus"] == "Independent segments"
    assert MODE_LABELS["long_document"] == "Ordered text"
    for value, label in MODE_LABELS.items():
        assert mode_from_label(label) == value


def test_unknown_mode_label_is_rejected():
    from simple_topic_modeling.config import mode_from_label

    with pytest.raises(ValueError, match="Unknown analysis mode"):
        mode_from_label("Long document")


def test_saved_config_with_old_values_still_loads():
    assert AppConfig(analyse_as="long_document").analyse_as == "long_document"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            (1, 327, "paragraph", "long_document"),
            "1 file → 327 paragraphs → analysed as an ordered text",
        ),
        ((1, 1, "segment", "corpus"), "1 file → 1 segment → analysed as independent segments"),
        (
            (1, 12, "paragraph", "long_document", "pasted text"),
            "1 pasted text → 12 paragraphs → analysed as an ordered text",
        ),
        ((2, 40, "line", "corpus"), "2 files → 40 lines → analysed as independent segments"),
        (
            (1, 9, "document", "corpus", "table"),
            "1 table → 9 documents → analysed as independent segments",
        ),
    ],
)
def test_describe_analysis(args, expected):
    from simple_topic_modeling.io import describe_analysis

    assert describe_analysis(*args) == expected


def test_segment_unit_follows_the_split_rule():
    from simple_topic_modeling.io import segment_unit

    assert segment_unit("blank_lines") == "paragraph"
    assert segment_unit("blank_lines", is_pdf=True) == "segment"
    assert segment_unit("lines", is_pdf=True) == "line"
    assert segment_unit("whole") == "document"


def test_the_book_demo_splits_into_ordered_paragraphs():
    text = demo_text()
    segments, identifiers, metadata = split_long_document(text, DEMO_TEXT_NAME)
    assert len(segments) == text.count("\n\n") + 1
    assert segments[0].startswith("A VINDICATION OF THE RIGHTS OF WOMAN")
    assert segments[-1].startswith("Be just then, O ye men of understanding!")
    assert identifiers[:2] == [f"{DEMO_TEXT_NAME}#1", f"{DEMO_TEXT_NAME}#2"]
    assert metadata["segment_number"].tolist() == list(range(1, len(segments) + 1))


def test_the_book_demo_holds_the_work_only():
    text = demo_text()
    for leftover in ("Gutenberg", "This etext", "CONTENTS.", "Footnote", "BIOGRAPHICAL SKETCH"):
        assert leftover not in text
    assert all("\n" not in segment for segment in text.rstrip("\n").split("\n\n"))


def test_the_book_demo_fits_as_an_ordered_text():
    segments, identifiers, metadata = split_long_document(demo_text(), DEMO_TEXT_NAME)
    corpus, stats = build_corpus(segments, identifiers, metadata)
    config = AppConfig(model=ModelConfig(n_topics=8), analyse_as="long_document", language="en")
    result = fit_topic_model(corpus, config)
    assert stats.kept == len(segments)
    assert result.document_topic.shape == (len(segments), 8)
