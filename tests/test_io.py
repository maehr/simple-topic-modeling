import pandas as pd
import pytest

from simple_topic_modeling.errors import DecodeError, NoUsableTextError, UnsupportedFileError
from simple_topic_modeling.io import (
    UploadedFile,
    build_corpus,
    corpus_stats,
    decode_text,
    detect_kind,
    read_table,
    split_text,
    strip_markup,
)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("a.csv", "csv"),
        ("a.tsv", "tsv"),
        ("a.tab", "tsv"),
        ("a.json", "json"),
        ("a.jsonl", "jsonl"),
        ("a.ndjson", "jsonl"),
        ("a.txt", "text"),
        ("a.md", "text"),
        ("a.rst", "text"),
        ("a.log", "text"),
        ("a.tex", "text"),
        ("a.html", "text"),
        ("a.xml", "text"),
        ("a.yaml", "text"),
        ("README", "text"),
    ],
)
def test_extension_maps_to_the_right_reader(name, kind):
    assert detect_kind(name) == kind


@pytest.mark.parametrize("name", ["a.pdf", "a.docx", "a.zip", "a.png", "a.mp4"])
def test_known_binary_extensions_are_rejected(name):
    with pytest.raises(UnsupportedFileError):
        decode_text(UploadedFile(name, b"anything"))


def test_a_nul_byte_marks_a_file_as_binary():
    with pytest.raises(UnsupportedFileError):
        decode_text(UploadedFile("mystery.dat", b"ab\x00cd"))


def test_invalid_utf8_raises_a_decode_error():
    with pytest.raises(DecodeError):
        decode_text(UploadedFile("notes.txt", b"\xff\xfe\xfd"))


def test_unknown_extension_decodes_when_it_is_valid_utf8():
    assert decode_text(UploadedFile("mystery.q7", "Grüße".encode())) == "Grüße"


def test_html_entities_are_resolved():
    assert strip_markup("<p>a &amp; b</p>", "page.html") == "a & b"


def test_xml_tags_are_stripped():
    assert strip_markup("<root><item>text</item></root>", "feed.xml") == "text"


def test_style_blocks_are_dropped():
    assert strip_markup("<style>p{color:red}</style><p>keep</p>", "page.html") == "keep"


def test_markdown_lists_and_quotes_lose_their_syntax():
    assert strip_markup("- one\n- two\n> quoted", "notes.md") == "one\ntwo\nquoted"


def test_markdown_rules_and_numbered_lists_are_stripped():
    assert strip_markup("---\n1. first", "notes.md") == "first"


def test_markdown_images_keep_their_alt_text():
    assert strip_markup("![alt](a.png)", "notes.md") == "alt"


def test_plain_text_passes_through_untouched():
    assert strip_markup("a < b and c > d", "notes.txt") == "a < b and c > d"


def test_whole_file_mode_returns_nothing_for_blank_input():
    assert split_text("   \n  ", "whole") == []


def test_line_mode_drops_blank_lines():
    assert split_text("a\n\n b \n", "lines") == ["a", "b"]


def test_csv_with_metadata_columns_is_parsed():
    file = UploadedFile("r.csv", b"text,group,date\nhello,a,2024-01-01\n")
    frame = read_table(file)
    assert frame.columns.tolist() == ["text", "group", "date"]


def test_json_object_becomes_one_row():
    assert read_table(UploadedFile("r.json", b'{"text":"a"}')).shape == (1, 1)


def test_jsonl_ignores_blank_lines():
    file = UploadedFile("r.jsonl", b'{"text":"a"}\n\n{"text":"b"}\n')
    assert read_table(file).shape == (2, 1)


def test_corpus_stats_counts_duplicates_once_per_extra_copy():
    stats = corpus_stats(["a", "a", "a", "b"])
    assert stats.duplicates == 2
    assert stats.kept == 4


def test_corpus_stats_median_uses_non_empty_documents_only():
    assert corpus_stats(["abc", "", "abcde"]).median_length == 4.0


def test_build_corpus_keeps_metadata_rows_aligned():
    metadata = pd.DataFrame({"group": ["x", "y", "z"]})
    corpus, stats = build_corpus(["a", "", "c"], ["1", "2", "3"], metadata)
    assert corpus.metadata["group"].tolist() == ["x", "z"]
    assert corpus.document_ids == ["1", "3"]
    assert stats.empty == 1


def test_build_corpus_strips_surrounding_whitespace():
    corpus, _ = build_corpus(["  a  "], ["1"])
    assert corpus.documents == ["a"]


def test_build_corpus_rejects_a_corpus_with_no_usable_text():
    with pytest.raises(NoUsableTextError):
        build_corpus([" ", ""], ["1", "2"])


def test_a_comfortable_corpus_gets_no_warning():
    from simple_topic_modeling.io import corpus_size_warning

    assert corpus_size_warning(["a short document"] * 100) is None


def test_a_large_document_count_warns():
    from simple_topic_modeling.io import LARGE_DOCUMENT_COUNT, corpus_size_warning

    warning = corpus_size_warning(["x"] * (LARGE_DOCUMENT_COUNT + 1))
    assert warning is not None
    assert "documents" in warning.detail
    assert "sample" in warning.recovery


def test_a_large_text_volume_warns_even_with_few_documents():
    from simple_topic_modeling.io import LARGE_TEXT_BYTES, corpus_size_warning

    warning = corpus_size_warning(["x" * (LARGE_TEXT_BYTES + 1)])
    assert warning is not None
    assert "MB of text" in warning.detail


def test_sampling_is_reproducible_and_ordered():
    from simple_topic_modeling.io import sample_corpus

    corpus, _ = build_corpus([f"doc {i}" for i in range(50)], [str(i) for i in range(50)])
    first = sample_corpus(corpus, 10)
    second = sample_corpus(corpus, 10)
    assert first.document_ids == second.document_ids
    assert first.document_ids == sorted(first.document_ids, key=int)


def test_sampling_keeps_metadata_aligned():
    from simple_topic_modeling.io import sample_corpus

    metadata = pd.DataFrame({"group": [f"g{i}" for i in range(20)]})
    corpus, _ = build_corpus([f"doc {i}" for i in range(20)], [str(i) for i in range(20)], metadata)
    sampled = sample_corpus(corpus, 5)
    assert sampled.metadata["group"].tolist() == [f"g{i}" for i in sampled.document_ids]


def test_a_sample_larger_than_the_corpus_changes_nothing():
    from simple_topic_modeling.io import sample_corpus

    corpus, _ = build_corpus(["a", "b"], ["1", "2"])
    assert sample_corpus(corpus, 10) is corpus
