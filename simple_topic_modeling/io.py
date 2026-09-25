"""Corpus import.

`SPECS.md` section 2 accepts structured tables, generic UTF-8 text, and text-based PDF files. This
module decodes bytes, parses the structured formats, extracts the text of a PDF, strips markup,
splits a single file into documents, and reports the corpus statistics.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib.resources import files
from statistics import median
from typing import Literal

import numpy as np
import pandas as pd
from pypdf import PdfReader
from pypdf.errors import DependencyError

from simple_topic_modeling.config import SplitMode
from simple_topic_modeling.errors import (
    DecodeError,
    FriendlyMessage,
    NoUsableTextError,
    PdfTextError,
    UnsupportedFileError,
    pdf_pages_without_text,
)

__all__ = [
    "MIN_PDF_CHARACTERS",
    "Corpus",
    "CorpusStats",
    "FileKind",
    "PdfExtraction",
    "SplitMode",
    "UploadedFile",
    "build_corpus",
    "corpus_size_warning",
    "corpus_stats",
    "decode_text",
    "demo_table",
    "detect_kind",
    "extract_pdf_text",
    "normalize_whitespace",
    "read_table",
    "read_text_document",
    "sample_corpus",
    "split_long_document",
    "split_text",
    "strip_markup",
]

FileKind = Literal["csv", "tsv", "json", "jsonl", "text"]

_TABLE_EXTENSIONS: dict[str, FileKind] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}
_MARKUP_EXTENSIONS = frozenset({".html", ".htm", ".xml"})
# A block element starts a new paragraph, so `split_text` can split an HTML page on blank lines.
# The TEI names `head` and `lg` cover the most common XML edition format.
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "lg",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_SKIPPED_TAGS = frozenset({"script", "style", "title"})
_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
_BINARY_EXTENSIONS = frozenset(
    {".docx", ".doc", ".xlsx", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".mp3", ".mp4"}
)
_PDF_EXTENSION = ".pdf"

MIN_PDF_CHARACTERS = 100
"""A PDF with fewer visible characters than this counts as scanned. A model needs more text."""

_BLANK_LINES = re.compile(r"\n\s*\n")
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_MARKDOWN_NOISE = re.compile(
    r"^[ \t]{0,3}#{1,6}[ \t]+|^[ \t]{0,3}>[ \t]?|^[ \t]{0,3}[-*+][ \t]+|^[ \t]{0,3}\d+\.[ \t]+"
    r"|^[ \t]*[-*_]{3,}[ \t]*$",
    re.MULTILINE,
)
_MARKDOWN_INLINE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)|[*_`~]{1,3}")


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """One file that a person added.

    >>> UploadedFile("notes.txt", b"hello").name
    'notes.txt'
    """

    name: str
    data: bytes

    @property
    def suffix(self) -> str:
        """Return the lowercase extension, including the dot.

        >>> UploadedFile("Report.CSV", b"").suffix
        '.csv'
        >>> UploadedFile("noextension", b"").suffix
        ''
        """
        _, dot, tail = self.name.rpartition(".")
        return f"{dot}{tail}".lower() if dot else ""


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """The counts that `SPECS.md` section 2 shows after import.

    >>> CorpusStats(4, 1, 1, 12.0).kept
    3
    """

    total: int
    empty: int
    duplicates: int
    median_length: float

    @property
    def kept(self) -> int:
        """Return the number of documents that the app models.

        >>> CorpusStats(10, 2, 3, 40.0).kept
        8
        """
        return self.total - self.empty


@dataclass(frozen=True, slots=True)
class Corpus:
    """A parsed corpus, ready for cleaning.

    >>> corpus = Corpus(["alpha", "beta"], ["a", "b"], pd.DataFrame(index=[0, 1]))
    >>> len(corpus)
    2
    """

    documents: list[str]
    document_ids: list[str]
    metadata: pd.DataFrame

    def __len__(self) -> int:
        """Return the document count.

        >>> len(Corpus(["x"], ["1"], pd.DataFrame(index=[0])))
        1
        """
        return len(self.documents)


def detect_kind(filename: str) -> FileKind:
    """Map a file name to a reader.

    An unknown extension falls back to the generic text reader, as `SPECS.md` section 2 requires.

    >>> detect_kind("rows.csv")
    'csv'
    >>> detect_kind("rows.NDJSON")
    'jsonl'
    >>> detect_kind("notes.md")
    'text'
    >>> detect_kind("mystery.q7")
    'text'
    """
    suffix = UploadedFile(filename, b"").suffix
    return _TABLE_EXTENSIONS.get(suffix, "text")


def decode_text(file: UploadedFile) -> str:
    r"""Decode a file as UTF-8.

    A known binary extension or a NUL byte raises `UnsupportedFileError`. Any other decoding
    failure raises `DecodeError`.

    >>> decode_text(UploadedFile("a.txt", b"caf\xc3\xa9"))
    'café'
    >>> decode_text(UploadedFile("a.docx", b"PK"))
    Traceback (most recent call last):
    simple_topic_modeling.errors.UnsupportedFileError: ...
    >>> decode_text(UploadedFile("a.txt", b"\xff\xfe\x00bad"))
    Traceback (most recent call last):
    simple_topic_modeling.errors.UnsupportedFileError: ...
    """
    if UploadedFile(file.name, b"").suffix in _BINARY_EXTENSIONS or b"\x00" in file.data:
        raise UnsupportedFileError(file.name)
    try:
        return file.data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DecodeError(file.name) from error


class _TagStripper(HTMLParser):
    r"""Collect the text of an HTML or XML document and drop the tags.

    A block element becomes a paragraph break. A `br` element becomes a line break. The page
    title is metadata, not text, so the stripper drops it with the scripts and the styles.

    >>> stripper = _TagStripper()
    >>> stripper.feed("<p>one</p><p>two<br>three</p>")
    >>> normalize_whitespace(" ".join(stripper.parts))
    'one\n\ntwo\nthree'
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        """Skip a script, style, or title element, and break the line at a block element."""
        # HTMLParser reads a script or style body as raw text, and a title holds no tags, so no
        # tag arrives while skipping.
        if tag in _SKIPPED_TAGS:
            self._skip += 1
        else:
            self._break(tag)

    def handle_endtag(self, tag: str) -> None:
        """Stop skipping at the end of a skipped element, and break the line at a block element."""
        if tag in _SKIPPED_TAGS and self._skip:
            self._skip -= 1
        elif tag != "br":
            self._break(tag)

    def _break(self, tag: str) -> None:
        """Add a paragraph break for a block element and a line break for `br`."""
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Keep the text of every element that is not skipped."""
        if not self._skip:
            self.parts.append(data)


def normalize_whitespace(text: str) -> str:
    r"""Collapse the spaces inside each line, and keep the line breaks.

    `split_text` needs the line breaks. A run of blank lines becomes one blank line.

    >>> normalize_whitespace("  a   b \r\n\n\n\n c\t d ")
    'a b\n\nc d'
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return _EXTRA_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def strip_markup(text: str, filename: str) -> str:
    r"""Remove HTML, XML, or Markdown syntax and keep the readable text.

    The app adds no parser dependency, as `SPECS.md` section 2 requires. The line breaks stay, so
    `split_text` can still split the text into paragraphs.

    >>> strip_markup("<p>Hello <b>world</b></p>", "page.html")
    'Hello world'
    >>> strip_markup("<p>a</p><script>var x=1</script>", "page.html")
    'a'
    >>> strip_markup("# Title\n\nSome **bold** text", "notes.md")
    'Title\n\nSome bold text'
    >>> strip_markup("[link](http://example.com) here", "notes.md")
    'link here'
    >>> strip_markup("plain text", "notes.txt")
    'plain text'
    """
    suffix = UploadedFile(filename, b"").suffix
    if suffix in _MARKUP_EXTENSIONS:
        stripper = _TagStripper()
        stripper.feed(text)
        stripper.close()
        text = " ".join(stripper.parts)
    elif suffix in _MARKDOWN_EXTENSIONS:
        text = _MARKDOWN_NOISE.sub(" ", text)
        text = _MARKDOWN_INLINE.sub(r"\1", text)
    return normalize_whitespace(text)


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    """The text of one PDF, and the pages that held no text.

    >>> extraction = PdfExtraction("book.pdf", "Some text.", 3, (2,))
    >>> extraction.notice.detail
    'The app found no text on 1 of 3 pages of "book.pdf": page 2.'
    >>> PdfExtraction("book.pdf", "Some text.", 3, ()).notice is None
    True
    """

    name: str
    text: str
    page_count: int
    pages_without_text: tuple[int, ...]

    @property
    def notice(self) -> FriendlyMessage | None:
        """Warn about the pages without text, or return `None` when every page held text.

        >>> PdfExtraction("a.pdf", "x", 1, ()).notice is None
        True
        """
        if not self.pages_without_text:
            return None
        return pdf_pages_without_text(self.name, self.pages_without_text, self.page_count)


def _pdf_string(text: str) -> str:
    r"""Escape a line for a PDF string literal.

    >>> _pdf_string("a (b) \\ c")
    'a \\(b\\) \\\\ c'
    """
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _example_pdf(pages: Sequence[str]) -> bytes:
    r"""Build a small PDF with one page per string, for the doctests and the tests.

    An empty string makes a page without text, like a scanned page. The builder writes the bytes
    by hand, so the tests need no PDF library besides pypdf and no binary file in git.

    >>> _example_pdf(["Hello"])[:8]
    b'%PDF-1.4'
    >>> len(PdfReader(io.BytesIO(_example_pdf(["a", "", "b"]))).pages)
    3
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        lines = [f"({_pdf_string(line)}) Tj T*" for line in text.split("\n")] if text else []
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td {' '.join(lines)} ET".encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % len(objects)
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    body = b"%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, content)
    table = b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    return (
        body
        + b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
        + table
        + b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
        + b"startxref\n%d\n%%%%EOF\n" % len(body)
    )


def extract_pdf_text(file: UploadedFile) -> PdfExtraction:
    r"""Extract the text of a PDF, page by page, in the browser or in CPython.

    pypdf is pure Python, so the same code runs under Pyodide and under the tests. A blank line
    separates two pages, so a page always starts a new paragraph. A page without text is skipped
    and reported. A PDF with too little text, a password, or a damaged structure raises
    `PdfTextError`.

    >>> text = "First paragraph of the report, long enough to count as real text for the model."
    >>> extraction = extract_pdf_text(UploadedFile("r.pdf", _example_pdf([text, "", text])))
    >>> extraction.page_count, extraction.pages_without_text
    (3, (2,))
    >>> extraction.text.count("\n\n")
    1
    >>> extract_pdf_text(UploadedFile("scan.pdf", _example_pdf(["", ""])))
    Traceback (most recent call last):
    simple_topic_modeling.errors.PdfTextError: The PDF "scan.pdf" does not contain ...
    >>> extract_pdf_text(UploadedFile("broken.pdf", b"not a pdf"))
    Traceback (most recent call last):
    simple_topic_modeling.errors.PdfTextError: The app cannot read the PDF "broken.pdf". ...
    """
    try:
        reader = PdfReader(io.BytesIO(file.data))
        # A PDF with an owner password only opens with the empty user password. Any other
        # password is out of scope, as issue #27 decided.
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfTextError(file.name, "encrypted")
        pages = [normalize_whitespace(page.extract_text() or "") for page in reader.pages]
    except PdfTextError:
        raise
    except DependencyError as error:
        # pypdf needs an extra crypto package for AES. The app does not ship one.
        raise PdfTextError(file.name, "encrypted") from error
    except Exception as error:
        # A damaged file can fail anywhere inside the parser, so every other failure maps to
        # one message.
        raise PdfTextError(file.name, "damaged") from error
    text = "\n\n".join(page for page in pages if page)
    if len("".join(text.split())) < MIN_PDF_CHARACTERS:
        raise PdfTextError(file.name, "no_text")
    empty = tuple(number for number, page in enumerate(pages, start=1) if not page)
    return PdfExtraction(file.name, text, len(pages), empty)


def read_text_document(file: UploadedFile) -> tuple[str, FriendlyMessage | None]:
    """Read one text-like file, and return its readable text with an optional notice.

    A PDF goes to `extract_pdf_text`. Any other file goes through `decode_text` and
    `strip_markup`.

    >>> read_text_document(UploadedFile("a.md", b"# Title"))
    ('Title', None)
    >>> page = "A page of a report. " * 10
    >>> text, notice = read_text_document(UploadedFile("r.PDF", _example_pdf([page, ""])))
    >>> text.startswith("A page of a report.")
    True
    >>> notice.detail
    'The app found no text on 1 of 2 pages of "r.PDF": page 2.'
    """
    if file.suffix == _PDF_EXTENSION:
        extraction = extract_pdf_text(file)
        return extraction.text, extraction.notice
    return strip_markup(decode_text(file), file.name), None


def split_text(text: str, mode: SplitMode) -> list[str]:
    r"""Split one text file into documents.

    >>> split_text("a\n\nb", "whole")
    ['a\n\nb']
    >>> split_text("a\n\nb", "blank_lines")
    ['a', 'b']
    >>> split_text("a\nb", "lines")
    ['a', 'b']
    >>> split_text("a\n\n\n b ", "blank_lines")
    ['a', 'b']
    """
    if mode == "whole":
        return [text.strip()] if text.strip() else []
    parts = _BLANK_LINES.split(text) if mode == "blank_lines" else text.splitlines()
    return [part.strip() for part in parts if part.strip()]


def split_long_document(
    text: str, name: str, mode: SplitMode = "blank_lines"
) -> tuple[list[str], list[str], pd.DataFrame]:
    r"""Split one long document into ordered segments, and keep the source position of each.

    `split_text` does the split. Each segment becomes one modelling document. The metadata names
    the parent document and the position of the segment, so the source order survives every
    filter and every export. `segment_index` counts from 0. `segment_number` counts from 1.

    >>> segments, ids, metadata = split_long_document("one\n\n\ntwo\n\nthree", "book.txt")
    >>> segments
    ['one', 'two', 'three']
    >>> ids
    ['book.txt#1', 'book.txt#2', 'book.txt#3']
    >>> metadata.columns.tolist()
    ['parent_document_id', 'segment_index', 'segment_number']
    >>> metadata["segment_index"].tolist(), metadata["segment_number"].tolist()
    ([0, 1, 2], [1, 2, 3])
    >>> split_long_document("a\nb", "notes.txt", "lines")[1]
    ['notes.txt#1', 'notes.txt#2']
    """
    segments = split_text(text, mode)
    count = len(segments)
    identifiers = [f"{name}#{number}" for number in range(1, count + 1)]
    metadata = pd.DataFrame(
        {
            "parent_document_id": [name] * count,
            "segment_index": list(range(count)),
            "segment_number": list(range(1, count + 1)),
        }
    )
    return segments, identifiers, metadata


def read_table(file: UploadedFile) -> pd.DataFrame:
    r"""Parse a CSV, TSV, JSON, or JSONL file into a frame.

    >>> read_table(UploadedFile("r.csv", b"text,group\nhello,a\n")).columns.tolist()
    ['text', 'group']
    >>> read_table(UploadedFile("r.tsv", b"text\tid\nhello\t1\n")).shape
    (1, 2)
    >>> read_table(UploadedFile("r.jsonl", b'{"text":"a"}\n{"text":"b"}\n')).shape
    (2, 1)
    >>> read_table(UploadedFile("r.json", b'[{"text":"a"}]')).shape
    (1, 1)
    """
    kind = detect_kind(file.name)
    text = decode_text(file)
    if kind == "csv":
        return pd.read_csv(io.StringIO(text))
    if kind == "tsv":
        return pd.read_csv(io.StringIO(text), sep="\t")
    if kind == "jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(rows)
    payload = json.loads(text)
    rows = payload if isinstance(payload, list) else [payload]
    return pd.DataFrame(rows)


def corpus_stats(documents: Sequence[str]) -> CorpusStats:
    """Report the counts that `SPECS.md` section 2 shows.

    The median length counts characters of the non-empty documents.

    >>> stats = corpus_stats(["alpha", "beta", "", "alpha"])
    >>> stats.total, stats.empty, stats.duplicates
    (4, 1, 1)
    >>> stats.median_length
    5.0
    >>> corpus_stats([]).median_length
    0.0
    """
    kept = [doc for doc in documents if doc.strip()]
    duplicates = len(kept) - len({doc for doc in kept})
    lengths = [len(doc) for doc in kept]
    return CorpusStats(
        total=len(documents),
        empty=len(documents) - len(kept),
        duplicates=duplicates,
        median_length=float(median(lengths)) if lengths else 0.0,
    )


def build_corpus(
    documents: Sequence[str],
    document_ids: Sequence[str],
    metadata: pd.DataFrame | None = None,
) -> tuple[Corpus, CorpusStats]:
    """Drop the empty documents and return the corpus with its statistics.

    `SPECS.md` section 2 excludes an empty document and shows the count.

    >>> corpus, stats = build_corpus(["alpha", "", "beta"], ["a", "b", "c"])
    >>> corpus.documents
    ['alpha', 'beta']
    >>> corpus.document_ids
    ['a', 'c']
    >>> stats.empty
    1
    >>> build_corpus(["", "  "], ["a", "b"])
    Traceback (most recent call last):
    simple_topic_modeling.errors.NoUsableTextError: ...
    """
    stats = corpus_stats(documents)
    keep = [index for index, doc in enumerate(documents) if doc.strip()]
    if not keep:
        raise NoUsableTextError
    frame = pd.DataFrame(index=range(len(documents))) if metadata is None else metadata
    return (
        Corpus(
            documents=[documents[index].strip() for index in keep],
            document_ids=[document_ids[index] for index in keep],
            metadata=frame.iloc[keep].reset_index(drop=True),
        ),
        stats,
    )


def demo_table() -> pd.DataFrame:
    """Read the demo corpus that ships inside the package.

    The corpus holds articles from two Swiss newspapers of 1914. `NOTICE` records the source and
    the licence. `scripts/build_demo_corpus.py` rebuilds the file.

    The app never fetches this file over the network, so it works offline in the browser.

    >>> frame = demo_table()
    >>> frame.columns.tolist()
    ['document_id', 'text', 'category', 'date']
    >>> len(frame)
    295
    >>> sorted(frame["category"].unique())[:2]
    ['Chronique financière', 'Chronique judiciaire']
    >>> frame["date"].min(), frame["date"].max()
    ('1914-01-01', '1914-12-31')
    """
    resource = files("simple_topic_modeling") / "data" / "demo_corpus.csv"
    return pd.read_csv(io.StringIO(resource.read_text(encoding="utf-8")))


LARGE_DOCUMENT_COUNT = 10_000
"""`SPECS.md` section 8 warns above this many documents."""

LARGE_TEXT_BYTES = 50 * 1024 * 1024
"""`SPECS.md` section 8 warns above this much text."""


def corpus_size_warning(documents: Sequence[str]) -> FriendlyMessage | None:
    """Warn when a corpus is large enough to strain the browser.

    Returns `None` when the corpus is a comfortable size.

    >>> corpus_size_warning(["short"]) is None
    True
    >>> warning = corpus_size_warning(["x"] * 20_000)
    >>> warning.detail
    'This corpus holds 20,000 documents.'
    """
    count = len(documents)
    size = sum(len(document) for document in documents)
    if count <= LARGE_DOCUMENT_COUNT and size <= LARGE_TEXT_BYTES:
        return None
    if count > LARGE_DOCUMENT_COUNT:
        detail = f"This corpus holds {count:,} documents."
    else:
        detail = f"This corpus holds {size / 1024 / 1024:.0f} MB of text."
    return FriendlyMessage(
        detail,
        "The browser may run slowly. Model a sample first, or lower the maximum vocabulary.",
    )


def sample_corpus(corpus: Corpus, size: int, seed: int = 42) -> Corpus:
    """Take a reproducible sample of a corpus, keeping the document order.

    A sample at or above the corpus size returns the corpus unchanged.

    >>> corpus, _ = build_corpus(["a", "b", "c", "d"], ["1", "2", "3", "4"])
    >>> len(sample_corpus(corpus, 2))
    2
    >>> sample_corpus(corpus, 99) is corpus
    True
    >>> sample_corpus(corpus, 2).document_ids == sample_corpus(corpus, 2).document_ids
    True
    """
    if size >= len(corpus):
        return corpus
    generator = np.random.default_rng(seed)
    keep = sorted(generator.choice(len(corpus), size=size, replace=False).tolist())
    return Corpus(
        documents=[corpus.documents[index] for index in keep],
        document_ids=[corpus.document_ids[index] for index in keep],
        metadata=corpus.metadata.iloc[keep].reset_index(drop=True),
    )
