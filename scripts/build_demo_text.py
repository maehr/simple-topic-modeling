"""Build the demo text that ships inside the package.

The text is Mary Wollstonecraft, *A Vindication of the Rights of Woman* (1792), from Project
Gutenberg ebook #3420. The author died in 1797, so the work is in the public domain. `NOTICE`
records the source and the licence of the Project Gutenberg file.

The Project Gutenberg edition prints an anonymous biographical sketch before the work. It has no
author and no date, so its copyright status is not certain. The script removes it, and keeps only
the work of Wollstonecraft.

The script also removes the table of contents and the bare chapter and section numbers, such as
`CHAPTER 5.` and `SECTION 5.1.`. It keeps each chapter title and each sentence of the work. A topic
model groups words that occur together, so these short repeated lines would form topics of their
own, such as `chapter 11 10 12 13`. They say nothing about the argument of the book. For the same
reason, the script removes the `*Footnote.` label before each note, and keeps the note.

The script downloads the plain-text file once, checks it against a known hash, and writes
`simple_topic_modeling/data/demo_text.txt`. The output holds one paragraph per block, with one blank
line between blocks, so the app can split it on blank lines. The script prints the source URL and
the SHA-256 of both the source and the output, and it writes the same report to
`scripts/demo_text.provenance.json`.

Project Gutenberg can change a file in place. The pinned hash then stops the build. The maintainer
reviews the new file and updates `SOURCE_SHA256`. The script uses no mirror.

Run it from the repository root:

    uv run python scripts/build_demo_text.py

The output is deterministic. A rerun writes the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

SOURCE_PAGE = "https://www.gutenberg.org/ebooks/3420"
"""The Project Gutenberg page for the ebook."""

SOURCE_URL = "https://www.gutenberg.org/cache/epub/3420/pg3420.txt"
"""The plain-text file itself."""

SOURCE_SHA256 = "7d79652f43541759d8ff0ce467372fa4a534d3f9dc32013e01ca99ea04b286a8"
"""The SHA-256 of the source file. The script stops when the download does not match."""

START_MARKER = re.compile(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*\*\*\*$", re.M)
"""The line that opens the ebook text."""

END_MARKER = re.compile(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*\*\*\*$", re.M)
"""The line that closes the ebook text."""

BOILERPLATE_PREFIXES = ("Ver.", "This etext was produced by")
"""Paragraph openings that mark Project Gutenberg boilerplate inside the markers.

`Ver.12.12.00*END*` is a leftover of the old header. `This etext was produced by` starts the
credit block for the volunteers, with their e-mail addresses.
"""

DATE_STAMP = " 8 April, 2001"
"""A transcriber's date stamp, not text of the book.

The source puts it on the line under the last entry of the contents, with no blank line between.
The paragraph split therefore joins it to that entry, and the script cuts it from the end.
"""

SKETCH_FIRST = "WITH A BIOGRAPHICAL SKETCH OF THE AUTHOR."
"""The title line that announces the sketch. The sketch starts here."""

DEDICATION_FIRST = "TO"
"""The first line of the dedication to Talleyrand. The work starts again here."""

CONTENTS_HEADING = "CONTENTS."
"""The heading of the table of contents. The entries follow it, one per paragraph."""

CONTENTS_ENTRY = re.compile(r"(?:INTRODUCTION|CHAPTER \d+)\. .*|INTRODUCTION\.")
"""One entry of the table of contents."""

STRUCTURE_MARKER = re.compile(r"CHAPTER \d+\.|SECTION \d+\.\d+\.|\*(?: \*)+")
"""A paragraph that holds only a chapter number, a section number, or a row of asterisks."""

OUTPUT = Path("simple_topic_modeling/data/demo_text.txt")
PROVENANCE = Path("scripts/demo_text.provenance.json")
CACHE = Path(".cache/pg3420.txt")


def download(url: str, expected_sha256: str, cache: Path) -> bytes:
    """Return the file bytes, from the cache when the cached copy matches the hash.

    Raises:
        ValueError: when the downloaded bytes do not match `expected_sha256`.
    """
    if cache.exists():
        payload = cache.read_bytes()
        if hashlib.sha256(payload).hexdigest() == expected_sha256:
            print(f"cache hit: {cache}", file=sys.stderr)
            return payload
    print(f"downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "simple-topic-modeling/2.0"})
    with urllib.request.urlopen(request) as response:
        payload = bytes(response.read())
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        message = f"The file hash {digest} does not match the expected {expected_sha256}."
        raise ValueError(message)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return payload


def decode(payload: bytes) -> str:
    r"""Return the text of the bytes, without a BOM and with `\n` line endings.

    >>> decode(b"\xef\xbb\xbfone\r\ntwo\rthree")
    'one\ntwo\nthree'
    """
    text = payload.decode("utf-8").removeprefix("\ufeff")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_body(text: str) -> str:
    r"""Return the text between the Project Gutenberg start and end markers.

    Raises:
        ValueError: when either marker is missing, or the end comes before the start.

    >>> extract_body("head\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n"
    ...              "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfoot")
    '\nbody\n'
    """
    start, end = START_MARKER.search(text), END_MARKER.search(text)
    if start is None:
        raise ValueError("The start marker of the Project Gutenberg ebook is missing.")
    if end is None:
        raise ValueError("The end marker of the Project Gutenberg ebook is missing.")
    if end.start() < start.end():
        raise ValueError("The end marker comes before the start marker.")
    return text[start.end() : end.start()]


def split_paragraphs(body: str) -> list[str]:
    r"""Split on blank lines, and join each hard-wrapped paragraph into one line.

    >>> split_paragraphs("\nfirst line\n  second line\n\n\n\nnext\n")
    ['first line second line', 'next']
    """
    blocks = re.split(r"\n[ \t]*\n", body)
    return [joined for block in blocks if (joined := " ".join(block.split()))]


def drop_boilerplate(paragraphs: list[str]) -> list[str]:
    """Remove the Project Gutenberg boilerplate that sits inside the markers.

    >>> drop_boilerplate(["Ver.1*END*", "This etext was produced by A", "End. 8 April, 2001"])
    ['End.']
    """
    return [
        paragraph.removesuffix(DATE_STAMP)
        for paragraph in paragraphs
        if not paragraph.startswith(BOILERPLATE_PREFIXES)
    ]


def drop_sketch(paragraphs: list[str]) -> list[str]:
    """Remove the anonymous biographical sketch, from its title line to the dedication.

    >>> drop_sketch(["Title.", SKETCH_FIRST, "Life.", "Death.", DEDICATION_FIRST, "M. T."])
    ['Title.', 'TO', 'M. T.']

    Raises:
        ValueError: when the source no longer holds both ends of the sketch.
    """
    if SKETCH_FIRST not in paragraphs:
        message = "The source no longer holds the biographical sketch. Check the new file."
        raise ValueError(message)
    first = paragraphs.index(SKETCH_FIRST)
    if DEDICATION_FIRST not in paragraphs[first:]:
        message = "The source no longer holds the dedication after the sketch. Check the file."
        raise ValueError(message)
    return paragraphs[:first] + paragraphs[paragraphs.index(DEDICATION_FIRST, first) :]


def drop_navigation(paragraphs: list[str]) -> list[str]:
    """Remove the table of contents and the bare chapter and section numbers.

    The chapter titles in the text stay. Only their repetition in the contents goes.

    >>> drop_navigation(
    ...     ["Title.", "CONTENTS.", "INTRODUCTION.", "CHAPTER 1. RIGHTS.", "A sketch.",
    ...      "CHAPTER 1.", "RIGHTS.", "SECTION 1.1.", "Text.", "* * *", "More text."]
    ... )
    ['Title.', 'A sketch.', 'RIGHTS.', 'Text.', 'More text.']
    """
    kept: list[str] = []
    in_contents = False
    for paragraph in paragraphs:
        if paragraph == CONTENTS_HEADING:
            in_contents = True
            continue
        if in_contents and CONTENTS_ENTRY.fullmatch(paragraph):
            continue
        in_contents = False
        if not STRUCTURE_MARKER.fullmatch(paragraph):
            kept.append(paragraph)
    return kept


def strip_italics(paragraph: str) -> str:
    """Remove the `_` marks that Project Gutenberg uses for italics. No word changes.

    >>> strip_italics("She read _Emile_ and _the rest_ twice.")
    'She read Emile and the rest twice.'
    >>> strip_italics("a snake_case name")
    'a snake_case name'
    """
    return re.sub(r"(?<!\w)_(?=\S)([^_]*?\S)_(?!\w)", r"\1", paragraph)


def strip_footnote_label(paragraph: str) -> str:
    """Remove the `*Footnote.` label that the transcribers put before each note. Keep the note.

    The label is not a word of the work. Left in, it forms a topic of its own.

    >>> strip_footnote_label("(*Footnote. See the essay.)")
    '(See the essay.)'
    >>> strip_footnote_label("(Footnote. Also this one.)")
    '(Also this one.)'
    """
    return re.sub(r"\(\*?Footnote\.\s*", "(", paragraph)


def build_text(payload: bytes) -> list[str]:
    r"""Return the cleaned paragraphs of the ebook.

    >>> blocks = ["*** START OF THE PROJECT GUTENBERG EBOOK X ***", "Ver.1", "A _b_\nc.",
    ...           SKETCH_FIRST, DEDICATION_FIRST, "*** END OF THE PROJECT GUTENBERG EBOOK X ***"]
    >>> build_text("\n\n".join(blocks).encode())
    ['A b c.', 'TO']
    """
    paragraphs = split_paragraphs(extract_body(decode(payload)))
    paragraphs = drop_navigation(drop_sketch(drop_boilerplate(paragraphs)))
    return [strip_footnote_label(strip_italics(paragraph)) for paragraph in paragraphs]


def render(paragraphs: list[str]) -> bytes:
    r"""Return the output bytes: paragraphs, one blank line apart, one final newline.

    >>> render(["a", "b"])
    b'a\n\nb\n'
    """
    return ("\n\n".join(paragraphs) + "\n").encode("utf-8")


def main() -> int:
    """Build the demo text and report the provenance."""
    payload = download(SOURCE_URL, SOURCE_SHA256, CACHE)
    paragraphs = build_text(payload)
    written = render(paragraphs)
    OUTPUT.write_bytes(written)
    report = {
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "output": str(OUTPUT),
        "output_sha256": hashlib.sha256(written).hexdigest(),
        "output_bytes": len(written),
        "paragraphs": len(paragraphs),
        "words": len(written.decode("utf-8").split()),
    }
    PROVENANCE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
