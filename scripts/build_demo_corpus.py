"""Build the demo corpus that ships inside the package.

The corpus holds articles from two Swiss newspapers of 1914. The EPFL DHLab published the
digitised archive for the 2015 Swiss Open Cultural Data Hackathon, under CC BY 4.0. The articles
themselves are anonymous newspaper text from 1914, so they left copyright long ago. `NOTICE`
records the source and the holder.

The script downloads the archive once, checks it against a known hash, and writes
`simple_topic_modeling/data/demo_corpus.csv`. It prints the source URL and the SHA-256 of both the
archive and the output, and it writes the same report to `scripts/demo_corpus.provenance.json`.

Run it from the repository root:

    uv run python scripts/build_demo_corpus.py

The output is deterministic. A rerun writes the same bytes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SOURCE_PAGE = "https://hack.glam.opendata.ch/project/234"
"""The hackathon project page that publishes the archive and states the licence."""

ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fi/7nntp81687zux5xz76oi3/data.zip"
    "?rlkey=3i0b7pn49o89nnttfxgom6vh3&dl=1"
)
"""The archive itself. The `?dl=1` query asks Dropbox for the bytes, not the preview page."""

ARCHIVE_SHA256 = "2a96b01494d9e7b4d814f9a333cf4fe07c39bc35cec3a0d4ad0f392e8b9e6458"
"""The SHA-256 of the archive. The script stops when the download does not match."""

CATEGORIES: dict[str, str] = {
    "CHRONIQUE MILITAIRE": "Chronique militaire",
    "LES SPORTS": "Les sports",
    "CHRONIQUE FINANCIERE": "Chronique financière",
    "LE TEMPS QU'IL FAIT": "Le temps qu'il fait",
    "LES LIVRES": "Les livres",
    "CHRONIQUE JUDICIAIRE": "Chronique judiciaire",
}
"""Each source rubric, and the label that the demo corpus shows for it.

A rubric is the newspaper's own section heading. The six below name six clear themes: the land
war, sport, finance, the weather, book reviews, and the courts.
"""

DOCUMENTS_PER_CATEGORY = 50
"""How many articles each category contributes. Six categories stay inside the 200 to 400 range."""

MAX_WORDS = 250
"""How many words each article keeps. This holds the packaged file near half a megabyte."""

MIN_WORDS = 80
"""The shortest article the corpus accepts, from `SPECS.md` section 9."""

OUTPUT = Path("simple_topic_modeling/data/demo_corpus.csv")
PROVENANCE = Path("scripts/demo_corpus.provenance.json")
CACHE = Path(".cache/glamhack-2015-data.zip")


@dataclass(frozen=True)
class Article:
    """One newspaper article, ready for the corpus."""

    document_id: str
    text: str
    category: str
    date: str


def download(url: str, expected_sha256: str, cache: Path) -> bytes:
    """Return the archive bytes, from the cache when the cached copy matches the hash.

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
        message = f"The archive hash {digest} does not match the expected {expected_sha256}."
        raise ValueError(message)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return payload


def normalize_rubric(name: str) -> str:
    """Return the comparable form of a section heading.

    >>> normalize_rubric("Les Sports .")
    'LES SPORTS'
    """
    return re.sub(r"[.\s]+$", "", re.sub(r"\s+", " ", name).strip().upper())


def clean_text(raw: str, title: str) -> str:
    """Return readable article text, without the repeated section heading.

    The heading has to go. It names the category, so a model that kept it would sort the corpus by
    a word that the category itself supplies.

    The archive puts a space before each comma and each full stop, because the text comes from an
    OCR process. The spacing is repaired here. No word is changed.

    >>> clean_text("Les Sports . Le match a eu lieu , hier .", "Les Sports")
    'Le match a eu lieu, hier.'
    """
    text = re.sub(r"\s+", " ", raw).strip()
    head = normalize_rubric(title)
    if head and normalize_rubric(text).startswith(head):
        text = text[len(title) :].lstrip(" .,-—:;")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def iso_date(source: str) -> str:
    """Convert a `DD/MM/YYYY` date to `YYYY-MM-DD`.

    >>> iso_date("17/11/1914")
    '1914-11-17'
    """
    day, month, year = source.split("/")
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def read_articles(archive: zipfile.ZipFile) -> list[Article]:
    """Return every article of an accepted category, in a stable order."""
    pattern = re.compile(r"^data/(GDL|JDG)/raw/1914/\d{2}/\d{2}/\d+\.xml$")
    found: list[Article] = []
    for name in sorted(archive.namelist()):
        if not pattern.match(name):
            continue
        try:
            root = ET.fromstring(archive.read(name).decode("utf-8"))
        except (ET.ParseError, UnicodeDecodeError):
            continue
        for entity in root.findall("entity"):
            article = _read_entity(entity)
            if article is not None:
                found.append(article)
    return found


def _read_entity(entity: ET.Element) -> Article | None:
    """Return one article, or `None` when the entity does not qualify."""
    meta, full_text = entity.find("meta"), entity.find("full_text")
    if meta is None or full_text is None or not full_text.text:
        return None

    def field(tag: str) -> str:
        node = meta.find(tag)
        return (node.text or "").strip() if node is not None else ""

    category = CATEGORIES.get(normalize_rubric(field("name")))
    if category is None:
        return None
    text = clean_text(full_text.text, field("name"))
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    return Article(
        document_id=f"{field('publication')}-{iso_date(field('issue_date'))}-{field('id')}",
        text=" ".join(words[:MAX_WORDS]),
        category=category,
        date=iso_date(field("issue_date")),
    )


def select(articles: list[Article]) -> list[Article]:
    """Return an even spread of articles across the year, for each category.

    The spread matters. A block of consecutive issues would tie each category to one month, and
    the date chart would then show the sampling, not the year.
    """
    grouped: dict[str, list[Article]] = defaultdict(list)
    for article in articles:
        grouped[article.category].append(article)
    chosen: list[Article] = []
    for category in CATEGORIES.values():
        ordered = sorted(grouped[category], key=lambda item: (item.date, item.document_id))
        if len(ordered) <= DOCUMENTS_PER_CATEGORY:
            chosen.extend(ordered)
            continue
        step = len(ordered) / DOCUMENTS_PER_CATEGORY
        chosen.extend(ordered[int(index * step)] for index in range(DOCUMENTS_PER_CATEGORY))
    return sorted(chosen, key=lambda item: (item.date, item.document_id))


def write_csv(articles: list[Article], target: Path) -> bytes:
    """Write the corpus and return the bytes written."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["document_id", "text", "category", "date"])
    for article in articles:
        writer.writerow([article.document_id, article.text, article.category, article.date])
    payload = buffer.getvalue().encode("utf-8")
    target.write_bytes(payload)
    return payload


def main() -> int:
    """Build the corpus and report the provenance."""
    payload = download(ARCHIVE_URL, ARCHIVE_SHA256, CACHE)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        articles = select(read_articles(archive))
    missing = set(CATEGORIES.values()) - {item.category for item in articles}
    if missing:
        print(f"These categories found no article: {sorted(missing)}", file=sys.stderr)
        return 1
    written = write_csv(articles, OUTPUT)
    counts: dict[str, int] = defaultdict(int)
    for article in articles:
        counts[article.category] += 1
    report = {
        "source_page": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "output": str(OUTPUT),
        "output_sha256": hashlib.sha256(written).hexdigest(),
        "output_bytes": len(written),
        "documents": len(articles),
        "documents_per_category": dict(sorted(counts.items())),
    }
    PROVENANCE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
