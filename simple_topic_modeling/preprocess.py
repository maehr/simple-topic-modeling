"""Text cleaning and the cheap frequent-terms preview.

`SPECS.md` section 3 applies cleaning before the vectorizer. The vectorizer therefore runs with
`lowercase=False` and a token pattern that carries the minimum token length.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from sklearn.feature_extraction.text import CountVectorizer

from simple_topic_modeling.config import PreprocessConfig

__all__ = ["clean_text", "clean_texts", "frequent_terms", "token_pattern"]

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUMBER = re.compile(r"\b\d[\d.,]*\b")
_WHITESPACE = re.compile(r"\s+")


def token_pattern(min_token_length: int) -> str:
    r"""Build the vectorizer token pattern for a minimum token length.

    >>> token_pattern(2)
    '(?u)\\b\\w{2,}\\b'
    >>> import re
    >>> re.findall(token_pattern(3), "a bc def")
    ['def']
    """
    return rf"(?u)\b\w{{{min_token_length},}}\b"


def clean_text(text: str, config: PreprocessConfig) -> str:
    """Apply the cleaning steps in a fixed order.

    The order is: strip URLs, strip emails, lowercase, strip numbers, normalize accents.
    URLs go first, so that a lowercase step cannot change what the URL pattern matches.

    >>> config = PreprocessConfig()
    >>> clean_text("Visit https://example.com NOW", config)
    'visit now'
    >>> clean_text("Mail me@example.com please", config)
    'mail please'
    >>> clean_text("Total 1,250 items", PreprocessConfig(strip_numbers=True))
    'total items'
    >>> clean_text("Café Zürich", PreprocessConfig(normalize_accents=True))
    'cafe zurich'
    >>> clean_text("Keep It Cased", PreprocessConfig(lowercase=False))
    'Keep It Cased'
    """
    if config.strip_urls:
        text = _URL.sub(" ", text)
    if config.strip_emails:
        text = _EMAIL.sub(" ", text)
    if config.lowercase:
        text = text.lower()
    if config.strip_numbers:
        text = _NUMBER.sub(" ", text)
    if config.normalize_accents:
        decomposed = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _WHITESPACE.sub(" ", text).strip()


def clean_texts(texts: Sequence[str], config: PreprocessConfig) -> list[str]:
    """Clean every document in a corpus.

    >>> clean_texts(["Hello WORLD", "  Second  DOC "], PreprocessConfig())
    ['hello world', 'second doc']
    """
    return [clean_text(text, config) for text in texts]


def _expand(stop_words: frozenset[str], min_token_length: int) -> list[str] | None:
    """Expand the stop words to match the tokenizer. See `stopwords.vectorizer_stopwords`."""
    from simple_topic_modeling.stopwords import vectorizer_stopwords

    return vectorizer_stopwords(stop_words, min_token_length)


def frequent_terms(
    texts: Sequence[str],
    stop_words: frozenset[str],
    min_token_length: int = 2,
    top_n: int = 25,
) -> list[tuple[str, int]]:
    """Count the most frequent terms after cleaning.

    This preview never fits a topic model, as `SPECS.md` section 3 requires. It returns an empty
    list when no term survives, so the caller can show an empty state instead of an error.

    >>> frequent_terms(["cat dog cat", "cat bird"], frozenset(), top_n=2)
    [('cat', 3), ('bird', 1)]
    >>> frequent_terms(["cat dog cat"], frozenset({"cat"}))
    [('dog', 1)]
    >>> frequent_terms(["cat"], frozenset({"cat"}))
    []
    """
    vectorizer = CountVectorizer(
        lowercase=False,
        stop_words=_expand(stop_words, min_token_length),
        token_pattern=token_pattern(min_token_length),
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, counts, strict=True), key=lambda pair: (-pair[1], pair[0]))
    return [(term, int(count)) for term, count in ranked[:top_n]]
