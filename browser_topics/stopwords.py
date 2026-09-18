"""Stop-word lists and the effective stop-word set.

The lists ship inside the package, as `SPECS.md` section 3 requires. The app never downloads a list.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files

from browser_topics.config import Language, StopWordConfig
from browser_topics.preprocess import token_pattern

__all__ = [
    "base_stopwords",
    "effective_stopwords",
    "parse_word_input",
    "vectorizer_stopwords",
]

_SEPARATORS = re.compile(r"[,;\s]+")


@lru_cache(maxsize=8)
def base_stopwords(language: Language) -> frozenset[str]:
    """Read the packaged stop-word list for one language.

    The words are lowercase. The vectorizer runs with `lowercase=False`, so the caller must
    lowercase its text itself.

    >>> "the" in base_stopwords("en")
    True
    >>> "und" in base_stopwords("de")
    True
    >>> all(word == word.lower() for word in base_stopwords("fr"))
    True
    """
    resource = files("browser_topics") / "data" / "stopwords" / f"{language}.txt"
    text = resource.read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def parse_word_input(text: str) -> frozenset[str]:
    r"""Split user input on commas, whitespace, or newlines.

    `SPECS.md` section 3 accepts all three separators.

    >>> sorted(parse_word_input("alpha, beta\ngamma  delta"))
    ['alpha', 'beta', 'delta', 'gamma']
    >>> parse_word_input("   ")
    frozenset()
    >>> sorted(parse_word_input("Alpha ALPHA alpha"))
    ['alpha']
    """
    parts = _SEPARATORS.split(text.strip().lower())
    return frozenset(part for part in parts if part)


def effective_stopwords(language: Language, config: StopWordConfig) -> frozenset[str]:
    """Build the effective set: `(base OR added) MINUS always_keep`.

    >>> config = StopWordConfig(added="widget", always_keep="the")
    >>> words = effective_stopwords("en", config)
    >>> "widget" in words
    True
    >>> "the" in words
    False
    >>> no_base = StopWordConfig(use_base_list=False, added="alpha beta")
    >>> sorted(effective_stopwords("en", no_base))
    ['alpha', 'beta']
    """
    base = base_stopwords(language) if config.use_base_list else frozenset()
    added = parse_word_input(config.added)
    keep = parse_word_input(config.always_keep)
    return frozenset((base | added) - keep)


def vectorizer_stopwords(words: frozenset[str], min_token_length: int) -> list[str] | None:
    """Expand a stop-word set into the tokens that the vectorizer produces.

    A list such as spaCy's English list holds contractions like `'ll`. The token pattern splits
    those into `ll`, which would survive as a term. Expanding each entry through the same pattern
    keeps the stop words consistent with the tokenizer, and silences the scikit-learn warning.

    The result is sorted, so a run stays reproducible. An empty set returns `None`, which is what
    the scikit-learn vectorizers expect for "no stop words".

    >>> vectorizer_stopwords(frozenset({"'ll", "the"}), 2)
    ["'ll", 'll', 'the']
    >>> vectorizer_stopwords(frozenset({"new york"}), 2)
    ['new', 'new york', 'york']
    >>> vectorizer_stopwords(frozenset(), 2) is None
    True
    """
    pattern = re.compile(token_pattern(min_token_length))
    expanded = set(words)
    for word in words:
        expanded.update(pattern.findall(word))
    return sorted(expanded) or None
