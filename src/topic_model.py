"""Model logic for the Simple Topic Modeling app.

This module contains everything that touches scikit-learn, pandas, and
pyLDAvis. `app.py` only calls into it and renders the results with
Streamlit widgets.
"""

import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.stopwords import french, german, spanish

# Stop word lists and n-gram options offered in the UI.
STOP_WORDS = {
    "english": "english",
    "german": german,
    "french": french,
    "spanish": spanish,
}

NGRAM_OPTIONS = {
    "unigram": (1, 1),
    "unigram + bigram": (1, 2),
    "unigram + bigram + trigram": (1, 3),
    "bigram": (2, 2),
    "trigram": (3, 3),
}

# scikit-learn's own default token pattern for the "word" analyzer.
DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
# Drops single- and two-character tokens and tokens starting with a digit.
# [^\W\d_] is a Unicode letter: a word character that is not a digit and not an
# underscore. An [a-zA-Z] class here would drop every accented word, and the app
# offers French, German, and Spanish.
SHORT_WORD_TOKEN_PATTERN = r"(?u)\b[^\W\d_][^\W_]{2,}\b"


class EmptyVocabularyError(Exception):
    """Raised when preprocessing removes every token from the corpus."""


class TooManyTopicsError(Exception):
    """Raised when the requested topic count exceeds the document count."""


def decode_uploaded_files(uploaded_files):
    """Decode uploaded files to text, tolerating non-UTF-8 encodings.

    Returns a tuple of (records, undecodable_filenames). Each record is a
    dict with "filename" and "content" keys. Files that cannot be decoded
    as UTF-8 are still included, with invalid bytes replaced, and their
    names are reported separately so the caller can warn the user.
    """
    records = []
    undecodable_filenames = []
    for file in uploaded_files:
        raw = file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
            undecodable_filenames.append(file.name)
        records.append({"filename": file.name, "content": content})
    return records, undecodable_filenames


def clean_custom_stop_words(raw_text):
    """Split a comma-separated stop word field into a clean list.

    Blank entries (from empty input, or trailing commas) are dropped.
    """
    return [word.strip() for word in raw_text.split(",") if word.strip()]


def resolve_stop_words(
    remove_stop_words, use_custom_stop_words, custom_stop_words, language
):
    """Determine the `stop_words` argument to pass to CountVectorizer."""
    if not remove_stop_words:
        return None
    if use_custom_stop_words:
        return custom_stop_words or None
    return STOP_WORDS[language]


def resolve_token_pattern(remove_short_words_and_numbers):
    """Determine the `token_pattern` argument to pass to CountVectorizer."""
    if remove_short_words_and_numbers:
        return SHORT_WORD_TOKEN_PATTERN
    return DEFAULT_TOKEN_PATTERN


@st.cache_data(show_spinner=False)
def fit_topic_model(
    texts,
    stop_words,
    token_pattern,
    ngram_range,
    num_topics,
    max_iter,
    random_state=42,
):
    """Vectorize the corpus and fit an LDA model.

    Arguments are plain, hashable values (a tuple of texts, a tuple/string
    of stop words or None, strings, ints) so Streamlit can cache the
    result.

    Raises EmptyVocabularyError if preprocessing leaves no vocabulary, and
    TooManyTopicsError if num_topics exceeds the number of documents.
    """
    if num_topics > len(texts):
        raise TooManyTopicsError(
            f"Number of topics ({num_topics}) exceeds the number of "
            f"documents ({len(texts)}). Choose a lower number of topics."
        )

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=list(stop_words) if isinstance(stop_words, tuple) else stop_words,
        token_pattern=token_pattern,
        ngram_range=ngram_range,
    )
    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise EmptyVocabularyError(
            "No words remained after preprocessing. Try a less aggressive "
            "stop word or filtering setting."
        ) from exc

    lda = LatentDirichletAllocation(
        n_components=num_topics, max_iter=max_iter, random_state=random_state
    )
    lda_output = lda.fit_transform(dtm)
    return vectorizer, dtm, lda, lda_output


def prepare_visualization(lda, dtm, vectorizer):
    """Build the pyLDAvis prepared data used for the interactive figure."""
    return pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)


def build_topic_order_mapping(topic_order):
    """Map scikit-learn's 0-based topic index to pyLDAvis's display label.

    `topic_order` is `pyLDAvis.PreparedData.topic_order`: a list, one entry
    per topic, in the order pyLDAvis displays them (most prevalent first),
    holding the *original* 0-based sklearn topic index plus pyLDAvis's
    1-based `start_index`. Position `i` in that list (0-based) is shown in
    the visualization as "Topic i + 1".

    So `topic_order[i] - 1` is the original sklearn topic index of the
    topic pyLDAvis displays in position `i`, i.e. as "Topic i + 1". This
    function inverts that: sklearn topic index -> pyLDAvis display number.
    """
    return {
        original - 1: display for display, original in enumerate(topic_order, start=1)
    }


def build_topic_distribution(filenames, lda_output, topic_order):
    """Build the downloadable topic distribution table.

    Columns are per-topic weights, labeled with the same topic numbers
    pyLDAvis shows in the figure (via `topic_order`), so `Dominant_Topic`
    always agrees with what the visualization displays.
    """
    mapping = build_topic_order_mapping(topic_order)
    weights = pd.DataFrame(lda_output, columns=range(lda_output.shape[1]))
    weights = weights.rename(columns=mapping)
    weights = weights[sorted(weights.columns)]
    dominant_topic = weights.idxmax(axis=1)

    df = pd.DataFrame(
        {
            "filename": pd.Series(filenames).reset_index(drop=True),
            "Dominant_Topic": dominant_topic.reset_index(drop=True),
        }
    )
    df = pd.concat([df, weights.reset_index(drop=True)], axis=1)
    return df
