import re

import numpy as np
import pytest

from topic_model import (
    NGRAM_OPTIONS,
    STOP_WORDS,
    EmptyVocabularyError,
    TooManyTopicsError,
    build_topic_distribution,
    build_topic_order_mapping,
    clean_custom_stop_words,
    fit_topic_model,
    resolve_stop_words,
    resolve_token_pattern,
)

# A small, clearly separable corpus: three themes, distinct vocabularies,
# repeated to give LDA enough signal. Mirrors the shape of the scratchpad
# test corpus (climate / medicine / history word-salad documents).
CLIMATE_DOCS = (
    "climate warming glacier ocean carbon emissions renewable solar "
    "policy temperature climate warming glacier ocean carbon emissions "
    "renewable solar policy temperature climate warming glacier ocean",
    "carbon emissions climate policy warming ocean glacier renewable "
    "solar temperature carbon emissions climate policy warming ocean "
    "glacier renewable solar temperature carbon emissions climate",
    "renewable solar energy climate warming carbon ocean glacier policy "
    "temperature renewable solar energy climate warming carbon ocean "
    "glacier policy temperature renewable solar energy climate",
)
MEDICINE_DOCS = (
    "therapy dosage hospital recovery treatment patient clinical trial "
    "diagnosis disease symptoms therapy dosage hospital recovery "
    "treatment patient clinical trial diagnosis disease symptoms",
    "patient diagnosis clinical trial hospital treatment dosage therapy "
    "recovery disease symptoms patient diagnosis clinical trial "
    "hospital treatment dosage therapy recovery disease symptoms",
    "clinical dosage therapy hospital recovery treatment disease "
    "symptoms patient trial diagnosis clinical dosage therapy hospital "
    "recovery treatment disease symptoms patient trial diagnosis",
)
HISTORY_DOCS = (
    "manuscript archive medieval scribe chronicle monastery century "
    "empire archduke treaty manuscript archive medieval scribe "
    "chronicle monastery century empire archduke treaty manuscript",
    "archduke treaty chronicle scribe century empire archive medieval "
    "manuscript monastery archduke treaty chronicle scribe century "
    "empire archive medieval manuscript monastery archduke treaty",
    "century monastery empire archive manuscript scribe chronicle "
    "medieval treaty archduke century monastery empire archive "
    "manuscript scribe chronicle medieval treaty archduke century",
)


def separable_corpus():
    return CLIMATE_DOCS + MEDICINE_DOCS + HISTORY_DOCS


def theme_labels():
    return (
        ["climate"] * len(CLIMATE_DOCS)
        + ["medicine"] * len(MEDICINE_DOCS)
        + ["history"] * len(HISTORY_DOCS)
    )


class TestVectorizerBuilds:
    @pytest.mark.parametrize("ngram_name", list(NGRAM_OPTIONS.keys()))
    def test_all_ngram_options(self, ngram_name):
        vectorizer, dtm, lda, lda_output = fit_topic_model(
            separable_corpus(),
            "english",
            resolve_token_pattern(True),
            NGRAM_OPTIONS[ngram_name],
            num_topics=2,
            max_iter=5,
            random_state=0,
        )
        assert dtm.shape[0] == len(separable_corpus())
        assert lda_output.shape == (len(separable_corpus()), 2)

    @pytest.mark.parametrize("language", list(STOP_WORDS.keys()))
    def test_all_stop_word_languages(self, language):
        vectorizer, dtm, lda, lda_output = fit_topic_model(
            separable_corpus(),
            STOP_WORDS[language]
            if isinstance(STOP_WORDS[language], str)
            else tuple(STOP_WORDS[language]),
            resolve_token_pattern(True),
            (1, 1),
            num_topics=2,
            max_iter=5,
            random_state=0,
        )
        assert dtm.shape[0] == len(separable_corpus())


class TestCustomStopWords:
    def test_splits_and_strips(self):
        assert clean_custom_stop_words("foo, bar ,  baz") == ["foo", "bar", "baz"]

    def test_empty_entries_are_dropped(self):
        # Trailing comma / blank textarea must not produce [""].
        assert clean_custom_stop_words("foo,,bar,") == ["foo", "bar"]
        assert clean_custom_stop_words("") == []
        assert clean_custom_stop_words("   ") == []

    def test_resolve_stop_words_empty_custom_falls_back_to_none(self):
        # An empty custom stop word list must not be passed to
        # CountVectorizer as [] (which would keep every token) silently
        # diverging from user intent; it resolves to None like "no
        # stop words".
        result = resolve_stop_words(
            remove_stop_words=True,
            use_custom_stop_words=True,
            custom_stop_words=[],
            language="english",
        )
        assert result is None

    def test_resolve_stop_words_off_ignores_language_and_custom(self):
        # Regression for item 1: language/use_custom_stop_words must not
        # be required when remove_stop_words is False.
        result = resolve_stop_words(
            remove_stop_words=False,
            use_custom_stop_words=False,
            custom_stop_words=[],
            language="english",
        )
        assert result is None

    def test_fit_with_custom_stop_words(self):
        vectorizer, dtm, lda, lda_output = fit_topic_model(
            separable_corpus(),
            tuple(["climate", "warming"]),
            resolve_token_pattern(True),
            (1, 1),
            num_topics=2,
            max_iter=5,
            random_state=0,
        )
        assert "climate" not in vectorizer.get_feature_names_out()
        assert "warming" not in vectorizer.get_feature_names_out()


class TestGuards:
    def test_empty_vocabulary_raises(self):
        stopwords_only = ("the and of to a in is it that was",)
        with pytest.raises(EmptyVocabularyError):
            fit_topic_model(
                stopwords_only,
                "english",
                resolve_token_pattern(True),
                (1, 1),
                num_topics=1,
                max_iter=5,
                random_state=0,
            )

    def test_too_many_topics_raises(self):
        docs = ("alpha beta gamma", "delta epsilon zeta")
        with pytest.raises(TooManyTopicsError):
            fit_topic_model(
                docs,
                None,
                resolve_token_pattern(True),
                (1, 1),
                num_topics=5,
                max_iter=5,
                random_state=0,
            )


class TestTopicOrderRemap:
    def test_dominant_topic_matches_pyldavis_topic_order(self):
        """Prove build_topic_distribution's Dominant_Topic column uses
        the same numbering pyLDAvis displays in the figure.

        This is the item 8 regression test. It has two independent
        checks:

        1. The most prevalent topic, per an independent recomputation of
           pyLDAvis's own ranking formula (topic weight summed over
           documents, weighted by document length), must be labeled
           "Topic 1" -- proving the mapping direction is not inverted.
        2. Every document's CSV Dominant_Topic must equal
           mapping[sklearn_argmax_topic] for that document -- proving the
           per-row remap is consistent with prepared_data.topic_order for
           the whole corpus, not just the top topic.
        """
        import pyLDAvis.lda_model

        texts = separable_corpus()
        vectorizer, dtm, lda, lda_output = fit_topic_model(
            texts,
            "english",
            resolve_token_pattern(True),
            (1, 1),
            num_topics=3,
            max_iter=25,
            random_state=1,
        )
        prepared = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)

        # Independent recomputation of pyLDAvis's topic_proportion
        # ranking (see pyLDAvis._prepare.prepare): topic_freq = doc_topic
        # weights summed over documents weighted by document length, then
        # sorted descending. This does not call our own mapping code.
        doc_lengths = np.asarray(dtm.sum(axis=1)).ravel()
        topic_freq = (lda_output * doc_lengths[:, None]).sum(axis=0)
        most_prevalent_sklearn_topic = int(np.argmax(topic_freq))

        mapping = build_topic_order_mapping(prepared.topic_order)
        assert mapping[most_prevalent_sklearn_topic] == 1

        # Full per-row consistency check.
        filenames = [f"doc_{i}.txt" for i in range(len(texts))]
        df = build_topic_distribution(filenames, lda_output, prepared.topic_order)
        sklearn_dominant = lda_output.argmax(axis=1)
        expected = [mapping[t] for t in sklearn_dominant]
        assert list(df["Dominant_Topic"]) == expected

        # And columns are labeled 1..n_topics, matching pyLDAvis's
        # 1-based, start_index=1 default numbering.
        topic_columns = sorted(c for c in df.columns if isinstance(c, int))
        assert topic_columns == [1, 2, 3]

    def test_documents_of_the_same_theme_get_the_same_dominant_topic(self):
        """On a corpus with clearly separable topics, same-theme
        documents should agree on their (remapped) dominant topic, and
        different themes should disagree -- a sanity check that the
        remap doesn't scramble anything semantically.
        """
        import pyLDAvis.lda_model

        texts = separable_corpus()
        labels = theme_labels()
        vectorizer, dtm, lda, lda_output = fit_topic_model(
            texts,
            "english",
            resolve_token_pattern(True),
            (1, 1),
            num_topics=3,
            max_iter=25,
            random_state=1,
        )
        prepared = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)
        filenames = [f"doc_{i}.txt" for i in range(len(texts))]
        df = build_topic_distribution(filenames, lda_output, prepared.topic_order)
        df["theme"] = labels

        topics_per_theme = df.groupby("theme")["Dominant_Topic"].nunique()
        assert (topics_per_theme == 1).all()

        dominant_by_theme = df.groupby("theme")["Dominant_Topic"].first()
        assert dominant_by_theme.nunique() == 3


class TestAccentedWordsSurvivePreprocessing:
    """The app offers French, German, and Spanish stop word lists.

    An ASCII-only token pattern silently dropped every accented word before
    LDA ever saw it, which quietly degraded every non-English corpus.
    """

    @pytest.mark.parametrize(
        "word",
        ["après", "Zürich", "café", "médecin", "über", "años", "naïf"],
    )
    def test_short_word_pattern_keeps_accented_words(self, word):
        pattern = resolve_token_pattern(True)
        assert re.fullmatch(pattern, word) is not None

    @pytest.mark.parametrize("token", ["a", "ab", "12", "3abc", "_x"])
    def test_short_word_pattern_still_drops_noise(self, token):
        pattern = resolve_token_pattern(True)
        assert re.fullmatch(pattern, token) is None

    def test_accented_words_reach_the_vocabulary(self):
        docs = ("après le café à Zürich",) * 4
        _, dtm, _, _ = fit_topic_model(
            docs,
            None,
            resolve_token_pattern(True),
            (1, 1),
            1,
            10,
        )
        assert dtm.shape[1] > 0
