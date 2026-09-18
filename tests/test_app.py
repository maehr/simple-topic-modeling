"""Smoke tests for the Streamlit app, using streamlit.testing.v1.AppTest.

These drive src/app.py the way a browser would: upload files, flip
checkboxes, move sliders, click the button. They mainly guard against
regressions that only show up when the whole script runs end to end,
which unit tests on topic_model.py functions cannot catch.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")

CLIMATE_TXT = b"climate warming ocean carbon emissions renewable solar policy"
MEDICINE_TXT = b"therapy dosage hospital recovery treatment patient clinical trial"
LATIN1_BYTES = "Zürich Genève Basel café naïve".encode("latin-1")
STOPWORDS_ONLY_TXT = b"the and of to a in is it that was"


def make_app_with_files(files):
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.file_uploader[0].set_value(files)
    at.run()
    return at


def get_checkbox(at, label):
    return [c for c in at.checkbox if c.label == label][0]


def get_slider(at, label):
    return [s for s in at.slider if s.label == label][0]


class TestAppStarts:
    def test_starts_without_uploads(self):
        at = AppTest.from_file(APP_PATH)
        at.run()
        assert not at.exception

    def test_starts_with_uploaded_files(self):
        at = make_app_with_files([("climate.txt", CLIMATE_TXT, "text/plain")])
        assert not at.exception
        assert get_checkbox(at, "Remove Stop Words").value is True


class TestRemoveStopWordsToggle:
    def test_toggling_off_does_not_raise(self):
        """Regression test for item 1: use_custom_stop_words and language
        used to be read unconditionally at button-click time even though
        they are only defined inside `if remove_stop_words:`. Unchecking
        "Remove Stop Words" crashed the app with a NameError.
        """
        at = make_app_with_files([("climate.txt", CLIMATE_TXT, "text/plain")])
        get_checkbox(at, "Remove Stop Words").set_value(False)
        at.run()
        assert not at.exception
        # The language/custom stop word controls disappear once the
        # checkbox is off.
        assert not any(c.label == "Use a Custom Stop Words List" for c in at.checkbox)

    def test_toggling_off_and_running_the_model_does_not_raise(self):
        at = make_app_with_files(
            [
                ("climate.txt", CLIMATE_TXT, "text/plain"),
                ("medicine.txt", MEDICINE_TXT, "text/plain"),
            ]
        )
        get_checkbox(at, "Remove Stop Words").set_value(False)
        at.run()
        get_slider(at, "Number of Topics").set_value(2)
        get_slider(at, "Max Iterations").set_value(10)
        at.run()
        at.button[0].set_value(True)
        at.run(timeout=60)
        assert not at.exception


class TestLatin1Upload:
    def test_non_utf8_upload_does_not_raise_and_warns(self):
        at = make_app_with_files(
            [
                ("latin1_sample.txt", LATIN1_BYTES, "text/plain"),
                ("climate.txt", CLIMATE_TXT, "text/plain"),
            ]
        )
        assert not at.exception
        warning_texts = " ".join(w.value for w in at.warning)
        assert "latin1_sample.txt" in warning_texts


class TestStopWordsOnlyCorpus:
    def test_empty_vocabulary_is_caught_with_a_clear_message(self):
        at = make_app_with_files(
            [("stopwords_only.txt", STOPWORDS_ONLY_TXT, "text/plain")]
        )
        get_slider(at, "Number of Topics").set_value(1)
        at.run()
        at.button[0].set_value(True)
        at.run(timeout=60)
        assert not at.exception
        assert len(at.error) == 1
        assert "No words remained" in at.error[0].value


class TestTooManyTopics:
    def test_more_topics_than_documents_is_caught_with_a_clear_message(self):
        at = make_app_with_files(
            [
                ("a.txt", CLIMATE_TXT, "text/plain"),
                ("b.txt", MEDICINE_TXT, "text/plain"),
            ]
        )
        get_slider(at, "Number of Topics").set_value(5)
        at.run()
        at.button[0].set_value(True)
        at.run(timeout=60)
        assert not at.exception
        assert len(at.error) == 1
        assert "exceeds the number of documents" in at.error[0].value
