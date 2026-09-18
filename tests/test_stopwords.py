from utils.stopwords import french, german, spanish


class TestStopWordLists:
    def test_german_non_empty_no_duplicates(self):
        assert len(german) > 0
        assert len(german) == len(set(german))

    def test_french_non_empty_no_duplicates(self):
        assert len(french) > 0
        assert len(french) == len(set(french))

    def test_spanish_non_empty_no_duplicates(self):
        assert len(spanish) > 0
        assert len(spanish) == len(set(spanish))
