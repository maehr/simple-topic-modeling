import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig
from simple_topic_modeling.io import build_corpus

ANIMAL_TEXTS = [
    "cat dog runs fast across the field",
    "cat sleeps often on the warm couch",
    "dog barks loudly at the postman",
    "cat and dog play together every day",
    "bird sings a song in the morning",
    "bird flies high above the tall trees",
    "fish swims deep in the cold water",
    "fish and bird share the quiet pond",
]


@pytest.fixture
def corpus():
    built, _ = build_corpus(ANIMAL_TEXTS, [f"d{index}" for index in range(len(ANIMAL_TEXTS))])
    return built


@pytest.fixture
def config():
    return AppConfig(model=ModelConfig(n_topics=3, min_df=1))
