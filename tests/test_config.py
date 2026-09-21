from importlib.metadata import version

import pytest
from pydantic import ValidationError

from simple_topic_modeling.config import (
    LANGUAGE_LABELS,
    UNKNOWN_VERSION,
    AppConfig,
    ModelConfig,
    PreprocessConfig,
    StopWordConfig,
    config_from_upload,
    normalize_language,
    package_version,
)
from simple_topic_modeling.errors import ConfigFileError
from simple_topic_modeling.exports import config_json


@pytest.mark.parametrize("code", sorted(LANGUAGE_LABELS))
def test_every_supported_language_normalizes_to_itself(code):
    assert normalize_language(code.upper()) == code


def test_sp_is_accepted_as_an_alias_for_es():
    assert normalize_language(" SP ") == "es"
    assert AppConfig(language="sp").language == "es"  # ty: ignore[invalid-argument-type]


def test_config_json_always_writes_es_never_sp():
    dumped = AppConfig(language="sp").model_dump()  # ty: ignore[invalid-argument-type]
    assert dumped["language"] == "es"


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_language("xx")


def test_unsupported_language_in_app_config_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig(language="xx")  # ty: ignore[invalid-argument-type]


def test_language_passes_through_when_absent_from_the_payload():
    assert AppConfig.model_validate({}).language == "en"


def test_non_dict_payload_reaches_the_model_validator():
    assert AppConfig.model_validate(AppConfig(language="de")).language == "de"


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_topics", 1), ("n_topics", 31), ("max_features", 0), ("min_df", 0), ("max_df", 0.4)],
)
def test_model_parameters_stay_inside_the_documented_range(field, value):
    with pytest.raises(ValidationError):
        ModelConfig(**{field: value})


def test_max_iter_must_be_positive():
    with pytest.raises(ValidationError):
        ModelConfig(max_iter=0)


def test_max_iter_accepts_an_explicit_value():
    assert ModelConfig(max_iter=7).resolved_max_iter == 7


def test_configs_are_frozen():
    with pytest.raises(ValidationError):
        PreprocessConfig().lowercase = False  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        StopWordConfig().added = "x"  # ty: ignore[invalid-assignment]


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        PreprocessConfig(unknown=True)  # ty: ignore[unknown-argument]


def test_summary_matches_the_spec_example():
    assert AppConfig().summary(2418) == (
        "2,418 documents · English · NMF · 10 topics · max 5,000 terms"
    )


def test_package_version_matches_the_project_version():
    assert package_version() == version("simple-topic-modeling")


def test_package_version_falls_back_when_the_distribution_is_missing():
    assert package_version("no-such-distribution-8f21c") == UNKNOWN_VERSION


def test_app_config_stamps_the_installed_version():
    assert AppConfig().app_version == package_version()


def test_config_from_upload_reads_a_file_the_app_wrote():
    payload = config_json(AppConfig(language="fr", model=ModelConfig(n_topics=6)))
    loaded = config_from_upload(payload)
    assert loaded.language == "fr"
    assert loaded.model.n_topics == 6


def test_config_from_upload_keeps_the_seed():
    payload = config_json(AppConfig(model=ModelConfig(random_seed=7)))
    assert config_from_upload(payload).model.random_seed == 7


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (b"\xff\xfe", "The file is not UTF-8 text."),
        (b'{"language": "fr"', "The file is not valid JSON."),
        (b"[1, 2, 3]", "The file does not hold a JSON object."),
        (b'"a string"', "The file does not hold a JSON object."),
        (b'{"language": "klingon"}', "The file holds a setting that the app cannot use."),
        (b'{"model": {"n_topics": 99}}', "The file holds a setting that the app cannot use."),
    ],
)
def test_config_from_upload_reports_a_recovery_action(payload, detail):
    with pytest.raises(ConfigFileError) as caught:
        config_from_upload(payload)
    assert caught.value.friendly.detail == detail
    assert caught.value.friendly.recovery == "Upload the config.json that this app wrote in Step 4."
