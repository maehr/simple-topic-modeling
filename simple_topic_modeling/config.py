"""Configuration models.

These models cross the `config.json` boundary, so they use Pydantic, as `AGENTS.md` section 4
requires. The defaults come from `SPECS.md` sections 3 and 4.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from simple_topic_modeling.errors import ConfigFileError

__all__ = [
    "LANGUAGE_ALIASES",
    "LANGUAGE_LABELS",
    "AnalysisMode",
    "AppConfig",
    "Language",
    "ModelConfig",
    "ModelType",
    "NgramChoice",
    "PreprocessConfig",
    "SplitMode",
    "StopWordConfig",
    "config_from_upload",
    "load_app_config",
    "package_version",
]

Language = Literal["en", "de", "fr", "it", "es"]
ModelType = Literal["nmf", "lda"]
NgramChoice = Literal["1", "1-2", "2"]
SplitMode = Literal["whole", "blank_lines", "lines"]
"""How the app splits one text into documents: not at all, on blank lines, or on each line."""
AnalysisMode = Literal["corpus", "long_document"]
"""How the app treats one text: as unrelated documents, or as ordered segments of one document."""

LANGUAGE_LABELS: dict[Language, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
}
"""User-facing label for each supported language."""

LANGUAGE_ALIASES: dict[str, Language] = {"sp": "es"}
"""Accepted import aliases. `SPECS.md` section 3 accepts `sp`, but the app always writes `es`."""

MAX_VOCABULARY_LIMIT = 20_000
"""Largest vocabulary that `SPECS.md` section 4 allows."""

_DEFAULT_MAX_ITER: dict[ModelType, int] = {"nmf": 400, "lda": 20}

DISTRIBUTION = "simple-topic-modeling"
"""The name of the installed distribution that carries this package."""

UNKNOWN_VERSION = "0.0.0+unknown"
"""The version to report when the distribution metadata is missing."""


def package_version(distribution: str = DISTRIBUTION) -> str:
    """Return the version of an installed distribution.

    The browser installs the app as a wheel, so the metadata is present there. A source tree that
    was never installed has no metadata, so the function reports `UNKNOWN_VERSION` instead.

    >>> package_version()[0].isdigit()
    True
    >>> package_version("no-such-distribution-8f21c")
    '0.0.0+unknown'
    """
    try:
        return version(distribution)
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def normalize_language(value: str) -> Language:
    """Map a language code or an accepted alias to a supported code.

    >>> normalize_language("DE")
    'de'
    >>> normalize_language("sp")
    'es'
    """
    code = value.strip().lower()
    code = LANGUAGE_ALIASES.get(code, code)  # type: ignore[arg-type]
    if code not in LANGUAGE_LABELS:
        supported = ", ".join(sorted(LANGUAGE_LABELS))
        message = f"Unsupported language {value!r}. Use one of: {supported}."
        raise ValueError(message)
    return code  # type: ignore[return-value]


class PreprocessConfig(BaseModel):
    """Cleaning options. The defaults are the safe baseline of `SPECS.md` section 3.

    >>> PreprocessConfig().lowercase
    True
    >>> PreprocessConfig().strip_numbers
    False
    >>> PreprocessConfig(min_token_length=3).min_token_length
    3
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lowercase: bool = True
    strip_urls: bool = True
    strip_emails: bool = True
    strip_numbers: bool = False
    normalize_accents: bool = False
    min_token_length: Annotated[int, Field(ge=1, le=10)] = 2


class StopWordConfig(BaseModel):
    """Stop-word choices.

    `use_base_list` false means the mixed-language case of `SPECS.md` section 3.

    >>> StopWordConfig().use_base_list
    True
    >>> StopWordConfig(added="foo, bar").added
    'foo, bar'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    use_base_list: bool = True
    added: str = ""
    always_keep: str = ""


class ModelConfig(BaseModel):
    """Model choice and parameters. The defaults come from `SPECS.md` section 4.

    >>> ModelConfig().model_type
    'nmf'
    >>> ModelConfig().n_topics
    10
    >>> ModelConfig().resolved_max_iter
    400
    >>> ModelConfig(model_type="lda").resolved_max_iter
    20
    >>> ModelConfig(model_type="lda", max_iter=50).resolved_max_iter
    50
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_type: ModelType = "nmf"
    n_topics: Annotated[int, Field(ge=2, le=30)] = 10
    max_features: Annotated[int, Field(ge=1, le=MAX_VOCABULARY_LIMIT)] = 5_000
    min_df: Annotated[int, Field(ge=1)] = 2
    max_df: Annotated[float, Field(ge=0.5, le=1.0)] = 0.95
    ngrams: NgramChoice = "1"
    random_seed: int = 42
    max_iter: int | None = None

    @property
    def resolved_max_iter(self) -> int:
        """Return the iteration limit, or the model default when none is set.

        >>> ModelConfig(max_iter=None).resolved_max_iter
        400
        """
        if self.max_iter is None:
            return _DEFAULT_MAX_ITER[self.model_type]
        return self.max_iter

    @property
    def ngram_range(self) -> tuple[int, int]:
        """Return the n-gram range that the vectorizer expects.

        >>> ModelConfig(ngrams="1").ngram_range
        (1, 1)
        >>> ModelConfig(ngrams="1-2").ngram_range
        (1, 2)
        >>> ModelConfig(ngrams="2").ngram_range
        (2, 2)
        """
        if self.ngrams == "1":
            return (1, 1)
        if self.ngrams == "1-2":
            return (1, 2)
        return (2, 2)

    @field_validator("max_iter")
    @classmethod
    def _check_max_iter(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            message = "max_iter must be at least 1."
            raise ValueError(message)
        return value


class AppConfig(BaseModel):
    """The full configuration that `config.json` stores.

    The version stamp names the app that wrote the file, so a second person can repeat a run with
    the same build.

    >>> AppConfig().language
    'en'
    >>> AppConfig(language="sp").language
    'es'
    >>> AppConfig().app_version == package_version()
    True
    >>> AppConfig().summary(2418)
    '2,418 documents · English · NMF · 10 topics · max 5,000 terms'

    `analyse_as` and `split_mode` record how the app split one text into documents. A
    `split_mode` of `None` means that the app did not split the input. Both fields have defaults,
    so a `config.json` from an older version still loads.

    >>> AppConfig().analyse_as, AppConfig().split_mode
    ('corpus', None)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_version: str = Field(default_factory=package_version)
    language: Language = "en"
    preprocess: PreprocessConfig = PreprocessConfig()
    stop_words: StopWordConfig = StopWordConfig()
    model: ModelConfig = ModelConfig()
    analyse_as: AnalysisMode = "corpus"
    split_mode: SplitMode | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_language(cls, data: object) -> object:
        if isinstance(data, dict) and "language" in data:
            return {**data, "language": normalize_language(str(data["language"]))}
        return data

    def summary(self, document_count: int) -> str:
        """Build the one-line run summary of `SPECS.md` section 4.

        >>> AppConfig(language="de", model=ModelConfig(n_topics=8)).summary(12)
        '12 documents · German · NMF · 8 topics · max 5,000 terms'

        A long document counts its segments, not its documents.

        >>> AppConfig(analyse_as="long_document").summary(40)
        '40 segments · English · NMF · 10 topics · max 5,000 terms'
        """
        unit = "segments" if self.analyse_as == "long_document" else "documents"
        return (
            f"{document_count:,} {unit}"
            f" · {LANGUAGE_LABELS[self.language]}"
            f" · {self.model.model_type.upper()}"
            f" · {self.model.n_topics} topics"
            f" · max {self.model.max_features:,} terms"
        )


def load_app_config(payload: dict[str, object]) -> AppConfig:
    """Build an `AppConfig` from an imported `config.json`.

    The export adds result fields such as `topic_names`. This loader drops any key that
    `AppConfig` does not own, so a file that the app wrote always imports again.

    >>> load_app_config({"language": "sp", "topic_names": ["Economy"]}).language
    'es'
    >>> load_app_config({}).model.n_topics
    10
    """
    known = {key: value for key, value in payload.items() if key in AppConfig.model_fields}
    return AppConfig.model_validate(known)


def config_from_upload(payload: bytes) -> AppConfig:
    """Build an `AppConfig` from the bytes of an uploaded `config.json`.

    The app reads an uploaded file as bytes, so the decoding belongs here and not in the notebook.
    Every failure carries a recovery action.

    >>> import json
    >>> data = json.dumps({"language": "fr", "topic_names": ["Sport"]}).encode("utf-8")
    >>> config_from_upload(data).language
    'fr'
    >>> config_from_upload(b'{"language": "fr"')
    Traceback (most recent call last):
    simple_topic_modeling.errors.ConfigFileError: The file is not valid JSON. ...
    >>> config_from_upload(b'[1, 2, 3]')
    Traceback (most recent call last):
    simple_topic_modeling.errors.ConfigFileError: The file does not hold a JSON object. ...
    >>> config_from_upload(bytes.fromhex("fffe"))
    Traceback (most recent call last):
    simple_topic_modeling.errors.ConfigFileError: The file is not UTF-8 text. ...
    >>> config_from_upload(b'{"language": "klingon"}')
    Traceback (most recent call last):
    simple_topic_modeling.errors.ConfigFileError: The file holds a setting ...

    Raises:
        ConfigFileError: when the bytes are not a readable `config.json`.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigFileError("The file is not UTF-8 text.") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigFileError("The file is not valid JSON.") from error
    if not isinstance(data, dict):
        raise ConfigFileError("The file does not hold a JSON object.")
    try:
        return load_app_config(data)
    except ValueError as error:
        raise ConfigFileError("The file holds a setting that the app cannot use.") from error
