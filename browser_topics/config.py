"""Configuration models.

These models cross the `config.json` boundary, so they use Pydantic, as `AGENTS.md` section 4
requires. The defaults come from `SPECS.md` sections 3 and 4.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "LANGUAGE_ALIASES",
    "LANGUAGE_LABELS",
    "AppConfig",
    "Language",
    "ModelConfig",
    "ModelType",
    "NgramChoice",
    "PreprocessConfig",
    "StopWordConfig",
]

Language = Literal["en", "de", "fr", "it", "es"]
ModelType = Literal["nmf", "lda"]
NgramChoice = Literal["1", "1-2", "2"]

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

    >>> AppConfig().language
    'en'
    >>> AppConfig(language="sp").language
    'es'
    >>> AppConfig().summary(2418)
    '2,418 documents · English · NMF · 10 topics · max 5,000 terms'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_version: str = "0.1.0"
    language: Language = "en"
    preprocess: PreprocessConfig = PreprocessConfig()
    stop_words: StopWordConfig = StopWordConfig()
    model: ModelConfig = ModelConfig()

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
        """
        return (
            f"{document_count:,} documents"
            f" · {LANGUAGE_LABELS[self.language]}"
            f" · {self.model.model_type.upper()}"
            f" · {self.model.n_topics} topics"
            f" · max {self.model.max_features:,} terms"
        )
