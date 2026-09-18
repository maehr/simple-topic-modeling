"""Friendly errors and notices.

`SPECS.md` section 8 requires a clear message and a recovery action for each failure.
Every message in this module carries both. `app.py` renders them with one function.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DecodeError",
    "EmptyVocabularyError",
    "FriendlyMessage",
    "NoUsableTextError",
    "TooFewDocumentsError",
    "TooManyTopicsError",
    "TopicError",
    "UnsupportedFileError",
    "browser_failure",
    "date_parsing_failed",
    "model_did_not_converge",
]

MIN_DOCUMENTS = 3
"""Smallest corpus that a topic model can use."""


@dataclass(frozen=True, slots=True)
class FriendlyMessage:
    """A message a person can act on.

    >>> message = FriendlyMessage("No terms remain.", "Lower the minimum document frequency.")
    >>> message.detail
    'No terms remain.'
    >>> message.recovery
    'Lower the minimum document frequency.'
    """

    detail: str
    recovery: str

    def __str__(self) -> str:
        """Join the detail and the recovery action into one line.

        >>> str(FriendlyMessage("No terms remain.", "Remove some stop words."))
        'No terms remain. Remove some stop words.'
        """
        return f"{self.detail} {self.recovery}"


class TopicError(Exception):
    """Base class for every failure that the app reports to a person.

    >>> error = TopicError(FriendlyMessage("It broke.", "Try again."))
    >>> error.friendly.recovery
    'Try again.'
    >>> str(error)
    'It broke. Try again.'
    """

    def __init__(self, friendly: FriendlyMessage) -> None:
        super().__init__(str(friendly))
        self.friendly = friendly


class UnsupportedFileError(TopicError):
    """The file is binary, or the app cannot read its format.

    >>> UnsupportedFileError("report.pdf").friendly.detail
    'The file "report.pdf" is not readable text.'
    """

    def __init__(self, filename: str) -> None:
        super().__init__(
            FriendlyMessage(
                f'The file "{filename}" is not readable text.',
                "Upload CSV, TSV, JSON, JSONL, or a plain-text file such as TXT or Markdown.",
            )
        )


class DecodeError(TopicError):
    """The bytes are not valid UTF-8.

    >>> DecodeError("notes.txt").friendly.detail
    'The app cannot read "notes.txt" as UTF-8 text.'
    """

    def __init__(self, filename: str) -> None:
        super().__init__(
            FriendlyMessage(
                f'The app cannot read "{filename}" as UTF-8 text.',
                "Save the file again with UTF-8 encoding, then upload it.",
            )
        )


class NoUsableTextError(TopicError):
    """Every document is empty after cleaning.

    >>> NoUsableTextError().friendly.detail
    'No usable text remains.'
    """

    def __init__(self) -> None:
        super().__init__(
            FriendlyMessage(
                "No usable text remains.",
                "Choose a different text column, or turn off some cleaning options.",
            )
        )


class TooFewDocumentsError(TopicError):
    """The corpus holds fewer documents than a model needs.

    >>> TooFewDocumentsError(2).friendly.recovery
    'Add more documents. A topic model needs at least 3.'
    """

    def __init__(self, count: int) -> None:
        super().__init__(
            FriendlyMessage(
                f"The corpus holds {count} documents.",
                f"Add more documents. A topic model needs at least {MIN_DOCUMENTS}.",
            )
        )


class EmptyVocabularyError(TopicError):
    """Filtering removed every term.

    >>> EmptyVocabularyError().friendly.detail
    'No terms remain after filtering.'
    """

    def __init__(self) -> None:
        super().__init__(
            FriendlyMessage(
                "No terms remain after filtering.",
                "Lower the minimum document frequency, raise the maximum document frequency, "
                "or remove some stop words.",
            )
        )


class TooManyTopicsError(TopicError):
    """The topic count exceeds what the data supports.

    >>> TooManyTopicsError(30, 12).friendly.detail
    'You asked for 30 topics, but the data supports at most 12.'
    """

    def __init__(self, requested: int, limit: int) -> None:
        super().__init__(
            FriendlyMessage(
                f"You asked for {requested} topics, but the data supports at most {limit}.",
                "Lower the number of topics, or add more documents.",
            )
        )


def model_did_not_converge(max_iter: int) -> FriendlyMessage:
    """Report that the model stopped before it converged.

    The app shows this message. It does not stop the run.

    >>> model_did_not_converge(200).recovery
    'Raise the iteration limit above 200 in the advanced settings.'
    """
    return FriendlyMessage(
        "The model stopped before it converged.",
        f"Raise the iteration limit above {max_iter} in the advanced settings.",
    )


def date_parsing_failed(unparsed: int, total: int) -> FriendlyMessage:
    """Report how many dates the app could not read.

    >>> date_parsing_failed(4, 10).detail
    'The app could not read 4 of 10 dates.'
    """
    return FriendlyMessage(
        f"The app could not read {unparsed} of {total} dates.",
        "Choose a different date column, or use the ISO format YYYY-MM-DD.",
    )


def browser_failure(step: str) -> FriendlyMessage:
    """Report that the browser ran out of memory or computation time.

    >>> browser_failure("the model fit").detail
    'The browser could not finish the model fit.'
    """
    return FriendlyMessage(
        f"The browser could not finish {step}.",
        "Use fewer documents, lower the maximum vocabulary, or ask for fewer topics.",
    )
