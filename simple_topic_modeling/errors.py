"""Friendly errors and notices.

`SPECS.md` section 8 requires a clear message and a recovery action for each failure.
Every message in this module carries both. `app.py` renders them with one function.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ConfigFileError",
    "DecodeError",
    "EmptyVocabularyError",
    "FriendlyMessage",
    "NoUsableTextError",
    "PdfProblem",
    "PdfTextError",
    "TooFewDocumentsError",
    "TooManyTopicsError",
    "TopicError",
    "UnsupportedFileError",
    "browser_failure",
    "date_parsing_failed",
    "model_did_not_converge",
    "pdf_pages_without_text",
]

MIN_DOCUMENTS = 3
"""Smallest corpus that a topic model can use."""

PdfProblem = Literal["no_text", "encrypted", "damaged"]
"""Why the app cannot take the text of a PDF."""

LISTED_PAGES = 10
"""A notice names this many pages at most."""


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

    >>> UnsupportedFileError("report.docx").friendly.detail
    'The file "report.docx" is not readable text.'
    """

    def __init__(self, filename: str) -> None:
        super().__init__(
            FriendlyMessage(
                f'The file "{filename}" is not readable text.',
                "Upload CSV, TSV, JSON, JSONL, a text-based PDF, or a plain-text file such as TXT"
                " or Markdown.",
            )
        )


class PdfTextError(TopicError):
    """The app cannot take the text of a PDF.

    A scanned PDF holds images of text, not text. The app does no OCR, so the message names the
    way out. The app does not support a password either, as issue #27 decided.

    >>> PdfTextError("scan.pdf", "no_text").friendly.detail
    'The PDF "scan.pdf" does not contain enough extractable text.'
    >>> PdfTextError("scan.pdf", "no_text").friendly.recovery
    'OCR is not currently supported. Convert it to a searchable PDF or text file and try again.'
    >>> PdfTextError("secret.pdf", "encrypted").friendly.detail
    'The PDF "secret.pdf" is protected by a password.'
    >>> PdfTextError("broken.pdf", "damaged").problem
    'damaged'
    """

    def __init__(self, filename: str, problem: PdfProblem) -> None:
        messages: dict[PdfProblem, FriendlyMessage] = {
            "no_text": FriendlyMessage(
                f'The PDF "{filename}" does not contain enough extractable text.',
                "OCR is not currently supported. Convert it to a searchable PDF or text file and"
                " try again.",
            ),
            "encrypted": FriendlyMessage(
                f'The PDF "{filename}" is protected by a password.',
                "The app does not open a protected PDF. Remove the password, or save the text as"
                " a TXT file, and try again.",
            ),
            "damaged": FriendlyMessage(
                f'The app cannot read the PDF "{filename}".',
                "Save the PDF again, or save its text as a TXT file, and try again.",
            ),
        }
        super().__init__(messages[problem])
        self.problem = problem


class ConfigFileError(TopicError):
    """The uploaded settings file is not usable.

    >>> ConfigFileError("The file is not valid JSON.").friendly.detail
    'The file is not valid JSON.'
    >>> ConfigFileError("The file is not valid JSON.").friendly.recovery
    'Upload the config.json that this app wrote in Step 4.'
    """

    def __init__(self, detail: str) -> None:
        super().__init__(
            FriendlyMessage(detail, "Upload the config.json that this app wrote in Step 4.")
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


def pdf_pages_without_text(filename: str, pages: Sequence[int], page_count: int) -> FriendlyMessage:
    """Report the pages of a PDF that hold no text.

    The app models the other pages. A long list names the first pages only.

    >>> pdf_pages_without_text("book.pdf", [3], 12).detail
    'The app found no text on 1 of 12 pages of "book.pdf": page 3.'
    >>> pdf_pages_without_text("book.pdf", range(1, 15), 20).detail
    'The app found no text on 14 of 20 pages of "book.pdf":
     pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and 4 more.'
    """
    shown = ", ".join(str(page) for page in list(pages)[:LISTED_PAGES])
    extra = len(pages) - LISTED_PAGES
    listed = f"{shown}, and {extra} more" if extra > 0 else shown
    noun = "page" if len(pages) == 1 else "pages"
    return FriendlyMessage(
        f'The app found no text on {len(pages)} of {page_count} pages of "{filename}":'
        f" {noun} {listed}.",
        "A scanned page needs OCR, which the app does not support. The app models the other pages.",
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
