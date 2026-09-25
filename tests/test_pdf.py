import io

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import DependencyError

from simple_topic_modeling.errors import PdfTextError, TopicError
from simple_topic_modeling.io import (
    MIN_PDF_CHARACTERS,
    UploadedFile,
    _example_pdf,
    build_corpus,
    decode_text,
    detect_kind,
    extract_pdf_text,
    read_text_document,
    split_long_document,
)

PAGE_ONE = "The harbour was full of ships.\nThe captain watched the storm over the coast."
PAGE_TWO = "The farmer brought in the wheat.\nThe harvest filled the barn before the rain."
PAGE_THREE = "The judge read the verdict.\nThe jury had heard every witness in the trial."


def encrypted(data: bytes, user: str, owner: str | None = None) -> bytes:
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(data)).pages:
        writer.add_page(page)
    writer.encrypt(user_password=user, owner_password=owner)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def book_pdf():
    return UploadedFile("book.pdf", _example_pdf([PAGE_ONE, PAGE_TWO, PAGE_THREE]))


def test_a_pdf_uses_the_text_reader():
    assert detect_kind("book.pdf") == "text"


def test_a_text_pdf_keeps_every_page_in_order(book_pdf):
    extraction = extract_pdf_text(book_pdf)
    assert extraction.page_count == 3
    assert extraction.pages_without_text == ()
    assert extraction.notice is None
    assert extraction.text.split("\n\n") == [PAGE_ONE, PAGE_TWO, PAGE_THREE]


def test_a_mixed_pdf_skips_and_names_the_empty_pages():
    file = UploadedFile("mixed.pdf", _example_pdf([PAGE_ONE, "", PAGE_TWO, ""]))
    extraction = extract_pdf_text(file)
    assert extraction.pages_without_text == (2, 4)
    assert extraction.text.split("\n\n") == [PAGE_ONE, PAGE_TWO]
    assert extraction.notice is not None
    assert "2 of 4 pages" in extraction.notice.detail
    assert "pages 2, 4" in extraction.notice.detail


def test_a_scanned_pdf_asks_for_ocr():
    with pytest.raises(PdfTextError) as caught:
        extract_pdf_text(UploadedFile("scan.pdf", _example_pdf(["", "", ""])))
    assert caught.value.problem == "no_text"
    assert "OCR is not currently supported" in caught.value.friendly.recovery


def test_a_pdf_with_almost_no_text_counts_as_scanned():
    short = "x" * (MIN_PDF_CHARACTERS - 1)
    with pytest.raises(PdfTextError) as caught:
        extract_pdf_text(UploadedFile("stamp.pdf", _example_pdf([short])))
    assert caught.value.problem == "no_text"


def test_a_pdf_with_a_user_password_is_refused(book_pdf):
    file = UploadedFile("secret.pdf", encrypted(book_pdf.data, user="secret"))
    with pytest.raises(PdfTextError) as caught:
        extract_pdf_text(file)
    assert caught.value.problem == "encrypted"


def test_a_pdf_with_an_owner_password_only_still_opens(book_pdf):
    file = UploadedFile("locked.pdf", encrypted(book_pdf.data, user="", owner="owner"))
    assert extract_pdf_text(file).page_count == 3


def test_a_pdf_that_needs_a_crypto_package_is_refused(book_pdf, monkeypatch):
    def refuse(_self, _password):
        raise DependencyError("cryptography is required for AES")

    file = UploadedFile("aes.pdf", encrypted(book_pdf.data, user="", owner="owner"))
    monkeypatch.setattr(PdfReader, "decrypt", refuse)
    with pytest.raises(PdfTextError) as caught:
        extract_pdf_text(file)
    assert caught.value.problem == "encrypted"


def test_a_damaged_pdf_gets_a_friendly_message():
    with pytest.raises(PdfTextError) as caught:
        extract_pdf_text(UploadedFile("broken.pdf", b"%PDF-1.4\nthis is not a real pdf"))
    assert caught.value.problem == "damaged"
    assert isinstance(caught.value, TopicError)


def test_the_router_sends_a_pdf_to_the_extractor(book_pdf):
    text, notice = read_text_document(book_pdf)
    assert text.startswith(PAGE_ONE)
    assert notice is None


def test_the_router_keeps_the_text_path_for_other_files():
    text, notice = read_text_document(UploadedFile("notes.txt", b"a\n\nb"))
    assert text == "a\n\nb"
    assert notice is None


def test_decode_text_no_longer_rejects_the_pdf_extension():
    assert decode_text(UploadedFile("plain.pdf", b"just text")) == "just text"


def test_a_pdf_enters_the_long_document_path(book_pdf):
    text, _ = read_text_document(book_pdf)
    corpus, stats = build_corpus(*split_long_document(text, book_pdf.name))
    assert stats.kept == 3
    assert corpus.document_ids == ["book.pdf#1", "book.pdf#2", "book.pdf#3"]
    assert corpus.metadata["parent_document_id"].unique().tolist() == ["book.pdf"]
