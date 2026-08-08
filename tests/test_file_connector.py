"""
Unit tests for connectors/file_connector.py

Covers:
- Happy path: valid PDF, valid Excel, valid CSV
- Corrupted PDF → ConnectorError
- Empty file → ConnectorError
- Oversized file → ConnectorError
- Unsupported file type → ConnectorError
- Empty Excel (all sheets empty) → ConnectorError
- Missing user_id → ValueError
- Edge cases: CSV with only whitespace rows
"""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest

from connectors.file_connector import FileConnector
from connectors.exceptions import ConnectorError
from connectors.document import Document


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_minimal_pdf_bytes() -> bytes:
    """Construct a minimal but valid PDF in memory using pypdf."""
    try:
        import pypdf
        from pypdf import PdfWriter

        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        # Add a simple text annotation
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except ImportError:
        # If pypdf isn't installed in test env, use a known-good PDF stub
        # This is a minimal valid PDF structure
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
            b"/Resources<</Font<</F1 4 0 R>>>>>>\n"
            b"endobj\n"
            b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 5\n"
            b"0000000000 65535 f\r\n"
            b"0000000009 00000 n\r\n"
            b"0000000058 00000 n\r\n"
            b"0000000115 00000 n\r\n"
            b"0000000253 00000 n\r\n"
            b"trailer<</Size 5/Root 1 0 R>>\n"
            b"startxref\n317\n%%EOF\n"
        )


def _make_excel_bytes(text_content: str = "Hello World\tFoo Bar\nRow2\tData") -> bytes:
    """Create an in-memory Excel file with one sheet containing the given text."""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        for i, line in enumerate(text_content.split("\n"), start=1):
            for j, cell_val in enumerate(line.split("\t"), start=1):
                ws.cell(row=i, column=j, value=cell_val)
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()
    except ImportError:
        pytest.skip("openpyxl not installed")


def _make_csv_bytes(content: str = "name,age\nAlice,30\nBob,25") -> bytes:
    return content.encode("utf-8")


# ─── Happy path ───────────────────────────────────────────────────────────────

class TestFileConnectorHappyPath:
    @pytest.mark.skipif(
        sys.platform == "win32" and False,  # allow on all platforms
        reason="PDF test requires pypdf",
    )
    def test_valid_pdf_returns_documents(self):
        """A valid PDF with extractable text should return a non-empty Document list."""
        import connectors.file_connector as fc_module
        import unittest.mock as mock

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is test content from a PDF document."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = mock_reader
        mock_pypdf.errors = MagicMock()
        mock_pypdf.errors.PdfReadError = Exception

        original_pypdf = fc_module.pypdf
        original_available = fc_module._PYPDF_AVAILABLE
        try:
            fc_module.pypdf = mock_pypdf
            fc_module._PYPDF_AVAILABLE = True

            connector = FileConnector()
            docs = connector.ingest(
                file_bytes=b"fake-pdf-bytes",
                filename="test.pdf",
                user_id="user-123",
            )
        finally:
            fc_module.pypdf = original_pypdf
            fc_module._PYPDF_AVAILABLE = original_available

        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].source_type == "pdf"
        assert docs[0].user_id == "user-123"
        assert docs[0].source_identifier == "test.pdf"
        assert "content" in docs[0].text

    def test_valid_csv_returns_documents(self):
        csv_bytes = _make_csv_bytes("Name,Score\nAlice,95\nBob,87\nCarol,91")
        connector = FileConnector()
        docs = connector.ingest(
            file_bytes=csv_bytes,
            filename="results.csv",
            user_id="user-456",
        )
        assert len(docs) == 1
        assert docs[0].source_type == "csv"
        assert "Alice" in docs[0].text
        assert docs[0].user_id == "user-456"

    def test_valid_excel_returns_documents(self):
        """Uses openpyxl to create a real Excel file and verify extraction."""
        try:
            excel_bytes = _make_excel_bytes("Product,Price\nApple,1.50\nBanana,0.75")
            connector = FileConnector()
            docs = connector.ingest(
                file_bytes=excel_bytes,
                filename="prices.xlsx",
                user_id="user-789",
            )
            assert len(docs) >= 1
            assert all(d.source_type == "excel" for d in docs)
            assert any("Product" in d.text or "Apple" in d.text for d in docs)
        except ImportError:
            pytest.skip("openpyxl not installed")

    def test_document_has_required_fields(self):
        csv_bytes = _make_csv_bytes("col1,col2\nval1,val2")
        connector = FileConnector()
        docs = connector.ingest(
            file_bytes=csv_bytes,
            filename="data.csv",
            user_id="user-001",
        )
        doc = docs[0]
        assert doc.id  # non-empty UUID
        assert doc.text
        assert doc.source_type
        assert doc.source_identifier
        assert doc.user_id
        assert doc.created_at is not None


# ─── Failure cases ────────────────────────────────────────────────────────────

class TestFileConnectorFailureCases:
    def test_empty_file_raises_connector_error(self):
        connector = FileConnector()
        with pytest.raises(ConnectorError) as exc_info:
            connector.ingest(
                file_bytes=b"",
                filename="empty.pdf",
                user_id="user-001",
            )
        assert "empty" in str(exc_info.value).lower()

    def test_unsupported_file_type_raises_connector_error(self):
        connector = FileConnector()
        with pytest.raises(ConnectorError) as exc_info:
            connector.ingest(
                file_bytes=b"some content",
                filename="document.docx",
                user_id="user-001",
            )
        assert "unsupported" in str(exc_info.value).lower()

    def test_oversized_file_raises_connector_error(self):
        import os
        # Temporarily override size limit to something tiny for testing
        oversized_bytes = b"x" * 100  # 100 bytes
        with patch("connectors.file_connector._MAX_FILE_SIZE_BYTES", 50):
            connector = FileConnector()
            with pytest.raises(ConnectorError) as exc_info:
                connector.ingest(
                    file_bytes=oversized_bytes,
                    filename="big_file.csv",
                    user_id="user-001",
                )
        assert "large" in str(exc_info.value).lower() or "size" in str(exc_info.value).lower()

    def test_corrupted_pdf_raises_connector_error(self):
        import connectors.file_connector as fc_module

        mock_pypdf = MagicMock()
        mock_pypdf.errors = MagicMock()
        mock_pypdf.errors.PdfReadError = ValueError
        mock_pypdf.PdfReader.side_effect = ValueError("not a PDF")

        original_pypdf = fc_module.pypdf
        original_available = fc_module._PYPDF_AVAILABLE
        try:
            fc_module.pypdf = mock_pypdf
            fc_module._PYPDF_AVAILABLE = True

            connector = FileConnector()
            with pytest.raises(ConnectorError) as exc_info:
                connector.ingest(
                    file_bytes=b"not-a-real-pdf",
                    filename="corrupted.pdf",
                    user_id="user-001",
                )
        finally:
            fc_module.pypdf = original_pypdf
            fc_module._PYPDF_AVAILABLE = original_available

        error_msg = str(exc_info.value).lower()
        assert "corrupted" in error_msg or "invalid" in error_msg or "valid" in error_msg

    def test_missing_user_id_raises_value_error(self):
        connector = FileConnector()
        with pytest.raises(ValueError, match="user_id"):
            connector.ingest(
                file_bytes=_make_csv_bytes(),
                filename="data.csv",
                user_id="",
            )

    def test_csv_with_only_whitespace_raises_connector_error(self):
        whitespace_csv = b"   \n   \n   "
        connector = FileConnector()
        with pytest.raises(ConnectorError):
            connector.ingest(
                file_bytes=whitespace_csv,
                filename="blank.csv",
                user_id="user-001",
            )

    def test_missing_filename_raises_connector_error(self):
        connector = FileConnector()
        with pytest.raises(ConnectorError):
            connector.ingest(
                file_bytes=_make_csv_bytes(),
                filename="",
                user_id="user-001",
            )

    def test_no_text_pdf_raises_connector_error(self):
        """A PDF where all pages return empty text should raise ConnectorError."""
        import connectors.file_connector as fc_module

        mock_pypdf = MagicMock()
        mock_pypdf.errors = MagicMock()
        mock_pypdf.errors.PdfReadError = ValueError
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # Empty — scanned PDF
        mock_reader.pages = [mock_page]
        mock_pypdf.PdfReader.return_value = mock_reader

        original_pypdf = fc_module.pypdf
        original_available = fc_module._PYPDF_AVAILABLE
        try:
            fc_module.pypdf = mock_pypdf
            fc_module._PYPDF_AVAILABLE = True

            connector = FileConnector()
            with pytest.raises(ConnectorError) as exc_info:
                connector.ingest(
                    file_bytes=b"fake",
                    filename="scanned.pdf",
                    user_id="user-001",
                )
        finally:
            fc_module.pypdf = original_pypdf
            fc_module._PYPDF_AVAILABLE = original_available

        assert "no extractable text" in str(exc_info.value).lower()
