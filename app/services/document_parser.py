from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt"}


class DocumentParsingError(ValueError):
    pass


def validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        raise DocumentParsingError(
            f"Unsupported file type. Allowed types: {allowed}."
        )
    return extension


def parse_file(file_path: Path) -> str:
    extension = validate_extension(file_path.name)

    if extension == ".txt":
        return _read_txt(file_path)
    if extension == ".csv":
        return _read_csv(file_path)
    if extension == ".pdf":
        return _read_pdf(file_path)

    raise DocumentParsingError("Unsupported file type.")


def _read_txt(file_path: Path) -> str:
    encodings = ["utf-8", "utf-16", "latin-1"]
    for encoding in encodings:
        try:
            text = file_path.read_text(encoding=encoding)
            return _ensure_non_empty_text(text)
        except UnicodeDecodeError:
            continue
    raise DocumentParsingError("Could not decode text file with supported encodings.")


def _read_csv(file_path: Path) -> str:
    rows: list[str] = []
    encodings = ["utf-8", "latin-1"]

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding, newline="") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    rows.append(" | ".join(cell.strip() for cell in row))
            return _ensure_non_empty_text("\n".join(rows))
        except UnicodeDecodeError:
            rows = []
            continue

    raise DocumentParsingError("Could not decode CSV file with supported encodings.")


def _read_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # pragma: no cover - depends on parser internals
        raise DocumentParsingError(f"Could not read PDF: {exc}") from exc

    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)

    return _ensure_non_empty_text("\n\n".join(parts))


def _ensure_non_empty_text(text: str) -> str:
    clean = text.strip()
    if not clean:
        raise DocumentParsingError("No readable text found in the uploaded document.")
    return clean
