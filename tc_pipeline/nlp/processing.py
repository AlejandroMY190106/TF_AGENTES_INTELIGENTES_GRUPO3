from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _serialize_text_content(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return "\n".join(_serialize_text_content(item) for item in value if _serialize_text_content(item))

    if isinstance(value, dict):
        if "content" in value:
            return _serialize_text_content(value["content"])
        return "\n".join(
            _serialize_text_content(v) for v in value.values() if _serialize_text_content(v)
        )

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""

        if (text.startswith("{") or text.startswith("[")) and "\"" in text:
            try:
                parsed = json.loads(text)
                return _serialize_text_content(parsed)
            except json.JSONDecodeError:
                pass

        return text

    return str(value)


def clean_text(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_for_chunking(record: dict[str, Any]) -> str:
    """Extrae y concatena los motivos de demanda y fundamentos del registro."""
    motivos = _serialize_text_content(record.get("motivos_demanda"))
    fundamentos = _serialize_text_content(record.get("fundamentos"))
    
    parts = []
    if motivos:
        parts.append(motivos)
    if fundamentos:
        parts.append(fundamentos)
        
    combined = "\n".join(parts)
    return clean_text(combined)


def _split_words(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", text) if token]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    if not text:
        return []

    tokens = _split_words(text)
    if len(tokens) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end]).strip()
        chunks.append(chunk)

        if end == len(tokens):
            break

        start += max(1, chunk_size - overlap)

    return chunks


def build_chunks_for_record(
    record: dict[str, Any],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    text = extract_text_for_chunking(record)
    if not text:
        return []

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    base_id = record.get("numero_expediente")
    base_id = str(base_id).strip() if base_id is not None else ""

    chunked_records = []
    for index, chunk in enumerate(chunks, start=1):
        item = {
            "chunk_id": f"{base_id}_{index}" if base_id else f"chunk_{index}",
            "numero_expediente": base_id,
            "chunk_index": index,
            "text": chunk,
            "metadata": {
                "tipo_expediente": record.get("tipo_expediente"),
                "sentido_resolucion": record.get("sentido_resolucion"),
                "url_archivo_TC": record.get("url_archivo_TC"),
                "url_archivo_original": record.get("url_archivo_original"),
            },
        }
        chunked_records.append(item)

    return chunked_records
