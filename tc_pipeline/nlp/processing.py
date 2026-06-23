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
    candidates = []
    for key in ("fundamentos", "FUNDAMENTOS"):
        if key in record:
            candidate = _serialize_text_content(record[key])
            if candidate:
                candidates.append(candidate)

    if candidates:
        return clean_text(candidates[0])

    attachment = record.get("attachment") or record.get("attachment", {})
    attachment_text = _serialize_text_content(attachment)
    if attachment_text:
        return clean_text(attachment_text)

    return ""


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


def extract_fecha_ingreso(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"con fecha\s+(.+?)(?:\s+en\b|\s+con\b|\s+para\b|\s+por\b|[.,;:\n]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))

    return ""


def extract_sala_origen(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"expedida por la\s+([^.,;:\n]+?)(?:\s+en\b|\s+por\b|[.,;:\n]|$)",
        r"expedida por el\s+([^.,;:\n]+?)(?:\s+en\b|\s+por\b|[.,;:\n]|$)",
        r"emitida por la\s+([^.,;:\n]+?)(?:\s+en\b|\s+por\b|[.,;:\n]|$)",
        r"emitida por el\s+([^.,;:\n]+?)(?:\s+en\b|\s+por\b|[.,;:\n]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))

    return ""


def classify_participante(name: Any) -> str:
    name_text = _serialize_text_content(name)
    if not name_text:
        return ""

    normalized = name_text.strip()
    if not normalized:
        return ""

    entidad_keywords = (
        r"\b(Gobierno|Gobierno Regional|Municipalidad|Ministerio|Dirección|Hospital|Instituto|Empresa|Sociedad|Fundaci[oó]n|Banco|Universidad|Agencia|Consejo|Comisi[oó]n|Corporaci[oó]n|Compa[nñ][ií]a|Distrito|Regional|Estatal|Nacional|Provincial|Entidad|Servicio|Superintendencia|Corte|Juzgado|Fiscal[ií]a|Ministerio|Comando|Caja|EIRL|S\.A\.|S\.R\.L\.|S\.A\.C\.|A\.C\.|S\.C\.|COOP|E\.P\.S\.)\b"
    )

    if re.search(entidad_keywords, normalized, re.IGNORECASE):
        return "Entidad Pública/Privada"

    if re.search(r"\b(S\.A\.|S\.R\.L\.|S\.A\.C\.|EIRL|C\.A\.|COOP|A\.C\.)\b", normalized, re.IGNORECASE):
        return "Entidad Pública/Privada"

    words = [w for w in re.split(r"\s+", normalized) if w]
    if len(words) <= 4 and all(
        re.match(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+$", w)
        for w in words
    ):
        return "Persona Natural"

    return "Entidad Pública/Privada"


def extract_secondary_fields(record: dict[str, Any]) -> dict[str, str]:
    text = extract_text_for_chunking(record)
    fields = {
        "FEC_INGRESO": extract_fecha_ingreso(text),
        "SALA_ORIGEN": extract_sala_origen(text),
    }

    demandante = record.get("nombre_demandante") or record.get("DEMANDANTE")
    demandado = record.get("nombre_demandado") or record.get("DEMANDADO")
    fields["TIPO_DEMANDANTE"] = classify_participante(demandante)
    fields["TIPO_DEMANDADO"] = classify_participante(demandado)

    return fields


def build_chunks_for_record(
    record: dict[str, Any],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    text = extract_text_for_chunking(record)
    if not text:
        return []

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    metadata = extract_secondary_fields(record)

    base_id = (
        record.get("numero_expediente")
        or record.get("NEXPEDIENTE")
        or str(record.get("id", ""))
    )
    base_id = str(base_id).strip()

    chunked_records = []
    for index, chunk in enumerate(chunks, start=1):
        item = {
            "chunk_id": f"{base_id}_{index}" if base_id else f"chunk_{index}",
            "numero_expediente": base_id,
            "chunk_index": index,
            "text": chunk,
            "metadata": {
                "sentencia_sala": record.get("sentencia_sala") or record.get("SALA"),
                "sentencia_sentido": record.get("sentencia_sentido") or record.get("FALLO"),
                "tipo_proceso": record.get("CDES_TIPOPROCESO"),
                "materia": record.get("MATERIA"),
                "sub_materia": record.get("SUB_MATERIA"),
                "especifica": record.get("ESPECIFICA"),
                **metadata,
            },
        }
        chunked_records.append(item)

    return chunked_records
