import json
from pathlib import Path

import pytest

from tc_pipeline.nlp.processing import (
    build_chunks_for_record,
    chunk_text,
    classify_participante,
    extract_fecha_ingreso,
    extract_sala_origen,
    extract_text_for_chunking,
)


def test_extract_text_for_chunking_prefers_fundamentos():
    record = {
        "fundamentos": [
            "Texto principal de fundamentos.",
        ],
        "attachment": {"content": "Contenido PDF alternativo."},
    }

    assert extract_text_for_chunking(record) == "Texto principal de fundamentos."


def test_extract_text_for_chunking_uses_attachment_if_no_fundamentos():
    record = {
        "attachment": {"content": "Texto extraído de PDF."},
    }

    assert extract_text_for_chunking(record) == "Texto extraído de PDF."


def test_chunk_text_creates_overlapping_chunks():
    text = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince"
    chunks = chunk_text(text, chunk_size=5, overlap=2)

    assert len(chunks) == 5
    assert chunks[0] == "uno dos tres cuatro cinco"
    assert chunks[1] == "cuatro cinco seis siete ocho"
    assert chunks[2] == "siete ocho nueve diez once"
    assert chunks[3] == "diez once doce trece catorce"
    assert chunks[4] == "trece catorce quince"


def test_classify_participante_persona_natural():
    assert classify_participante("Juan Pérez") == "Persona Natural"


def test_classify_participante_entidad_publica():
    assert classify_participante("Municipalidad Provincial de Urubamba") == "Entidad Pública/Privada"


def test_extract_fecha_ingreso_from_text():
    text = "El demandante interpone la demanda con fecha 15 de enero de 2023 en la sede..."
    assert extract_fecha_ingreso(text) == "15 de enero de 2023"


def test_extract_sala_origen_from_text():
    text = "La resolución fue expedida por la Sala Civil de Lima en fecha..."
    assert extract_sala_origen(text) == "Sala Civil de Lima"


def test_build_chunks_for_record_includes_metadata():
    record = {
        "numero_expediente": "00001-2025-AA/TC",
        "fundamentos": ["Primera parte. Segunda parte. Tercera parte."],
        "sentencia_sala": "Sala 1",
        "sentencia_sentido": "Fundada",
        "CDES_TIPOPROCESO": "Acción de Amparo",
        "MATERIA": "Derecho Constitucional",
        "SUB_MATERIA": "Derechos Fundamentales",
        "ESPECIFICA": "Habeas Corpus",
        "nombre_demandante": "Juan Pérez",
        "nombre_demandado": "Ministerio de Justicia",
    }

    chunks = build_chunks_for_record(record, chunk_size=5, overlap=1)
    assert len(chunks) >= 1
    assert chunks[0]["numero_expediente"] == "00001-2025-AA/TC"
    assert chunks[0]["metadata"]["sentencia_sala"] == "Sala 1"
    assert chunks[0]["metadata"]["tipo_proceso"] == "Acción de Amparo"
    assert chunks[0]["metadata"]["TIPO_DEMANDANTE"] == "Persona Natural"
    assert chunks[0]["metadata"]["TIPO_DEMANDADO"] == "Entidad Pública/Privada"
