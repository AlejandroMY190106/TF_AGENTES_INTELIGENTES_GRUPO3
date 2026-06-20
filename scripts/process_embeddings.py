"""
scripts/process_embeddings.py
─────────────────────────────
Pipeline de chunking y embeddings para el corpus de expedientes.

Este script extrae texto de los campos `fundamentos` o `attachment`, realiza
chunking con solapamiento, calcula embeddings usando SentenceTransformers,
 y genera métricas de calidad de embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tc_pipeline.nlp.embeddings import EmbeddingModel, compute_embedding_quality
from tc_pipeline.nlp.processing import build_chunks_for_record
from tc_pipeline.storage.parquet_store import ParquetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera chunks y embeddings para expedientes del TC."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/expedientes_tc.parquet",
        help="Ruta al Parquet de expedientes.",
    )
    parser.add_argument(
        "--output-chunks",
        type=str,
        default="data/raw/chunks.jsonl",
        help="Ruta de salida para los chunks generados.",
    )
    parser.add_argument(
        "--output-embeddings",
        type=str,
        default="data/raw/embeddings.jsonl",
        help="Ruta de salida para los embeddings generados.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Nombre del modelo de SentenceTransformers.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Número de tokens por chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Número de tokens superpuestos entre chunks.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Número de registros a procesar como muestra (0 = todos).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_path}")

    logger.info("Cargando dataset de expedientes desde %s", input_path)
    df = pd.read_parquet(input_path, engine="pyarrow")

    if args.sample > 0:
        df = df.head(args.sample)

    chunks: list[dict[str, Any]] = []
    for _, record in df.to_dict(orient="records"):
        chunks.extend(build_chunks_for_record(record, chunk_size=args.chunk_size, overlap=args.overlap))

    if not chunks:
        logger.warning("No se generaron chunks para el dataset especificado.")
        return

    output_chunks_path = Path(args.output_chunks)
    output_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with output_chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Generados %d chunks y guardados en %s", len(chunks), output_chunks_path)

    model = EmbeddingModel(model_name=args.model)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.embed_texts(texts)

    output_embeddings_path = Path(args.output_embeddings)
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with output_embeddings_path.open("w", encoding="utf-8") as file:
        for chunk, embedding in zip(chunks, embeddings):
            record = {**chunk, "embedding": embedding}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = compute_embedding_quality(embeddings)
    metrics_path = output_embeddings_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Embeddings generados: %d. Métricas guardadas en %s", len(embeddings), metrics_path)


if __name__ == "__main__":
    main()
