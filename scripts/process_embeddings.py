"""
scripts/process_embeddings.py
─────────────────────────────
[SCRIPT DE DEPURACIÓN / OBSOLETO PARA EL FLUJO PRINCIPAL]

Responsabilidad Arquitectónica:
Este script genera archivos intermedios JSONL con chunks y embeddings en crudo.
Es útil exclusivamente como herramienta de depuración para inspeccionar vectores.
NO es requerido ni utilizado por el flujo principal de indexación en ChromaDB (`chroma_pipeline.py`),
el cual extrae, procesa e indexa directamente desde los archivos CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import glob
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tc_pipeline.nlp.embeddings import EmbeddingModel, compute_embedding_quality
from tc_pipeline.nlp.processing import build_chunks_for_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera chunks y embeddings para expedientes del TC a partir de CSVs limpios."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/merged",
        help="Directorio que contiene los CSVs limpios generados por clean_and_merge.py",
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
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Nombre del modelo de SentenceTransformers (Multilingüe por defecto).",
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

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"El directorio de entrada no existe: {input_dir}")

    # Buscar todos los CSVs en el directorio merged
    csv_files = glob.glob(str(input_dir / "*.csv"))
    if not csv_files:
        logger.warning("No se encontraron archivos CSV en el directorio %s", input_dir)
        return

    logger.info("Se encontraron %d archivos CSV para procesar.", len(csv_files))
    
    chunks: list[dict[str, Any]] = []

    # Iterar y cargar cada uno de los CSVs generados
    for file_path in csv_files:
        logger.info("Procesando archivo: %s", Path(file_path).name)
        try:
            # Forzamos que lea todo como strings para evitar pérdidas de ceros en números de expediente
            df = pd.read_csv(file_path, dtype=str)
            # Rellenar nulos en fundamentos para evitar que falle el chunker
            df["fundamentos"] = df["fundamentos"].fillna("")
            
            for _, record in df.to_dict(orient="records"):
                if record["fundamentos"].strip():  # Solo procesar si tiene texto observable
                    chunks.extend(build_chunks_for_record(record, chunk_size=args.chunk_size, overlap=args.overlap))
        except Exception as e:
            logger.error("Error al leer el archivo %s: %s", file_path, str(e))

    if not chunks:
        logger.warning("No se generaron chunks válidos del corpus de datos.")
        return

    # Guardar chunks en JSONL
    output_chunks_path = Path(args.output_chunks)
    output_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with output_chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Generados %d chunks y guardados en %s", len(chunks), output_chunks_path)

    # Inicializar modelo y generar embeddings vectoriales
    model = EmbeddingModel(model_name=args.model)
    texts = [chunk["text"] for chunk in chunks]
    
    logger.info("Calculando vectores de embedding... (Esto puede tomar unos minutos)")
    embeddings = model.embed_texts(texts)

    # Guardar embeddings combinados en JSONL
    output_embeddings_path = Path(args.output_embeddings)
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with output_embeddings_path.open("w", encoding="utf-8") as file:
        for chunk, embedding in zip(chunks, embeddings):
            record = {**chunk, "embedding": embedding}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Guardar Métricas de calidad
    metrics = compute_embedding_quality(embeddings)
    metrics_path = output_embeddings_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Embeddings generados exitosamente: %d. Métricas guardadas en %s", len(embeddings), metrics_path)


if __name__ == "__main__":
    main()