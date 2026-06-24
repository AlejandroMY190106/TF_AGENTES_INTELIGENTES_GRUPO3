"""
tc_pipeline/nlp/embeddings.py
─────────────────────────────
Módulo Lógico de Vectorización.

Responsabilidad Arquitectónica: 
Este módulo abstrae la generación de embeddings utilizando `SentenceTransformers`.
Está diseñado para ser importado por orquestadores (como `chroma_pipeline.py`) para servir
como función de embedding (EmbeddingFunction), así como proveer utilidades estandarizadas 
para el cálculo de métricas de calidad de los vectores (dimensiones, similitud, etc.).

NO realiza operaciones de Entrada/Salida (I/O) sobre bases de datos ni lee archivos CSV.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        logger.info("Cargando modelo de embeddings multilingüe: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return vectors.tolist()


def compute_embedding_quality(
    embeddings: list[list[float]],
    sample_size: int = 100,
) -> dict[str, Any]:
    if not embeddings:
        return {
            "count": 0,
            "dimension": 0,
            "zero_vectors": 0,
            "mean_norm": 0.0,
            "similarity_mean": 0.0,
            "similarity_min": 0.0,
            "similarity_max": 0.0,
        }

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Embeddings deben ser una matriz 2D")

    norms = np.linalg.norm(matrix, axis=1)
    zero_vectors = int(np.sum(norms == 0.0))
    
    if np.all(norms == 0.0):
        similarity_mean = similarity_min = similarity_max = 0.0
    else:
        matrix_normalized = matrix / np.maximum(norms[:, None], 1e-12)
        size = matrix_normalized.shape[0]
        if size == 1:
            similarity_mean = similarity_min = similarity_max = 1.0
        else:
            if size <= sample_size:
                similarities = cosine_similarity(matrix_normalized)
                mask = ~np.eye(size, dtype=bool)
                values = similarities[mask]
            else:
                rng = np.random.default_rng(0)
                pairs = rng.choice(size * size, size=sample_size * 2, replace=False)
                i = pairs // size
                j = pairs % size
                mask = i != j
                values = cosine_similarity(
                    matrix_normalized[i[mask]], matrix_normalized[j[mask]]
                ).diagonal()

            similarity_mean = float(np.mean(values)) if len(values) else 0.0
            similarity_min = float(np.min(values)) if len(values) else 0.0
            similarity_max = float(np.max(values)) if len(values) else 0.0

    return {
        "count": matrix.shape[0],
        "dimension": matrix.shape[1],  # Se adapta automáticamente a las 384 dimensiones del nuevo modelo
        "zero_vectors": zero_vectors,
        "mean_norm": float(np.mean(norms)),
        "similarity_mean": similarity_mean,
        "similarity_min": similarity_min,
        "similarity_max": similarity_max,
    }