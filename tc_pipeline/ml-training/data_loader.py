"""
tc_pipeline/ml-training/data_loader.py
───────────────────────────────────────
Módulo de Extracción y Consolidación de Datos Vectoriales.

Responsabilidad Arquitectónica:
    Conecta al cliente persistente de ChromaDB, recupera la totalidad de los
    embeddings y metadatas de la colección de jurisprudencia, y aplica una
    estrategia de *Mean Pooling* para consolidar los múltiples chunks que
    comparten el mismo ``numero_expediente`` en una única representación
    semántica densa por caso judicial.

    Este módulo NO conoce cómo se entrena el clasificador ni cómo se evalúa.
    Su única responsabilidad es transformar los datos crudos del vector store
    en matrices NumPy limpias y reproducibles.

Uso:
    from tc_pipeline.ml_training.data_loader import load_dataset
    from tc_pipeline.config import MLConfig

    cfg = MLConfig()
    X, y = load_dataset(cfg)
"""

from __future__ import annotations

import logging
import sys
import os
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import chromadb

# ─── Resolución dinámica de la raíz del proyecto ─────────────────────────────
# Garantiza que los imports de tc_pipeline funcionen tanto cuando se ejecuta
# desde el directorio del módulo como desde la raíz del repositorio.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tc_pipeline.config import MLConfig  # noqa: E402

# ─── Logger del módulo ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Tipo de retorno explícito ────────────────────────────────────────────────

class DatasetResult(NamedTuple):
    """Par de matrices NumPy resultado del proceso de carga y consolidación.

    Attributes:
        X: Matriz de características ``(n_expedientes, embedding_dim)``.
            Cada fila es el promedio vectorial de todos los chunks del expediente.
        y: Vector de etiquetas ``(n_expedientes,)`` con el ``sentido_resolucion``
            correspondiente a cada fila de ``X``.
    """
    X: np.ndarray
    y: np.ndarray


# ─── Función principal ────────────────────────────────────────────────────────

def load_dataset(cfg: MLConfig | None = None) -> DatasetResult:
    """Extrae embeddings de ChromaDB y retorna matrices consolidadas por documento.

    El proceso ejecuta tres pasos secuenciales:

    1. **Conexión**: Inicializa el ``PersistentClient`` de ChromaDB apuntando a
       ``cfg.chroma_db_path``.
    2. **Extracción**: Descarga la colección completa —embeddings, documentos y
       metadatas— en una sola llamada paginada.
    3. **Mean Pooling**: Agrupa los vectores por ``numero_expediente`` y calcula
       el centroide aritmético de cada grupo, eliminando la redundancia de chunks
       y garantizando que el clasificador reciba exactamente una muestra por caso
       judicial.

    Args:
        cfg: Instancia de :class:`~tc_pipeline.config.MLConfig`. Si es ``None``,
             se utiliza la configuración por defecto.

    Returns:
        :class:`DatasetResult` con los arrays ``X`` e ``y`` listos para
        ``train_test_split``.

    Raises:
        RuntimeError: Si no se puede conectar a ChromaDB o la colección está vacía.
        KeyError: Si algún metadato carece de las claves requeridas por la config.
    """
    if cfg is None:
        cfg = MLConfig()

    # ── 1. Conexión a ChromaDB ────────────────────────────────────────────────
    logger.info(
        "Conectando al cliente persistente de ChromaDB en '%s'...",
        cfg.chroma_db_path,
    )
    try:
        client = chromadb.PersistentClient(path=cfg.chroma_db_path)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo inicializar el cliente ChromaDB en '{cfg.chroma_db_path}': {exc}"
        ) from exc

    try:
        collection = client.get_collection(name=cfg.chroma_collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"La colección '{cfg.chroma_collection_name}' no existe o no es accesible: {exc}"
        ) from exc

    total_items: int = collection.count()
    logger.info(
        "Colección '%s' encontrada — %d chunks totales.",
        cfg.chroma_collection_name,
        total_items,
    )

    if total_items == 0:
        raise RuntimeError(
            f"La colección '{cfg.chroma_collection_name}' está vacía. "
            "Ejecuta primero el pipeline de indexación (chroma_pipeline.py)."
        )

    # ── 2. Extracción completa ────────────────────────────────────────────────
    # ChromaDB limita internamente las consultas; se itera en batches seguros
    # de 5 000 registros para no saturar la memoria en colecciones grandes.
    BATCH_SIZE = 5_000
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []
    offset = 0

    logger.info("Iniciando extracción de embeddings en batches de %d...", BATCH_SIZE)

    while offset < total_items:
        try:
            result = collection.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=["embeddings", "metadatas"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Error al recuperar embeddings (offset={offset}): {exc}"
            ) from exc

        # ChromaDB devuelve embeddings como numpy array; usar `or []` o `not`
        # sobre un array lanza ValueError. Se usan comprobaciones explícitas.
        _emb = result.get("embeddings")
        fetched_embeddings = _emb if _emb is not None else []
        _meta = result.get("metadatas")
        fetched_metadatas = _meta if _meta is not None else []

        if len(fetched_embeddings) == 0:
            # La paginación se agotó antes de llegar al total declarado
            logger.warning(
                "La respuesta de ChromaDB devolvió 0 embeddings en offset=%d. "
                "Deteniendo la extracción anticipadamente.",
                offset,
            )
            break

        # Convertir a lista por si ChromaDB devuelve numpy array
        all_embeddings.extend(
            fetched_embeddings.tolist()
            if hasattr(fetched_embeddings, "tolist")
            else fetched_embeddings
        )
        all_metadatas.extend(fetched_metadatas)
        offset += len(fetched_embeddings)

        logger.info(
            "  Recuperados %d/%d chunks...", offset, total_items
        )

    logger.info(
        "Extracción completada: %d embeddings recuperados de %d reportados.",
        len(all_embeddings),
        total_items,
    )

    # ── 3. Mean Pooling por numero_expediente ─────────────────────────────────
    logger.info(
        "Aplicando Mean Pooling agrupado por '%s'...",
        cfg.metadata_groupby_key,
    )

    # Acumuladores: expediente_id → (lista de vectores, etiqueta de resolución)
    vectors_by_exp: dict[str, list[list[float]]] = defaultdict(list)
    label_by_exp: dict[str, str] = {}
    skipped_chunks = 0

    for embedding, metadata in zip(all_embeddings, all_metadatas):
        # Validar presencia de claves obligatorias
        group_key = metadata.get(cfg.metadata_groupby_key)
        label_value = metadata.get(cfg.metadata_label_key)

        if not group_key or not label_value:
            skipped_chunks += 1
            continue

        vectors_by_exp[group_key].append(embedding)

        # Si un expediente tuviese labels inconsistentes entre chunks
        # (no debería ocurrir, pero es defensivo), conservamos el primero.
        if group_key not in label_by_exp:
            label_by_exp[group_key] = str(label_value).strip()

    if skipped_chunks > 0:
        logger.warning(
            "%d chunks omitidos por carecer de '%s' o '%s' en sus metadatas.",
            skipped_chunks,
            cfg.metadata_groupby_key,
            cfg.metadata_label_key,
        )

    n_expedientes = len(vectors_by_exp)
    if n_expedientes == 0:
        raise RuntimeError(
            "No se pudieron construir muestras consolidadas. Verifica que los "
            f"metadatas contengan las claves '{cfg.metadata_groupby_key}' y "
            f"'{cfg.metadata_label_key}'."
        )

    logger.info(
        "Consolidando %d expedientes únicos mediante Mean Pooling...",
        n_expedientes,
    )

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for exp_id, vectors in vectors_by_exp.items():
        # Centroide aritmético de todos los chunk-embeddings del expediente
        pooled_vector: np.ndarray = np.mean(
            np.array(vectors, dtype=np.float32), axis=0
        )
        X_list.append(pooled_vector)
        y_list.append(label_by_exp[exp_id])

    X: np.ndarray = np.vstack(X_list).astype(np.float32)
    y: np.ndarray = np.array(y_list)

    logger.info(
        "Dataset listo — X: %s | y: %s | Clases únicas: %s",
        X.shape,
        y.shape,
        sorted(set(y_list)),
    )

    return DatasetResult(X=X, y=y)


# ─── Ejecución directa (smoke test) ──────────────────────────────────────────
if __name__ == "__main__":
    result = load_dataset()
    print(f"\n✅ X shape : {result.X.shape}")
    print(f"✅ y shape : {result.y.shape}")
    print(f"✅ Clases  : {sorted(set(result.y.tolist()))}")
