"""
tc_pipeline/ml-training/model_trainer_XGBoost.py
─────────────────────────────────────────────────
Módulo de Entrenamiento y Serialización del Clasificador XGBoost.

Responsabilidad Arquitectónica:
    Recibe las matrices ``X`` e ``y`` pre-procesadas externamente por
    ``data_loader.py``, realiza la codificación categórica de etiquetas,
    ejecuta una partición estratificada 80/20, entrena el clasificador
    ``XGBClassifier`` con los hiperparámetros definidos en ``MLConfig``, y
    serializa los artefactos resultantes (modelo JSON + LabelEncoder joblib).

    Este módulo NO conoce cómo se extraen los datos de ChromaDB ni cómo se
    computan las métricas de evaluación.

Uso:
    from tc_pipeline.ml_training.model_trainer import train
    from tc_pipeline.config import MLConfig

    cfg = MLConfig()
    result = train(X, y, cfg)
    # → result.model, result.encoder, result.X_test, result.y_test
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ─── Resolución dinámica de la raíz del proyecto ─────────────────────────────
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

@dataclass
class TrainingResult:
    """Contiene todos los artefactos producidos por el entrenamiento.

    Attributes:
        model: Clasificador ``XGBClassifier`` ya entrenado.
        encoder: ``LabelEncoder`` ajustado sobre el conjunto de entrenamiento.
            Necesario para recuperar los nombres de clase originales durante la
            evaluación y la inferencia.
        X_test: Subconjunto de prueba de la matriz de características (20 %).
        y_test: Etiquetas numéricas codificadas del subconjunto de prueba.
        class_names: Lista de nombres de clase en el orden asignado por el encoder.
    """
    model: xgb.XGBClassifier
    encoder: LabelEncoder
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


# ─── Función principal ────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    cfg: MLConfig | None = None,
) -> TrainingResult:
    """Entrena el clasificador XGBoost y serializa los artefactos resultantes.

    Flujo de ejecución:

    1. **Codificación de etiquetas**: ``LabelEncoder`` convierte las clases de
       texto (``"FUNDADA"``, ``"INFUNDADA"``, ``"IMPROCEDENTE"``, etc.) en
       índices enteros contiguos requeridos por XGBoost (0, 1, 2, ...).
    2. **Partición estratificada**: ``train_test_split(..., stratify=y_encoded)``
       preserva la distribución original de clases en ambos subconjuntos, lo que
       previene sesgos estadísticos derivados del desbalance de clases.
    3. **Entrenamiento**: ``XGBClassifier`` con ``objective='multi:softprob'`` y
       ``eval_metric='mlogloss'``.
    4. **Serialización**: El modelo se exporta en formato JSON nativo de XGBoost
       y el encoder en formato ``joblib`` comprimido.

    Args:
        X: Matriz de características ``(n_samples, embedding_dim)``.
        y: Vector de etiquetas de texto ``(n_samples,)``.
        cfg: Instancia de :class:`~tc_pipeline.config.MLConfig`. Si es ``None``,
             se utiliza la configuración por defecto.

    Returns:
        :class:`TrainingResult` con el modelo, el encoder y el conjunto de test.

    Raises:
        ValueError: Si ``X`` o ``y`` están vacíos o tienen dimensiones incompatibles.
        OSError: Si no es posible crear el directorio de salida de los modelos.
    """
    if cfg is None:
        cfg = MLConfig()

    # ── Validaciones de entrada ───────────────────────────────────────────────
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(
            f"X debe ser una matriz 2-D no vacía. Shape recibido: {X.shape}"
        )
    if y.ndim != 1 or len(y) == 0:
        raise ValueError(
            f"y debe ser un vector 1-D no vacío. Shape recibido: {y.shape}"
        )
    if len(X) != len(y):
        raise ValueError(
            f"X e y deben tener el mismo número de muestras. "
            f"X={len(X)}, y={len(y)}"
        )

    logger.info(
        "Dataset recibido — X: %s | y: %s | Clases: %s",
        X.shape,
        y.shape,
        sorted(set(y.tolist())),
    )

    # ── 1. Codificación categórica de etiquetas ───────────────────────────────
    logger.info("Codificando etiquetas con LabelEncoder...")
    encoder = LabelEncoder()
    y_encoded: np.ndarray = encoder.fit_transform(y)
    class_names: list[str] = list(encoder.classes_)
    n_classes = len(class_names)

    logger.info(
        "Clases codificadas (%d): %s → índices 0..%d",
        n_classes,
        class_names,
        n_classes - 1,
    )

    # ── 2. Filtrar clases singleton (< 2 muestras) ────────────────────────────
    # train_test_split con stratify exige al menos 2 muestras por clase.
    # Las clases con 1 sola muestra suelen ser typos o combinaciones rarísimas
    # que no aportan información estadística al modelo.
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]
    n_dropped_classes = len(unique_classes) - len(valid_classes)

    if n_dropped_classes > 0:
        mask = np.isin(y_encoded, valid_classes)
        n_dropped_samples = int((~mask).sum())
        logger.warning(
            "Se eliminaron %d clases singleton y %d muestras asociadas "
            "para permitir la partición estratificada.",
            n_dropped_classes,
            n_dropped_samples,
        )
        X = X[mask]
        y_encoded = y_encoded[mask]

        # Re-codificar para que los índices sean contiguos (0..N-1).
        # Usamos el encoder ORIGINAL (ya fiteado) para recuperar los strings,
        # luego fiteamos uno nuevo sobre el subconjunto filtrado.
        y_labels_filtered = encoder.inverse_transform(y_encoded)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_labels_filtered)
        class_names = list(encoder.classes_)
        n_classes = len(class_names)
        logger.info(
            "Clases tras filtrado: %d → índices 0..%d", n_classes, n_classes - 1
        )

    # ── 3. Partición estratificada 80 / 20 ────────────────────────────────────
    logger.info(
        "Dividiendo el dataset (test_size=%.0f%%, random_state=%d, stratify=True)...",
        cfg.test_size * 100,
        cfg.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_encoded,
    )

    logger.info(
        "Partición completada — Entrenamiento: %d muestras | Prueba: %d muestras",
        len(X_train),
        len(X_test),
    )

    # ── 3. Configuración y entrenamiento del clasificador ─────────────────────
    params = {**cfg.xgb_params, "num_class": n_classes}
    logger.info("Inicializando XGBClassifier con parámetros: %s", params)

    model = xgb.XGBClassifier(**params)

    # Compensación de desbalance de clases mediante sample_weight inversamente
    # proporcional a la frecuencia de cada clase (estrategia 'balanced').
    # Equivale a class_weight='balanced' de scikit-learn pero en la API de XGBoost.
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    logger.info(
        "sample_weight calculado (balanced) — min=%.4f | max=%.4f | mean=%.4f",
        sample_weights.min(),
        sample_weights.max(),
        sample_weights.mean(),
    )

    logger.info("Entrenando XGBClassifier con sample_weight balanceado...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    logger.info("✅ Entrenamiento completado.")

    # ── 4. Serialización de artefactos ────────────────────────────────────────
    output_dir: Path = cfg.models_output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"No se pudo crear el directorio de modelos '{output_dir}': {exc}"
        ) from exc

    # 4a. Modelo XGBoost en formato nativo JSON
    model_path: Path = cfg.model_artifact_path
    model.save_model(str(model_path))
    logger.info("Modelo XGBoost guardado en: '%s'", model_path)

    # 4b. LabelEncoder serializado con joblib
    encoder_path: Path = cfg.encoder_artifact_path
    joblib.dump(encoder, str(encoder_path), compress=3)
    logger.info("LabelEncoder serializado en: '%s'", encoder_path)

    return TrainingResult(
        model=model,
        encoder=encoder,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )


# ─── Ejecución directa (smoke test) ──────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Importar el data_loader para la ejecución autónoma del módulo
    _ML_DIR = os.path.dirname(__file__)
    if _ML_DIR not in sys.path:
        sys.path.insert(0, _ML_DIR)

    from data_loader import load_dataset  # type: ignore[import-not-found]

    cfg = MLConfig()
    dataset = load_dataset(cfg)
    result = train(dataset.X, dataset.y, cfg)

    print(f"\n✅ Modelo entrenado: {type(result.model).__name__}")
    print(f"✅ Clases           : {result.class_names}")
    print(f"✅ X_test shape     : {result.X_test.shape}")
    print(f"✅ y_test shape     : {result.y_test.shape}")
