"""
tc_pipeline/ml-training/model_trainer_SVM.py
─────────────────────────────────────────────────
Módulo de Entrenamiento y Serialización del Clasificador SVM con Kernel RBF.

Responsabilidad Arquitectónica:
    Recibe las matrices ``X`` e ``y`` pre-procesadas externamente por
    ``data_loader.py``, realiza la codificación categórica de etiquetas,
    ejecuta una partición estratificada 80/20, entrena el clasificador
    ``SVC`` (SVM con Kernel RBF), y serializa los artefactos resultantes
    (modelo joblib + LabelEncoder joblib).
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


@dataclass
class TrainingResult:
    """Contiene todos los artefactos producidos por el entrenamiento."""
    model: SVC
    encoder: LabelEncoder
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


def train(
    X: np.ndarray,
    y: np.ndarray,
    cfg: MLConfig | None = None,
) -> TrainingResult:
    """Entrena el clasificador SVM y serializa los artefactos."""
    if cfg is None:
        cfg = MLConfig()

    # Validaciones de entrada
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"X debe ser una matriz 2-D no vacía. Shape recibido: {X.shape}")
    if y.ndim != 1 or len(y) == 0:
        raise ValueError(f"y debe ser un vector 1-D no vacío. Shape recibido: {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X e y deben tener el mismo número de muestras. X={len(X)}, y={len(y)}")

    logger.info(
        "Dataset recibido — X: %s | y: %s | Clases: %s",
        X.shape,
        y.shape,
        sorted(set(y.tolist())),
    )

    # 1. Codificación categórica de etiquetas
    logger.info("Codificando etiquetas con LabelEncoder...")
    encoder = LabelEncoder()
    y_encoded: np.ndarray = encoder.fit_transform(y)
    class_names: list[str] = list(encoder.classes_)
    n_classes = len(class_names)

    # 2. Filtrar clases singleton (< 2 muestras)
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]
    n_dropped_classes = len(unique_classes) - len(valid_classes)

    if n_dropped_classes > 0:
        mask = np.isin(y_encoded, valid_classes)
        n_dropped_samples = int((~mask).sum())
        logger.warning(
            "Se eliminaron %d clases singleton y %d muestras asociadas para permitir la partición estratificada.",
            n_dropped_classes,
            n_dropped_samples,
        )
        X = X[mask]
        y_encoded = y_encoded[mask]

        y_labels_filtered = encoder.inverse_transform(y_encoded)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_labels_filtered)
        class_names = list(encoder.classes_)
        n_classes = len(class_names)

    # 3. Partición estratificada 80 / 20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_encoded,
    )

    # 4. Configuración y entrenamiento de SVC
    logger.info("Inicializando SVC con Kernel RBF...")
    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=cfg.random_state
    )

    logger.info("Entrenando SVC (esto puede tardar unos momentos)...")
    model.fit(X_train, y_train)
    logger.info("✅ Entrenamiento completado.")

    # 5. Serialización de artefactos
    output_dir: Path = cfg.models_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.svm_model_artifact_path
    encoder_path = cfg.svm_encoder_artifact_path

    joblib.dump(model, str(model_path), compress=3)
    logger.info("Modelo SVM guardado en: '%s'", model_path)

    joblib.dump(encoder, str(encoder_path), compress=3)
    logger.info("LabelEncoder de SVM guardado en: '%s'", encoder_path)

    return TrainingResult(
        model=model,
        encoder=encoder,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )


if __name__ == "__main__":
    from data_loader import load_dataset  # type: ignore[import-not-found]
    cfg = MLConfig()
    dataset = load_dataset(cfg)
    result = train(dataset.X, dataset.y, cfg)

    print(f"\n✅ Modelo entrenado: {type(result.model).__name__}")
    print(f"✅ Clases           : {result.class_names}")
    print(f"✅ X_test shape     : {result.X_test.shape}")
    print(f"✅ y_test shape     : {result.y_test.shape}")
