"""
tc_pipeline/ml-training/model_trainer_RandomForest.py
───────────────────────────────────────────────────────
Modulo de Entrenamiento y Serializacion del Clasificador Random Forest.

Responsabilidad Arquitectonica:
    Recibe las matrices X e y pre-procesadas externamente por
    data_loader.py, realiza la codificacion categorica de etiquetas,
    ejecuta una particion estratificada 80/20, entrena el clasificador
    RandomForestClassifier con los hiperparametros definidos en
    MLConfig, y serializa los artefactos resultantes (modelo joblib +
    LabelEncoder joblib).

    Este modulo NO conoce como se extraen los datos de ChromaDB ni como se
    computan las metricas de evaluacion.

Uso:
    from tc_pipeline.ml_training.model_trainer_RandomForest import train
    from tc_pipeline.config import MLConfig

    cfg = MLConfig()
    result = train(X, y, cfg)
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tc_pipeline.config import MLConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    encoder: LabelEncoder
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


def train(
    X: np.ndarray,
    y: np.ndarray,
    cfg: MLConfig | None = None,
) -> TrainingResult:
    if cfg is None:
        cfg = MLConfig()

    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"X debe ser una matriz 2-D no vacia. Shape recibido: {X.shape}")
    if y.ndim != 1 or len(y) == 0:
        raise ValueError(f"y debe ser un vector 1-D no vacio. Shape recibido: {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X e y deben tener el mismo numero de muestras. X={len(X)}, y={len(y)}")

    logger.info("Dataset recibido -- X: %s | y: %s | Clases: %s", X.shape, y.shape, sorted(set(y.tolist())))

    encoder = LabelEncoder()
    y_encoded: np.ndarray = encoder.fit_transform(y)
    class_names: list[str] = list(encoder.classes_)
    n_classes = len(class_names)
    logger.info("Clases codificadas (%d): %s -> indices 0..%d", n_classes, class_names, n_classes - 1)

    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]
    n_dropped_classes = len(unique_classes) - len(valid_classes)

    if n_dropped_classes > 0:
        mask = np.isin(y_encoded, valid_classes)
        n_dropped_samples = int((~mask).sum())
        logger.warning("Se eliminaron %d clases singleton y %d muestras asociadas para permitir la particion estratificada.", n_dropped_classes, n_dropped_samples)
        X = X[mask]
        y_encoded = y_encoded[mask]
        y_labels_filtered = encoder.inverse_transform(y_encoded)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_labels_filtered)
        class_names = list(encoder.classes_)
        n_classes = len(class_names)
        logger.info("Clases tras filtrado: %d -> indices 0..%d", n_classes, n_classes - 1)

    logger.info("Dividiendo el dataset (test_size=%.0f%%, random_state=%d, stratify=True)...", cfg.test_size * 100, cfg.random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_encoded)
    logger.info("Particion completada -- Entrenamiento: %d muestras | Prueba: %d muestras", len(X_train), len(X_test))

    params = cfg.rf_params
    logger.info("Inicializando RandomForestClassifier con parametros: %s", params)
    model = RandomForestClassifier(**params)

    logger.info("Entrenando RandomForestClassifier...")
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado.")

    output_dir: Path = cfg.models_output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"No se pudo crear el directorio de modelos '{output_dir}': {exc}") from exc

    model_path: Path = cfg.rf_model_artifact_path
    joblib.dump(model, str(model_path), compress=3)
    logger.info("Modelo RandomForest guardado en: '%s'", model_path)

    encoder_path: Path = cfg.rf_encoder_artifact_path
    joblib.dump(encoder, str(encoder_path), compress=3)
    logger.info("LabelEncoder de RandomForest serializado en: '%s'", encoder_path)

    return TrainingResult(model=model, encoder=encoder, X_test=X_test, y_test=y_test, class_names=class_names)


if __name__ == "__main__":
    _ML_DIR = os.path.dirname(__file__)
    if _ML_DIR not in sys.path:
        sys.path.insert(0, _ML_DIR)

    from data_loader import load_dataset  # type: ignore[import-not-found]

    cfg = MLConfig()
    dataset = load_dataset(cfg)
    result = train(dataset.X, dataset.y, cfg)

    print(f"\nModelo entrenado: {type(result.model).__name__}")
    print(f"Clases           : {result.class_names}")
    print(f"X_test shape     : {result.X_test.shape}")
    print(f"y_test shape     : {result.y_test.shape}")
