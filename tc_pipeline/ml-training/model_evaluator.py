"""
tc_pipeline/ml-training/model_evaluator.py
───────────────────────────────────────────
Módulo de Métricas de Fiabilidad Avanzadas.

Responsabilidad Arquitectónica:
    Recibe el modelo entrenado, el codificador de etiquetas y el conjunto de
    prueba de manera totalmente desacoplada. Calcula y muestra en consola un
    conjunto riguroso de métricas de clasificación multiclase:

        1. Accuracy (exactitud global)
        2. Matriz de Confusión completa
        3. Classification Report (Precisión, Recall, F1 por clase + promedios)
        4. ROC-AUC multiclase One-vs-Rest con promedio macro

    Este módulo NO entrena ni ajusta parámetros. Es agnóstico al origen de los
    datos y al método de entrenamiento utilizado.

Uso:
    from tc_pipeline.ml_training.model_evaluator import evaluate
    from tc_pipeline.config import MLConfig

    evaluate(model, encoder, X_test, y_test)
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional, Any

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ─── Resolución dinámica de la raíz del proyecto ─────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─── Logger del módulo ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Constantes de formato ────────────────────────────────────────────────────
_SEPARATOR = "=" * 68
_SUBSEP = "-" * 68


# ─── Función principal ────────────────────────────────────────────────────────

def evaluate(
    model: Any,
    encoder: LabelEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    output_file: Optional[str] = None,
    model_name: str = "XGBoost",
) -> dict:
    """Evalúa el modelo entrenado sobre el conjunto de prueba y reporta métricas.

    Ejecuta la evaluación completa del clasificador mediante cuatro métricas
    complementarias:

    1. **Accuracy**: Proporción de predicciones correctas sobre el total.
    2. **Matriz de Confusión**: Tabla ``(n_clases × n_clases)`` con la distribución
       de predicciones correctas e incorrectas por clase.
    3. **Classification Report**: Precisión, Recall, F1-Score y soporte por clase,
       más promedios macro y ponderado.
    4. **ROC-AUC OvR Macro**: Área bajo la curva ROC calculada con la estrategia
       One-vs-Rest y promedio macro, operando sobre las probabilidades de clase
       devueltas por ``predict_proba``.

    Args:
        model: Clasificador ``XGBClassifier`` entrenado.
        encoder: ``LabelEncoder`` ajustado con las mismas clases del entrenamiento.
            Usado para mapear índices numéricos a nombres de clase legibles.
        X_test: Matriz de características del conjunto de prueba ``(n, d)``.
        y_test: Vector de etiquetas numéricas del conjunto de prueba ``(n,)``.
        output_file: Ruta opcional a un archivo ``.txt`` donde se exportará el
            reporte completo además de mostrarse en consola. Si es ``None``, sólo
            se registra en el logger.

    Returns:
        Diccionario con las métricas clave:
        ``{"accuracy": float, "roc_auc_macro": float | None,
           "confusion_matrix": np.ndarray, "class_names": list[str]}``.

    Raises:
        ValueError: Si las dimensiones de ``X_test`` e ``y_test`` son incompatibles.
    """
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test e y_test deben tener el mismo número de muestras. "
            f"X_test={len(X_test)}, y_test={len(y_test)}"
        )

    class_names: list[str] = list(encoder.classes_)
    n_classes = len(class_names)

    logger.info(
        "Iniciando evaluación sobre %d muestras de prueba (%d clases)...",
        len(X_test),
        n_classes,
    )

    # ── Predicciones ──────────────────────────────────────────────────────────
    y_pred: np.ndarray = model.predict(X_test)
    y_proba: np.ndarray = model.predict_proba(X_test)

    # ── 1. Accuracy ───────────────────────────────────────────────────────────
    accuracy: float = float(accuracy_score(y_test, y_pred))

    # ── 2. Matriz de Confusión ────────────────────────────────────────────────
    conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)

    # ── 3. Classification Report ──────────────────────────────────────────────
    # El test set puede no contener todas las clases (algunas son muy raras).
    # Se filtra target_names a solo las clases presentes en y_test y y_pred.
    present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    present_names = [class_names[i] for i in present_labels]
    clf_report: str = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        digits=4,
        zero_division=0,
    )

    # ── 4. ROC-AUC OvR Macro ─────────────────────────────────────────────────
    roc_auc: float | None = None
    roc_auc_warning = ""

    try:
        if n_classes == 2:
            # Caso binario: sólo la columna de probabilidades de la clase positiva
            roc_auc = float(
                roc_auc_score(y_test, y_proba[:, 1])
            )
        else:
            # Caso multiclase: OvR con promedio macro sobre todas las clases
            roc_auc = float(
                roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except Exception as exc:
        roc_auc_warning = f"  ⚠  No se pudo calcular ROC-AUC: {exc}"
        logger.warning("ROC-AUC no calculable: %s", exc)

    # ── Formateo del reporte completo ─────────────────────────────────────────
    report_lines: list[str] = [
        "",
        _SEPARATOR,
        f"  📊  REPORTE DE EVALUACIÓN DEL CLASIFICADOR {model_name.upper()}",
        _SEPARATOR,
        "",
        f"  Muestras de prueba  : {len(X_test)}",
        f"  Clases              : {class_names}",
        "",
        _SUBSEP,
        "  1. ACCURACY (Exactitud Global)",
        _SUBSEP,
        f"  Accuracy            : {accuracy:.4f}  ({accuracy * 100:.2f}%)",
        "",
        _SUBSEP,
        "  2. MATRIZ DE CONFUSIÓN",
        _SUBSEP,
    ]

    # Encabezado de la matriz
    col_header = "  " + " " * 16 + "  ".join(
        f"{cn[:12]:>12}" for cn in class_names
    )
    report_lines.append(col_header)

    for i, row in enumerate(conf_matrix):
        row_label = f"  [{class_names[i][:12]:>12}]"
        row_values = "  ".join(f"{v:>12}" for v in row)
        report_lines.append(f"{row_label}  {row_values}")

    report_lines += [
        "",
        _SUBSEP,
        "  3. CLASSIFICATION REPORT (Precisión / Recall / F1)",
        _SUBSEP,
        clf_report,
        _SUBSEP,
        "  4. ROC-AUC (One-vs-Rest · Promedio Macro)",
        _SUBSEP,
    ]

    if roc_auc is not None:
        report_lines.append(
            f"  ROC-AUC (macro-OvR) : {roc_auc:.4f}"
        )
    else:
        report_lines.append(roc_auc_warning)

    report_lines += ["", _SEPARATOR, ""]

    full_report = "\n".join(report_lines)

    # ── Salida en consola (a través del logger) ───────────────────────────────
    for line in report_lines:
        logger.info(line)

    # ── Exportación opcional a archivo ────────────────────────────────────────
    if output_file is not None:
        try:
            out_path = os.path.abspath(output_file)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(full_report)
            logger.info("Reporte exportado a: '%s'", out_path)
        except OSError as exc:
            logger.warning("No se pudo exportar el reporte a disco: %s", exc)

    return {
        "accuracy": accuracy,
        "roc_auc_macro": roc_auc,
        "confusion_matrix": conf_matrix,
        "class_names": class_names,
    }


def load_and_evaluate(
    model_path: str,
    encoder_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    output_file: Optional[str] = None,
    model_type: str = "xgboost",
) -> dict:
    """Carga los artefactos serializados desde disco y ejecuta la evaluación.

    Función de conveniencia que permite reutilizar el evaluador sin necesidad
    de retener los objetos en memoria, útil para pipelines de inferencia batch
    o CI/CD de validación de modelos.

    Args:
        model_path: Ruta al archivo del modelo serializado.
        encoder_path: Ruta al archivo del ``LabelEncoder``.
        X_test: Matriz de características de prueba.
        y_test: Etiquetas numéricas de prueba.
        output_file: Ruta opcional para exportar el reporte en texto plano.
        model_type: Tipo de modelo ("xgboost" o "sklearn").

    Returns:
        Mismo diccionario de métricas que :func:`evaluate`.

    Raises:
        FileNotFoundError: Si alguno de los artefactos no se encuentra en disco.
    """
    import joblib  # Importación local para no generar dependencia circular

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado en: '{model_path}'."
        )
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(
            f"LabelEncoder no encontrado en: '{encoder_path}'."
        )

    logger.info("Cargando modelo (%s) desde: '%s'", model_type, model_path)
    if model_type.lower() == "xgboost":
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        model = joblib.load(model_path)

    logger.info("Cargando LabelEncoder desde: '%s'", encoder_path)
    encoder: LabelEncoder = joblib.load(encoder_path)

    return evaluate(model, encoder, X_test, y_test, output_file=output_file, model_name=model_type)


# ─── Ejecución directa (smoke test + pipeline completo) ──────────────────────
if __name__ == "__main__":
    _ML_DIR = os.path.dirname(__file__)
    if _ML_DIR not in sys.path:
        sys.path.insert(0, _ML_DIR)

    from data_loader import load_dataset      # type: ignore[import-not-found]
    from model_trainer_XGBoost import train   # type: ignore[import-not-found]

    # ─── Importar config desde la raíz
    from tc_pipeline.config import MLConfig  # type: ignore[import-not-found]

    cfg = MLConfig()

    # Carga y entrenamiento
    dataset = load_dataset(cfg)
    training_result = train(dataset.X, dataset.y, cfg)

    # Evaluación independiente
    metrics = evaluate(
        model=training_result.model,
        encoder=training_result.encoder,
        X_test=training_result.X_test,
        y_test=training_result.y_test,
        output_file="models/evaluation_report.txt",
    )

    print(f"\n✅ Accuracy     : {metrics['accuracy']:.4f}")
    print(f"✅ ROC-AUC (OvR): {metrics['roc_auc_macro']}")
