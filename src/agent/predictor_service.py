"""
src/agent/predictor_service.py
──────────────────────────────
Servicio de Inferencia para el Clasificador Predictivo (SVM).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# Asegurar rutas de importación del proyecto antes de cargar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tc_pipeline.config import MLConfig
from tc_pipeline.nlp.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PredictorService:
    """
    Servicio encargado de cargar el modelo SVM y el codificador de etiquetas
    desde disco para realizar predicciones del sentido de la resolución a partir
    de fundamentos o motivos de la demanda.
    """

    def __init__(self, model_path: str | Path | None = None, encoder_path: str | Path | None = None):
        cfg = MLConfig()
        
        # Asignar rutas por defecto si no se especifican
        self.model_path = Path(model_path) if model_path else cfg.svm_model_artifact_path
        self.encoder_path = Path(encoder_path) if encoder_path else cfg.svm_encoder_artifact_path
        
        logger.info("Inicializando PredictorService...")
        
        # Validar existencia de artefactos en disco
        if not self.model_path.is_file() or not self.encoder_path.is_file():
            raise FileNotFoundError(
                f"No se encontraron los artefactos necesarios para el PredictorService:\n"
                f"  - Modelo esperado: '{self.model_path}'\n"
                f"  - Encoder esperado: '{self.encoder_path}'\n"
                f"Por favor, ejecuta primero el pipeline de entrenamiento ejecutando:\n"
                f"  python tc_pipeline/ml-training/model_evaluator.py"
            )
            
        logger.info(f"Cargando clasificador SVM desde: '{self.model_path}'")
        try:
            self.model = joblib.load(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo SVM desde '{self.model_path}': {e}")
            
        logger.info(f"Cargando codificador de etiquetas desde: '{self.encoder_path}'")
        try:
            self.encoder = joblib.load(str(self.encoder_path))
        except Exception as e:
            raise RuntimeError(f"Error al cargar el codificador desde '{self.encoder_path}': {e}")
            
        # Instanciar modelo de embeddings para la inferencia
        logger.info(f"Cargando modelo de embeddings para inferencia: '{cfg.embedding_model_name}'")
        self.embedding_model = EmbeddingModel(model_name=cfg.embedding_model_name)

    def predict(self, texto: str) -> dict[str, Any]:
        """
        Genera el embedding del texto provisto y realiza la predicción de clase
        junto con las probabilidades del sentido del fallo.
        """
        if not texto or not texto.strip():
            raise ValueError("El texto provisto para la predicción no puede estar vacío.")

        # 1. Vectorizar el texto de consulta
        logger.info("Vectorizando texto para inferencia predictiva...")
        embeddings = self.embedding_model.embed_texts([texto.strip()])
        X = np.array(embeddings, dtype=np.float32)

        # 2. Obtener probabilidades
        probs = self.model.predict_proba(X)[0]
        winner_idx = int(np.argmax(probs))
        
        # 3. Decodificar etiqueta ganadora
        prediccion_str = str(self.encoder.inverse_transform([winner_idx])[0])
        confianza = float(probs[winner_idx])
        
        # Mapear probabilidades por clase
        probabilidades = {}
        for idx, class_name in enumerate(self.encoder.classes_):
            probabilidades[str(class_name)] = float(probs[idx])

        logger.info(f"Predicción completada: '{prediccion_str}' (Confianza: {confianza:.4f})")
        return {
            "prediccion": prediccion_str,
            "probabilidades": probabilidades,
            "confianza": confianza
        }


if __name__ == "__main__":
    # Prueba local de humo (smoke test)
    try:
        predictor = PredictorService()
        texto_prueba = "Se interpone recurso de amparo constitucional contra la resolución por falta de motivación adecuada."
        res = predictor.predict(texto_prueba)
        print("\n🔮 [RESULTADO DE PREDICCIÓN CON ÉXITO] 🔮")
        print(res)
    except FileNotFoundError as fnf:
        print(f"\n⚠️  Aviso de configuración: {fnf}")
    except Exception as e:
        print(f"\n❌ Error al ejecutar el smoke test: {e}")
