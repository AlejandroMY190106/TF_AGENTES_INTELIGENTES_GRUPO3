"""
src/agent/orchestrator.py
─────────────────────────
Coordinador de ejecución de los agentes RAG y Predictor para el TC.
"""

import os
import sys
import logging
import asyncio

# Asegurar rutas de importación del proyecto antes de cargar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent.rag_service import RAGService
from src.agent.predictor_service import PredictorService
from tc_pipeline.api.schemas import AnalisisCompletoResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def analizar_caso(
    texto_demanda: str,
    rag: RAGService | None,
    predictor: PredictorService | None
) -> AnalisisCompletoResponse:
    """
    Coordinador asíncrono para ejecutar concurrentemente RAGService y PredictorService.
    Consolida las respuestas en AnalisisCompletoResponse y maneja fallos individuales de forma tolerante.
    """
    logger.info("Iniciando análisis coordinado del caso...")

    # 1. Ejecutar PredictorService primero (inferencia local rápida)
    pred_result = None
    if predictor is not None:
        try:
            pred_result = await asyncio.to_thread(predictor.predict, texto_demanda)
        except Exception as e:
            logger.error(f"Error durante la ejecución del PredictorService: {e}")
            pred_result = e
    else:
        logger.warning("PredictorService no está inicializado. Se omitirá su ejecución.")

    # Extraer variables para anclar el RAG
    prediccion_clase = None
    confianza_valor = None
    if pred_result is not None and not isinstance(pred_result, Exception):
        prediccion_clase = pred_result.get("prediccion")
        confianza_valor = pred_result.get("confianza")

    # 2. Ejecutar RAGService secuencialmente pasando el resultado del Predictor si existe
    rag_result = None
    if rag is not None:
        try:
            rag_result = await asyncio.to_thread(
                rag.generate_answer, 
                texto_demanda, 
                prediccion=prediccion_clase, 
                confianza_prediccion=confianza_valor
            )
        except Exception as e:
            logger.error(f"Error durante la ejecución de RAGService: {e}")
            rag_result = e
    else:
        logger.warning("RAGService no está inicializado. Se omitirá su ejecución.")

    # --- Procesamiento de resultados del RAG ---
    brief_text = ""
    sources = []
    metadata = {}

    if isinstance(rag_result, Exception):
        logger.error(f"Error durante la ejecución de RAGService: {rag_result}")
        brief_text = (
            "⚠️ El análisis del Agente RAG no se encuentra disponible debido a un error interno "
            "o falta de conexión con la base de datos de jurisprudencia."
        )
    elif rag_result is None:
        brief_text = (
            "⚠️ El análisis del Agente RAG no está disponible porque el servicio no fue inicializado "
            "(ChromaDB vacía o sin conexión)."
        )
    else:
        # Se obtuvo una respuesta exitosa (instancia de GlobalBriefResponse)
        brief_text = rag_result.brief
        sources = rag_result.sources
        metadata = rag_result.metadata

    # --- Procesamiento de resultados del Predictor ---
    prediccion = "No disponible"
    probabilidades = {}
    confianza = 0.0
    modelo = "XGBoost (Baseline)"

    if isinstance(pred_result, Exception):
        logger.error(f"Error durante la ejecución del PredictorService: {pred_result}")
    elif pred_result is None:
        logger.warning("PredictorService no devolvió resultados o no fue inicializado.")
    else:
        # Se obtuvo una respuesta exitosa (diccionario devuelto por PredictorService.predict)
        prediccion = pred_result.get("prediccion", "No disponible")
        probabilidades = pred_result.get("probabilidades", {})
        confianza = pred_result.get("confianza", 0.0)

    # --- Construcción y Retorno de la Respuesta Combinada ---
    return AnalisisCompletoResponse(
        query=texto_demanda,
        brief=brief_text,
        sources=sources,
        metadata=metadata,
        prediccion=prediccion,
        probabilidades=probabilidades,
        modelo=modelo,
        confianza=confianza
    )
