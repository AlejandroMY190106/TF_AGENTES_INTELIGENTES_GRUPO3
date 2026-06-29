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

    # Tareas asíncronas
    task_rag = None
    task_predictor = None

    # Inicializar llamadas concurrentes en hilos independientes (para evitar bloquear el loop principal)
    if rag is not None:
        task_rag = asyncio.to_thread(rag.generate_answer, texto_demanda)
    else:
        logger.warning("RAGService no está inicializado. Se omitirá su ejecución.")

    if predictor is not None:
        task_predictor = asyncio.to_thread(predictor.predict, texto_demanda)
    else:
        logger.warning("PredictorService no está inicializado. Se omitirá su ejecución.")

    # Ejecutar en paralelo
    rag_result = None
    pred_result = None

    # Construimos la lista de corutinas y hacemos gather tolerando excepciones individuales
    if task_rag and task_predictor:
        results = await asyncio.gather(task_rag, task_predictor, return_exceptions=True)
        rag_result, pred_result = results[0], results[1]
    elif task_rag:
        try:
            rag_result = await task_rag
        except Exception as e:
            rag_result = e
    elif task_predictor:
        try:
            pred_result = await task_predictor
        except Exception as e:
            pred_result = e

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
