"""
tc_pipeline/api/routes.py
─────────────────────────
Router de FastAPI con endpoints del pipeline TC.

Fase 1 (operativos):
  - GET /health         — health check de todos los componentes
  - GET /expedientes    — listar expedientes con paginación y filtros
  - GET /expedientes/{numero} — obtener un expediente específico
  - GET /stats          — estadísticas del pipeline

Scraping Masivo (nuevos):
  - POST /scraping/start      — iniciar pipeline masivo (1992-2026)
  - GET  /scraping/status/{id} — consultar progreso de tarea
  - POST /scraping/cancel/{id} — cancelar tarea en ejecución
  - GET  /datasets             — listar CSVs generados

Stubs para fases posteriores:
  - POST /ingest        — ingesta de expedientes (Fase 2)
  - POST /query         — consulta al Agente RAG (Fase 3)
  - POST /prediccion    — predicción del Agente Predictivo (Fase 4)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from tc_pipeline.api.schemas import (
    BriefResponse,
    DatasetInfo,
    DatasetListResponse,
    ExpedienteEnriquecido,
    HealthResponse,
    PaginatedResponse,
    PipelineStats,
    PrediccionRequest,
    PrediccionResponse,
    QueryRequest,
    ScrapingProgress,
    ScrapingRequest,
    ScrapingStatus,
    AnalisisCompletoRequest,
    AnalisisCompletoResponse,
)
from tc_pipeline.config import PipelineConfig

# Imports de Servicios e Inferencia
from src.agent.rag_service import RAGService
from src.agent.predictor_service import PredictorService
from src.agent.orchestrator import analizar_caso


logger = logging.getLogger(__name__)

router = APIRouter()

# ── Almacenamiento ──────────────────────────────────────────────────────
_config = PipelineConfig()

# ── Instanciación de Servicios (Singletons tolerantes a fallos) ────────
_rag_service: RAGService | None = None
_predictor_service: PredictorService | None = None

try:
    _rag_service = RAGService()
except Exception as _e:
    logger.warning(
        "⚠️ RAGService no pudo inicializarse durante el arranque del router: %s. "
        "El endpoint /query no estará funcional.",
        _e,
    )

try:
    _predictor_service = PredictorService()
except Exception as _e:
    logger.warning(
        "⚠️ PredictorService no pudo inicializarse durante el arranque del router: %s. "
        "El endpoint /prediccion no estará funcional.",
        _e,
    )

# ── Estado de tareas en memoria ─────────────────────────────────────────
_tasks: dict[str, ScrapingStatus] = {}
_cancel_flags: dict[str, bool] = {}


# ── Helpers de serialización ────────────────────────────────────────────


def _sanitize_value(v: Any) -> Any:
    """Convierte valores numpy/pandas a tipos Python nativos."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Sanitiza un dict para ser JSON-serializable."""
    return {k: _sanitize_value(v) for k, v in record.items()}


def _sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitiza una lista de dicts para ser JSON-serializable."""
    return [_sanitize_record(r) for r in records]


# ─────────────────────────────────────────────────────────────────────────
# Fase 1: Endpoints operativos
# ─────────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check() -> HealthResponse:
    """Verifica el estado de salud de la API y sus componentes."""
    components: dict[str, str] = {}

    # Verificar directorios de datos
    for name, path in [
        ("csv_output", _config.csv_output_root),
        ("sentencia_raw", _config.sentencia_raw_root),
        ("auto_resolucion_raw", _config.auto_resolucion_raw_root),
        ("sentencia_extract", _config.sentencia_extract_root),
        ("auto_resolucion_extract", _config.auto_resolucion_extract_root),
    ]:
        if path.exists():
            count = len(list(path.rglob("*.csv"))) + len(list(path.rglob("*.pdf")))
            components[name] = f"ok ({count} archivos)"
        else:
            components[name] = "no inicializado"

    # Tareas activas
    active = sum(1 for t in _tasks.values() if t.status == "running")
    components["active_tasks"] = str(active)

    return HealthResponse(components=components)



# ─────────────────────────────────────────────────────────────────────────
# Scraping Masivo
# ─────────────────────────────────────────────────────────────────────────


def _run_scraping_pipeline(task_id: str, request: ScrapingRequest) -> None:
    """Ejecuta el pipeline de scraping en background.

    Esta función corre en un hilo separado vía BackgroundTasks.
    Actualiza el estado de la tarea en ``_tasks`` conforme avanza.
    """
    from tc_pipeline.extraction.pdf_extractor import process_year as extract_year
    from tc_pipeline.scraping import PDFDownloader, TribunalAPIClient

    task = _tasks[task_id]
    task.status = "running"
    task.started_at = datetime.now(timezone.utc).isoformat()

    config = PipelineConfig()
    total_years = request.end_year - request.start_year + 1
    task.progress.total_years = total_years

    try:
        years_done = 0

        for year in range(request.end_year, request.start_year - 1, -1):
            # Verificar cancelación
            if _cancel_flags.get(task_id, False):
                task.status = "cancelled"
                return

            task.progress.current_year = year

            # Fase JSON
            if "json" in request.phases:
                task.progress.current_phase = "json"
                try:
                    with TribunalAPIClient(config) as api:
                        csv_path = api.fetch_year_to_csv(year)
                        with open(csv_path, encoding="utf-8") as f:
                            count = sum(1 for _ in f) - 1
                        task.progress.records_processed += max(0, count)
                except Exception as e:
                    task.errors.append(f"JSON {year}: {e}")

            # Fase Download
            if "download" in request.phases:
                task.progress.current_phase = "download"
                try:
                    downloader = PDFDownloader(config)
                    with TribunalAPIClient(config) as api:
                        items, records = api.get_items_with_metadata(year)
                    if items:
                        metrics = downloader.download_year(
                            items, records, year, show_progress=False
                        )
                        downloader.save_id_map(year)
                        task.progress.pdfs_downloaded += metrics.descargados
                except Exception as e:
                    task.errors.append(f"Download {year}: {e}")

            # Fase Extract
            if "extract" in request.phases:
                task.progress.current_phase = "extract"
                try:
                    downloader = PDFDownloader(config)
                    id_map = downloader.load_id_map(year)
                    for doc_type in ("sentencia", "auto-resolucion"):
                        extract_year(year, doc_type, config, id_map)
                        task.progress.pdfs_extracted += 1
                except Exception as e:
                    task.errors.append(f"Extract {year}: {e}")

            years_done += 1
            task.progress.years_done = years_done

            time.sleep(config.page_delay)

        task.status = "completed"

    except Exception as e:
        task.status = "failed"
        task.errors.append(f"Error fatal: {e}")
        logger.error("Pipeline falló: %s", e)

    finally:
        task.finished_at = datetime.now(timezone.utc).isoformat()
        task.progress.current_phase = None
        task.progress.current_year = None


@router.post(
    "/scraping/start",
    response_model=ScrapingStatus,
    tags=["Scraping Masivo"],
    status_code=202,
)
async def start_scraping(
    request: ScrapingRequest,
    background_tasks: BackgroundTasks,
) -> ScrapingStatus:
    """Inicia el pipeline de scraping masivo (1992-2026).

    Valida el request, crea la estructura de carpetas y lanza
    la tarea en segundo plano con BackgroundTasks.
    """
    # Validar que no hay otra tarea corriendo
    running = [t for t in _tasks.values() if t.status == "running"]
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Ya hay una tarea en ejecución: {running[0].task_id}",
        )

    # Validar fases
    valid_phases = {"json", "download", "extract"}
    invalid = set(request.phases) - valid_phases
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Fases inválidas: {invalid}. Opciones: {valid_phases}",
        )

    # Crear estructura de carpetas
    config = PipelineConfig()
    for path in [
        config.csv_output_root,
        config.sentencia_raw_root,
        config.auto_resolucion_raw_root,
        config.sentencia_extract_root,
        config.auto_resolucion_extract_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    # Crear tarea
    task_id = str(uuid.uuid4())[:8]
    task = ScrapingStatus(
        task_id=task_id,
        status="pending",
        progress=ScrapingProgress(
            total_years=request.end_year - request.start_year + 1,
        ),
    )
    _tasks[task_id] = task
    _cancel_flags[task_id] = False

    # Lanzar en background
    background_tasks.add_task(_run_scraping_pipeline, task_id, request)

    logger.info(
        "Tarea de scraping creada: %s (años %d-%d, fases: %s)",
        task_id,
        request.start_year,
        request.end_year,
        request.phases,
    )

    return task


@router.get(
    "/scraping/status/{task_id}",
    response_model=ScrapingStatus,
    tags=["Scraping Masivo"],
)
async def get_scraping_status(task_id: str) -> ScrapingStatus:
    """Consulta el estado de una tarea de scraping."""
    if task_id not in _tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Tarea '{task_id}' no encontrada.",
        )
    return _tasks[task_id]


@router.post(
    "/scraping/cancel/{task_id}",
    tags=["Scraping Masivo"],
)
async def cancel_scraping(task_id: str) -> dict[str, Any]:
    """Cancela una tarea de scraping en ejecución."""
    if task_id not in _tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Tarea '{task_id}' no encontrada.",
        )

    task = _tasks[task_id]
    if task.status != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Tarea no está en ejecución (estado: {task.status}).",
        )

    _cancel_flags[task_id] = True
    return {"task_id": task_id, "message": "Cancelación solicitada"}


@router.get(
    "/scraping/tasks",
    response_model=list[ScrapingStatus],
    tags=["Scraping Masivo"],
)
async def list_scraping_tasks() -> list[ScrapingStatus]:
    """Lista todas las tareas de scraping (historial)."""
    return list(_tasks.values())


# ─────────────────────────────────────────────────────────────────────────
# Datasets generados
# ─────────────────────────────────────────────────────────────────────────


def _scan_csvs(directory: Path, doc_type: str) -> list[DatasetInfo]:
    """Escanea un directorio recursivamente buscando CSVs."""
    datasets: list[DatasetInfo] = []

    if not directory.exists():
        return datasets

    for csv_file in sorted(directory.rglob("*.csv")):
        year_match = re.search(r"(\d{4})", csv_file.stem)
        year = int(year_match.group(1)) if year_match else 0

        datasets.append(
            DatasetInfo(
                filename=csv_file.name,
                path=str(csv_file),
                year=year,
                doc_type=doc_type,
                size_bytes=csv_file.stat().st_size if csv_file.exists() else 0,
            )
        )

    return datasets


@router.get(
    "/datasets",
    response_model=DatasetListResponse,
    tags=["Datasets"],
)
async def list_datasets() -> DatasetListResponse:
    """Lista todos los CSVs generados por el pipeline."""
    config = PipelineConfig()

    json_csvs = _scan_csvs(config.csv_output_root, "json-metadata")
    sentencia_csvs = _scan_csvs(config.sentencia_extract_root, "sentencia-texto")
    auto_csvs = _scan_csvs(config.auto_resolucion_extract_root, "auto-resolucion-texto")

    return DatasetListResponse(
        json_csvs=json_csvs,
        sentencia_csvs=sentencia_csvs,
        auto_csvs=auto_csvs,
        total_files=len(json_csvs) + len(sentencia_csvs) + len(auto_csvs),
    )


# ─────────────────────────────────────────────────────────────────────────
# Stubs para Fase 2+
# ─────────────────────────────────────────────────────────────────────────


@router.post("/ingest", tags=["Ingesta (Fase 2)"], status_code=202)
async def ingest_expedientes() -> dict[str, str]:
    """[Fase 2] Endpoint para ingesta de expedientes al dataset curado."""
    raise HTTPException(
        status_code=501,
        detail="Endpoint de ingesta pendiente de implementación (Fase 2).",
    )


@router.post("/query", response_model=BriefResponse, tags=["RAG (Fase 3)"])
async def query_rag(request: QueryRequest) -> BriefResponse:
    """[Fase 3] Consulta al Agente RAG para generar un brief ejecutivo."""
    raise HTTPException(
        status_code=501,
        detail="Agente RAG pendiente de implementación (Fase 3).",
    )


@router.post(
    "/prediccion",
    response_model=PrediccionResponse,
    tags=["Predictivo (Fase 4)"],
)
async def prediccion(request: PrediccionRequest) -> PrediccionResponse:
    """[Fase 4] Predicción del sentido del fallo con el Agente Predictivo."""
    raise HTTPException(
        status_code=501,
        detail="Agente Predictivo pendiente de implementación (Fase 4).",
    )
