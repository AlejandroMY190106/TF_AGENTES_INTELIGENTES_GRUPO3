"""
tc_pipeline/api/routes.py
─────────────────────────
Router de FastAPI con endpoints del pipeline TC.

Fase 1 (operativos ahora):
  - GET /health         — health check de todos los componentes
  - GET /expedientes    — listar expedientes con paginación y filtros
  - GET /expedientes/{numero} — obtener un expediente específico
  - GET /stats          — estadísticas del pipeline

Stubs para fases posteriores:
  - POST /ingest        — ingesta de expedientes (Fase 2)
  - POST /query         — consulta al Agente RAG (Fase 3)
  - POST /prediccion    — predicción del Agente Predictivo (Fase 4)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from tc_pipeline.api.schemas import (
    BriefResponse,
    ExpedienteEnriquecido,
    HealthResponse,
    PaginatedResponse,
    PipelineStats,
    PrediccionRequest,
    PrediccionResponse,
    QueryRequest,
)
from tc_pipeline.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Almacenamiento ──────────────────────────────────────────────────────
# Se inicializan con rutas por defecto. El main.py puede sobrescribirlas.
_parquet_store = ParquetStore(Path("data/raw/expedientes_tc.parquet"))


def set_parquet_store(store: ParquetStore) -> None:
    """Permite al main.py inyectar el store configurado."""
    global _parquet_store
    _parquet_store = store


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

    # Verificar Parquet store
    if _parquet_store.exists:
        try:
            df = _parquet_store.load()
            components["parquet"] = f"ok ({len(df)} expedientes)"
        except Exception as e:
            components["parquet"] = f"error: {e}"
    else:
        components["parquet"] = "no inicializado"

    # Verificar manifest DB
    manifest_path = Path("data/manifests/pipeline_state.db")
    if manifest_path.exists():
        components["manifest_db"] = "ok"
    else:
        components["manifest_db"] = "no inicializado"

    return HealthResponse(components=components)


@router.get("/expedientes", response_model=PaginatedResponse, tags=["Expedientes"])
async def listar_expedientes(
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(20, ge=1, le=100, description="Tamaño de página"),
    sala: str | None = Query(None, description="Filtrar por sala"),
    fallo: str | None = Query(None, description="Filtrar por sentido del fallo"),
    tipo_proceso: str | None = Query(None, description="Filtrar por tipo de proceso"),
    anio: int | None = Query(None, description="Filtrar por año del expediente"),
) -> PaginatedResponse:
    """Lista expedientes del dataset con paginación y filtros opcionales.

    Los datos se cargan desde el archivo Parquet generado por el pipeline
    de extracción.
    """
    if not _parquet_store.exists:
        raise HTTPException(
            status_code=404,
            detail="No se ha generado el dataset de expedientes aún. "
                   "Ejecute primero: python scripts/collect_metadata.py",
        )

    df = _parquet_store.load()

    # Aplicar filtros
    if sala:
        col = "SALA" if "SALA" in df.columns else "sentencia_sala"
        if col in df.columns:
            df = df[df[col].str.contains(sala, case=False, na=False)]

    if fallo:
        col = "FALLO" if "FALLO" in df.columns else "sentencia_sentido"
        if col in df.columns:
            df = df[df[col].str.contains(fallo, case=False, na=False)]

    if tipo_proceso:
        if "CDES_TIPOPROCESO" in df.columns:
            df = df[df["CDES_TIPOPROCESO"].str.contains(tipo_proceso, case=False, na=False)]

    if anio:
        if "numero_expediente" in df.columns:
            df = df[df["numero_expediente"].str.contains(str(anio), na=False)]

    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size

    page_data = _sanitize_records(
        df.iloc[start:end].fillna("").to_dict(orient="records")
    )

    return PaginatedResponse(
        data=page_data,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/expedientes/{numero}",
    response_model=dict[str, Any],
    tags=["Expedientes"],
)
async def obtener_expediente(numero: str) -> dict[str, Any]:
    """Obtiene un expediente específico por número."""
    if not _parquet_store.exists:
        raise HTTPException(status_code=404, detail="Dataset no disponible.")

    df = _parquet_store.load()
    col = "numero_expediente" if "numero_expediente" in df.columns else "NEXPEDIENTE"

    if col not in df.columns:
        raise HTTPException(status_code=500, detail="Columna de expediente no encontrada.")

    match = df[df[col] == numero]
    if match.empty:
        # Intentar busqueda parcial
        match = df[df[col].str.contains(numero, case=False, na=False)]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Expediente '{numero}' no encontrado.",
        )

    record = match.iloc[0].fillna("").to_dict()
    return _sanitize_record(record)


@router.get("/stats", response_model=PipelineStats, tags=["Sistema"])
async def pipeline_stats() -> PipelineStats:
    """Retorna estadísticas del pipeline de extracción."""
    stats = PipelineStats()

    if _parquet_store.exists:
        store_stats = _parquet_store.get_stats()
        stats.expedientes_parquet = store_stats.get("total_expedientes", 0)
        stats.total_expedientes = stats.expedientes_parquet
        stats.cobertura_temporal = {
            k: v for k, v in store_stats.items()
            if k in ("fecha_min", "fecha_max", "anio_expediente_min",
                      "anio_expediente_max", "expedientes_por_anio")
        }

    return stats


# ─────────────────────────────────────────────────────────────────────────
# Stubs para Fase 2+
# ─────────────────────────────────────────────────────────────────────────


@router.post("/ingest", tags=["Ingesta (Fase 2)"], status_code=202)
async def ingest_expedientes() -> dict[str, str]:
    """[Fase 2] Endpoint para ingesta de expedientes al dataset curado.

    Aún no implementado — se activará en la Fase 2 del plan.
    """
    raise HTTPException(
        status_code=501,
        detail="Endpoint de ingesta pendiente de implementación (Fase 2).",
    )


@router.post("/query", response_model=BriefResponse, tags=["RAG (Fase 3)"])
async def query_rag(request: QueryRequest) -> BriefResponse:
    """[Fase 3] Consulta al Agente RAG para generar un brief ejecutivo.

    Aún no implementado — se activará en la Fase 3 del plan.
    """
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
    """[Fase 4] Predicción del sentido del fallo con el Agente Predictivo.

    Aún no implementado — se activará en la Fase 4 del plan.
    """
    raise HTTPException(
        status_code=501,
        detail="Agente Predictivo pendiente de implementación (Fase 4).",
    )
