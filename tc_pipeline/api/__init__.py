"""
tc_pipeline.api
───────────────
Paquete de la API FastAPI del Sistema Multiagente TC.

Exporta la aplicación FastAPI y los esquemas Pydantic.
"""

from tc_pipeline.api.main import app
from tc_pipeline.api.schemas import (
    BriefResponse,
    ExpedienteAPI,
    ExpedienteBase,
    ExpedienteEnriquecido,
    HealthResponse,
    PaginatedResponse,
    PipelineStats,
    PrediccionRequest,
    PrediccionResponse,
    QueryRequest,
)

__all__ = [
    "app",
    "ExpedienteBase",
    "ExpedienteAPI",
    "ExpedienteEnriquecido",
    "PaginatedResponse",
    "HealthResponse",
    "PipelineStats",
    "QueryRequest",
    "BriefResponse",
    "PrediccionRequest",
    "PrediccionResponse",
]
