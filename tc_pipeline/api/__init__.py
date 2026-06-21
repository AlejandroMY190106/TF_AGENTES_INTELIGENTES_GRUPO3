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
    HealthResponse,
    PrediccionRequest,
    PrediccionResponse,
    QueryRequest,
)

__all__ = [
    "app",
    "ExpedienteBase",
    "ExpedienteAPI",
    "HealthResponse",
    "QueryRequest",
    "BriefResponse",
    "PrediccionRequest",
    "PrediccionResponse",
]
