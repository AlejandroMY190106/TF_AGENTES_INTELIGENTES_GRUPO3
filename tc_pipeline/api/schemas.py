"""
tc_pipeline/api/schemas.py
──────────────────────────
Esquemas Pydantic — contrato de datos entre agentes del SMA.

Define los modelos que validan la entrada/salida de:
- Agente de Curación: ExpedienteAPI → ExpedienteEnriquecido
- Agente RAG: QueryRequest → BriefResponse
- Agente Predictivo: PrediccionRequest → PrediccionResponse

Estos esquemas son la "interfaz" compartida entre los tres agentes y
garantizan consistencia de tipos en toda la pipeline.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────
# Expediente base (campos comunes)
# ─────────────────────────────────────────────────────────────────────────


class ExpedienteBase(BaseModel):
    """Campos compartidos por todas las representaciones de un expediente."""

    numero_expediente: str = Field(
        ..., description="Número de expediente (ej: '01234-2025-AA/TC')"
    )
    fecha_publicacion: str | None = Field(
        None, description="Fecha de publicación en la web del TC"
    )
    sentencia_sala: str | None = Field(
        None, description="Sala que emitió la sentencia"
    )
    sentencia_sentido: str | None = Field(
        None, description="Sentido del fallo (Fundada, Infundada, etc.)"
    )
    sentencia_tipo: int | str | None = Field(
        None, description="Tipo de resolución (código numérico o texto)"
    )


# ─────────────────────────────────────────────────────────────────────────
# Respuesta de la API del TC (datos crudos)
# ─────────────────────────────────────────────────────────────────────────


class ExpedienteAPI(ExpedienteBase):
    """Expediente tal como llega del endpoint de la API del TC.

    Incluye todos los campos del JSON ``_source`` que devuelve la API
    cronológica de sentencias.
    """

    nombre_demandante: str | None = Field(None, description="Nombre del demandante")
    nombre_demandado: str | None = Field(None, description="Nombre del demandado")
    url_archivo: str | None = Field(None, description="URL al PDF de la sentencia")
    fundamentos: str | None = Field(None, description="Texto de los fundamentos")
    palabras: list[dict[str, Any]] | None = Field(
        None, description="Palabras clave / materias jerárquicas"
    )
    distrito_judicial: str | None = Field(None, description="Distrito judicial de origen")

    # Campos adicionales que pueden venir de la API
    sentencia_sala_id: int | None = None
    sentencia_sentido_id: int | None = None
    sentencia_tipo_id: int | None = None

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────────────────────────────────────
# Expediente enriquecido (después de curación)
# ─────────────────────────────────────────────────────────────────────────


class ExpedienteEnriquecido(BaseModel):
    """Expediente después de pasar por el Agente de Curación.

    Contiene las columnas del esquema xlsx histórico, con campos
    derivados y transformados.
    """

    NEXPEDIENTE: str = Field(..., description="Número de expediente normalizado")
    PUB_PAGWEB: str | None = Field(None, description="Fecha de publicación web")
    SALA: str | None = Field(None, description="Sala del TC")
    FALLO: str | None = Field(None, description="Sentido del fallo")
    TIPO_RESOLUCION: str | None = Field(None, description="Tipo de resolución (etiqueta)")
    CDES_TIPOPROCESO: str | None = Field(None, description="Tipo de proceso constitucional")
    MATERIA: str | None = Field(None, description="Materia (nivel 1)")
    SUB_MATERIA: str | None = Field(None, description="Sub-materia (nivel 2)")
    ESPECIFICA: str | None = Field(None, description="Específica (nivel 3)")
    DEPARTAMENTO: str | None = Field(None, description="Departamento (derivado de distrito judicial)")
    DEMANDANTE: str | None = Field(None, description="Nombre del demandante")
    DEMANDADO: str | None = Field(None, description="Nombre del demandado")
    FUNDAMENTOS: str | None = Field(None, description="Texto de los fundamentos")
    url_archivo: str | None = Field(None, description="URL al PDF")

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────────────────────────────────────
# Paginación
# ─────────────────────────────────────────────────────────────────────────


class PaginatedResponse(BaseModel):
    """Wrapper genérico de paginación para respuestas de la API."""

    data: list[dict[str, Any]] = Field(default_factory=list, description="Lista de resultados")
    total: int = Field(0, description="Total de registros")
    page: int = Field(1, description="Página actual")
    page_size: int = Field(20, description="Tamaño de página")
    total_pages: int = Field(0, description="Total de páginas")


# ─────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Estado de cada componente (parquet, sqlite, etc.)"
    )


# ─────────────────────────────────────────────────────────────────────────
# Pipeline Stats
# ─────────────────────────────────────────────────────────────────────────


class PipelineStats(BaseModel):
    """Estadísticas del pipeline de extracción."""

    total_expedientes: int = 0
    expedientes_parquet: int = 0
    expedientes_manifest: dict[str, int] = Field(default_factory=dict)
    cobertura_temporal: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Stubs para Fase 2+ (RAG & Predictivo)
# ─────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request para consulta al Agente RAG (Fase 3)."""

    query: str = Field(..., description="Pregunta en lenguaje natural")
    top_k: int = Field(5, description="Número de fragmentos a recuperar")
    filters: dict[str, Any] | None = Field(
        None, description="Filtros de metadata (sala, materia, sentido, etc.)"
    )


class BriefResponse(BaseModel):
    """Respuesta del Agente RAG con el brief ejecutivo (Fase 3)."""

    query: str
    brief: str = Field(..., description="Brief ejecutivo generado por el LLM")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Fragmentos fuente utilizados"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrediccionRequest(BaseModel):
    """Request para el Agente Predictivo (Fase 4)."""

    numero_expediente: str | None = None
    tipo_proceso: str | None = None
    sala: str | None = None
    materia: str | None = None
    departamento: str | None = None
    features: dict[str, Any] | None = Field(
        None, description="Features calculados para predicción"
    )


class PrediccionResponse(BaseModel):
    """Respuesta del Agente Predictivo (Fase 4)."""

    prediccion: str = Field(..., description="Fallo predicho")
    probabilidades: dict[str, float] = Field(
        default_factory=dict, description="Probabilidad por clase"
    )
    modelo: str = Field("", description="Modelo utilizado")
    confianza: float = Field(0.0, description="Confianza de la predicción")
