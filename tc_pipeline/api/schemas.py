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


# ─────────────────────────────────────────────────────────────────────────
# Pipeline de Scraping Masivo
# ─────────────────────────────────────────────────────────────────────────


class ScrapingRequest(BaseModel):
    """Request para iniciar el scraping masivo."""

    start_year: int = Field(
        1992,
        ge=1992,
        le=2026,
        description="Año de inicio",
    )
    end_year: int = Field(
        2026,
        ge=1992,
        le=2026,
        description="Año de fin",
    )
    phases: list[str] = Field(
        default_factory=lambda: ["json", "download", "extract"],
        description="Fases a ejecutar: json, download, extract",
    )


class ScrapingProgress(BaseModel):
    """Progreso de una tarea de scraping."""

    current_year: int | None = Field(None, description="Año en proceso")
    current_phase: str | None = Field(None, description="Fase actual")
    years_done: int = Field(0, description="Años completados")
    total_years: int = Field(0, description="Total de años a procesar")
    records_processed: int = Field(0, description="Registros procesados")
    pdfs_downloaded: int = Field(0, description="PDFs descargados")
    pdfs_extracted: int = Field(0, description="PDFs con texto extraído")


class ScrapingStatus(BaseModel):
    """Estado de una tarea de scraping."""

    task_id: str = Field(..., description="ID de la tarea")
    status: str = Field(
        "pending",
        description="Estado: pending, running, completed, failed, cancelled",
    )
    progress: ScrapingProgress = Field(
        default_factory=ScrapingProgress,
        description="Progreso actual",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errores acumulados",
    )
    started_at: str | None = Field(None, description="Timestamp de inicio")
    finished_at: str | None = Field(None, description="Timestamp de fin")


class DatasetInfo(BaseModel):
    """Información de un CSV generado."""

    filename: str
    path: str
    year: int
    doc_type: str
    size_bytes: int = 0


class DatasetListResponse(BaseModel):
    """Lista de datasets/CSVs disponibles."""

    json_csvs: list[DatasetInfo] = Field(
        default_factory=list, description="CSVs de metadata JSON"
    )
    sentencia_csvs: list[DatasetInfo] = Field(
        default_factory=list, description="CSVs de texto de sentencias"
    )
    auto_csvs: list[DatasetInfo] = Field(
        default_factory=list, description="CSVs de texto de autos/resoluciones"
    )
    total_files: int = Field(0, description="Total de archivos")

