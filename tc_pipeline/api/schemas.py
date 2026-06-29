"""
tc_pipeline/api/schemas.py
──────────────────────────
Esquemas Pydantic — contrato de datos entre agentes del SMA (RAG / Predictivo).

Define los modelos que validan la entrada/salida de:
- API / RAG Input: MotivosDemandaRequest
- Representación unificada: ExpedienteBase
- Agente RAG: QueryRequest → BriefResponse
- Agente Predictivo: PrediccionRequest → PrediccionResponse
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ─── Expediente unificado (Acoplado a las columnas del CSV) ───────────────

class ExpedienteBase(BaseModel):
    """Representación limpia de un expediente según el pipeline consolidado."""

    numero_expediente: str = Field(..., description="Número de expediente (ej: '01234-2025-AA/TC')")
    url_archivo_TC: str | None = Field(None, description="URL del archivo en el servidor del TC")
    url_archivo_original: str | None = Field(None, description="URL de origen del archivo original")
    tipo_expediente: str | None = Field(None, description="Tipo de proceso (Amparo, Habeas Corpus, etc.)")
    motivos_demanda: str | None = Field(None, description="Resumen o texto de los motivos de la demanda")
    sentido_resolucion: str | None = Field(None, description="Sentido del fallo (Fundada, Infundada, etc.)")
    fundamentos: str | None = Field(None, description="Texto limpio extraído de los fundamentos de la sentencia")


class ExpedienteAPI(ExpedienteBase):
    """Expediente tal como llega del endpoint de la API del TC."""

    nombre_demandante: str | None = Field(None, description="Nombre del demandante")
    nombre_demandado: str | None = Field(None, description="Nombre del demandado")
    url_archivo: str | None = Field(None, description="URL al PDF de la sentencia")
    fundamentos: str | None = Field(None, description="Texto de los fundamentos")
    palabras: list[dict[str, Any]] | None = Field(
        None, description="Palabras clave / materias jerárquicas"
    )
    distrito_judicial: str | None = Field(None, description="Distrito judicial de origen")

    sentencia_sala_id: int | None = None
    sentencia_sentido_id: int | None = None
    sentencia_tipo_id: int | None = None

    model_config = {"extra": "allow"}


# ─── Inputs de Usuario / Frontend ─────────────────────────────────────────

class MotivosDemandaRequest(BaseModel):
    """Input desde la interfaz de usuario para recibir la descripción del caso."""

    motivos: str = Field(
        ..., 
        min_length=10,
        description="Descripción detallada de los motivos por los que el usuario interpone o evalúa la demanda."
    )


# ─── Health check ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Respuesta del endpoint /health optimizada para la infraestructura RAG."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Estado de los componentes activos (chroma_storage, vector_model)"
    )


# ─── Esquemas de Operación de Agentes (RAG & Predictivo) ──────────────────

class QueryRequest(BaseModel):
    """Request para consulta al Agente RAG (Fase 3)."""

    query: str = Field(..., description="Pregunta en lenguaje natural")
    top_k: int = Field(5, description="Número de fragmentos a recuperar")
    filters: dict[str, Any] | None = Field(None, description="Filtros por metadata (tipo_expediente, sentido_resolucion)")


class BriefResponse(BaseModel):
    """Respuesta del Agente RAG con el brief ejecutivo (Fase 3)."""

    query: str
    brief: str = Field(..., description="Brief ejecutivo generado por el LLM")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Fragmentos fuente utilizados")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrediccionRequest(BaseModel):
    """Request para el Agente Predictivo (Fase 4)."""

    numero_expediente: str | None = None
    tipo_expediente: str | None = None
    motivos_demanda: str = Field(..., description="Texto de los motivos de la demanda a vectorizar y clasificar")
    features: dict[str, Any] | None = Field(None, description="Features calculados para predicción")


class PrediccionResponse(BaseModel):
    """Respuesta del Agente Predictivo (Fase 4)."""

    prediccion: str = Field(..., description="Fallo predicho")
    probabilidades: dict[str, float] = Field(default_factory=dict, description="Probabilidad por clase")
    modelo: str = Field("", description="Modelo utilizado")
    confianza: float = Field(0.0, description="Confianza de la predicción")


# ─── Schemas Combinados (RAG + Predictivo) ────────────────────────────────

class AnalisisCompletoRequest(BaseModel):
    """Request combinado para el análisis completo (RAG + Predictivo)."""

    tipo_demanda: str = Field(
        ...,
        min_length=20,
        description="Descripción de la acción constitucional que se quiere interponer (ej: Acción de Amparo por vulneración al debido proceso)"
    )
    motivos: str = Field(
        ...,
        min_length=20,
        description="Argumentos y fundamentos fácticos/jurídicos del peticionario."
    )


class AnalisisCompletoResponse(BaseModel):
    """Respuesta unificada del agente constitucional (RAG + Predictivo)."""

    query: str = Field(..., description="Consulta original realizada")
    brief: str = Field(..., description="Brief ejecutivo generado por el LLM RAG")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Fragmentos fuente recuperados de la base vectorial")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata adicional de la consulta RAG")
    prediccion: str = Field(..., description="Fallo predicho por el modelo de ML")
    probabilidades: dict[str, float] = Field(default_factory=dict, description="Probabilidad estimada por cada clase de fallo")
    modelo: str = Field(..., description="Modelo predictivo utilizado")
    confianza: float = Field(..., description="Nivel de confianza de la predicción (0.0 a 1.0)")


# ─── Pipeline de Scraping Masivo ─────────────────────────────────────────

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

