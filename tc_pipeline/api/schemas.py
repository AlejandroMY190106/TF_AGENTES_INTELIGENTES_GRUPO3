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
