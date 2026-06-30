"""
tc_pipeline/api/main.py
───────────────────────
Aplicación FastAPI — Sistema Multiagente para análisis de jurisprudencia del TC.

Punto de entrada principal del backend. Configura:
- CORS (para desarrollo frontend)
- Metadata de la API (título, versión, descripción)
- Montaje del router con todos los endpoints
- Eventos de startup/shutdown

Para ejecutar:
    uvicorn tc_pipeline.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from tc_pipeline.api.routes import router
from tc_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ── Ruta al directorio de la interfaz de usuario ────────────────────────
_UI_DIR = Path(__file__).resolve().parents[2] / "src" / "ui"


# ─────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación."""
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("🚀 Iniciando API del Sistema Multiagente TC...")

    # Crear estructura de directorios para el pipeline CSV
    config = PipelineConfig()
    for dir_path in [
        config.csv_output_root,
        config.sentencia_raw_root,
        config.auto_resolucion_raw_root,
        config.sentencia_extract_root,
        config.auto_resolucion_extract_root,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("📁 Estructura de directorios del pipeline verificada.")

    # ── Inicialización asíncrona diferida de Agentes (RAG y Inferencia) ──
    from src.agent.rag_service import RAGService
    from src.agent.predictor_service import PredictorService
    import tc_pipeline.api.routes as routes

    # RAG Service (Groq + ChromaDB)
    try:
        logger.info("Cargando RAGService en lifespan...")
        rag_instance = RAGService()
        app.state.rag_service = rag_instance
        routes._rag_service = rag_instance
        logger.info("✅ RAGService inicializado correctamente.")
    except Exception as exc:
        logger.warning(
            "⚠️ RAGService no pudo inicializarse (ChromaDB no creada/vacia o Groq desconfigurada): %s. "
            "El servicio RAG de la API no estará disponible temporalmente.",
            exc,
        )
        app.state.rag_service = None
        routes._rag_service = None

    # Predictor Service (XGBoost + Embeddings)
    try:
        logger.info("Cargando PredictorService en lifespan...")
        predictor_instance = PredictorService()
        app.state.predictor_service = predictor_instance
        routes._predictor_service = predictor_instance
        logger.info("✅ PredictorService inicializado correctamente.")
    except Exception as exc:
        logger.warning(
            "⚠️ PredictorService no pudo inicializarse (Falta modelo o codificador en models/): %s. "
            "El servicio Predictor de la API no estará disponible temporalmente.",
            exc,
        )
        app.state.predictor_service = None
        routes._predictor_service = None

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("🛑 API detenida.")


# ─────────────────────────────────────────────────────────────────────────
# Aplicación FastAPI
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sistema Multiagente TC — Análisis de Jurisprudencia",
    description=(
        "API backend para el sistema multiagente de análisis de jurisprudencia "
        "del Tribunal Constitucional del Perú.\n\n"
        "**Agentes del sistema:**\n"
        "- 🔧 **Agente de Curación**: Pipeline de extracción, limpieza e imputación\n"
        "- 🔍 **Agente RAG**: Recuperación semántica y generación de briefs ejecutivos\n"
        "- 📊 **Agente Predictivo**: Predicción del sentido del fallo\n\n"
        "**Estado actual:** Fase 4 — Agentes RAG y Predictivo activos (tolerantes a fallos de arranque)"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Montar router ────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Montar interfaz de usuario como archivos estáticos ───────────────────
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
    logger.info("🌐 UI montada en /ui → %s", _UI_DIR)
else:
    logger.warning("⚠️ Directorio UI no encontrado en %s. La interfaz no estará disponible en /ui.", _UI_DIR)


# ── Ruta raíz ────────────────────────────────────────────────────────────


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Redirige a la interfaz de usuario si está disponible."""
    if _UI_DIR.exists():
        return RedirectResponse(url="/ui")
    return {"service": "Sistema Multiagente TC", "version": "0.1.0", "docs": "/docs"}
