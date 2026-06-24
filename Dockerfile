# ─────────────────────────────────────────────────────────────────────────
# Dockerfile — Pipeline TC (API + Workers)
# ─────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Variables de entorno para optimizar Python en Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema esenciales para compilación y pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias de Python ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ────────────────────────────────────────────────────────
COPY tc_pipeline/ tc_pipeline/
COPY scripts/ scripts/

# ── Crear estructura de directorios activa y RAG ─────────────────────────
# Las carpetas internas del pipeline se autogeneran en el lifespan de FastAPI,
# aquí aseguramos el almacenamiento persistente de Chroma y datos unificados.
RUN mkdir -p \
    data/merged \
    data/raw \
    data/chroma_storage

# ── Exponer puerto de la API ─────────────────────────────────────────────
EXPOSE 8000

# ── Comando por defecto: API FastAPI ─────────────────────────────────────
CMD ["uvicorn", "tc_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]