# ─────────────────────────────────────────────────────────────────────────
# Dockerfile — Pipeline TC (API + Workers)
# ─────────────────────────────────────────────────────────────────────────
# Imagen multi-propósito: puede ejecutarse como API FastAPI o como
# worker de pipeline (JSON/Download/Extract) según el CMD.
#
# Uso:
#   docker build -t tc-pipeline .
#   docker run -p 8000:8000 tc-pipeline                     # API
#   docker run tc-pipeline python scripts/run_pipeline.py   # Worker
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema (necesarias para pdfplumber/Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ─────────────────────────────────────────────────
COPY requirements.txt .

# Instalar solo las dependencias esenciales para el pipeline
# (excluir torch, tensorflow y otros paquetes pesados de ML)
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv \
    pandas openpyxl pyarrow \
    requests httpx tqdm \
    pdfplumber \
    python-multipart \
    numpy

# ── Código fuente ────────────────────────────────────────────────────────
COPY tc_pipeline/ tc_pipeline/
COPY scripts/ scripts/

# ── Crear estructura de directorios ──────────────────────────────────────
RUN mkdir -p \
    data/csv \
    data/sentencia-raw \
    data/auto-resolucion-raw \
    data/sentencia-Extract \
    data/auto-resolucion-Extract \
    data/raw \
    data/manifests

# ── Exponer puerto ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Comando por defecto: API FastAPI ─────────────────────────────────────
CMD ["uvicorn", "tc_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
