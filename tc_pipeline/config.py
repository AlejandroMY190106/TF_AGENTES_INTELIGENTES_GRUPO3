"""
tc_pipeline/config.py
─────────────────────
Configuración centralizada del pipeline de scraping.

Reemplaza las variables globales de scraper.py (URL_API, HEADERS,
CARPETA_RAIZ_DESCARGAS, MAX_WORKERS, etc.) con un dataclass inmutable
que sirve como única fuente de verdad para todos los módulos.

Uso:
    from tc_pipeline.config import PipelineConfig
    config = PipelineConfig()                        # valores por defecto
    config = PipelineConfig(max_workers=20)          # override parcial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# Códigos HTTP que disparan reintentos automáticos
# ─────────────────────────────────────────────────────────────────────────
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración inmutable del pipeline de scraping del Tribunal Constitucional.

    Attributes:
        api_url: Endpoint de búsqueda cronológica de sentencias.
        user_agent: User-Agent para las peticiones HTTP.
        accept_header: Accept header para las peticiones HTTP.
        connect_timeout: Timeout de conexión en segundos (aplica a API y PDFs).
        api_read_timeout: Timeout de lectura para llamadas a la API.
        pdf_read_timeout: Timeout de lectura para descarga de PDFs.
        max_retries: Número máximo de reintentos por petición fallida.
        retry_base_delay: Delay base en segundos para backoff exponencial (1s, 2s, 4s, 8s).
        page_delay: Pausa entre páginas de la API para no saturar el servidor.
        retryable_status_codes: Códigos HTTP que disparan reintentos.
        download_root: Directorio raíz para PDFs descargados.
        max_workers: Número de workers concurrentes para descargas.
        manifest_db: Ruta a la base de datos SQLite del manifiesto.
    """

    # ── API del Tribunal Constitucional ──────────────────────────────────
    api_url: str = (
        "https://jurisbackend.sedetc.gob.pe"
        "/api/visitor/sentencia/busqueda/cronologico"
    )
    api_url_avanzada: str = (
        "https://jurisbackend.sedetc.gob.pe"
        "/api/visitor/sentencia/busqueda/avanzada"
    )
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    accept_header: str = "application/json"

    # ── Timeouts (segundos) ──────────────────────────────────────────────
    # Separados para fallar rápido en conexión y tolerar PDFs grandes.
    connect_timeout: float = 5.0
    api_read_timeout: float = 20.0
    pdf_read_timeout: float = 45.0

    # ── Política de reintentos ───────────────────────────────────────────
    max_retries: int = 10
    retry_base_delay: float = 5.0
    page_delay: float = 2.5
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: RETRYABLE_STATUS_CODES
    )

    # ── Descargas (legacy) ───────────────────────────────────────────────
    download_root: Path = Path("EXPEDIENTES")
    max_workers: int = 10

    # ── Nuevas rutas de datos (pipeline CSV) ─────────────────────────────
    csv_output_root: Path = Path("data/csv")
    sentencia_raw_root: Path = Path("data/sentencia-raw")
    auto_resolucion_raw_root: Path = Path("data/auto-resolucion-raw")
    sentencia_extract_root: Path = Path("data/sentencia-Extract")
    auto_resolucion_extract_root: Path = Path("data/auto-resolucion-Extract")

    # ── Extracción de PDF ────────────────────────────────────────────────
    pdf_extraction_timeout: float = 30.0  # Timeout por PDF individual (segundos)

    # ── Manifiesto SQLite ────────────────────────────────────────────────
    manifest_db: Path = Path("data/manifests/pipeline_state.db")

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def headers(self) -> dict[str, str]:
        """Headers HTTP para todas las peticiones."""
        return {
            "User-Agent": self.user_agent,
            "Accept": self.accept_header,
        }

    @property
    def api_timeout(self) -> tuple[float, float]:
        """Tupla (connect, read) para peticiones a la API."""
        return (self.connect_timeout, self.api_read_timeout)

    @property
    def pdf_timeout(self) -> tuple[float, float]:
        """Tupla (connect, read) para descarga de PDFs."""
        return (self.connect_timeout, self.pdf_read_timeout)
