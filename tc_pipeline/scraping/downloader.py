"""
tc_pipeline/scraping/downloader.py
──────────────────────────────────
Administrador de descargas concurrentes de PDFs.

Migra desde scraper.py:
- descargar_un_pdf()  (L38-66)
- ThreadPoolExecutor  (L110-121)
- os.makedirs()       (L29, L35)

Responsabilidades exclusivas:
- Recibir lista de expedientes (dicts con _source)
- Construir rutas con pathlib: EXPEDIENTES/YYYY/MM/expediente.pdf
- Verificar existencia antes de descargar
- Descargas concurrentes con ThreadPoolExecutor
- Retornar DownloadMetrics con contadores
- Crear carpetas automáticamente

NO hace:
- Consultar la API
- Registrar manifiestos
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from tc_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Dataclass de métricas
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class DownloadMetrics:
    """Contadores de resultados de una operación de descarga.

    Attributes:
        descargados: PDFs descargados exitosamente en esta ejecución.
        existentes: PDFs que ya existían en disco (skip).
        errores: PDFs que fallaron al descargar.
        detalles_errores: Lista de tuplas (expediente, error) para diagnóstico.
    """

    descargados: int = 0
    existentes: int = 0
    errores: int = 0
    detalles_errores: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total de items procesados."""
        return self.descargados + self.existentes + self.errores

    def merge(self, other: DownloadMetrics) -> DownloadMetrics:
        """Combina las métricas de otra instancia en esta."""
        self.descargados += other.descargados
        self.existentes += other.existentes
        self.errores += other.errores
        self.detalles_errores.extend(other.detalles_errores)
        return self


# ─────────────────────────────────────────────────────────────────────────
# Resultado individual de descarga
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _DownloadResult:
    """Resultado interno de una descarga individual."""

    expediente: str
    status: str  # "descargado" | "existente" | "invalido" | "error_..."
    path: Path | None = None
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Descargador
# ─────────────────────────────────────────────────────────────────────────


class PDFDownloader:
    """Descargador concurrente de PDFs de sentencias del TC.

    Recibe listas de expedientes (como dicts con estructura ``_source``)
    y los descarga en paralelo usando ``ThreadPoolExecutor``.

    La estructura de directorios generada es::

        EXPEDIENTES/
        └── YYYY/
            └── MM/
                ├── 01234-2025-AA.pdf
                └── 05678-2025-HC.pdf

    Args:
        config: Configuración del pipeline.  Si no se provee, se usan
                los valores por defecto de ``PipelineConfig``.

    Example:
        >>> downloader = PDFDownloader()
        >>> metrics = downloader.download_batch(items, "2025-01")
        >>> print(f"Descargados: {metrics.descargados}")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    # ── Construcción de rutas ────────────────────────────────────────────

    def build_path(self, expediente: str, periodo: str) -> Path:
        """Construye la ruta de destino para un PDF.

        Sanitiza el nombre del expediente reemplazando ``/`` por ``_``
        y eliminando espacios, luego genera la ruta completa.

        Args:
            expediente: Número de expediente (ej: ``"01234-2025-AA"``).
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            Path: ``EXPEDIENTES/YYYY/MM/expediente_limpio.pdf``
        """
        anio, mes = periodo.split("-")
        nombre_limpio = expediente.replace("/", "_").replace(" ", "")
        return self._config.download_root / anio / mes / f"{nombre_limpio}.pdf"

    # ── Descarga individual ──────────────────────────────────────────────

    def download_pdf(self, url: str, dest: Path) -> str:
        """Descarga un PDF individual a la ruta especificada.

        Verifica existencia antes de descargar.  Crea directorios
        intermedios automáticamente si no existen.

        Args:
            url: URL directa al archivo PDF.
            dest: Ruta de destino local.

        Returns:
            Status string: ``"descargado"``, ``"existente"`` o
            ``"error_..."``.
        """
        if dest.exists():
            return "existente"

        # Crear directorios padre si no existen
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(
                url,
                headers=self._config.headers,
                timeout=self._config.pdf_timeout,
            )
            if response.status_code == 200:
                dest.write_bytes(response.content)
                return "descargado"
            else:
                return f"error_status_{response.status_code}"
        except requests.exceptions.Timeout:
            return "error_timeout"
        except requests.exceptions.ConnectionError:
            return "error_conexion"
        except requests.exceptions.RequestException as exc:
            return f"error_{type(exc).__name__}"

    # ── Procesamiento de un item ─────────────────────────────────────────

    def _process_item(
        self,
        item: dict[str, Any],
        periodo: str,
    ) -> _DownloadResult:
        """Procesa un item individual del listado de la API.

        Extrae expediente y URL del dict con estructura ``_source``,
        construye la ruta y ejecuta la descarga.

        Args:
            item: Dict con estructura ``{"_source": {"numero_expediente": ..., "url_archivo": ...}}``.
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            _DownloadResult con el status de la operación.
        """
        source = item.get("_source", {})
        expediente = source.get("numero_expediente", "")
        url_pdf = source.get("url_archivo", "")

        if not expediente or not url_pdf:
            return _DownloadResult(
                expediente=expediente or "desconocido",
                status="invalido",
                error="Expediente o URL faltante",
            )

        dest = self.build_path(expediente, periodo)
        status = self.download_pdf(url_pdf, dest)

        error = status if "error" in status else None
        return _DownloadResult(
            expediente=expediente,
            status=status,
            path=dest if status == "descargado" else None,
            error=error,
        )

    # ── Descarga en lote ─────────────────────────────────────────────────

    def download_batch(
        self,
        items: list[dict[str, Any]],
        periodo: str,
        progress_bar: tqdm | None = None,
    ) -> DownloadMetrics:
        """Descarga un lote de PDFs concurrentemente.

        Usa ``ThreadPoolExecutor`` con ``max_workers`` configurados.
        Opcionalmente actualiza una barra de progreso ``tqdm``.

        Args:
            items: Lista de expedientes (dicts con estructura ``_source``).
            periodo: Periodo en formato ``"YYYY-MM"``.
            progress_bar: Barra tqdm externa para actualizar progreso.

        Returns:
            DownloadMetrics con contadores de resultados.
        """
        metrics = DownloadMetrics()

        if not items:
            return metrics

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {
                executor.submit(self._process_item, item, periodo): item
                for item in items
            }

            for future in as_completed(futures):
                result = future.result()

                if result.status == "descargado":
                    metrics.descargados += 1
                elif result.status == "existente":
                    metrics.existentes += 1
                else:
                    metrics.errores += 1
                    metrics.detalles_errores.append(
                        (result.expediente, result.error or result.status)
                    )
                    tqdm.write(
                        f"     ✗ Falló: {result.expediente} ({result.status})"
                    )

                if progress_bar is not None:
                    progress_bar.update(1)

        return metrics

    def download_period(
        self,
        items: list[dict[str, Any]],
        periodo: str,
        show_progress: bool = True,
    ) -> DownloadMetrics:
        """Descarga todos los PDFs de un periodo con barra de progreso.

        Wrapper de conveniencia sobre ``download_batch`` que crea
        automáticamente una barra ``tqdm`` si se solicita.

        Args:
            items: Lista de expedientes del periodo.
            periodo: Periodo en formato ``"YYYY-MM"``.
            show_progress: Si True, muestra barra de progreso tqdm.

        Returns:
            DownloadMetrics con contadores finales.
        """
        if not items:
            logger.info("Periodo %s: sin items para descargar.", periodo)
            return DownloadMetrics()

        logger.info(
            "Periodo %s: descargando %d PDFs con %d workers.",
            periodo,
            len(items),
            self._config.max_workers,
        )

        # Asegurar que exista el directorio del periodo
        anio, mes = periodo.split("-")
        period_dir = self._config.download_root / anio / mes
        period_dir.mkdir(parents=True, exist_ok=True)

        if show_progress:
            with tqdm(
                total=len(items),
                desc=f"Descargando {periodo}",
                unit="pdf",
            ) as pbar:
                metrics = self.download_batch(items, periodo, progress_bar=pbar)
        else:
            metrics = self.download_batch(items, periodo)

        logger.info(
            "Periodo %s completado: %d descargados, %d existentes, %d errores.",
            periodo,
            metrics.descargados,
            metrics.existentes,
            metrics.errores,
        )
        return metrics
