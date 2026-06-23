"""
tc_pipeline/scraping/downloader.py
──────────────────────────────────
Administrador de descargas concurrentes de PDFs.

Responsabilidades:
- Recibir lista de expedientes (dicts con _source)
- Construir rutas: sentencia-raw/YYYY/{id_interno}.pdf o
  auto-resolucion-raw/YYYY/{id_interno}.pdf
- Verificar existencia antes de descargar
- Descargas concurrentes con ThreadPoolExecutor
- Retornar DownloadMetrics con contadores
- Crear carpetas automáticamente
- Mantener registro ID interno ↔ numero_expediente

NO hace:
- Consultar la API
- Registrar manifiestos
"""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from tc_pipeline.cleaning.mapping import classify_document_type
from tc_pipeline.config import PipelineConfig
from tc_pipeline.scraping.api_client import TribunalAPIClient

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
    id_interno: str
    status: str  # "descargado" | "existente" | "invalido" | "error_..."
    path: Path | None = None
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Descargador
# ─────────────────────────────────────────────────────────────────────────


class PDFDownloader:
    """Descargador concurrente de PDFs de sentencias del TC.

    Soporta dos modos:
    - **Legacy**: Descarga por periodo (YYYY-MM) a ``EXPEDIENTES/YYYY/MM/``
    - **Nuevo**: Descarga por tipo a ``sentencia-raw/YYYY/`` o
      ``auto-resolucion-raw/YYYY/`` usando el ID interno como nombre.

    Args:
        config: Configuración del pipeline.  Si no se provee, se usan
                los valores por defecto de ``PipelineConfig``.

    Example:
        >>> downloader = PDFDownloader()
        >>> metrics = downloader.download_year(items, records, 2025)
        >>> print(f"Descargados: {metrics.descargados}")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        # Registro interno: id_interno → numero_expediente
        self._id_map: dict[str, str] = {}

    @property
    def id_map(self) -> dict[str, str]:
        """Mapeo id_interno → numero_expediente acumulado."""
        return dict(self._id_map)

    # ── Construcción de rutas ──────────────────────────

    def build_path(
        self,
        id_interno: str,
        doc_type: str,
        year: int,
    ) -> Path:
        """Construye la ruta de destino para un PDF (nuevo paradigma).

        Los archivos se nombran con el ID interno único y se organizan
        por tipo de documento y año.

        Args:
            id_interno: Identificador único del registro.
            doc_type: ``"sentencia"`` o ``"auto-resolucion"``.
            year: Año del registro.

        Returns:
            Path: ``sentencia-raw/YYYY/{id_interno}.pdf`` o
                  ``auto-resolucion-raw/YYYY/{id_interno}.pdf``
        """
        if doc_type == "auto-resolucion":
            root = self._config.auto_resolucion_raw_root
        else:
            root = self._config.sentencia_raw_root

        return root / str(year) / f"{id_interno}.pdf"

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

    # ── Procesamiento de un item ───────────────────────

    def _process_item(
        self,
        item: dict[str, Any],
        record: dict[str, Any],
        year: int,
    ) -> _DownloadResult:
        """Procesa un item con el nuevo paradigma de rutas por tipo.

        Args:
            item: Dict original de ``data[]`` con ``_source``.
            record: Dict procesado con campos del CSV.
            year: Año de agrupación.

        Returns:
            _DownloadResult con el status de la operación.
        """
        source = item.get("_source", {})
        expediente = record.get("numero_expediente", "")
        url_pdf = source.get("url_archivo", "") or record.get("url_archivo", "")
        if url_pdf and not url_pdf.startswith("http"):
            url_pdf = f"https://{url_pdf}"
        id_interno = record.get("id_interno", "")
        doc_type = record.get("_doc_type", "sentencia")

        if not url_pdf or not id_interno:
            return _DownloadResult(
                expediente=expediente or "desconocido",
                id_interno=id_interno or "sin_id",
                status="invalido",
                error="URL o ID interno faltante",
            )

        # Registrar mapeo ID → expediente
        if id_interno and expediente:
            self._id_map[id_interno] = expediente

        dest = self.build_path(id_interno, doc_type, year)
        status = self.download_pdf(url_pdf, dest)

        error = status if "error" in status else None
        return _DownloadResult(
            expediente=expediente,
            id_interno=id_interno,
            status=status,
            path=dest if status == "descargado" else None,
            error=error,
        )

    # ── Descarga en lote ───────────────────────────────

    def download_batch(
        self,
        items: list[dict[str, Any]],
        records: list[dict[str, Any]],
        year: int,
        progress_bar: tqdm | None = None,
    ) -> DownloadMetrics:
        """Descarga un lote de PDFs separados por tipo de documento.

        Los archivos se nombran con el ID interno y se organizan en
        ``sentencia-raw/YYYY/`` y ``auto-resolucion-raw/YYYY/``.

        Args:
            items: Lista de items originales de ``data[]``.
            records: Lista de records procesados (del CSV mapping).
            year: Año de agrupación.
            progress_bar: Barra tqdm externa.

        Returns:
            DownloadMetrics con contadores de resultados.
        """
        metrics = DownloadMetrics()

        if not items or not records:
            return metrics

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_item, item, record, year
                ): (item, record)
                for item, record in zip(items, records)
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
                        f"     [X] Falló: {result.expediente} "
                        f"(ID: {result.id_interno}) ({result.status})"
                    )

                if progress_bar is not None:
                    progress_bar.update(1)

        return metrics

    def download_year(
        self,
        items: list[dict[str, Any]],
        records: list[dict[str, Any]],
        year: int,
        show_progress: bool = True,
    ) -> DownloadMetrics:
        """Descarga todos los PDFs de un año, separados por tipo.

        Wrapper de conveniencia que crea barras de progreso y
        delega a ``download_batch_v2``.

        Args:
            items: Lista de items originales de la API.
            records: Lista de records procesados con CSV mapping.
            year: Año.
            show_progress: Si True, muestra barra tqdm.

        Returns:
            DownloadMetrics con contadores finales.
        """
        if not items:
            logger.info("Año %d: sin items para descargar.", year)
            return DownloadMetrics()

        logger.info(
            "Año %d: descargando %d PDFs con %d workers.",
            year,
            len(items),
            self._config.max_workers,
        )

        # Asegurar directorios
        (self._config.sentencia_raw_root / str(year)).mkdir(
            parents=True, exist_ok=True
        )
        (self._config.auto_resolucion_raw_root / str(year)).mkdir(
            parents=True, exist_ok=True
        )

        if show_progress:
            with tqdm(
                total=len(items),
                desc=f"Descargando {year}",
                unit="pdf",
            ) as pbar:
                metrics = self.download_batch(
                    items, records, year, progress_bar=pbar
                )
        else:
            metrics = self.download_batch(items, records, year)

        logger.info(
            "Año %d completado: %d descargados, %d existentes, %d errores.",
            year,
            metrics.descargados,
            metrics.existentes,
            metrics.errores,
        )
        return metrics

    # ── Persistencia del mapeo ID ↔ expediente ───────────────────────────

    def save_id_map(self, year: int) -> Path:
        """Guarda el mapeo id_interno → numero_expediente a JSON.

        Args:
            year: Año para el nombre del archivo.

        Returns:
            Path al archivo JSON generado.
        """
        map_path = self._config.csv_output_root / f"id_map-{year}.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)

        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self._id_map, f, ensure_ascii=False, indent=2)

        logger.info(
            "Mapeo ID guardado: %s (%d entradas)",
            map_path,
            len(self._id_map),
        )
        return map_path

    def load_id_map(self, year: int) -> dict[str, str]:
        """Carga el mapeo id_interno → numero_expediente desde JSON.

        Args:
            year: Año del archivo a cargar.

        Returns:
            Dict id_interno → numero_expediente.
        """
        map_path = self._config.csv_output_root / f"id_map-{year}.json"

        if not map_path.exists():
            logger.warning("Archivo de mapeo no encontrado: %s", map_path)
            return {}

        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._id_map.update(data)
        logger.info(
            "Mapeo ID cargado: %s (%d entradas)",
            map_path,
            len(data),
        )
        return data

    def clear_id_map(self) -> None:
        """Limpia el mapeo interno ID ↔ expediente."""
        self._id_map.clear()
