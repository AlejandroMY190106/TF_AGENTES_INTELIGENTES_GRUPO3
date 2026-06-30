"""
tc_pipeline/scraping/api_client.py
──────────────────────────────────
Cliente centralizado para comunicación con la API del Tribunal Constitucional.

Responsabilidades:
- Construir requests con headers y params
- Timeout separado (connect=5s, read=20s)
- Retries exponenciales (4 intentos: 1s, 2s, 4s, 8s)
- Parsear JSON → APIResponse dataclass
- Validar status_code y lanzar excepciones claras
- Logging estructurado
- Extraer registros listos para CSV anual (nuevo paradigma)
- Agrupar por año con fallback fecha_publicacion → fecha_sentencia

NO hace:
- Descargar PDFs
- Escribir archivos directamente (delega a csv.DictWriter / pandas)
- Registrar manifiestos
"""

from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from tc_pipeline.cleaning.mapping import (
    apply_csv_mapping,
    classify_document_type,
    extract_year_from_record,
)
from tc_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Excepciones
# ─────────────────────────────────────────────────────────────────────────


class APIError(Exception):
    """Error genérico de la API del Tribunal Constitucional."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIRetryExhaustedError(APIError):
    """Todos los reintentos se agotaron."""


class APINonRetryableError(APIError):
    """Error HTTP que no admite reintento (400, 401, 403, 404)."""


# ─────────────────────────────────────────────────────────────────────────
# Dataclass de respuesta
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class APIResponse:
    """Respuesta normalizada de una página de la API.

    Attributes:
        data: Lista de expedientes (dicts con estructura ``_source``).
        total_pages: Número total de páginas para la consulta.
        total_items: Número total de registros en la consulta.
    """

    data: list[dict[str, Any]]
    total_pages: int
    total_items: int


# ─────────────────────────────────────────────────────────────────────────
# Columnas del CSV anual
# ─────────────────────────────────────────────────────────────────────────

CSV_COLUMNS: list[str] = [
    "id_interno",
    "numero_sentencia",
    "numero_expediente",
    "url_archivo",
    "sentencia_sala",
    "sentencia_distrito",
    "tipo_expediente",
    "sentido_resolucion",
    "nombre_demandante",
    "nombre_demandado",
    "fecha_publicacion",
    "fecha_sentencia",
    "fundamentos",
]


# ─────────────────────────────────────────────────────────────────────────
# Cliente
# ─────────────────────────────────────────────────────────────────────────


class TribunalAPIClient:
    """Cliente HTTP para el endpoint cronológico de sentencias del TC.

    Encapsula toda la comunicación con la API, incluyendo paginación,
    timeouts y reintentos exponenciales.

    Args:
        config: Configuración del pipeline.  Si no se provee, se usan
                los valores por defecto de ``PipelineConfig``.

    Example:
        >>> client = TribunalAPIClient()
        >>> response = client.fetch_page("2025-01", page=1)
        >>> print(response.total_items)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._session: requests.Session | None = None

    # ── Gestión de sesión ────────────────────────────────────────────────

    def create_session(self) -> requests.Session:
        """Crea y configura una sesión HTTP reutilizable.

        Configura headers por defecto y devuelve la sesión para
        reutilización de conexiones TCP (keep-alive).

        Returns:
            Sesión de requests configurada con los headers del pipeline.
        """
        session = requests.Session()
        session.headers.update(self._config.headers)
        logger.debug("Sesión HTTP creada con headers: %s", self._config.headers)
        return session

    @property
    def session(self) -> requests.Session:
        """Sesión HTTP lazy — se crea en el primer acceso."""
        if self._session is None:
            self._session = self.create_session()
        return self._session

    # ── Peticiones con reintentos ────────────────────────────────────────

    def _request_with_retry(
        self,
        params: dict[str, Any],
        url: str | None = None,
    ) -> dict[str, Any]:
        """Ejecuta un GET a la API con reintentos exponenciales.

        Solo reintenta en códigos: 429, 500, 502, 503, 504.
        Lanza inmediatamente en: 400, 401, 403, 404.

        Args:
            params: Parámetros de query string para la petición.
             url: URL del endpoint a consultar. Si no se provee,
                  usa ``api_url`` (búsqueda avanzada).

        Returns:
            Diccionario parseado del JSON de respuesta.

        Raises:
            APINonRetryableError: Si el código HTTP no es reintentable.
            APIRetryExhaustedError: Si se agotan todos los reintentos.
            APIError: Si hay un error inesperado de conexión.
        """
        target_url = url or self._config.api_url
        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                response = self.session.get(
                    target_url,
                    params=params,
                    timeout=self._config.api_timeout,
                )

                # ── Éxito ────────────────────────────────────────────────
                if response.status_code == 200:
                    return response.json()

                # ── Error no reintentable ────────────────────────────────
                if response.status_code not in self._config.retryable_status_codes:
                    raise APINonRetryableError(
                        f"API respondió con status {response.status_code} "
                        f"(no reintentable) para params={params}",
                        status_code=response.status_code,
                    )

                # ── Error reintentable ───────────────────────────────────
                delay = self._config.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "API respondió %d (intento %d/%d). "
                    "Reintentando en %.1fs…",
                    response.status_code,
                    attempt + 1,
                    self._config.max_retries,
                    delay,
                )
                last_exception = APIError(
                    f"Status {response.status_code}",
                    status_code=response.status_code,
                )
                time.sleep(delay)

            except requests.exceptions.RequestException as exc:
                delay = self._config.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Error de conexión (intento %d/%d): %s. "
                    "Reintentando en %.1fs…",
                    attempt + 1,
                    self._config.max_retries,
                    exc,
                    delay,
                )
                last_exception = exc
                time.sleep(delay)

        raise APIRetryExhaustedError(
            f"Reintentos agotados ({self._config.max_retries} intentos) "
            f"para params={params}. Último error: {last_exception}"
        )

    # ── Parseo de respuesta ──────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: dict[str, Any]) -> APIResponse:
        """Convierte la respuesta JSON cruda en un ``APIResponse`` tipado.

        Args:
            raw: Diccionario parseado del JSON de la API.

        Returns:
            APIResponse con data, total_pages y total_items.
        """
        pagination = raw.get("pagination", {})
        return APIResponse(
            data=raw.get("data", []),
            total_pages=pagination.get("num_pages", 0),
            total_items=pagination.get("total_item", 0),
        )

    # ── Métodos públicos (API avanzada — principal) ──────────────────────

    def fetch_page_avanzada(
        self,
        fecha_publicacion: str,
        page: int = 1,
    ) -> APIResponse:
        """Consulta una página del endpoint de **búsqueda avanzada**.

        Este endpoint devuelve campos enriquecidos: ``sentido``,
        ``sistematizacion``, ``tesaurio``, ``tipo``, etc.

        Args:
            fecha_publicacion: Periodo en formato ``"YYYY-MM"``.
            page: Número de página (1-indexed).

        Returns:
            APIResponse con datos enriquecidos.
        """
        params: dict[str, Any] = {
            "page": page,
            "search": "",
            "numero_expediente": "",
            "nombre_demandante": "",
            "nombre_demandado": "",
            "fecha_publicacion": fecha_publicacion,
            "sentencia_sentido": "",
            "id_sentencia_distrito": "",
            "id_sentencia_sala": "",
            "id_sentencia_tipo": "",
            "palabras_claves": "",
            "palabras": "",
        }

        raw = self._request_with_retry(params)
        return self._parse_response(raw)

    def fetch_period(self, periodo: str) -> list[dict[str, Any]]:
        """Descarga todas las páginas de un periodo usando búsqueda avanzada.

        Usa el endpoint ``/busqueda/avanzada`` que retorna campos
        enriquecidos incluyendo ``sentido`` para el sentido de resolución.

        Args:
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            Lista de todos los expedientes del periodo.
        """
        all_items: list[dict[str, Any]] = []
        first_page = self.fetch_page_avanzada(periodo, page=1)

        if first_page.total_items == 0:
            logger.info("Periodo %s: sin registros.", periodo)
            return all_items

        all_items.extend(first_page.data)
        total_pages = first_page.total_pages

        logger.info(
            "Periodo %s: %d items en %d páginas.",
            periodo,
            first_page.total_items,
            total_pages,
        )

        for page in range(2, total_pages + 1):
            time.sleep(self._config.page_delay)
            response = self.fetch_page_avanzada(periodo, page=page)
            all_items.extend(response.data)

        logger.info(
            "Periodo %s completado: %d items recolectados.",
            periodo,
            len(all_items),
        )
        return all_items

    # ═════════════════════════════════════════════════════════════════════
    #  NUEVO PARADIGMA: Métodos para pipeline CSV anual
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_internal_id(item: dict[str, Any]) -> str:
        """Extrae el ID interno único del item de la API.

        El ``_id`` se encuentra directamente en el item del array ``data``,
        no dentro de ``_source``.

        Args:
            item: Elemento individual de ``data[]``.

        Returns:
            ID interno como string.
        """
        raw_id = item.get("_id", "") or item.get("id", "")
        return str(raw_id).strip() if raw_id else ""

    @staticmethod
    def extract_record_for_csv(item: dict[str, Any]) -> dict[str, Any]:
        """Extrae los campos de un item de la API para exportar a CSV.

        Combina el ``_id`` del nivel superior con los campos de
        ``_source``, aplicando el mapeo CSV de 12 columnas.

        Args:
            item: Elemento individual de ``data[]`` con ``_id`` y ``_source``.

        Returns:
            Dict con las columnas del CSV + metadatos internos.
        """
        internal_id = TribunalAPIClient.extract_internal_id(item)
        source = item.get("_source", {})
        record = apply_csv_mapping(source, internal_id)

        # Metadata adicional para clasificación/routing
        record["_doc_type"] = classify_document_type(source)
        record["_year"] = extract_year_from_record(source)

        return record

    def fetch_year_records(self, year: int) -> list[dict[str, Any]]:
        """Descarga todos los registros de un año completo (12 meses).

        Args:
            year: Año a descargar (ej: 2025).

        Returns:
            Lista de records procesados (listos para CSV).
        """
        all_records: list[dict[str, Any]] = []

        for month in range(1, 13):
            periodo = f"{year}-{month:02d}"
            logger.info("Consultando periodo: %s", periodo)

            try:
                items = self.fetch_period(periodo)
            except APIError as e:
                logger.error("Error en periodo %s: %s", periodo, e)
                continue

            if not items:
                logger.info("Periodo %s: sin registros.", periodo)
                continue

            for item in items:
                record = self.extract_record_for_csv(item)
                all_records.append(record)

            logger.info(
                "Periodo %s: %d registros procesados (acumulado: %d).",
                periodo,
                len(items),
                len(all_records),
            )

            # Pausa cortés entre meses
            time.sleep(self._config.page_delay)

        logger.info(
            "Año %d completado: %d registros totales.",
            year,
            len(all_records),
        )
        return all_records

    def fetch_year_to_csv(self, year: int) -> Path:
        """Descarga un año completo y exporta a CSV.

        Genera el archivo ``expedientes-json-YYYY.csv`` en el directorio
        configurado en ``csv_output_root``.

        Args:
            year: Año a procesar.

        Returns:
            Path al archivo CSV generado.
        """
        records = self.fetch_year_records(year)

        if not records:
            logger.warning("Año %d: sin registros para exportar.", year)
            # Crear CSV vacío con headers
            csv_path = self._config.csv_output_root / f"expedientes-json-{year}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            return csv_path

        return self._write_records_to_csv(records, year)

    def _write_records_to_csv(
        self,
        records: list[dict[str, Any]],
        year: int,
    ) -> Path:
        """Escribe registros a un CSV anual.

        Args:
            records: Lista de dicts con las columnas del CSV.
            year: Año para el nombre del archivo.

        Returns:
            Path al CSV generado.
        """
        csv_path = self._config.csv_output_root / f"expedientes-json-{year}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        logger.info(
            "CSV generado: %s (%d registros)",
            csv_path,
            len(records),
        )
        return csv_path

    def fetch_all_years_to_csv(
        self,
        start_year: int = 1992,
        end_year: int = 2026,
        progress_callback: Any | None = None,
    ) -> list[Path]:
        """Descarga todos los años y genera CSVs anuales.

        Itera de ``end_year`` a ``start_year`` (más reciente primero).

        Args:
            start_year: Año de inicio (inclusive).
            end_year: Año de fin (inclusive).
            progress_callback: Callable opcional ``(year, total_years, records_count) -> None``
                para reportar progreso.

        Returns:
            Lista de Paths a los CSVs generados.
        """
        csv_paths: list[Path] = []
        total_years = end_year - start_year + 1

        for i, year in enumerate(range(end_year, start_year - 1, -1), 1):
            logger.info(
                "[%d/%d] Procesando año: %d",
                i,
                total_years,
                year,
            )

            try:
                csv_path = self.fetch_year_to_csv(year)
                csv_paths.append(csv_path)

                if progress_callback:
                    # Count records in the generated CSV
                    record_count = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
                    progress_callback(year, total_years, max(0, record_count))

            except Exception as e:
                logger.error("Error procesando año %d: %s", year, e)

            # Pausa entre años
            time.sleep(self._config.page_delay * 2)

        logger.info(
            "Pipeline CSV completado: %d CSVs generados (%d-%d).",
            len(csv_paths),
            start_year,
            end_year,
        )
        return csv_paths

    def get_items_with_metadata(
        self,
        year: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Descarga un año y devuelve los items originales + records CSV.

        Útil para el downloader que necesita tanto los items originales
        (con URL de PDF) como los records procesados (con ID interno).

        Args:
            year: Año a procesar.

        Returns:
            Tupla de (items_originales, records_csv).
        """
        all_items: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []

        for month in range(1, 13):
            periodo = f"{year}-{month:02d}"
            try:
                items = self.fetch_period(periodo)
            except APIError as e:
                logger.error("Error en periodo %s: %s", periodo, e)
                continue

            for item in items:
                record = self.extract_record_for_csv(item)
                all_items.append(item)
                all_records.append(record)

            if items:
                time.sleep(self._config.page_delay)

        return all_items, all_records

    # ── Limpieza ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Cierra la sesión HTTP, liberando conexiones."""
        if self._session is not None:
            self._session.close()
            self._session = None
            logger.debug("Sesión HTTP cerrada.")

    def __enter__(self) -> TribunalAPIClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
