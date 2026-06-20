"""
tc_pipeline/scraping/api_client.py
──────────────────────────────────
Cliente centralizado para comunicación con la API del Tribunal Constitucional.

Migra desde scraper.py:
- URL_API  (L19)
- HEADERS  (L20-23)
- requests.get() con parseo de paginación  (L83-97)

Responsabilidades exclusivas:
- Construir requests con headers y params
- Timeout separado (connect=5s, read=20s)
- Retries exponenciales (4 intentos: 1s, 2s, 4s, 8s)
- Parsear JSON → APIResponse dataclass
- Validar status_code y lanzar excepciones claras
- Logging estructurado

NO hace:
- Descargar PDFs
- Escribir archivos
- Registrar manifiestos
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

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
    ) -> dict[str, Any]:
        """Ejecuta un GET a la API con reintentos exponenciales.

        Solo reintenta en códigos: 429, 500, 502, 503, 504.
        Lanza inmediatamente en: 400, 401, 403, 404.

        Args:
            params: Parámetros de query string para la petición.

        Returns:
            Diccionario parseado del JSON de respuesta.

        Raises:
            APINonRetryableError: Si el código HTTP no es reintentable.
            APIRetryExhaustedError: Si se agotan todos los reintentos.
            APIError: Si hay un error inesperado de conexión.
        """
        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                response = self.session.get(
                    self._config.api_url,
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

    # ── Métodos públicos ─────────────────────────────────────────────────

    def fetch_page(
        self,
        fecha_publicacion: str,
        page: int = 1,
        extra_filters: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Consulta una página específica de resultados para un periodo.

        Args:
            fecha_publicacion: Periodo en formato ``"YYYY-MM"``.
            page: Número de página (1-indexed).
            extra_filters: Filtros adicionales para la query string.

        Returns:
            APIResponse con los datos de la página solicitada.

        Raises:
            APIError: Si la API responde con error.
        """
        params: dict[str, Any] = {
            "fecha_publicacion": fecha_publicacion,
            "page": page,
        }
        if extra_filters:
            params.update(extra_filters)

        logger.debug(
            "Consultando API: periodo=%s, página=%d",
            fecha_publicacion,
            page,
        )
        raw = self._request_with_retry(params)
        response = self._parse_response(raw)

        logger.debug(
            "Respuesta: %d items en página %d/%d (total: %d)",
            len(response.data),
            page,
            response.total_pages,
            response.total_items,
        )
        return response

    def fetch_period(self, periodo: str) -> list[dict[str, Any]]:
        """Descarga todas las páginas de un periodo dado.

        Itera página por página hasta cubrir ``total_pages``, acumulando
        todos los expedientes.  Respeta ``page_delay`` entre peticiones.

        Args:
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            Lista de todos los expedientes del periodo.
        """
        all_items: list[dict[str, Any]] = []
        first_page = self.fetch_page(periodo, page=1)

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
            response = self.fetch_page(periodo, page=page)
            all_items.extend(response.data)

        logger.info(
            "Periodo %s completado: %d items recolectados.",
            periodo,
            len(all_items),
        )
        return all_items

    def fetch_until_end(self, periodo: str) -> list[dict[str, Any]]:
        """Descarga páginas de un periodo hasta que no haya más datos.

        Similar a ``fetch_period``, pero no depende de ``total_pages``
        del primer response — sigue consultando mientras haya datos.
        Útil cuando la paginación de la API no es confiable.

        Args:
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            Lista acumulada de todos los expedientes del periodo.
        """
        all_items: list[dict[str, Any]] = []
        page = 1

        while True:
            response = self.fetch_page(periodo, page=page)

            if not response.data:
                logger.debug(
                    "Periodo %s: página %d vacía, finalizando.",
                    periodo,
                    page,
                )
                break

            all_items.extend(response.data)

            if page >= response.total_pages:
                break

            page += 1
            time.sleep(self._config.page_delay)

        logger.info(
            "Periodo %s (fetch_until_end): %d items recolectados.",
            periodo,
            len(all_items),
        )
        return all_items

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
