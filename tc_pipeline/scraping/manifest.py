"""
tc_pipeline/scraping/manifest.py
────────────────────────────────
Persistencia operacional del pipeline de scraping con SQLite.

Módulo completamente nuevo — no existía en scraper.py.

Responsabilidades exclusivas:
- Checkpoint de descargas (qué se descargó, qué falló)
- Reanudación tras interrupciones
- Auditoría de operaciones

Ubicación de la BD: data/manifests/pipeline_state.db

NO hace:
- Descargar PDFs
- Llamar a la API
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS download_manifest (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    expediente          TEXT    NOT NULL UNIQUE,
    periodo             TEXT    NOT NULL,
    estado              TEXT    NOT NULL DEFAULT 'pending',
    ruta_pdf            TEXT,
    error               TEXT,
    fecha               TEXT    NOT NULL,
    ultima_actualizacion TEXT   NOT NULL
);
"""

_CREATE_INDEX_PERIODO = """
CREATE INDEX IF NOT EXISTS idx_manifest_periodo
    ON download_manifest (periodo);
"""

_CREATE_INDEX_ESTADO = """
CREATE INDEX IF NOT EXISTS idx_manifest_estado
    ON download_manifest (estado);
"""


# ─────────────────────────────────────────────────────────────────────────
# Repositorio
# ─────────────────────────────────────────────────────────────────────────


class ManifestRepository:
    """Repositorio SQLite para tracking del estado de descargas.

    Gestiona la tabla ``download_manifest`` que registra cada expediente
    descargado, fallido o pendiente, permitiendo reanudación y auditoría.

    Args:
        db_path: Ruta al archivo de base de datos SQLite.
                 Se crean directorios intermedios automáticamente.

    Example:
        >>> manifest = ManifestRepository(Path("data/manifests/pipeline_state.db"))
        >>> manifest.initialize()
        >>> manifest.register_success("01234-2025-AA", "2025-01", "EXPEDIENTES/2025/01/01234-2025-AA.pdf")
        >>> manifest.already_processed("01234-2025-AA")
        True
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ── Conexión ─────────────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        """Obtiene o crea la conexión a SQLite.

        Returns:
            Conexión configurada con row_factory para acceso por nombre.
        """
        if self._conn is None:
            # Crear directorios padre si no existen
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            # Habilitar WAL para mejor concurrencia de lectura
            self._conn.execute("PRAGMA journal_mode=WAL;")
            logger.debug("Conexión SQLite abierta: %s", self._db_path)
        return self._conn

    # ── Inicialización ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Crea la tabla ``download_manifest`` si no existe.

        También crea índices sobre ``periodo`` y ``estado`` para
        optimizar las consultas de reanudación.
        """
        conn = self._get_connection()
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX_PERIODO)
        conn.execute(_CREATE_INDEX_ESTADO)
        conn.commit()
        logger.info(
            "Manifiesto inicializado en: %s", self._db_path
        )

    # ── Consultas ────────────────────────────────────────────────────────

    def already_processed(self, expediente: str) -> bool:
        """Verifica si un expediente ya fue procesado exitosamente.

        Args:
            expediente: Número de expediente a consultar.

        Returns:
            True si el expediente tiene estado ``"success"``.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM download_manifest "
            "WHERE expediente = ? AND estado = 'success' "
            "LIMIT 1;",
            (expediente,),
        )
        return cursor.fetchone() is not None

    def get_pending(self, periodo: str) -> list[dict[str, Any]]:
        """Obtiene expedientes pendientes o fallidos de un periodo.

        Útil para reanudación: retorna los expedientes que aún no
        se han descargado exitosamente.

        Args:
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            Lista de dicts con campos de la tabla manifest.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, expediente, periodo, estado, ruta_pdf, error, "
            "       fecha, ultima_actualizacion "
            "FROM download_manifest "
            "WHERE periodo = ? AND estado != 'success' "
            "ORDER BY id;",
            (periodo,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_failed(self) -> list[dict[str, Any]]:
        """Obtiene todos los expedientes con estado ``"failed"``.

        Útil para el flag ``--retry-failed`` del orquestador.

        Returns:
            Lista de dicts con campos de la tabla manifest.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, expediente, periodo, estado, ruta_pdf, error, "
            "       fecha, ultima_actualizacion "
            "FROM download_manifest "
            "WHERE estado = 'failed' "
            "ORDER BY periodo, id;",
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Registros ────────────────────────────────────────────────────────

    @staticmethod
    def _now_iso() -> str:
        """Retorna timestamp ISO 8601 en UTC."""
        return datetime.now(timezone.utc).isoformat()

    def register_success(
        self,
        expediente: str,
        periodo: str,
        ruta_pdf: str,
    ) -> None:
        """Registra una descarga exitosa.

        Usa ``INSERT OR REPLACE`` para actualizar registros previos
        (ej: un expediente que antes falló y ahora se descargó).

        Args:
            expediente: Número de expediente.
            periodo: Periodo en formato ``"YYYY-MM"``.
            ruta_pdf: Ruta relativa al PDF descargado.
        """
        now = self._now_iso()
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO download_manifest "
            "  (expediente, periodo, estado, ruta_pdf, error, fecha, ultima_actualizacion) "
            "VALUES (?, ?, 'success', ?, NULL, ?, ?) "
            "ON CONFLICT(expediente) DO UPDATE SET "
            "  estado = 'success', "
            "  ruta_pdf = excluded.ruta_pdf, "
            "  error = NULL, "
            "  ultima_actualizacion = excluded.ultima_actualizacion;",
            (expediente, periodo, ruta_pdf, now, now),
        )
        conn.commit()
        logger.debug("Registrado éxito: %s", expediente)

    def register_failure(
        self,
        expediente: str,
        periodo: str,
        error: str,
    ) -> None:
        """Registra una descarga fallida.

        Args:
            expediente: Número de expediente.
            periodo: Periodo en formato ``"YYYY-MM"``.
            error: Descripción del error.
        """
        now = self._now_iso()
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO download_manifest "
            "  (expediente, periodo, estado, ruta_pdf, error, fecha, ultima_actualizacion) "
            "VALUES (?, ?, 'failed', NULL, ?, ?, ?) "
            "ON CONFLICT(expediente) DO UPDATE SET "
            "  estado = 'failed', "
            "  error = excluded.error, "
            "  ultima_actualizacion = excluded.ultima_actualizacion;",
            (expediente, periodo, error, now, now),
        )
        conn.commit()
        logger.debug("Registrado fallo: %s — %s", expediente, error)

    def register_pending(
        self,
        expediente: str,
        periodo: str,
    ) -> None:
        """Registra un expediente como pendiente (descubierto pero no descargado).

        No sobrescribe registros existentes.

        Args:
            expediente: Número de expediente.
            periodo: Periodo en formato ``"YYYY-MM"``.
        """
        now = self._now_iso()
        conn = self._get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO download_manifest "
            "  (expediente, periodo, estado, fecha, ultima_actualizacion) "
            "VALUES (?, ?, 'pending', ?, ?);",
            (expediente, periodo, now, now),
        )
        conn.commit()

    # ── Operaciones de periodo ───────────────────────────────────────────

    def mark_period_complete(self, periodo: str) -> None:
        """Marca un periodo como completamente procesado.

        Actualiza la ``ultima_actualizacion`` de todos los registros
        exitosos del periodo.  Los registros fallidos mantienen su
        estado para reintentos futuros.

        Args:
            periodo: Periodo en formato ``"YYYY-MM"``.
        """
        now = self._now_iso()
        conn = self._get_connection()
        conn.execute(
            "UPDATE download_manifest "
            "SET ultima_actualizacion = ? "
            "WHERE periodo = ? AND estado = 'success';",
            (now, periodo),
        )
        conn.commit()
        logger.info("Periodo %s marcado como completado.", periodo)

    def get_period_stats(self, periodo: str) -> dict[str, int]:
        """Obtiene estadísticas de un periodo.

        Args:
            periodo: Periodo en formato ``"YYYY-MM"``.

        Returns:
            Dict con contadores: ``{"success": N, "failed": N, "pending": N, "total": N}``.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT estado, COUNT(*) as cantidad "
            "FROM download_manifest "
            "WHERE periodo = ? "
            "GROUP BY estado;",
            (periodo,),
        )
        stats: dict[str, int] = {"success": 0, "failed": 0, "pending": 0}
        for row in cursor.fetchall():
            stats[row["estado"]] = row["cantidad"]
        stats["total"] = sum(stats.values())
        return stats

    # ── Limpieza ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Cierra la conexión a SQLite."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Conexión SQLite cerrada.")

    def __enter__(self) -> ManifestRepository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
