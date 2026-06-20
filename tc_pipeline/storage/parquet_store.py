"""
tc_pipeline/storage/parquet_store.py
────────────────────────────────────
Persistencia de expedientes en formato Parquet (Apache Arrow).

Responsabilidades exclusivas:
- Guardar DataFrames de expedientes a Parquet
- Cargar Parquet existentes como DataFrame
- Append incremental (merge deduplicado por numero_expediente)
- Estadísticas básicas del dataset almacenado

NO hace:
- Consultar APIs
- Transformar o limpiar datos
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStore:
    """Almacenamiento de expedientes en formato Parquet.

    Proporciona operaciones CRUD básicas sobre archivos Parquet,
    con deduplicación automática por ``numero_expediente``.

    Args:
        path: Ruta al archivo .parquet. Se crean directorios intermedios
              automáticamente si no existen.

    Example:
        >>> store = ParquetStore(Path("data/raw/expedientes_tc.parquet"))
        >>> store.save(df)
        >>> loaded = store.load()
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Ruta al archivo Parquet."""
        return self._path

    @property
    def exists(self) -> bool:
        """Verifica si el archivo Parquet existe."""
        return self._path.exists()

    def save(self, df: pd.DataFrame) -> int:
        """Guarda un DataFrame completo a Parquet (sobrescribe).

        Args:
            df: DataFrame con los expedientes a guardar.

        Returns:
            Número de registros guardados.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path, engine="pyarrow", index=False)
        count = len(df)
        logger.info(
            "Guardados %d expedientes en %s (%.2f MB)",
            count,
            self._path,
            self._path.stat().st_size / (1024 * 1024),
        )
        return count

    def load(self) -> pd.DataFrame:
        """Carga el archivo Parquet como DataFrame.

        Returns:
            DataFrame con los expedientes almacenados.

        Raises:
            FileNotFoundError: Si el archivo no existe.
        """
        if not self.exists:
            raise FileNotFoundError(f"No existe el archivo: {self._path}")

        df = pd.read_parquet(self._path, engine="pyarrow")
        logger.info("Cargados %d expedientes desde %s", len(df), self._path)
        return df

    def append(self, new_df: pd.DataFrame) -> int:
        """Agrega expedientes nuevos, deduplicando por numero_expediente.

        Si el archivo ya existe, carga los datos previos, concatena con
        los nuevos y deduplica.  Si no existe, guarda directamente.

        Args:
            new_df: DataFrame con expedientes nuevos a agregar.

        Returns:
            Número total de registros después del append.
        """
        if self.exists:
            existing = self.load()
            combined = pd.concat([existing, new_df], ignore_index=True)
            if "numero_expediente" in combined.columns:
                before = len(combined)
                combined = combined.drop_duplicates(
                    subset=["numero_expediente"], keep="last"
                )
                dupes = before - len(combined)
                if dupes > 0:
                    logger.info("Eliminados %d duplicados por numero_expediente.", dupes)
        else:
            combined = new_df

        return self.save(combined)

    def get_stats(self) -> dict:
        """Obtiene estadísticas básicas del dataset almacenado.

        Returns:
            Dict con total_expedientes, columnas, rango de años, etc.
        """
        if not self.exists:
            return {"total_expedientes": 0, "exists": False}

        df = self.load()
        stats: dict = {
            "exists": True,
            "total_expedientes": len(df),
            "columnas": list(df.columns),
            "size_mb": round(self._path.stat().st_size / (1024 * 1024), 2),
        }

        # Intentar extraer rango temporal
        if "fecha_publicacion" in df.columns:
            stats["fecha_min"] = str(df["fecha_publicacion"].min())
            stats["fecha_max"] = str(df["fecha_publicacion"].max())

        if "numero_expediente" in df.columns:
            # Extraer año del expediente (formato: NNNNN-YYYY-XX)
            years = df["numero_expediente"].str.extract(
                r"-(\d{4})-", expand=False
            )
            years = pd.to_numeric(years, errors="coerce").dropna().astype(int)
            if len(years) > 0:
                stats["anio_expediente_min"] = int(years.min())
                stats["anio_expediente_max"] = int(years.max())
                stats["expedientes_por_anio"] = (
                    years.value_counts().sort_index().to_dict()
                )

        return stats
