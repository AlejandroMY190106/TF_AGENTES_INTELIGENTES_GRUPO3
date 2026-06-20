"""
tc_pipeline.storage
───────────────────
Paquete de almacenamiento del pipeline TC.

Exporta:
- ParquetStore: Persistencia de expedientes en formato Parquet
"""

from tc_pipeline.storage.parquet_store import ParquetStore

__all__ = ["ParquetStore"]
