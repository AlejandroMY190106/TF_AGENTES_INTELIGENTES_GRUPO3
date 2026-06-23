"""
tc_pipeline.cleaning
────────────────────
Paquete de limpieza y mapeo del pipeline TC.

Exporta:
- apply_csv_mapping: Función principal de mapeo JSON API → esquema CSV
- Constantes de mapeo (SENTENCIA_TIPO_MAP, TIPO_PROCESO_MAP, etc.)
"""

from tc_pipeline.cleaning.mapping import (
    SENTENCIA_TIPO_MAP,
    TIPO_PROCESO_MAP,
    TIPO_EXPEDIENTE_CANONICO,
    SENTIDO_CANONICO,
    apply_csv_mapping,
    classify_document_type,
    extract_sentido_resolucion,
    extract_tipo_expediente,
    extract_tipo_proceso,
    extract_year_from_record,
    resolve_tipo_resolucion,
)

__all__ = [
    "SENTENCIA_TIPO_MAP",
    "TIPO_PROCESO_MAP",
    "TIPO_EXPEDIENTE_CANONICO",
    "SENTIDO_CANONICO",
    "apply_csv_mapping",
    "classify_document_type",
    "extract_sentido_resolucion",
    "extract_tipo_expediente",
    "extract_tipo_proceso",
    "extract_year_from_record",
    "resolve_tipo_resolucion",
]
