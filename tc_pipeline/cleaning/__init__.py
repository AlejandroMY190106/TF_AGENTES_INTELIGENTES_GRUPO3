"""
tc_pipeline.cleaning
────────────────────
Paquete de limpieza y mapeo del pipeline TC.

Exporta:
- apply_mapping: Función principal de mapeo JSON API → esquema xlsx
- Constantes de mapeo (FIELD_MAP, SENTENCIA_TIPO_MAP, etc.)
"""

from tc_pipeline.cleaning.mapping import (
    DISTRITO_JUDICIAL_DEPARTAMENTO,
    FIELD_MAP,
    SENTENCIA_TIPO_MAP,
    TIPO_PROCESO_MAP,
    apply_mapping,
    extract_materias,
    extract_tipo_proceso,
    resolve_departamento,
    resolve_tipo_resolucion,
)

__all__ = [
    "apply_mapping",
    "FIELD_MAP",
    "SENTENCIA_TIPO_MAP",
    "TIPO_PROCESO_MAP",
    "DISTRITO_JUDICIAL_DEPARTAMENTO",
    "extract_tipo_proceso",
    "extract_materias",
    "resolve_departamento",
    "resolve_tipo_resolucion",
]
