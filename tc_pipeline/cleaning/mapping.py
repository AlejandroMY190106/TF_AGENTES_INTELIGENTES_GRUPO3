"""
tc_pipeline/cleaning/mapping.py
───────────────────────────────
Diccionario de mapeo consolidado: JSON API del TC → variables del xlsx histórico.

Mapea los campos del endpoint cronológico del TC hacia las columnas del
dataset maestro ``dataset_tc.xlsx``, incluyendo:
- Mapeo directo de campos renombrados
- Tabla de equivalencias de ``sentencia_tipo`` (código → etiqueta)
- Tabla de equivalencias de tipo de proceso (sufijo → nombre)
- Tabla de equivalencias ``Distrito Judicial → Departamento``
- Jerarquía ``MATERIA/SUB_MATERIA/ESPECIFICA`` a partir de ``palabras``

Uso:
    from tc_pipeline.cleaning.mapping import apply_mapping, FIELD_MAP
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Mapeo directo: campo API (_source) → columna xlsx
# ─────────────────────────────────────────────────────────────────────────

FIELD_MAP: dict[str, str] = {
    # Campos directos
    "numero_expediente": "NEXPEDIENTE",
    "fecha_publicacion": "PUB_PAGWEB",
    "sentencia_sala": "SALA",
    "sentencia_sentido": "FALLO",
    "sentencia_tipo": "TIPO_RESOLUCION",
    "nombre_demandante": "DEMANDANTE",
    "nombre_demandado": "DEMANDADO",
    "fundamentos": "FUNDAMENTOS",
    # Campos que requieren transformación (se procesan aparte)
    # "numero_expediente" → también alimenta CDES_TIPOPROCESO (sufijo)
    # "palabras" → MATERIA, SUB_MATERIA, ESPECIFICA
}


# ─────────────────────────────────────────────────────────────────────────
# Tabla de equivalencias: sentencia_tipo (código → etiqueta)
# ─────────────────────────────────────────────────────────────────────────
# Basado en el combobox "Tipo de Resolución" del frontend del TC.
# Se debe validar empíricamente con los datos reales.

SENTENCIA_TIPO_MAP: dict[int, str] = {
    1: "Auto",
    2: "Auto de Vista",
    3: "Resolución",
    4: "Sentencia",
    5: "Sentencia Interlocutoria",
    6: "Sentencia de Vista",
}


# ─────────────────────────────────────────────────────────────────────────
# Tabla de equivalencias: sufijo de expediente → tipo de proceso
# ─────────────────────────────────────────────────────────────────────────
# El número de expediente tiene formato: NNNNN-YYYY-XX/TC
# donde XX indica el tipo de proceso constitucional.

TIPO_PROCESO_MAP: dict[str, str] = {
    "AA": "Acción de Amparo",
    "AC": "Acción de Cumplimiento",
    "HC": "Hábeas Corpus",
    "HD": "Hábeas Data",
    "AI": "Acción de Inconstitucionalidad",
    "CC": "Conflicto de Competencia",
    "CI": "Cuestión de Inconstitucionalidad",
    "PA": "Proceso de Amparo",
    "PHC": "Proceso de Hábeas Corpus",
    "PHD": "Proceso de Hábeas Data",
    "PC": "Proceso de Cumplimiento",
    "PI": "Proceso de Inconstitucionalidad",
    "PCC": "Proceso Competencial",
    "Q": "Queja",
}


# ─────────────────────────────────────────────────────────────────────────
# Tabla de equivalencias: Distrito Judicial → Departamento
# ─────────────────────────────────────────────────────────────────────────

DISTRITO_JUDICIAL_DEPARTAMENTO: dict[str, str] = {
    "AMAZONAS": "AMAZONAS",
    "ÁNCASH": "ÁNCASH",
    "ANCASH": "ÁNCASH",
    "SANTA": "ÁNCASH",
    "DEL SANTA": "ÁNCASH",
    "APURÍMAC": "APURÍMAC",
    "APURIMAC": "APURÍMAC",
    "AREQUIPA": "AREQUIPA",
    "AYACUCHO": "AYACUCHO",
    "CAJAMARCA": "CAJAMARCA",
    "CALLAO": "CALLAO",
    "CUSCO": "CUSCO",
    "CUZCO": "CUSCO",
    "HUANCAVELICA": "HUANCAVELICA",
    "HUÁNUCO": "HUÁNUCO",
    "HUANUCO": "HUÁNUCO",
    "ICA": "ICA",
    "JUNÍN": "JUNÍN",
    "JUNIN": "JUNÍN",
    "LA LIBERTAD": "LA LIBERTAD",
    "LAMBAYEQUE": "LAMBAYEQUE",
    "LIMA": "LIMA",
    "LIMA ESTE": "LIMA",
    "LIMA NORTE": "LIMA",
    "LIMA SUR": "LIMA",
    "LORETO": "LORETO",
    "MADRE DE DIOS": "MADRE DE DIOS",
    "MOQUEGUA": "MOQUEGUA",
    "PASCO": "PASCO",
    "PIURA": "PIURA",
    "SULLANA": "PIURA",
    "PUNO": "PUNO",
    "SAN MARTÍN": "SAN MARTÍN",
    "SAN MARTIN": "SAN MARTÍN",
    "TACNA": "TACNA",
    "TUMBES": "TUMBES",
    "UCAYALI": "UCAYALI",
    "CAÑETE": "LIMA",
    "HUAURA": "LIMA",
    "VENTANILLA": "CALLAO",
    "SELVA CENTRAL": "JUNÍN",
}


# ─────────────────────────────────────────────────────────────────────────
# Funciones de extracción
# ─────────────────────────────────────────────────────────────────────────


def extract_tipo_proceso(numero_expediente: str) -> str:
    """Extrae el tipo de proceso del número de expediente.

    Formato esperado: ``NNNNN-YYYY-XX/TC`` o ``NNNNN-YYYY-XX``
    donde XX es el código del tipo de proceso.

    Args:
        numero_expediente: Número de expediente completo.

    Returns:
        Nombre del tipo de proceso o el sufijo crudo si no se reconoce.

    Example:
        >>> extract_tipo_proceso("01234-2025-AA/TC")
        'Acción de Amparo'
        >>> extract_tipo_proceso("05678-2020-HC")
        'Hábeas Corpus'
    """
    if not numero_expediente:
        return ""

    # Limpiar /TC al final si existe
    clean = numero_expediente.strip()
    if clean.endswith("/TC"):
        clean = clean[:-3]
    elif clean.endswith("/tc"):
        clean = clean[:-3]

    # Buscar el sufijo después del último guión
    match = re.search(r"-(\d{4})-([A-Za-z]+)$", clean)
    if match:
        sufijo = match.group(2).upper()
        return TIPO_PROCESO_MAP.get(sufijo, sufijo)

    return ""


def extract_materias(palabras: list[dict] | None) -> dict[str, str]:
    """Extrae la jerarquía MATERIA/SUB_MATERIA/ESPECIFICA de ``palabras``.

    El campo ``palabras`` del JSON de la API es una lista de dicts con
    estructura jerárquica.  Se mapea al esquema plano del xlsx:
    - Nivel 1 → MATERIA
    - Nivel 2 → SUB_MATERIA
    - Nivel 3 → ESPECIFICA

    Args:
        palabras: Lista de dicts del campo ``palabras`` del JSON.

    Returns:
        Dict con claves ``MATERIA``, ``SUB_MATERIA``, ``ESPECIFICA``.
    """
    result = {"MATERIA": "", "SUB_MATERIA": "", "ESPECIFICA": ""}

    if not palabras or not isinstance(palabras, list):
        return result

    # Tomar las primeras palabras como jerarquía
    nombres = []
    for p in palabras:
        if isinstance(p, dict):
            nombre = p.get("nombre", p.get("name", ""))
            if nombre:
                nombres.append(str(nombre).strip())
        elif isinstance(p, str):
            nombres.append(p.strip())

    if len(nombres) >= 1:
        result["MATERIA"] = nombres[0]
    if len(nombres) >= 2:
        result["SUB_MATERIA"] = nombres[1]
    if len(nombres) >= 3:
        result["ESPECIFICA"] = " | ".join(nombres[2:])

    return result


def resolve_departamento(distrito_judicial: str) -> str:
    """Resuelve el departamento a partir del distrito judicial.

    Args:
        distrito_judicial: Nombre del distrito judicial.

    Returns:
        Nombre del departamento correspondiente, o el distrito original
        si no se encuentra en la tabla de equivalencias.
    """
    if not distrito_judicial:
        return ""
    key = distrito_judicial.strip().upper()
    return DISTRITO_JUDICIAL_DEPARTAMENTO.get(key, distrito_judicial)


def resolve_tipo_resolucion(sentencia_tipo: int | str | None) -> str:
    """Resuelve el tipo de resolución desde el código numérico.

    Args:
        sentencia_tipo: Código numérico de ``sentencia_tipo``.

    Returns:
        Nombre legible del tipo de resolución.
    """
    if sentencia_tipo is None:
        return ""
    try:
        code = int(sentencia_tipo)
    except (ValueError, TypeError):
        return str(sentencia_tipo)
    return SENTENCIA_TIPO_MAP.get(code, f"Desconocido ({code})")


# ─────────────────────────────────────────────────────────────────────────
# Función principal de mapeo
# ─────────────────────────────────────────────────────────────────────────


def apply_mapping(source: dict[str, Any]) -> dict[str, Any]:
    """Transforma un registro ``_source`` del JSON de la API al esquema xlsx.

    Aplica el mapeo directo de campos, extrae tipo de proceso del
    número de expediente, resuelve tipo de resolución, y extrae
    la jerarquía de materias.

    Args:
        source: Dict con los campos ``_source`` de un expediente.

    Returns:
        Dict con claves del esquema xlsx (NEXPEDIENTE, SALA, FALLO, etc.).

    Example:
        >>> source = {
        ...     "numero_expediente": "01234-2025-AA/TC",
        ...     "sentencia_sala": "Segunda Sala",
        ...     "sentencia_sentido": "Fundada",
        ...     "sentencia_tipo": 4,
        ...     "fecha_publicacion": "2025-01-15",
        ...     "palabras": [{"nombre": "Derecho Laboral"}],
        ... }
        >>> mapped = apply_mapping(source)
        >>> mapped["CDES_TIPOPROCESO"]
        'Acción de Amparo'
    """
    mapped: dict[str, Any] = {}

    # 1. Mapeo directo de campos
    for api_field, xlsx_col in FIELD_MAP.items():
        value = source.get(api_field)
        if value is not None:
            mapped[xlsx_col] = value

    # 2. Tipo de proceso (derivado del número de expediente)
    expediente = source.get("numero_expediente", "")
    mapped["CDES_TIPOPROCESO"] = extract_tipo_proceso(expediente)

    # 3. Tipo de resolución (transformar código → etiqueta)
    mapped["TIPO_RESOLUCION"] = resolve_tipo_resolucion(
        source.get("sentencia_tipo")
    )

    # 4. Jerarquía de materias
    materias = extract_materias(source.get("palabras"))
    mapped.update(materias)

    # 5. Campos adicionales del JSON que se preservan tal cual
    for extra_field in [
        "url_archivo",
        "sentencia_sala_id",
        "sentencia_sentido_id",
        "sentencia_tipo_id",
        "distrito_judicial",
    ]:
        if extra_field in source:
            mapped[extra_field] = source[extra_field]

    # 6. Departamento (si hay distrito judicial)
    if "distrito_judicial" in source:
        mapped["DEPARTAMENTO"] = resolve_departamento(
            source.get("distrito_judicial", "")
        )

    return mapped
