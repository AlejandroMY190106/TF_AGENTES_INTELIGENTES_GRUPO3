"""
tc_pipeline/cleaning/mapping.py
───────────────────────────────
Diccionario de mapeo consolidado: JSON API del TC → variables del CSV anual.

Incluye:
- Mapeo directo de campos renombrados (legacy xlsx + nuevo CSV)
- Tabla de equivalencias de ``sentencia_tipo`` (código → etiqueta)
- Tabla de equivalencias de tipo de proceso (sufijo → nombre)
- Tabla de equivalencias ``Distrito Judicial → Departamento``
- Extracción de ``tipo_expediente`` con fallback a tesaurio
- Extracción de ``sentido_resolucion`` desde objeto anidado ``sentido``
- Clasificación de tipo de documento (sentencia vs auto/resolución)
- Jerarquía ``MATERIA/SUB_MATERIA/ESPECIFICA`` a partir de ``palabras``

Uso:
    from tc_pipeline.cleaning.mapping import apply_csv_mapping, apply_mapping
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)



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

# Códigos que corresponden a Autos/Resoluciones vs Sentencias
_AUTO_TIPO_CODES: frozenset[int] = frozenset({1, 2, 3})
_SENTENCIA_TIPO_CODES: frozenset[int] = frozenset({4, 5, 6})


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
# Mapeo canónico de tipo de expediente (6 categorías objetivo)
# ─────────────────────────────────────────────────────────────────────────
# Normaliza las variantes históricas a las 6 categorías canónicas.

TIPO_EXPEDIENTE_CANONICO: dict[str, str] = {
    # Amparo
    "Acción de Amparo": "Proceso de Amparo",
    "Proceso de Amparo": "Proceso de Amparo",
    "AA": "Proceso de Amparo",
    "PA": "Proceso de Amparo",
    # Hábeas Corpus
    "Hábeas Corpus": "Proceso de Habeas Corpus",
    "Proceso de Hábeas Corpus": "Proceso de Habeas Corpus",
    "HC": "Proceso de Habeas Corpus",
    "PHC": "Proceso de Habeas Corpus",
    # Hábeas Data
    "Hábeas Data": "Proceso de Habeas Data",
    "Proceso de Hábeas Data": "Proceso de Habeas Data",
    "HD": "Proceso de Habeas Data",
    "PHD": "Proceso de Habeas Data",
    # Cumplimiento
    "Acción de Cumplimiento": "Proceso de Cumplimiento",
    "Proceso de Cumplimiento": "Proceso de Cumplimiento",
    "AC": "Proceso de Cumplimiento",
    "PC": "Proceso de Cumplimiento",
    # Competencial
    "Conflicto de Competencia": "Proceso Competencial",
    "Proceso Competencial": "Proceso Competencial",
    "CC": "Proceso Competencial",
    "PCC": "Proceso Competencial",
    # Inconstitucionalidad
    "Acción de Inconstitucionalidad": "Proceso de Inconstitucionalidad",
    "Proceso de Inconstitucionalidad": "Proceso de Inconstitucionalidad",
    "Cuestión de Inconstitucionalidad": "Proceso de Inconstitucionalidad",
    "AI": "Proceso de Inconstitucionalidad",
    "PI": "Proceso de Inconstitucionalidad",
    "CI": "Proceso de Inconstitucionalidad",
    # Queja (no es uno de los 6, se mantiene)
    "Queja": "Queja",
    "Q": "Queja",
}


# ─────────────────────────────────────────────────────────────────────────
# Sentido de resolución canónico
# ─────────────────────────────────────────────────────────────────────────

SENTIDO_CANONICO: dict[str, str] = {
    "improcedente": "Improcedente",
    "fundada": "Fundada",
    "fundado": "Fundada",
    "fundada en parte": "Fundada en parte",
    "infundada": "Infundada",
    "infundado": "Infundada",
    "infundada / improcedente": "Infundada / improcedente",
    "improcedente / infundada": "Improcedente / Infundada",
    "fundada / improcedente": "Fundada / Improcedente",
    "fundada / infundada": "Fundada / Infundada",
    "infundada / fundada": "Infundada / Fundada",
    "improcedente / fundada": "Improcedente / Fundada",
    "improcedente el rac": "Improcedente el RAC",
    "improcedente la demanda (autos)": "Improcedente la demanda (Autos)",
    "improcedente la demanda": "Improcedente la demanda (Autos)",
    "fundado el desistimiento": "Fundado el desistimiento",
    "nulo": "Nulo",
    "nula": "Nulo",
    "nulo y admitase la demanda en el pj": "Nulo y admítase la demanda en el PJ",
    "nulo y admítase la demanda en el pj": "Nulo y admítase la demanda en el PJ",
    "admitase la demanda en el pj": "Nulo y admítase la demanda en el PJ",
    "admítase la demanda en el pj": "Nulo y admítase la demanda en el PJ",
    "admitase la demanda en el tc": "Admítase la demanda en el TC",
    "admítase la demanda en el tc": "Admítase la demanda en el TC",
    "nulo el concesorio del rac": "Nulo el Concesorio del RAC",
    "admite la demanda (pi-ccc)": "Admite la demanda (PI-CCC)",
    "improcedente la demanda (pi-ccc)": "Improcedente la demanda (PI-CCC)",
    "inadmisible la demanda (pi-ccc)": "Inadmisible la demanda (PI-CCC)",
    "admite la medida cautelar": "Admite la medida cautelar",
    "inadmisible la medida cautelar": "Inadmisible la medida cautelar",
    "improcedente la medida cautelar": "Improcedente la medida cautelar",
    "otros": "Otros",
}


# ─────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
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



def extract_tipo_expediente(source: dict[str, Any]) -> str:
    """Extrae el tipo de expediente con cadena de fallback.

    Cadena de resolución:
    1. Sufijo del ``numero_expediente`` → ``TIPO_PROCESO_MAP`` → canónico
    2. ``tesaurio.nombre`` o ``tesaurio.slug``
    3. ``tesaurio_auto[0].nombre`` o ``tesaurio_auto[0].slug``
    4. ``tesaurio_inter[0].nombre`` o ``tesaurio_inter[0].slug``

    Mapea al conjunto canónico de 6 tipos:
    - Proceso de Amparo
    - Proceso de Habeas Corpus
    - Proceso de Habeas Data
    - Proceso de Cumplimiento
    - Proceso Competencial
    - Proceso de Inconstitucionalidad

    Args:
        source: Dict ``_source`` del expediente.

    Returns:
        Tipo de expediente canónico, o el valor crudo si no se puede mapear.
    """
    # 1. Intentar desde el sufijo del expediente
    expediente = source.get("numero_expediente", "")
    tipo_raw = extract_tipo_proceso(expediente)
    if tipo_raw:
        canonico = TIPO_EXPEDIENTE_CANONICO.get(tipo_raw)
        if canonico:
            return canonico
        # Si el sufijo crudo no está en canónico, intentar el propio sufijo
        canonico = TIPO_EXPEDIENTE_CANONICO.get(tipo_raw.upper())
        if canonico:
            return canonico

    # 2. Intentar desde tesaurio (objeto)
    tesaurio = source.get("tesaurio")
    if isinstance(tesaurio, dict):
        nombre = tesaurio.get("nombre", "") or tesaurio.get("slug", "")
        if nombre:
            canonico = TIPO_EXPEDIENTE_CANONICO.get(nombre)
            return canonico or nombre

    # 3. Intentar desde tesaurio_auto (lista)
    tesaurio_auto = source.get("tesaurio_auto")
    if isinstance(tesaurio_auto, list) and tesaurio_auto:
        first = tesaurio_auto[0]
        if isinstance(first, dict):
            nombre = first.get("nombre", "") or first.get("slug", "")
            if nombre:
                canonico = TIPO_EXPEDIENTE_CANONICO.get(nombre)
                return canonico or nombre

    # 4. Intentar desde tesaurio_inter (lista)
    tesaurio_inter = source.get("tesaurio_inter")
    if isinstance(tesaurio_inter, list) and tesaurio_inter:
        first = tesaurio_inter[0]
        if isinstance(first, dict):
            nombre = first.get("nombre", "") or first.get("slug", "")
            if nombre:
                canonico = TIPO_EXPEDIENTE_CANONICO.get(nombre)
                return canonico or nombre

    return ""


def extract_sentido_resolucion(source: dict[str, Any]) -> str:
    """Extrae el sentido de la resolución desde el objeto ``sentido``.

    Cadena de resolución:
    1. ``sentido.nombre``
    2. ``sentido.slug``
    3. ``sentencia_sentido`` (campo directo, fallback)

    Normaliza al conjunto canónico:
    - Improcedente
    - Fundada
    - Improcedente la demanda (Autos)
    - Infundada
    - Nulo
    - admítase la demanda en el PJ

    Args:
        source: Dict ``_source`` del expediente.

    Returns:
        Sentido de resolución canónico.
    """
    valor = ""

    # 1. Objeto anidado "sentido"
    sentido = source.get("sentido")
    if isinstance(sentido, dict):
        valor = sentido.get("nombre", "") or sentido.get("slug", "") or ""

    # 2. Objeto anidado "sentencia_sentido"
    if not valor:
        sentencia_sentido = source.get("sentencia_sentido")
        if isinstance(sentencia_sentido, dict):
            valor = sentencia_sentido.get("nombre", "") or sentencia_sentido.get("slug", "") or ""
        elif isinstance(sentencia_sentido, str):
            valor = sentencia_sentido

    # 3. En caso de estar dentro de sistematizacion
    if not valor:
        sistematizacion = source.get("sistematizacion")
        if isinstance(sistematizacion, list):
            for s in sistematizacion:
                if isinstance(s, dict):
                    s_sentido = s.get("sentido") or s.get("sentencia_sentido")
                    if isinstance(s_sentido, dict):
                        valor = s_sentido.get("nombre", "") or s_sentido.get("slug", "") or ""
                        if valor:
                            break

    if not valor:
        return ""

    # Normalizar al canónico
    key = str(valor).strip().lower()
    return SENTIDO_CANONICO.get(key, str(valor).strip())


def classify_document_type(source: dict[str, Any]) -> str:
    """Clasifica un expediente como 'sentencia' o 'auto-resolucion'.

    Usa ``sentencia_tipo`` como señal primaria. Si no está disponible,
    intenta inferir desde el tipo de resolución resuelto o el nombre.

    Args:
        source: Dict ``_source`` del expediente.

    Returns:
        ``"sentencia"`` o ``"auto-resolucion"``.
    """
    tipo_code = source.get("sentencia_tipo")

    if tipo_code is not None:
        try:
            code = int(tipo_code)
            if code in _AUTO_TIPO_CODES:
                return "auto-resolucion"
            if code in _SENTENCIA_TIPO_CODES:
                return "sentencia"
        except (ValueError, TypeError):
            pass

    # Fallback: buscar en el nombre del tipo
    tipo_nombre = resolve_tipo_resolucion(tipo_code).lower()
    if any(kw in tipo_nombre for kw in ("auto", "resolución", "resolucion")):
        return "auto-resolucion"

    # Default: sentencia
    return "sentencia"


def extract_year_from_record(source: dict[str, Any]) -> int | None:
    """Extrae el año del registro con fallback fecha_publicacion → fecha_sentencia.

    Args:
        source: Dict ``_source`` del expediente.

    Returns:
        Año como entero, o None si no se puede determinar.
    """
    for date_field in ("fecha_publicacion", "fecha_sentencia"):
        raw = source.get(date_field)
        if raw:
            raw_str = str(raw).strip()
            # Intentar extraer YYYY del inicio (formato ISO o similar)
            match = re.match(r"(\d{4})", raw_str)
            if match:
                return int(match.group(1))

    # Último recurso: año del expediente
    expediente = source.get("numero_expediente", "")
    match = re.search(r"-(\d{4})-", expediente)
    if match:
        return int(match.group(1))

    return None


def apply_csv_mapping(
    source: dict[str, Any],
    internal_id: str,
) -> dict[str, Any]:
    """Transforma un registro ``_source`` al esquema CSV de 12 columnas.

    Columnas de salida:
    - ``id_interno``: Identificador único del registro
    - ``numero_sentencia``: Número o código de la sentencia
    - ``numero_expediente``: Número de expediente completo
    - ``url_archivo``: URL al PDF
    - ``sentencia_sala``: Sala que emitió
    - ``sentencia_distrito``: Distrito judicial
    - ``tipo_expediente``: Tipo canónico (6 categorías + Queja)
    - ``sentido_resolucion``: Sentido canónico
    - ``nombre_demandante``: Demandante
    - ``nombre_demandado``: Demandado
    - ``fecha_publicacion``: Fecha de publicación
    - ``fecha_sentencia``: Fecha de la sentencia
    - ``fundamentos``: Texto de fundamentos

    Args:
        source: Dict ``_source`` del expediente.
        internal_id: ID interno único (``_id`` del JSON de la API).

    Returns:
        Dict con las columnas del esquema CSV.
    """
    return {
        "id_interno": internal_id,
        "numero_sentencia": source.get("numero_sentencia", "")
            or source.get("sentencia_numero", "")
            or "",
        "numero_expediente": source.get("numero_expediente", "") or "",
        "url_archivo": source.get("url_archivo", "") or "",
        "sentencia_sala": source.get("sentencia_sala", "") or "",
        "sentencia_distrito": source.get("sentencia_distrito", "")
            or source.get("distrito_judicial", "")
            or "",
        "tipo_expediente": extract_tipo_expediente(source),
        "sentido_resolucion": extract_sentido_resolucion(source),
        "nombre_demandante": source.get("nombre_demandante", "") or "",
        "nombre_demandado": source.get("nombre_demandado", "") or "",
        "fecha_publicacion": source.get("fecha_publicacion", "") or "",
        "fecha_sentencia": source.get("fecha_sentencia", "") or "",
        "fundamentos": source.get("fundamentos", "") or "",
    }
