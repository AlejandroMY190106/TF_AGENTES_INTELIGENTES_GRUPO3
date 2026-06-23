"""
tc_pipeline/extraction/pdf_extractor.py
───────────────────────────────────────
Extracción robusta de texto de PDFs del Tribunal Constitucional.

Usa ``pdfplumber`` como motor principal para extraer texto sin OCR
(los PDFs ya están saneados). Aplica regex para capturar secciones
específicas de sentencias y autos/resoluciones.

Para Sentencias:
    - Extraer texto debajo de "ANTECEDENTES"
    - Extraer texto debajo de "ASUNTO"
    - Fin: cuando aparece "HA RESUELTO" o "FALLA"

Para Autos/Resoluciones:
    - Extraer texto debajo de "VISTO"
    - Extraer texto debajo de "ATENDIENDO A QUE"
    - Fin: cuando aparece "RESUELVE"

Uso:
    from tc_pipeline.extraction.pdf_extractor import process_year
    csv_path = process_year(2025, "sentencia", config)
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pdfplumber

from tc_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Normalización de texto
# ─────────────────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Normaliza texto extraído de PDF para facilitar el matching regex.

    - Normaliza Unicode (NFC)
    - Reemplaza múltiples espacios/tabs por uno solo
    - Reemplaza múltiples saltos de línea por dos (preserva párrafos)
    - Elimina espacios al inicio/fin de cada línea

    Args:
        text: Texto crudo extraído del PDF.

    Returns:
        Texto normalizado.
    """
    if not text:
        return ""

    # Normalizar Unicode
    text = unicodedata.normalize("NFC", text)

    # Reemplazar tabs por espacios
    text = text.replace("\t", " ")

    # Normalizar múltiples espacios (pero no saltos de línea)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Eliminar espacios al inicio/fin de cada línea
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Reducir múltiples saltos de línea a máximo 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_extracted_section(text: str) -> str:
    """Limpieza adicional para texto de secciones extraídas.

    - Elimina números de página sueltos
    - Elimina encabezados/pies de página repetitivos del TC
    - Normaliza espacios finales

    Args:
        text: Texto de una sección extraída.

    Returns:
        Texto limpio.
    """
    if not text:
        return ""

    # Eliminar líneas que son solo números (números de página)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Saltar líneas que son solo números o vacías
        if stripped and not re.match(r"^\d{1,3}$", stripped):
            # Saltar encabezados típicos del TC
            if not re.match(
                r"^(TRIBUNAL CONSTITUCIONAL|EXP\.?\s*N\.?°?\s*\d)",
                stripped,
                re.IGNORECASE,
            ):
                cleaned.append(line)

    text = "\n".join(cleaned)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────
# Extractor de texto PDF
# ─────────────────────────────────────────────────────────────────────────


class PDFTextExtractor:
    """Extrae texto completo de un PDF usando pdfplumber.

    Concatena el texto de todas las páginas con separadores de página.
    No aplica OCR — diseñado para PDFs con texto digital embebido.

    Example:
        >>> extractor = PDFTextExtractor()
        >>> text = extractor.extract("path/to/sentencia.pdf")
        >>> print(text[:200])
    """

    @staticmethod
    def extract(pdf_path: str | Path) -> str:
        """Extrae todo el texto de un PDF.

        Args:
            pdf_path: Ruta al archivo PDF.

        Returns:
            Texto completo concatenado de todas las páginas.

        Raises:
            FileNotFoundError: Si el PDF no existe.
            Exception: Si pdfplumber no puede abrir el archivo.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        pages_text: list[str] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
        except Exception as e:
            logger.error("Error extrayendo texto de %s: %s", pdf_path.name, e)
            return ""

        full_text = "\n".join(pages_text)
        return normalize_text(full_text)

    @staticmethod
    def extract_expediente_from_text(text: str) -> str:
        """Intenta extraer el número de expediente del texto del PDF.

        Busca patrones como ``EXP. N.° XXXXX-YYYY-XX/TC`` en las
        primeras líneas del documento.

        Args:
            text: Texto completo del PDF.

        Returns:
            Número de expediente encontrado, o cadena vacía.
        """
        if not text:
            return ""

        # Buscar en las primeras 1000 caracteres
        header = text[:1000]
        pattern = re.compile(
            r"EXP\.?\s*N\.?°?\s*(\d{3,5}-\d{4}-[A-Z]{1,3}(?:/TC)?)",
            re.IGNORECASE,
        )
        match = pattern.search(header)
        if match:
            return match.group(1).strip()

        # Fallback: buscar patrón de expediente sin prefijo
        pattern2 = re.compile(r"(\d{3,5}-\d{4}-[A-Z]{1,3}/TC)")
        match2 = pattern2.search(header)
        if match2:
            return match2.group(1).strip()

        return ""


# ─────────────────────────────────────────────────────────────────────────
# Parser de Sentencias
# ─────────────────────────────────────────────────────────────────────────


class SentenciaParser:
    """Parser para sentencias del TC.

    Extrae el texto de las secciones "ANTECEDENTES" y "ASUNTO".
    La extracción termina cuando encuentra "HA RESUELTO" o "FALLA".

    Example:
        >>> parser = SentenciaParser()
        >>> sections = parser.parse(text)
        >>> print(sections["antecedentes"][:100])
    """

    # Patrones regex para títulos de inicio (tolerantes a variaciones)
    ANTECEDENTES_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:I+\.?\s*)?ANTECEDENTES?\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )

    ASUNTO_PATTERN = re.compile(
        r"(?:^|\n)\s*ASUNTO\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )

    # Patrones de fin (donde termina la extracción)
    END_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:HA\s+RESUELTO|FALLA|RESUELVE)\s*[:\n]?",
        re.IGNORECASE | re.MULTILINE,
    )

    # Patrones de secciones intermedias que delimitan las anteriores
    FUNDAMENTOS_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:FUNDAMENTOS?|I+\.?\s*FUNDAMENTOS?)\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )

    def parse(self, text: str) -> dict[str, str]:
        """Extrae las secciones de una sentencia.

        Args:
            text: Texto completo del PDF normalizado.

        Returns:
            Dict con claves ``antecedentes`` y ``asunto``.
        """
        result = {"antecedentes": "", "asunto": ""}

        if not text:
            return result

        result["antecedentes"] = self._extract_section(
            text,
            start_pattern=self.ANTECEDENTES_PATTERN,
            end_patterns=[
                self.FUNDAMENTOS_PATTERN,
                self.END_PATTERN,
                self.ASUNTO_PATTERN,
            ],
        )

        result["asunto"] = self._extract_section(
            text,
            start_pattern=self.ASUNTO_PATTERN,
            end_patterns=[
                self.ANTECEDENTES_PATTERN,
                self.FUNDAMENTOS_PATTERN,
                self.END_PATTERN,
            ],
        )

        return result

    @staticmethod
    def _extract_section(
        text: str,
        start_pattern: re.Pattern,
        end_patterns: list[re.Pattern],
    ) -> str:
        """Extrae texto entre un patrón de inicio y el primer patrón de fin.

        Args:
            text: Texto completo.
            start_pattern: Regex compilado del título de inicio.
            end_patterns: Lista de regexes que delimitan el fin.

        Returns:
            Texto de la sección, limpio.
        """
        match_start = start_pattern.search(text)
        if not match_start:
            return ""

        # El contenido empieza después del título
        start_pos = match_start.end()

        # Buscar el fin más cercano
        end_pos = len(text)
        for pattern in end_patterns:
            match_end = pattern.search(text, pos=start_pos)
            if match_end and match_end.start() < end_pos:
                end_pos = match_end.start()

        section_text = text[start_pos:end_pos]
        return clean_extracted_section(section_text)


# ─────────────────────────────────────────────────────────────────────────
# Parser de Autos/Resoluciones
# ─────────────────────────────────────────────────────────────────────────


class AutoResolucionParser:
    """Parser para autos y resoluciones del TC.

    Extrae el texto de las secciones "VISTO" y "ATENDIENDO A QUE".
    La extracción termina cuando encuentra "RESUELVE".

    Example:
        >>> parser = AutoResolucionParser()
        >>> sections = parser.parse(text)
        >>> print(sections["visto"][:100])
    """

    VISTO_PATTERN = re.compile(
        r"(?:^|\n)\s*VISTOS?\s*[:\n]",
        re.IGNORECASE | re.MULTILINE,
    )

    ATENDIENDO_PATTERN = re.compile(
        r"(?:^|\n)\s*ATENDIENDO\s+A\s+QUE\s*[:\n]",
        re.IGNORECASE | re.MULTILINE,
    )

    END_PATTERN = re.compile(
        r"(?:^|\n)\s*RESUELVE\s*[:\n]?",
        re.IGNORECASE | re.MULTILINE,
    )

    # Patrón intermedio
    CONSIDERANDO_PATTERN = re.compile(
        r"(?:^|\n)\s*CONSIDERANDO\s*[:\n]",
        re.IGNORECASE | re.MULTILINE,
    )

    def parse(self, text: str) -> dict[str, str]:
        """Extrae las secciones de un auto/resolución.

        Args:
            text: Texto completo del PDF normalizado.

        Returns:
            Dict con claves ``visto`` y ``atendiendo_a_que``.
        """
        result = {"visto": "", "atendiendo_a_que": ""}

        if not text:
            return result

        result["visto"] = self._extract_section(
            text,
            start_pattern=self.VISTO_PATTERN,
            end_patterns=[
                self.ATENDIENDO_PATTERN,
                self.CONSIDERANDO_PATTERN,
                self.END_PATTERN,
            ],
        )

        result["atendiendo_a_que"] = self._extract_section(
            text,
            start_pattern=self.ATENDIENDO_PATTERN,
            end_patterns=[
                self.END_PATTERN,
            ],
        )

        return result

    @staticmethod
    def _extract_section(
        text: str,
        start_pattern: re.Pattern,
        end_patterns: list[re.Pattern],
    ) -> str:
        """Extrae texto entre un patrón de inicio y el primer patrón de fin.

        Args:
            text: Texto completo.
            start_pattern: Regex compilado del título de inicio.
            end_patterns: Lista de regexes que delimitan el fin.

        Returns:
            Texto de la sección, limpio.
        """
        match_start = start_pattern.search(text)
        if not match_start:
            return ""

        start_pos = match_start.end()

        end_pos = len(text)
        for pattern in end_patterns:
            match_end = pattern.search(text, pos=start_pos)
            if match_end and match_end.start() < end_pos:
                end_pos = match_end.start()

        section_text = text[start_pos:end_pos]
        return clean_extracted_section(section_text)


# ─────────────────────────────────────────────────────────────────────────
# Orquestador de extracción anual
# ─────────────────────────────────────────────────────────────────────────


def process_year(
    year: int,
    doc_type: str,
    config: PipelineConfig | None = None,
    id_map: dict[str, str] | None = None,
) -> Path:
    """Extrae texto de todos los PDFs de un año y genera un CSV.

    Itera sobre los PDFs en el directorio crudo correspondiente,
    extrae las secciones relevantes y genera un CSV en el directorio
    de extracción.

    Args:
        year: Año a procesar.
        doc_type: ``"sentencia"`` o ``"auto-resolucion"``.
        config: Configuración del pipeline.
        id_map: Mapeo ``id_interno → numero_expediente``. Si no se
                proporciona, intenta extraer el expediente del PDF.

    Returns:
        Path al CSV generado.

    Example:
        >>> csv_path = process_year(2025, "sentencia")
        >>> print(csv_path)
        data/sentencia-Extract/2025/sentencia-pdf-2025.csv
    """
    config = config or PipelineConfig()
    id_map = id_map or {}

    # Determinar directorios de entrada y salida
    if doc_type == "auto-resolucion":
        raw_dir = config.auto_resolucion_raw_root / str(year)
        extract_dir = config.auto_resolucion_extract_root / str(year)
        csv_name = f"auto-resolucion-{year}.csv"
        parser = AutoResolucionParser()
        columns = ["numero_expediente", "id_interno", "visto", "atendiendo_a_que"]
    else:
        raw_dir = config.sentencia_raw_root / str(year)
        extract_dir = config.sentencia_extract_root / str(year)
        csv_name = f"sentencia-pdf-{year}.csv"
        parser = SentenciaParser()
        columns = ["numero_expediente", "id_interno", "antecedentes", "asunto"]

    extract_dir.mkdir(parents=True, exist_ok=True)
    csv_path = extract_dir / csv_name

    # Verificar que el directorio raw existe
    if not raw_dir.exists():
        logger.warning(
            "Directorio raw no encontrado: %s. Creando CSV vacío.",
            raw_dir,
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return csv_path

    # Listar PDFs
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("Año %d (%s): sin PDFs encontrados.", year, doc_type)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return csv_path

    logger.info(
        "Año %d (%s): procesando %d PDFs.",
        year,
        doc_type,
        len(pdf_files),
    )

    extractor = PDFTextExtractor()
    records: list[dict[str, str]] = []
    errors: list[str] = []

    for pdf_file in pdf_files:
        id_interno = pdf_file.stem  # El nombre del archivo ES el ID interno

        try:
            # Extraer texto
            text = extractor.extract(pdf_file)

            if not text:
                logger.warning(
                    "PDF sin texto extraíble: %s",
                    pdf_file.name,
                )
                errors.append(pdf_file.name)
                continue

            # Resolver numero_expediente
            numero_expediente = id_map.get(id_interno, "")
            if not numero_expediente:
                # Intentar extraer del propio texto del PDF
                numero_expediente = extractor.extract_expediente_from_text(text)

            # Parsear secciones
            sections = parser.parse(text)

            # Construir registro
            record: dict[str, str] = {
                "numero_expediente": numero_expediente,
                "id_interno": id_interno,
            }
            record.update(sections)
            records.append(record)

        except Exception as e:
            logger.error(
                "Error procesando PDF %s: %s",
                pdf_file.name,
                e,
            )
            errors.append(pdf_file.name)

    # Escribir CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    logger.info(
        "CSV generado: %s (%d registros, %d errores de %d PDFs)",
        csv_path,
        len(records),
        len(errors),
        len(pdf_files),
    )

    # Guardar log de errores si hay
    if errors:
        errors_path = extract_dir / f"errores-{year}.txt"
        errors_path.write_text("\n".join(errors), encoding="utf-8")
        logger.info("Log de errores: %s", errors_path)

    return csv_path


def process_all_years(
    start_year: int = 1992,
    end_year: int = 2026,
    config: PipelineConfig | None = None,
    progress_callback: Any | None = None,
) -> list[Path]:
    """Extrae texto de PDFs para todos los años en ambos tipos.

    Args:
        start_year: Año de inicio.
        end_year: Año de fin.
        config: Configuración del pipeline.
        progress_callback: ``(year, doc_type, csv_path) -> None``

    Returns:
        Lista de Paths a los CSVs generados.
    """
    config = config or PipelineConfig()
    csv_paths: list[Path] = []

    for year in range(end_year, start_year - 1, -1):
        for doc_type in ("sentencia", "auto-resolucion"):
            # Intentar cargar el id_map para este año
            id_map: dict[str, str] = {}
            map_path = config.csv_output_root / f"id_map-{year}.json"
            if map_path.exists():
                try:
                    with open(map_path, "r", encoding="utf-8") as f:
                        id_map = json.load(f)
                except Exception:
                    pass

            csv_path = process_year(year, doc_type, config, id_map)
            csv_paths.append(csv_path)

            if progress_callback:
                progress_callback(year, doc_type, csv_path)

    return csv_paths
