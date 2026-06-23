"""
tc_pipeline.extraction
──────────────────────
Paquete de extracción de texto de PDFs del Tribunal Constitucional.

Exporta los componentes principales:
- PDFTextExtractor   → extracción de texto crudo de PDFs
- SentenciaParser    → parser para sentencias (Antecedentes, Asunto)
- AutoResolucionParser → parser para autos/resoluciones (Visto, Atendiendo A Que)
- process_year       → orquestador de extracción anual a CSV
"""

from tc_pipeline.extraction.pdf_extractor import (
    AutoResolucionParser,
    PDFTextExtractor,
    SentenciaParser,
    process_year,
)

__all__ = [
    "PDFTextExtractor",
    "SentenciaParser",
    "AutoResolucionParser",
    "process_year",
]
