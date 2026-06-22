"""
scripts/run_pipeline.py
───────────────────────
Orquestador CLI para el pipeline completo del Tribunal Constitucional.

Fases disponibles:
  --phase json      → Recoger JSON y generar CSVs anuales (expedientes-json-YYYY.csv)
  --phase download  → Descargar PDFs separados por tipo (sentencia-raw/ y auto-resolucion-raw/)
  --phase extract   → Extraer texto de PDFs a CSVs (sentencia-Extract/ y auto-resolucion-Extract/)
  --phase all       → Ejecutar las tres fases secuencialmente

Uso:
    python scripts/run_pipeline.py --phase all --start-year 2024 --end-year 2025
    python scripts/run_pipeline.py --phase json --year 2025
    python scripts/run_pipeline.py --phase download --year 2025
    python scripts/run_pipeline.py --phase extract --year 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Asegurar que el path del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tc_pipeline.config import PipelineConfig
from tc_pipeline.extraction.pdf_extractor import process_year as extract_year
from tc_pipeline.scraping import PDFDownloader, TribunalAPIClient

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Fase 1: JSON → CSV
# ─────────────────────────────────────────────────────────────────────────


def phase_json(
    start_year: int,
    end_year: int,
    config: PipelineConfig,
) -> list[Path]:
    """Descarga metadata JSON de la API y genera CSVs anuales.

    Args:
        start_year: Año de inicio.
        end_year: Año de fin.
        config: Configuración del pipeline.

    Returns:
        Lista de Paths a los CSVs generados.
    """
    logger.info("═" * 60)
    logger.info("  FASE 1: JSON → CSV (%d - %d)", start_year, end_year)
    logger.info("═" * 60)

    csv_paths: list[Path] = []

    with TribunalAPIClient(config) as api:
        total = end_year - start_year + 1
        for i, year in enumerate(range(end_year, start_year - 1, -1), 1):
            logger.info("[%d/%d] Procesando año: %d", i, total, year)
            try:
                csv_path = api.fetch_year_to_csv(year)
                csv_paths.append(csv_path)
                logger.info("  ✓ CSV generado: %s", csv_path)
            except Exception as e:
                logger.error("  ✗ Error en año %d: %s", year, e)

            if i < total:
                time.sleep(config.page_delay * 2)

    logger.info("Fase 1 completada: %d CSVs generados.", len(csv_paths))
    return csv_paths


# ─────────────────────────────────────────────────────────────────────────
# Fase 2: Descarga de PDFs
# ─────────────────────────────────────────────────────────────────────────


def phase_download(
    start_year: int,
    end_year: int,
    config: PipelineConfig,
) -> None:
    """Descarga PDFs separados en sentencia-raw/ y auto-resolucion-raw/.

    Args:
        start_year: Año de inicio.
        end_year: Año de fin.
        config: Configuración del pipeline.
    """
    logger.info("═" * 60)
    logger.info("  FASE 2: DESCARGA PDFs (%d - %d)", start_year, end_year)
    logger.info("═" * 60)

    downloader = PDFDownloader(config)
    total = end_year - start_year + 1

    for i, year in enumerate(range(end_year, start_year - 1, -1), 1):
        logger.info("[%d/%d] Descargando PDFs del año: %d", i, total, year)

        # Limpiar id_map para este año
        downloader.clear_id_map()

        with TribunalAPIClient(config) as api:
            try:
                items, records = api.get_items_with_metadata(year)
            except Exception as e:
                logger.error("  ✗ Error consultando API para %d: %s", year, e)
                continue

        if not items:
            logger.info("  -> Sin registros para %d", year)
            continue

        logger.info(
            "  -> %d items obtenidos (%d sentencias, %d autos)",
            len(items),
            sum(1 for r in records if r.get("_doc_type") == "sentencia"),
            sum(1 for r in records if r.get("_doc_type") == "auto-resolucion"),
        )

        metrics = downloader.download_year(items, records, year)

        # Guardar mapeo ID ↔ expediente
        downloader.save_id_map(year)

        logger.info(
            "  [OK] Año %d: %d descargados, %d existentes, %d errores.",
            year,
            metrics.descargados,
            metrics.existentes,
            metrics.errores,
        )

        if i < total:
            time.sleep(config.page_delay)

    logger.info("Fase 2 completada.")


# ─────────────────────────────────────────────────────────────────────────
# Fase 3: Extracción de texto PDF → CSV
# ─────────────────────────────────────────────────────────────────────────


def phase_extract(
    start_year: int,
    end_year: int,
    config: PipelineConfig,
) -> list[Path]:
    """Extrae texto de PDFs y genera CSVs anuales por tipo.

    Args:
        start_year: Año de inicio.
        end_year: Año de fin.
        config: Configuración del pipeline.

    Returns:
        Lista de Paths a los CSVs generados.
    """
    logger.info("═" * 60)
    logger.info("  FASE 3: EXTRACCIÓN PDF → CSV (%d - %d)", start_year, end_year)
    logger.info("═" * 60)

    csv_paths: list[Path] = []
    total = end_year - start_year + 1
    downloader = PDFDownloader(config)

    for i, year in enumerate(range(end_year, start_year - 1, -1), 1):
        logger.info("[%d/%d] Extrayendo texto del año: %d", i, total, year)

        # Cargar mapeo ID ↔ expediente
        id_map = downloader.load_id_map(year)

        for doc_type in ("sentencia", "auto-resolucion"):
            try:
                csv_path = extract_year(year, doc_type, config, id_map)
                csv_paths.append(csv_path)
                logger.info("  [OK] %s -> %s", doc_type, csv_path)
            except Exception as e:
                logger.error(
                    "  [X] Error extrayendo %s-%d: %s",
                    doc_type,
                    year,
                    e,
                )

    logger.info("Fase 3 completada: %d CSVs generados.", len(csv_paths))
    return csv_paths


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo del Tribunal Constitucional: JSON → CSV → PDF → Texto.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Fase completa para un rango de años
  python scripts/run_pipeline.py --phase all --start-year 1992 --end-year 2026

  # Solo JSON → CSV para un año
  python scripts/run_pipeline.py --phase json --year 2025

  # Solo descargar PDFs
  python scripts/run_pipeline.py --phase download --year 2025

  # Solo extraer texto de PDFs ya descargados
  python scripts/run_pipeline.py --phase extract --year 2025
        """,
    )

    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["json", "download", "extract", "all"],
        help="Fase del pipeline a ejecutar",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Año específico a procesar")
    group.add_argument(
        "--start-year",
        type=int,
        help="Año de inicio (requiere --end-year)",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        help="Año de fin (requerido con --start-year)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Número de workers concurrentes para descargas",
    )

    args = parser.parse_args()

    if args.start_year and not args.end_year:
        parser.error("--start-year requiere --end-year")

    # Determinar rango
    if args.year:
        start_year = args.year
        end_year = args.year
    else:
        start_year = args.start_year
        end_year = args.end_year

    # Configuración
    config_kwargs: dict = {}
    if args.max_workers:
        config_kwargs["max_workers"] = args.max_workers
    config = PipelineConfig(**config_kwargs)

    start_time = time.time()

    # Ejecutar fases
    if args.phase in ("json", "all"):
        phase_json(start_year, end_year, config)

    if args.phase in ("download", "all"):
        phase_download(start_year, end_year, config)

    if args.phase in ("extract", "all"):
        phase_extract(start_year, end_year, config)

    elapsed = time.time() - start_time

    # Resumen final
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETADO")
    print("═" * 60)
    print(f"  Fase:        {args.phase}")
    print(f"  Rango:       {start_year} - {end_year}")
    print(f"  Tiempo:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
