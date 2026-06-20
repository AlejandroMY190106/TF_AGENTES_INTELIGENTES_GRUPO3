"""
scripts/download_pdfs.py
────────────────────────
Orquestador CLI para la descarga de PDFs del Tribunal Constitucional.

Flujo principal:
  API -> Downloader -> Manifest

Permite configuraciones de CLI híbrida y usa valores de PipelineConfig.
Soporta:
- Descarga de un año completo: --year 2025
- Rango de años: --start-year 1992 --end-year 2026
- Un mes específico: --year 2025 --month 01
- Reintento de fallidos previos: --retry-failed
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Asegurar que el path del proyecto esté en sys.path para importaciones
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tc_pipeline.config import PipelineConfig
from tc_pipeline.scraping import (
    ManifestRepository,
    PDFDownloader,
    TribunalAPIClient,
)

# Configurar logging básico para la consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_periods_from_args(args: argparse.Namespace) -> list[str]:
    """Genera la lista de periodos (YYYY-MM) basados en los argumentos del CLI."""
    periods: list[str] = []

    if args.retry_failed:
        # Los periodos vendrán directo del manifiesto.
        return periods

    if args.start_year and args.end_year:
        if args.start_year > args.end_year:
            logger.error("--start-year debe ser menor o igual a --end-year")
            sys.exit(1)
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                periods.append(f"{year}-{month:02d}")
    elif args.year:
        if args.month:
            # Ejemplo: "2025-01"
            periods.append(f"{args.year}-{int(args.month):02d}")
        else:
            # Todo el año: "2025-01" hasta "2025-12"
            for month in range(1, 13):
                periods.append(f"{args.year}-{month:02d}")
    else:
        logger.error("Debe especificar --year, rango de años o --retry-failed.")
        sys.exit(1)

    return periods


def _process_items_for_period(
    periodo: str,
    items_to_process: list[dict[str, Any]],
    downloader: PDFDownloader,
    manifest: ManifestRepository,
) -> None:
    """Descarga items y registra los resultados en el manifiesto."""
    if not items_to_process:
        return

    # Descargar
    metrics = downloader.download_period(items_to_process, periodo, show_progress=True)

    # El downloader solo da un métricas agregadas por ahora,
    # pero para el manifiesto necesitamos registrar qué expediente fue bien y cuál no.
    # Dado que download_batch y download_period devuelven contadores, actualicemos
    # individualmente repitiendo lógica mínima, o modifiquemos el orquestador
    # para usar download_batch nosotros o simplemente verificar si el path existe ahora.
    
    # Re-evaluar estado y registrar en manifiesto (para no romper la abstracción del downloader,
    # verificamos localmente qué se descargó y qué no).
    logger.info("Registrando resultados en el manifiesto para el periodo %s...", periodo)
    for item in items_to_process:
        source = item.get("_source", {})
        expediente = source.get("numero_expediente")
        
        if not expediente:
            continue
            
        dest = downloader.build_path(expediente, periodo)
        
        # Ojo: esto asume que si existe, fue exitoso. Los errores del downloader
        # se registran si queremos extraerlos de los details de métricas.
        error_details = dict(metrics.detalles_errores)
        
        if expediente in error_details:
            manifest.register_failure(expediente, periodo, error_details[expediente])
        elif dest.exists():
            manifest.register_success(expediente, periodo, str(dest))
        else:
            manifest.register_failure(expediente, periodo, "Desconocido: archivo no existe y no hay error registrado")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga de PDFs del Tribunal Constitucional."
    )
    group_year = parser.add_mutually_exclusive_group(required=True)
    group_year.add_argument("--year", type=int, help="Año a descargar (ej: 2025)")
    group_year.add_argument(
        "--start-year",
        type=int,
        help="Año de inicio para descarga de rango múltiple (ej: 1992)",
    )
    group_year.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reintentar descargas que fallaron previamente en la BD de manifiestos",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        help="Año de fin para descarga de rango múltiple (requerido con --start-year)",
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Mes específico a descargar (ej: 01) (solo aplica con --year)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Número de workers concurrentes (sobrescribe la config)",
    )

    args = parser.parse_args()

    if args.start_year and not args.end_year:
        parser.error("--start-year requiere --end-year")

    # 1. Configuración
    config_kwargs = {}
    if args.max_workers:
        config_kwargs["max_workers"] = args.max_workers
    config = PipelineConfig(**config_kwargs)

    # 2. Inicializar componentes
    manifest = ManifestRepository(config.manifest_db)
    manifest.initialize()

    downloader = PDFDownloader(config)

    # Flujo especial: Reintentar fallidos
    if args.retry_failed:
        failed_items = manifest.get_failed()
        if not failed_items:
            logger.info("No hay items fallidos para reintentar.")
            return

        # Agrupar por periodo
        items_by_period = defaultdict(list)
        for row in failed_items:
            items_by_period[row["periodo"]].append(row)

        for periodo, items_db in items_by_period.items():
            logger.info(f"Reintentando {len(items_db)} expedientes fallidos del periodo {periodo}")
            
            # Reconstruir la estructura esperada por downloader (dict con _source)
            # Para reintentar, necesitamos la URL. La BD de manifiestos no guarda la URL original,
            # así que tenemos que consultar la API de nuevo para ese periodo o tenerla en DB.
            # En nuestro diseño actual de ManifestRepository, no hay URL. 
            # Por lo tanto, consultaremos la API completa para el periodo,
            # pero filtramos solo los fallidos.
            
            with TribunalAPIClient(config) as api:
                all_api_items = api.fetch_until_end(periodo)
                
            failed_expedientes = {row["expediente"] for row in items_db}
            items_to_process = [
                i for i in all_api_items 
                if i.get("_source", {}).get("numero_expediente") in failed_expedientes
            ]
            
            _process_items_for_period(periodo, items_to_process, downloader, manifest)

        logger.info("Operación de reintento completada.")
        return

    # Flujo Normal: Por periodos
    periods = get_periods_from_args(args)
    logger.info("Periodos a procesar: %s", periods)

    for periodo in periods:
        logger.info(f"\n=========================================")
        logger.info(f"Procesando periodo: {periodo}")
        logger.info(f"=========================================")

        # 3. Consultar API
        with TribunalAPIClient(config) as api:
            all_items = api.fetch_until_end(periodo)

        if not all_items:
            continue

        # 4. Registrar en manifiesto como pendientes (los nuevos)
        for item in all_items:
            expediente = item.get("_source", {}).get("numero_expediente")
            if expediente:
                manifest.register_pending(expediente, periodo)

        # 5. Filtrar ya procesados
        items_to_process = []
        for item in all_items:
            expediente = item.get("_source", {}).get("numero_expediente")
            if not expediente:
                continue
                
            if not manifest.already_processed(expediente):
                items_to_process.append(item)

        logger.info(
            "Periodo %s: Total en API=%d, Ya procesados=%d, Por descargar=%d",
            periodo,
            len(all_items),
            len(all_items) - len(items_to_process),
            len(items_to_process),
        )

        # 6. Enviar lotes
        _process_items_for_period(periodo, items_to_process, downloader, manifest)
        
        # 7. Marcar periodo completo
        manifest.mark_period_complete(periodo)
        
        # Imprimir stats
        stats = manifest.get_period_stats(periodo)
        logger.info(f"Estadísticas del periodo {periodo}: {stats}")

    manifest.close()
    logger.info("¡Operación finalizada por completo!")


if __name__ == "__main__":
    main()
