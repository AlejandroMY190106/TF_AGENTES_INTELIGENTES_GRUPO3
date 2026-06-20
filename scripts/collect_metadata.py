"""
scripts/collect_metadata.py
───────────────────────────
Recolector de metadatos de expedientes del Tribunal Constitucional.

A diferencia de download_pdfs.py (que descarga PDFs físicos), este script
extrae y persiste los METADATOS JSON de la API del TC en formato Parquet
para análisis posterior.

Uso:
    # Recolectar unos meses específicos (rápido, ~500 expedientes)
    python scripts/collect_metadata.py --start-year 2024 --end-year 2024

    # Incluir periodo histórico 1992-2012 para validar cobertura
    python scripts/collect_metadata.py --start-year 1996 --end-year 2024

    # Recolectar un año específico
    python scripts/collect_metadata.py --year 2023

    # Recolectar un mes específico
    python scripts/collect_metadata.py --year 2024 --month 01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Asegurar que el path del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tc_pipeline.cleaning.mapping import apply_mapping
from tc_pipeline.config import PipelineConfig
from tc_pipeline.scraping import TribunalAPIClient
from tc_pipeline.storage.parquet_store import ParquetStore

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────


def flatten_source(item: dict[str, Any]) -> dict[str, Any]:
    """Aplana un item de la API extrayendo los campos de _source.

    Args:
        item: Dict con estructura ``{"_source": {...}, ...}``.

    Returns:
        Dict plano con todos los campos de _source.
    """
    source = item.get("_source", {})

    # Serializar campos complejos (listas, dicts) a JSON string para Parquet
    flat: dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value

    return flat


def generate_periods(
    start_year: int,
    end_year: int,
    month: int | None = None,
) -> list[str]:
    """Genera lista de periodos en formato YYYY-MM."""
    periods = []
    for year in range(start_year, end_year + 1):
        if month:
            periods.append(f"{year}-{month:02d}")
        else:
            for m in range(1, 13):
                periods.append(f"{year}-{m:02d}")
    return periods


def generate_coverage_report(df: pd.DataFrame, output_path: Path) -> None:
    """Genera un reporte de cobertura temporal del dataset.

    Args:
        df: DataFrame con los expedientes recolectados.
        output_path: Ruta para guardar el reporte.
    """
    report_lines = [
        "# Reporte de Cobertura — Pipeline de Extracción TC\n",
        f"**Total de expedientes recolectados:** {len(df)}\n",
    ]

    if "numero_expediente" in df.columns:
        # Extraer año del expediente
        years = df["numero_expediente"].str.extract(r"-(\d{4})-", expand=False)
        years = pd.to_numeric(years, errors="coerce").dropna().astype(int)

        report_lines.append(f"**Rango temporal:** {years.min()} - {years.max()}\n")
        report_lines.append("\n## Expedientes por año\n")
        report_lines.append("| Año | Cantidad |")
        report_lines.append("|-----|----------|")

        counts = years.value_counts().sort_index()
        for year, count in counts.items():
            marker = " ⚠️" if count < 10 else ""
            report_lines.append(f"| {year} | {count}{marker} |")

        # Validación especial 1992-2012
        historical = years[(years >= 1992) & (years <= 2012)]
        report_lines.append(f"\n## Validación periodo 1992-2012\n")
        report_lines.append(f"- Expedientes del periodo histórico: **{len(historical)}**")

        if len(historical) > 0:
            report_lines.append(f"- Años cubiertos: {sorted(historical.unique().tolist())}")
        else:
            report_lines.append("- ⚠️ **Sin cobertura** en el periodo 1992-2012")

    if "sentencia_sala" in df.columns:
        report_lines.append("\n## Distribución por sala\n")
        report_lines.append("| Sala | Cantidad |")
        report_lines.append("|------|----------|")
        for sala, count in df["sentencia_sala"].value_counts().head(10).items():
            report_lines.append(f"| {sala} | {count} |")

    if "sentencia_sentido" in df.columns:
        report_lines.append("\n## Distribución por sentido del fallo\n")
        report_lines.append("| Fallo | Cantidad |")
        report_lines.append("|-------|----------|")
        for fallo, count in df["sentencia_sentido"].value_counts().items():
            report_lines.append(f"| {fallo} | {count} |")

    # Verificar campos vacíos
    report_lines.append("\n## Campos con datos faltantes\n")
    report_lines.append("| Campo | % Vacío |")
    report_lines.append("|-------|---------|")
    for col in df.columns:
        try:
            na_mask = df[col].isna()
            # Only compare to "" for string-like columns
            if df[col].dtype == object:
                empty_mask = df[col].astype(str).eq("")
                pct_empty = (na_mask | empty_mask).mean() * 100
            else:
                pct_empty = na_mask.mean() * 100
        except Exception:
            pct_empty = df[col].isna().mean() * 100
        if pct_empty > 0:
            report_lines.append(f"| {col} | {pct_empty:.1f}% |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Reporte de cobertura guardado en: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recolecta metadatos de expedientes del TC y exporta a Parquet."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Año a recolectar")
    group.add_argument("--start-year", type=int, help="Año de inicio (requiere --end-year)")

    parser.add_argument("--end-year", type=int, help="Año de fin (con --start-year)")
    parser.add_argument("--month", type=int, help="Mes específico (1-12)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/expedientes_tc.parquet",
        help="Ruta del archivo Parquet de salida",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Agregar al dataset existente en vez de sobrescribir",
    )
    parser.add_argument(
        "--with-mapping",
        action="store_true",
        default=True,
        help="Aplicar diccionario de mapeo a los campos (por defecto: True)",
    )

    args = parser.parse_args()

    if args.start_year and not args.end_year:
        parser.error("--start-year requiere --end-year")

    # Generar periodos
    if args.start_year:
        periods = generate_periods(args.start_year, args.end_year, args.month)
    else:
        periods = generate_periods(args.year, args.year, args.month)

    logger.info("Periodos a procesar: %d (%s ... %s)", len(periods), periods[0], periods[-1])

    # Inicializar componentes
    config = PipelineConfig()
    store = ParquetStore(Path(args.output))

    all_records: list[dict[str, Any]] = []
    total_api_items = 0

    with TribunalAPIClient(config) as api:
        for i, periodo in enumerate(periods, 1):
            logger.info(
                "[%d/%d] Recolectando periodo: %s",
                i, len(periods), periodo,
            )

            try:
                items = api.fetch_period(periodo)
            except Exception as e:
                logger.error("Error en periodo %s: %s", periodo, e)
                continue

            if not items:
                logger.info("  -> Sin registros para %s", periodo)
                continue

            total_api_items += len(items)

            # Aplanar y opcionalmente mapear
            for item in items:
                flat = flatten_source(item)

                if args.with_mapping:
                    source = item.get("_source", {})
                    mapped = apply_mapping(source)
                    # Fusionar: los campos mapeados tienen prioridad
                    record = {**flat, **mapped}
                else:
                    record = flat

                record["_periodo_recoleccion"] = periodo
                all_records.append(record)

            logger.info(
                "  -> %d expedientes recolectados (acumulado: %d)",
                len(items), len(all_records),
            )

            # Pausa cortés entre periodos
            if i < len(periods):
                time.sleep(0.5)

    if not all_records:
        logger.warning("No se recolectaron expedientes. Verifique los periodos.")
        return

    # Crear DataFrame
    df = pd.DataFrame(all_records)
    logger.info(
        "DataFrame creado: %d filas x %d columnas",
        len(df), len(df.columns),
    )

    # Guardar
    if args.append:
        count = store.append(df)
    else:
        count = store.save(df)

    logger.info("Dataset guardado: %d expedientes en %s", count, args.output)

    # Generar reporte de cobertura
    report_path = Path("docs/coverage_report.md")
    # Cargar el dataset completo para el reporte
    full_df = store.load()
    generate_coverage_report(full_df, report_path)

    # Resumen final
    print("\n" + "=" * 60)
    print("  RESUMEN DE RECOLECCIÓN")
    print("=" * 60)
    print(f"  Periodos consultados:  {len(periods)}")
    print(f"  Items totales de API:  {total_api_items}")
    print(f"  Registros guardados:   {count}")
    print(f"  Archivo de salida:     {args.output}")
    print(f"  Reporte de cobertura:  {report_path}")
    print("=" * 60)

    # Validar meta de ≥500
    if count >= 500:
        print(f"  ✅ META CUMPLIDA: {count} ≥ 500 expedientes")
    else:
        print(f"  ⚠️  META PENDIENTE: {count} < 500 expedientes")
        print(f"     Sugerencia: amplíe el rango de periodos.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
