"""
scripts/analyze_class_distribution.py
───────────────────────────────────────
Análisis de distribución de clases en los CSVs de data/merged/.

Escanea todos los archivos expedientes_cleaned_*.csv, cuenta registros por
valor de 'sentido_resolucion', identifica outliers/singletons, y genera un
reporte completo en consola y en un archivo de texto.

Uso:
    python scripts/analyze_class_distribution.py
    python scripts/analyze_class_distribution.py --threshold 20
    python scripts/analyze_class_distribution.py --output reports/class_distribution.txt
"""

from __future__ import annotations

import argparse
import os
import sys

# ─── Forzar UTF-8 en la consola de Windows ───────────────────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import pandas as pd

# ─── Configuración de rutas ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MERGED_DIR = _PROJECT_ROOT / "data" / "merged"
_REPORT_DIR = _PROJECT_ROOT / "reports"

# Columna objetivo
LABEL_COL = "sentido_resolucion"

# Las 4 macroetiquetas canónicas objetivo del proyecto
TARGET_CLASSES = {"Fundada", "Infundada", "Improcedente", "Procedente"}

# ─── Mapeo canónico: variantes → macroetiqueta ────────────────────────────────
CANONICAL_MAP = {
    # Fundada y variantes directas (1 solo fallo dominante)
    "Fundada": "Fundada",
    "Fundada en parte": "Fundada",
    "Fundada en mayoría": "Fundada",
    "Fundada por mayoría": "Fundada",
    "Fundado en parte": "Fundada",
    "Fundado el desistimiento": "Fundada",
    "Fundada el desistimiento": "Fundada",
    "Fundada en parte/ Infundada": "Fundada",
    "Fundada/ Fundada": "Fundada",
    "Fundada extremo": "Fundada",
    "Fundadas/ Improcedente": "Fundada",

    # Infundada y variantes
    "Infundada": "Infundada",
    "Infundadas": "Infundada",
    "Infundada en parte": "Infundada",
    "Infundada por mayoría": "Infundada",
    "Iinfundada": "Infundada",
    "Infunfada": "Infundada",
    "Infudada": "Infundada",
    "Infudanda": "Infundada",
    "Infudado": "Infundada",
    "Infunda/ Improcedente": "Infundada",
    "Infundada/ Infundada": "Infundada",
    "Infundada y/o Improcedente": "Infundada",

    # Improcedente y variantes
    "Improcedente": "Improcedente",
    "Improcedencia": "Improcedente",
    "Improcedente el RAC": "Improcedente",
    "Inadmisible": "Improcedente",
    "Improccedente": "Improcedente",
    "Improdecente": "Improcedente",
    "Improcednte": "Improcedente",
    "Improcendente / Infundada": "Improcedente",
    "Improcredente": "Improcedente",
    "Improcedente/ Improcedente": "Improcedente",
    "Inadmisible la demanda (PI-CC)": "Improcedente",
    "Improcedente la demanda (Autos)": "Improcedente",
    "Improcedente la demanda (PI-CC)": "Improcedente",

    # Procedente — no aparece en los datos originales
    "Procedente": "Procedente",
}


def load_all_csvs(merged_dir: Path) -> pd.DataFrame:
    """Carga todos los CSVs de data/merged/ en un único DataFrame."""
    csv_files = sorted(merged_dir.glob("expedientes_cleaned_*.csv"))
    if not csv_files:
        print(f"[ERROR] No se encontraron archivos CSV en: {merged_dir}")
        sys.exit(1)

    frames = []
    print(f"\n  Escaneando {len(csv_files)} archivos en '{merged_dir}'...\n")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(
                csv_path,
                usecols=[LABEL_COL],
                dtype={LABEL_COL: str},
                low_memory=False,
            )
            df["_source"] = csv_path.name
            frames.append(df)
        except Exception as exc:
            print(f"  [WARN] No se pudo leer '{csv_path.name}': {exc}")

    return pd.concat(frames, ignore_index=True)


def build_report(df: pd.DataFrame, threshold: int) -> str:
    """Genera el análisis completo y devuelve el reporte como string."""

    lines: list[str] = []
    SEP = "=" * 80
    SUB = "-" * 80

    total_records = len(df)
    null_count = int(df[LABEL_COL].isna().sum())
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    valid_total = len(df)

    # ── Distribución completa ─────────────────────────────────────────────────
    class_counts = df[LABEL_COL].value_counts()
    n_classes = len(class_counts)

    # ── Segmentación por tamaño ───────────────────────────────────────────────
    singletons   = class_counts[class_counts == 1]
    few_samples  = class_counts[(class_counts >= 2) & (class_counts < threshold)]
    large_enough = class_counts[class_counts >= threshold]

    keep_count      = int(large_enough.sum())
    eliminate_count = valid_total - keep_count

    # ── Proyección canónica ───────────────────────────────────────────────────
    df["canonical"] = df[LABEL_COL].map(CANONICAL_MAP)
    canonical_counts = df["canonical"].dropna().value_counts()

    # ─────────────────────────────────────────────────────────────────────────
    lines += [
        "",
        SEP,
        "  ANÁLISIS DE DISTRIBUCIÓN DE CLASES — data/merged/",
        SEP,
        "",
        f"  Total de registros escaneados : {total_records:>10,}",
        f"  Registros con etiqueta nula   : {null_count:>10,}",
        f"  Registros con etiqueta válida : {valid_total:>10,}",
        f"  Clases únicas encontradas     : {n_classes:>10,}",
        "",
        SUB,
        "  1. DISTRIBUCIÓN COMPLETA POR CLASE (orden descendente de frecuencia)",
        SUB,
        f"  {'Rango':>5}  {'Count':>9}  {'%Total':>7}  Clase",
        f"  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*55}",
    ]

    for rank, (label, count) in enumerate(class_counts.items(), start=1):
        pct = count / valid_total * 100
        flag = "  " if count >= threshold else ("S " if count == 1 else "F ")
        lines.append(f"  {rank:>5}  {count:>9,}  {pct:>6.2f}%  [{flag}] {label}")

    lines += [
        "",
        "  Leyenda: [S ] = Singleton (1 registro) | [F ] = Few (<threshold) | [  ] = Viable",
        "",
        SUB,
        "  2. MACROETIQUETAS CANÓNICAS OBJETIVO DEL PROYECTO",
        SUB,
    ]

    for label in sorted(TARGET_CLASSES):
        count = int(class_counts.get(label, 0))
        pct = count / valid_total * 100 if valid_total > 0 else 0.0
        status = "OK " if count > threshold else ("AUSENTE" if count == 0 else "CRITICO")
        lines.append(f"  [{status}]  {label:<30}  {count:>9,}  ({pct:.2f}%)")

    lines += [
        "",
        "  NOTA: Si 'Procedente' tiene 0 registros, la etiqueta no existe en el corpus",
        "        o está expresada con otra variante. Verificar manualmente.",
        "",
        SUB,
        f"  3. OUTLIERS Y SINGLETONS (threshold = {threshold} registros mínimos)",
        SUB,
        f"  Singletons (exactamente 1 registro)    : {len(singletons):>5} clases → {int(singletons.sum()):>9,} registros",
        f"  Pocos datos (2 a {threshold-1} registros)          : {len(few_samples):>5} clases → {int(few_samples.sum()):>9,} registros",
        f"  Clases viables (>= {threshold} registros)       : {len(large_enough):>5} clases → {keep_count:>9,} registros",
        "",
        "  SINGLETONS (1 registro exacto — se eliminan sin duda):",
    ]
    for label in singletons.index:
        lines.append(f"    • {label}")

    lines += [
        "",
        f"  POCOS DATOS (2–{threshold-1} registros — se recomiendan eliminar):",
    ]
    for label, count in few_samples.sort_values().items():
        lines.append(f"    • ({count:>3} reg.)  {label}")

    lines += [
        "",
        SUB,
        "  4. IMPACTO CUANTITATIVO DE LA ELIMINACIÓN DE OUTLIERS",
        SUB,
        f"  Registros CONSERVADOS tras limpieza (>= {threshold} reg.)  : {keep_count:>9,}  ({keep_count/valid_total*100:.2f}%)",
        f"  Registros ELIMINADOS  (<  {threshold} reg.)               : {eliminate_count:>9,}  ({eliminate_count/valid_total*100:.2f}%)",
        f"  Clases eliminadas                                  : {len(singletons) + len(few_samples):>9,}",
        f"  Clases supervivientes                              : {len(large_enough):>9,}",
        "",
        SUB,
        "  5. PROYECCIÓN: CONTEOS TRAS NORMALIZACIÓN CANÓNICA (mapeo a 3 macroetiquetas)",
        SUB,
        "  Aplicando CANONICAL_MAP sobre el corpus completo:",
        "",
    ]
    proj_total = int(canonical_counts.sum())
    for label, count in canonical_counts.sort_values(ascending=False).items():
        pct = count / valid_total * 100
        lines.append(f"  {label:<30}  {count:>9,}  ({pct:.2f}%)")
    lines += [
        "",
        f"  Total recuperado tras normalización parcial : {proj_total:>9,}",
        f"  Registros sin mapeo (descartados)           : {valid_total - proj_total:>9,}",
        "",
        SUB,
        "  6. INSTRUCCIONES DE RELIMPIEZA PARA expedientes_cleaned_*.csv",
        SUB,
        "  Columna a operar: 'sentido_resolucion'",
        "",
        "  PASO 1 — ELIMINAR filas donde 'sentido_resolucion':",
        "    a) Sea NaN / vacío",
        "    b) Pertenezca a cualquier clase con < {th} ocurrencias en el corpus total".format(th=threshold),
        "    c) Sea una combinación compuesta que no puede mapearse unívocamente",
        "       (ej: 'Fundada/ Improcedente', 'Infundada / Fundada en parte')",
        "",
        "  PASO 2 — NORMALIZAR los valores restantes:",
        "    'Fundada en parte'      → 'Fundada'",
        "    'Infundada en parte'    → 'Infundada'",
        "    'Iinfundada'/'Infunfada'/'Infudada' → 'Infundada'  (typos)",
        "    'Improcedencia'/'Improdecente' → 'Improcedente'",
        "    'Inadmisible'           → 'Improcedente'",
        "",
        "  PASO 3 — FILTRO FINAL: conservar solo filas con:",
        "    sentido_resolucion IN ('Fundada', 'Infundada', 'Improcedente')",
        "",
        "  Archivos afectados: expedientes_cleaned_1992.csv → expedientes_cleaned_2026.csv",
        "",
        SEP,
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza distribución de clases en data/merged/"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Mínimo de registros para clase viable (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del archivo de salida (default: reports/class_distribution_report.txt)",
    )
    args = parser.parse_args()

    if not _MERGED_DIR.exists():
        print(f"[ERROR] Directorio no encontrado: {_MERGED_DIR}")
        sys.exit(1)

    df = load_all_csvs(_MERGED_DIR)
    report = build_report(df, threshold=args.threshold)
    print(report)

    out_path = Path(args.output) if args.output else _REPORT_DIR / "class_distribution_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"  [OK] Reporte guardado en: {out_path}")


if __name__ == "__main__":
    main()
