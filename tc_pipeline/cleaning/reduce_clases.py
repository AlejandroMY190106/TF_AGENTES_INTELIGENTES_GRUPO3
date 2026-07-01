"""
tc_pipeline/cleaning/reduce_clases.py
──────────────────────────────────────
Reducción Canónica de Clases — 217 etiquetas → 3 macroetiquetas.

Responsabilidad:
    Lee todos los archivos expedientes_cleaned_*.csv de data/merged/,
    normaliza la columna 'sentido_resolucion' aplicando un mapa canónico
    exhaustivo derivado del análisis de distribución de clases, y persiste
    los CSVs limpios en data/merged_cleaned/ conservando el esquema completo
    de columnas del pipeline.

Macroetiquetas objetivo:
    • Fundada      (incluye variantes directas y correcciones de typos)
    • Infundada    (incluye variantes directas y correcciones de typos)
    • Improcedente (incluye variantes directas, Inadmisible, Improcedencia)

Registros descartados:
    • Etiquetas compuestas ambiguas (ej: 'Fundada/ Improcedente')
    • Clases sin mapeo a ninguna macroetiqueta
    • Etiquetas NaN / vacías

Uso:
    python tc_pipeline/cleaning/reduce_clases.py
    python tc_pipeline/cleaning/reduce_clases.py --input-dir data/merged --output-dir data/merged_cleaned
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

# ─── Forzar UTF-8 en la consola de Windows ────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── Rutas por defecto ────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_MERGED_DIR   = _PROJECT_ROOT / "data" / "merged"
_OUTPUT_DIR   = _PROJECT_ROOT / "data" / "merged_cleaned"

LABEL_COL = "sentido_resolucion"

# ─── Mapa canónico exhaustivo ──────────────────────────────────────────────────
# Derivado directamente del reporte reports/class_distribution_report.txt.
# Sólo se incluyen etiquetas que corresponden a UN fallo dominante sin ambigüedad.
# Las etiquetas compuestas (dos fallos separados por '/' o ',') NO se mapean aquí
# y quedarán como None → serán descartadas.
CANONICAL_MAP: dict[str, str] = {

    # ── FUNDADA ────────────────────────────────────────────────────────────────
    "Fundada":                          "Fundada",
    "Fundada en parte":                 "Fundada",
    "Fundada en parte/ Infundada":      "Fundada",    # fallo principal: Fundada
    "Fundada en mayoría":               "Fundada",
    "Fundada por mayoría":              "Fundada",
    "Fundado en parte":                 "Fundada",
    "Fundado el desistimiento":         "Fundada",
    "Fundada el desistimiento":         "Fundada",
    "Fundada extremo":                  "Fundada",
    "Fundada/ Fundada":                 "Fundada",

    # ── INFUNDADA ─────────────────────────────────────────────────────────────
    "Infundada":                        "Infundada",
    "Infundadas":                       "Infundada",
    "Infundada en parte":               "Infundada",
    "Infundada por mayoría":            "Infundada",
    "Infundada/ Infundada":             "Infundada",
    # Typos
    "Iinfundada":                       "Infundada",
    "Infunfada":                        "Infundada",
    "Infudada":                         "Infundada",
    "Infudanda":                        "Infundada",
    "Infudado":                         "Infundada",

    # ── IMPROCEDENTE ──────────────────────────────────────────────────────────
    "Improcedente":                     "Improcedente",
    "Improcedencia":                    "Improcedente",
    "Improcedente el RAC":              "Improcedente",
    "Inadmisible":                      "Improcedente",
    "Inadmisible la demanda (PI-CC)":   "Improcedente",
    "Improcedente la demanda (Autos)":  "Improcedente",
    "Improcedente la demanda (PI-CC)":  "Improcedente",
    "Improcedente la incorporación":    "Improcedente",
    "Improcedente/ Improcedente":       "Improcedente",
    # Typos
    "Improccedente":                    "Improcedente",
    "Improdecente":                     "Improcedente",
    "Improcednte":                      "Improcedente",
    "Improcredente":                    "Improcedente",
    "Improcendente / Infundada":        "Improcedente",  # typo severo — fallo único
    "Improcednete/ Infundada":          "Improcedente",  # typo severo — fallo único
}

# Conjunto final de etiquetas aceptadas
KEEP_LABELS: frozenset[str] = frozenset({"Fundada", "Infundada", "Improcedente"})


def _normalize_label(raw: str) -> str | None:
    """Normaliza una etiqueta cruda a su macroetiqueta canónica.

    Returns:
        La macroetiqueta canónica, o ``None`` si la etiqueta no tiene
        mapeo unívoco (compuesta, singleton, desconocida).
    """
    if not isinstance(raw, str):
        return None
    label = raw.strip()
    if not label:
        return None
    return CANONICAL_MAP.get(label, None)


def process_csv(src: Path, dst: Path) -> tuple[int, int, Counter]:
    """Procesa un CSV individual, aplica la normalización y lo guarda en dst.

    Returns:
        (filas_leidas, filas_guardadas, counter_por_clase)
    """
    try:
        df = pd.read_csv(src, dtype=str, low_memory=False, keep_default_na=False)
    except Exception as exc:
        print(f"  [WARN] No se pudo leer '{src.name}': {exc}")
        return 0, 0, Counter()

    rows_in = len(df)

    # Normalizar etiqueta
    df[LABEL_COL] = df[LABEL_COL].apply(_normalize_label)

    # Eliminar filas sin mapeo (None → descartado)
    df = df[df[LABEL_COL].notna()].copy()
    df = df[df[LABEL_COL].isin(KEEP_LABELS)].copy()

    rows_out = len(df)
    class_counter: Counter = Counter(df[LABEL_COL].tolist())

    if rows_out == 0:
        print(f"  [SKIP] '{src.name}' → sin registros válidos tras normalización.")
        return rows_in, 0, class_counter

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8")

    return rows_in, rows_out, class_counter


def main(input_dir: Path, output_dir: Path) -> None:
    csv_files = sorted(input_dir.glob("expedientes_cleaned_*.csv"))
    if not csv_files:
        print(f"[ERROR] No se encontraron CSVs en: {input_dir}")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  reduce_clases.py — Reducción canónica a 3 macroetiquetas")
    print("=" * 72)
    print(f"  Origen  : {input_dir}")
    print(f"  Destino : {output_dir}")
    print(f"  Archivos: {len(csv_files)} CSVs a procesar")
    print()

    total_in   = 0
    total_out  = 0
    global_cnt: Counter = Counter()

    for src in csv_files:
        dst = output_dir / src.name
        rows_in, rows_out, cnt = process_csv(src, dst)
        total_in  += rows_in
        total_out += rows_out
        global_cnt.update(cnt)

        retained_pct = (rows_out / rows_in * 100) if rows_in > 0 else 0.0
        print(
            f"  {src.name:<40}  {rows_in:>6} → {rows_out:>6}  "
            f"({retained_pct:.1f}% retenido)"
        )

    discarded = total_in - total_out

    print()
    print("=" * 72)
    print("  RESUMEN GLOBAL")
    print("=" * 72)
    print(f"  Registros leídos         : {total_in:>10,}")
    print(f"  Registros retenidos      : {total_out:>10,}  ({total_out/total_in*100:.2f}%)")
    print(f"  Registros descartados    : {discarded:>10,}  ({discarded/total_in*100:.2f}%)")
    print()
    print("  Distribución por macroetiqueta:")
    for label in ("Improcedente", "Infundada", "Fundada"):
        cnt = global_cnt.get(label, 0)
        pct = cnt / total_out * 100 if total_out > 0 else 0.0
        print(f"    {label:<20}  {cnt:>8,}  ({pct:.2f}%)")
    print()
    print(f"  CSVs limpios guardados en: {output_dir}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce sentido_resolucion a 3 macroetiquetas canónicas."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(_MERGED_DIR),
        help=f"Directorio de CSVs originales (default: {_MERGED_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_OUTPUT_DIR),
        help=f"Directorio de salida con CSVs limpios (default: {_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    main(Path(args.input_dir), Path(args.output_dir))
