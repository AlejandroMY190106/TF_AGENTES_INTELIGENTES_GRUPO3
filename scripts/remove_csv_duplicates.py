"""
remove_csv_duplicates.py
------------------------
Elimina filas duplicadas de todos los archivos CSV en la carpeta data/merged.
Un duplicado se define como cualquier fila cuyo 'numero_expediente' ya aparecio
antes en el mismo archivo. Se conserva unicamente la PRIMERA ocurrencia.

Uso:
    python scripts/remove_csv_duplicates.py

El script opera IN-PLACE: sobreescribe cada CSV con su version sin duplicados.
Se muestra un resumen de filas originales, filas eliminadas y filas conservadas
para cada archivo.
"""

import os
import csv
import io
import sys

# Algunos campos de texto (fundamentos, motivos) superan el limite por defecto
# de 131,072 bytes del modulo csv. Esta linea elimina dicho limite.
csv.field_size_limit(sys.maxsize)

# ----------------------------------------------
# CONFIGURACION
# ----------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MERGED_DIR   = os.path.join(PROJECT_ROOT, "data", "merged")

KEY_COLUMN   = "numero_expediente"   # columna que identifica duplicados
ENCODING     = "utf-8-sig"           # maneja BOM si existiera
# ----------------------------------------------


def process_file(filepath: str):
    """
    Lee el CSV, elimina duplicados por KEY_COLUMN (conservando la primera
    aparicion) y lo sobreescribe.

    Retorna:
        (total_rows, kept_rows, removed_rows)
    """
    # --- Lectura completa en memoria ---
    with open(filepath, "r", encoding=ENCODING, newline="") as f:
        content = f.read()

    reader = csv.DictReader(io.StringIO(content))

    if KEY_COLUMN not in (reader.fieldnames or []):
        print(f"  WARNING  Columna '{KEY_COLUMN}' no encontrada -- archivo omitido.")
        return 0, 0, 0

    fieldnames = reader.fieldnames  # preservar orden original de columnas

    seen_ids = set()
    kept_rows = []
    total = 0

    for row in reader:
        total += 1
        exp_id = row[KEY_COLUMN].strip()
        if exp_id not in seen_ids:
            seen_ids.add(exp_id)
            kept_rows.append(row)

    removed = total - len(kept_rows)

    # --- Escritura (sobreescritura in-place) ---
    with open(filepath, "w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,   # mantener consistencia con el formato original
        )
        writer.writeheader()
        writer.writerows(kept_rows)

    return total, len(kept_rows), removed


def main():
    if not os.path.isdir(MERGED_DIR):
        print(f"[ERROR] No se encontro la carpeta: {MERGED_DIR}")
        sys.exit(1)

    csv_files = sorted(
        f for f in os.listdir(MERGED_DIR) if f.lower().endswith(".csv")
    )

    if not csv_files:
        print(f"[INFO] No se encontraron archivos CSV en: {MERGED_DIR}")
        sys.exit(0)

    print()
    print("=" * 65)
    print("  Eliminacion de duplicados -- carpeta: data/merged")
    print(f"  Columna clave: '{KEY_COLUMN}'")
    print("=" * 65)
    print()

    grand_total = grand_kept = grand_removed = 0

    for filename in csv_files:
        filepath = os.path.join(MERGED_DIR, filename)
        print(f">> {filename}")

        try:
            total, kept, removed = process_file(filepath)
        except Exception as exc:
            print(f"   [ERROR] {exc}")
            print()
            continue

        grand_total   += total
        grand_kept    += kept
        grand_removed += removed

        if removed == 0:
            status = "sin duplicados"
        else:
            status = f"{removed} duplicados eliminados"

        print(f"   Filas originales : {total:>7,}")
        print(f"   Filas conservadas: {kept:>7,}")
        print(f"   Filas eliminadas : {removed:>7,}   [{status}]")
        print()

    print("=" * 65)
    print("  RESUMEN GLOBAL")
    print(f"  Archivos procesados : {len(csv_files)}")
    print(f"  Filas originales    : {grand_total:>9,}")
    print(f"  Filas conservadas   : {grand_kept:>9,}")
    print(f"  Filas eliminadas    : {grand_removed:>9,}")
    print("=" * 65)
    print()
    print("Proceso completado. Los archivos han sido sobreescritos.")


if __name__ == "__main__":
    main()
