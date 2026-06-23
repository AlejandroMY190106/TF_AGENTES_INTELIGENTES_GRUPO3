from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from tc_pipeline.extraction.pdf_extractor import normalize_text, clean_extracted_section
from tc_pipeline.nlp.processing import clean_text


PROCESS_YEARS = set(range(1991,2027))  
ROOT = Path("data")
SENTENCIA_EXTRACT = ROOT / "sentencia-Extract"
AUTO_EXTRACT = ROOT / "auto-resolucion-Extract"
JSON_CSV_DIR = ROOT / "csv"
MERGED_DIR = ROOT / "merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

# Esquema final de columnas para los CSVs cleaned
FINAL_COLUMNS = [
    "numero_expediente",
    "url_archivo_TC",
    "url_archivo_original",
    "tipo_expediente",
    "motivos_demanda",
    "sentido_resolucion",
    "fundamentos",
]


COMMON_TOKENS = [
    "ANTECEDENTES",
    "FUNDAMENTOS",
    "HA RESUELTO",
    "FALLA",
    "RESUELVE",
    "VISTO",
    "EXPEDIENTE",
]


def detect_and_fix_reversed(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    up = text.upper()
    for tok in COMMON_TOKENS:
        if tok in up:
            return text, False

    rev = text[::-1]
    rev_up = rev.upper()
    for tok in COMMON_TOKENS:
        if tok in rev_up:
            # probable full reversal
            return rev, True

    return text, False


def is_noisy(text: str) -> bool:
    if not text:
        return True
    # replacement character or many non-letter characters
    if "�" in text or "\ufffd" in text:
        return True
    letters = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]+", text)
    total_chars = len(text)
    letters_chars = sum(len(w) for w in letters)
    if total_chars == 0:
        return True
    if (total_chars - letters_chars) / total_chars > 0.45:
        return True
    return False


def extract_text_from_record(rec: dict) -> str:
    # Prefer 'antecedentes' or 'asunto' (secciones PDF extraídas), then other text sources
    for k in ("antecedentes", "ANTECEDENTES", "asunto", "ASUNTO"):
        if k in rec and rec[k]:
            val = str(rec[k]).strip()
            if val and val.lower() != "nan":
                return val

    # Fallback: try fundamentos or visto (campos JSON alternativos)
    for k in ("fundamentos", "FUNDAMENTOS", "visto", "VISTO"):
        if k in rec and rec[k]:
            val = str(rec[k]).strip()
            if val and val.lower() != "nan":
                return val

    att = rec.get("attachment") or rec.get("attachment.content")
    if isinstance(att, dict) and att.get("content"):
        return str(att.get("content"))
    if isinstance(att, str) and att:
        return str(att)

    # fallback: find any key with 'texto' or 'content'
    for k, v in rec.items():
        if v and ("texto" in k.lower() or "content" in k.lower()):
            val = str(v).strip()
            if val and val.lower() != "nan":
                return val

    return ""


def clean_text_block(text: str) -> str:
    """Limpia un bloque de texto extraído de PDF y devuelve el texto limpio."""
    orig = text or ""
    # normalize unicode and basic PDF cleaning
    t = normalize_text(orig)
    t = clean_extracted_section(t)
    t = clean_text(t)

    t, _ = detect_and_fix_reversed(t)

    # final whitespace collapse
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


# ---------------------------------------------------------------------------
# Fuente 1: CSVs extraídos de PDFs (sentencia-Extract / auto-resolucion-Extract)
# ---------------------------------------------------------------------------

def process_pdf_dir(extract_root: Path) -> list[dict]:
    """Procesa CSVs de sentencia-Extract o auto-resolucion-Extract.

    Devuelve filas con:
      - numero_expediente
      - url_archivo_original  (antes source_file)
      - motivos_demanda       (antes cleaned_text)
    """
    rows: list[dict] = []
    for year_path in sorted(extract_root.iterdir() if extract_root.exists() else []):
        if not year_path.is_dir() or not year_path.name.isdigit():
            continue
        year = int(year_path.name)
        if year not in PROCESS_YEARS:
            continue

        for csv_file in sorted(year_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, dtype=str, encoding="utf-8", keep_default_na=False)
            except Exception:
                try:
                    df = pd.read_csv(csv_file, dtype=str, encoding="latin-1", keep_default_na=False)
                except Exception:
                    continue

            for _, rec in df.iterrows():
                recd = rec.to_dict()
                text_raw = extract_text_from_record(recd)
                cleaned = clean_text_block(text_raw)

                rows.append(
                    {
                        "numero_expediente": (
                            recd.get("numero_expediente")
                            or recd.get("numero_sentencia")
                            or ""
                        ),
                        "url_archivo_original": str(csv_file),
                        "motivos_demanda": cleaned,
                        "year_origen": str(year),
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Fuente 2: CSVs extraídos de JSONs (data/csv/expedientes-json-YYYY.csv)
# ---------------------------------------------------------------------------

def load_json_csvs() -> pd.DataFrame:
    """Carga los CSVs de la fuente JSON y selecciona / renombra columnas.

    Columnas extraídas:
      numero_expediente          (clave de vinculación)
      url_archivo  -> url_archivo_TC
      tipo_expediente
      sentido_resolucion
      fundamentos
    """
    frames: list[pd.DataFrame] = []

    for csv_file in sorted(JSON_CSV_DIR.glob("expedientes-json-*.csv")):
        # Extraer año del nombre del archivo
        match = re.search(r"(\d{4})", csv_file.stem)
        if not match:
            continue
        year = int(match.group(1))
        if year not in PROCESS_YEARS:
            continue

        try:
            df = pd.read_csv(csv_file, dtype=str, encoding="utf-8", keep_default_na=False)
        except Exception:
            try:
                df = pd.read_csv(csv_file, dtype=str, encoding="latin-1", keep_default_na=False)
            except Exception:
                continue

        # Seleccionar solo las columnas necesarias (las que existan)
        cols_needed = [
            "numero_expediente",
            "url_archivo",
            "tipo_expediente",
            "sentido_resolucion",
            "fundamentos",
        ]
        available = [c for c in cols_needed if c in df.columns]
        df_subset = df[available].copy()

        # Renombrar url_archivo -> url_archivo_TC
        if "url_archivo" in df_subset.columns:
            df_subset = df_subset.rename(columns={"url_archivo": "url_archivo_TC"})

        frames.append(df_subset)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # Eliminar duplicados por numero_expediente (conservar primera aparición)
    if "numero_expediente" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["numero_expediente"], keep="first")

    return df_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Procesar PDFs de ambas fuentes ──────────────────────────────────
    pdf_rows: list[dict] = []
    pdf_rows.extend(process_pdf_dir(SENTENCIA_EXTRACT))
    pdf_rows.extend(process_pdf_dir(AUTO_EXTRACT))

    if not pdf_rows:
        print("[WARNING] No se procesaron registros PDF. Revisa las rutas de extracción.")
        return

    df_pdf = pd.DataFrame(pdf_rows)
    print(f"  PDF: {len(df_pdf)} registros procesados")

    # ── 2. Cargar datos de la fuente JSON ──────────────────────────────────
    df_json = load_json_csvs()
    print(f"  JSON: {len(df_json)} registros cargados")

    # ── 3. Merge: vincular PDF con JSON por numero_expediente ──────────────
    if not df_json.empty:
        df_merged = pd.merge(df_pdf, df_json, on="numero_expediente", how="left")
    else:
        df_merged = df_pdf.copy()

    # Asegurar que todas las columnas finales existan (rellenar con "" si faltan)
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = ""

    # Reordenar al esquema final manteniendo la columna temporal
    cols = FINAL_COLUMNS + (["year_origen"] if "year_origen" in df_merged.columns else [])
    df_merged = df_merged[cols]

    # ── 4. Agrupar y guardar CSVs por año ──────────────────────────────────
    if "year_origen" in df_merged.columns:
        df_merged["_year"] = df_merged["year_origen"]
    else:
        df_merged["_year"] = "sin_año"

    generated_files: list[str] = []
    for year, group in sorted(df_merged.groupby("_year")):
        year_csv = MERGED_DIR / f"expedientes_cleaned_{year}.csv"
        group.drop(columns=["_year", "year_origen"], errors="ignore").to_csv(year_csv, index=False, encoding="utf-8")
        generated_files.append(str(year_csv))

    # ── 5. Resumen ─────────────────────────────────────────────────────────
    summary = {
        "total_records": len(df_merged),
        "by_year": df_merged.groupby("_year").size().to_dict(),
        "processed_years": sorted(list(PROCESS_YEARS)),
        "generated_files": generated_files,
        "columns": FINAL_COLUMNS,
    }

    print("\n=== Limpieza y merge completados ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()