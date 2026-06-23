from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from tc_pipeline.extraction.pdf_extractor import normalize_text, clean_extracted_section
from tc_pipeline.nlp.processing import clean_text


PROCESS_YEARS = set(range(2004, 2011))  # Only process 2004-2010
ROOT = Path("data")
SENTENCIA_EXTRACT = ROOT / "sentencia-Extract"
AUTO_EXTRACT = ROOT / "auto-resolucion-Extract"
MERGED_DIR = ROOT / "merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)


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


def clean_text_block(text: str) -> Tuple[str, dict]:
    orig = text or ""
    # normalize unicode and basic PDF cleaning
    t = normalize_text(orig)
    t = clean_extracted_section(t)
    t = clean_text(t)

    reversed_fixed = False
    t, reversed_fixed = detect_and_fix_reversed(t)

    noisy = is_noisy(t)

    # final whitespace collapse
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip(), {"original_len": len(orig), "cleaned_len": len(t), "noisy": noisy, "reversed_fixed": reversed_fixed}


def process_dir(extract_root: Path, doc_type: str):
    rows = []
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
                cleaned, meta = clean_text_block(text_raw)

                rows.append(
                    {
                        "numero_expediente": recd.get("numero_expediente") or recd.get("numero_sentencia") or "",
                        "year": year,
                        "doc_type": doc_type,
                        "source_file": str(csv_file),
                        "original_snippet": (text_raw[:300] if text_raw else ""),
                        "cleaned_text": cleaned,
                        "original_len": meta["original_len"],
                        "cleaned_len": meta["cleaned_len"],
                        "noisy": meta["noisy"],
                        "reversed_fixed": meta["reversed_fixed"],
                    }
                )

    return rows


def main() -> None:
    all_rows = []
    # sentencia
    all_rows.extend(process_dir(SENTENCIA_EXTRACT, "sentencia"))
    # autos
    all_rows.extend(process_dir(AUTO_EXTRACT, "auto-resolucion"))

    if not all_rows:
        print("No se procesaron registros. Revisa las rutas de extracción.")
        return

    df_merged = pd.DataFrame(all_rows)

    generated_files = []
    for year, group in sorted(df_merged.groupby("year")):
        year_csv = MERGED_DIR / f"expedientes_cleaned_{year}.csv"
        group.to_csv(year_csv, index=False, encoding="utf-8")
        generated_files.append(str(year_csv))

    summary = {
        "total_records": len(df_merged),
        "by_year": df_merged.groupby("year").size().to_dict(),
        "noisy_count": int(df_merged["noisy"].sum()),
        "reversed_fixed": int(df_merged["reversed_fixed"].sum()),
        "processed_years": sorted(list(PROCESS_YEARS)),
        "generated_files": generated_files,
    }

    print("Limpieza y merge completados.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
