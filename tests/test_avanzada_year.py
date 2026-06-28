"""Quick test: fetch one year (2025) using the new avanzada endpoint and verify sentido_resolucion."""
import csv
import logging
from tc_pipeline.scraping.api_client import TribunalAPIClient
from tc_pipeline.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = PipelineConfig(page_delay=0.5)

with TribunalAPIClient(config) as client:
    logger.info("Fetching year 2025 using avanzada endpoint...")
    csv_path = client.fetch_year_to_csv(2025)
    logger.info("CSV generated: %s", csv_path)

# Now read back and check sentido_resolucion
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    total = 0
    filled = 0
    samples = []
    for row in reader:
        total += 1
        if row.get("sentido_resolucion", "").strip():
            filled += 1
            if len(samples) < 5:
                samples.append((row["numero_expediente"], row["sentido_resolucion"]))

print(f"\nTotal records: {total}")
print(f"With sentido_resolucion: {filled} ({100*filled/total:.1f}%)")
print(f"Empty: {total - filled}")
print("\nSample values:")
for exp, sent in samples:
    print(f"  {exp}: {sent}")
