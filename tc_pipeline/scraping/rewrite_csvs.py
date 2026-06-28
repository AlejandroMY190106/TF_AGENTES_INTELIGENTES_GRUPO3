import sys
import logging
from tc_pipeline.scraping.api_client import TribunalAPIClient
from tc_pipeline.config import PipelineConfig

def run_csv_rewrite():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    # We will use lower page_delay to speed up if possible, but keeping it safe
    config = PipelineConfig(page_delay=0.5)
    
    with TribunalAPIClient(config) as client:
        logger.info("Iniciando reescritura de todos los CSVs desde 1992 hasta 2026...")
        client.fetch_all_years_to_csv(start_year=1992, end_year=2026)
        logger.info("Reescritura de CSVs completada.")

if __name__ == "__main__":
    run_csv_rewrite()
