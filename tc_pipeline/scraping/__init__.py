"""
tc_pipeline.scraping
────────────────────
Paquete de scraping del pipeline del Tribunal Constitucional.

Exporta los tres componentes principales:
- TribunalAPIClient / APIResponse  → comunicación con la API
- PDFDownloader / DownloadMetrics   → descarga concurrente de PDFs
"""

from tc_pipeline.scraping.api_client import APIResponse, TribunalAPIClient
from tc_pipeline.scraping.downloader import DownloadMetrics, PDFDownloader

__all__ = [
    "TribunalAPIClient",
    "APIResponse",
    "PDFDownloader",
    "DownloadMetrics",
]
