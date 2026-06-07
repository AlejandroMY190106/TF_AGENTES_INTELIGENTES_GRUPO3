from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from tc_pipeline.config import PipelineConfig
from tc_pipeline.scraping.downloader import PDFDownloader, DownloadMetrics

@pytest.fixture
def config(tmp_path):
    return PipelineConfig(download_root=tmp_path)

def test_build_path(config):
    downloader = PDFDownloader(config)
    path = downloader.build_path("01234/2025 AA", "2025-01")
    assert path == config.download_root / "2025" / "01" / "01234_2025AA.pdf"

@patch("requests.get")
def test_download_pdf_success(mock_get, config):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"pdf_content"
    mock_get.return_value = mock_resp
    
    downloader = PDFDownloader(config)
    dest = config.download_root / "test.pdf"
    
    status = downloader.download_pdf("http://test.com/1.pdf", dest)
    
    assert status == "descargado"
    assert dest.exists()
    assert dest.read_bytes() == b"pdf_content"

def test_download_pdf_skip_existing(config, tmp_path):
    dest = tmp_path / "test.pdf"
    dest.write_bytes(b"existing")
    
    downloader = PDFDownloader(config)
    status = downloader.download_pdf("http://test.com/1.pdf", dest)
    
    assert status == "existente"
    assert dest.read_bytes() == b"existing"

@patch("requests.get")
def test_download_batch(mock_get, config):
    # Mockear un éxito y un error
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.content = b"content"
    
    mock_error = MagicMock()
    mock_error.status_code = 404
    
    def side_effect(url, **kwargs):
        if "1.pdf" in url:
            return mock_success
        return mock_error
        
    mock_get.side_effect = side_effect
    
    downloader = PDFDownloader(config)
    items = [
        {"_source": {"numero_expediente": "1", "url_archivo": "http://1.pdf"}},
        {"_source": {"numero_expediente": "2", "url_archivo": "http://2.pdf"}},
        {"_source": {"numero_expediente": "", "url_archivo": ""}} # Inválido
    ]
    
    metrics = downloader.download_batch(items, "2025-01")
    
    assert metrics.descargados == 1
    assert metrics.errores == 2
    assert metrics.existentes == 0
