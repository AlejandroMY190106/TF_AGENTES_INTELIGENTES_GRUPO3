from unittest.mock import MagicMock, patch

import pytest
import requests

from tc_pipeline.config import PipelineConfig
from tc_pipeline.scraping.api_client import (
    APIError,
    APINonRetryableError,
    APIRetryExhaustedError,
    APIResponse,
    TribunalAPIClient,
)

@pytest.fixture
def config():
    return PipelineConfig(retry_base_delay=0.01)

@pytest.fixture
def mock_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "pagination": {"num_pages": 2, "total_item": 20},
        "data": [{"_source": {"numero_expediente": "123"}}]
    }
    return resp

def test_parse_response():
    client = TribunalAPIClient()
    raw = {
        "pagination": {"num_pages": 5, "total_item": 50},
        "data": [{"id": 1}, {"id": 2}]
    }
    response = client._parse_response(raw)
    assert response.total_pages == 5
    assert response.total_items == 50
    assert len(response.data) == 2

@patch("requests.Session.get")
def test_fetch_page_success(mock_get, config, mock_response):
    mock_get.return_value = mock_response
    
    with TribunalAPIClient(config) as client:
        response = client.fetch_page("2025-01", page=1)
        
    assert response.total_items == 20
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert kwargs["params"]["fecha_publicacion"] == "2025-01"
    assert kwargs["params"]["page"] == 1

@patch("requests.Session.get")
def test_retry_on_5xx(mock_get, config):
    error_resp = MagicMock()
    error_resp.status_code = 502
    
    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = {"pagination": {}, "data": []}
    
    # Falla 2 veces, éxito a la 3ra
    mock_get.side_effect = [error_resp, error_resp, success_resp]
    
    with TribunalAPIClient(config) as client:
        response = client.fetch_page("2025-01", page=1)
        
    assert mock_get.call_count == 3

@patch("requests.Session.get")
def test_non_retryable_error(mock_get, config):
    error_resp = MagicMock()
    error_resp.status_code = 404
    mock_get.return_value = error_resp
    
    with TribunalAPIClient(config) as client:
        with pytest.raises(APINonRetryableError):
            client.fetch_page("2025-01")

@patch("requests.Session.get")
def test_retry_exhausted(mock_get, config):
    error_resp = MagicMock()
    error_resp.status_code = 502
    mock_get.return_value = error_resp
    
    with TribunalAPIClient(config) as client:
        with pytest.raises(APIRetryExhaustedError):
            client.fetch_page("2025-01")
            
    assert mock_get.call_count == config.max_retries
