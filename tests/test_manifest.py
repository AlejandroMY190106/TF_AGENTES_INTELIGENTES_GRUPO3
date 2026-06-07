from pathlib import Path
import sqlite3

import pytest

from tc_pipeline.scraping.manifest import ManifestRepository

@pytest.fixture
def manifest(tmp_path):
    db_path = tmp_path / "test.db"
    repo = ManifestRepository(db_path)
    repo.initialize()
    yield repo
    repo.close()

def test_initialize_creates_table(manifest):
    conn = sqlite3.connect(manifest._db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='download_manifest';")
    assert cursor.fetchone() is not None
    conn.close()

def test_register_and_check_success(manifest):
    manifest.register_success("123", "2025-01", "path/123.pdf")
    
    assert manifest.already_processed("123") is True
    assert manifest.already_processed("999") is False

def test_register_failure(manifest):
    manifest.register_failure("123", "2025-01", "Timeout")
    
    assert manifest.already_processed("123") is False
    
    failed = manifest.get_failed()
    assert len(failed) == 1
    assert failed[0]["expediente"] == "123"
    assert failed[0]["error"] == "Timeout"

def test_update_status_from_failed_to_success(manifest):
    manifest.register_failure("123", "2025-01", "Timeout")
    assert manifest.already_processed("123") is False
    
    manifest.register_success("123", "2025-01", "path.pdf")
    assert manifest.already_processed("123") is True
    
    failed = manifest.get_failed()
    assert len(failed) == 0

def test_get_pending(manifest):
    manifest.register_pending("1", "2025-01")
    manifest.register_failure("2", "2025-01", "Err")
    manifest.register_success("3", "2025-01", "path")
    
    pending = manifest.get_pending("2025-01")
    assert len(pending) == 2
    expedientes = [p["expediente"] for p in pending]
    assert "1" in expedientes
    assert "2" in expedientes
    assert "3" not in expedientes

def test_period_stats(manifest):
    manifest.register_success("1", "2025-01", "path")
    manifest.register_success("2", "2025-01", "path")
    manifest.register_failure("3", "2025-01", "Err")
    manifest.register_pending("4", "2025-01")
    
    stats = manifest.get_period_stats("2025-01")
    assert stats["success"] == 2
    assert stats["failed"] == 1
    assert stats["pending"] == 1
    assert stats["total"] == 4
