from tc_pipeline.scraping.api_client import TribunalAPIClient
import json

def test_api():
    client = TribunalAPIClient()
    print("Consultando API (2026-01)...")
    
    first_page = client.fetch_page("2026-01", page=1)
    if first_page.total_items == 0:
        return
        
def test_api():
    client = TribunalAPIClient()
    for p in range(1, 20):
        response = client.fetch_page("2023-01", page=p)
        for item in response.data:
            source = item.get("_source", {})
            has_sent = "sentido" in source or "sentencia_sentido" in source or "sistematizacion" in source
            if has_sent:
                print(f"FOUND in {source.get('numero_expediente')}")
                return
    print("NO RECORD WITH SENTIDO FOUND IN 20 PAGES")

if __name__ == "__main__":
    test_api()
