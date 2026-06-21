from tc_pipeline.scraping.api_client import TribunalAPIClient
import json

def test_api():
    client = TribunalAPIClient()
    print("Consultando API (2026-01)...")
    
    first_page = client.fetch_page("2026-01", page=1)
    if first_page.total_items == 0:
        return
        
    for page in range(1, first_page.total_pages + 1):
        print(f"Checking page {page}...")
        response = client.fetch_page("2026-01", page=page)
        for item in response.data:
            source = item.get("_source", {})
            has_sentencia_sentido = "sentencia_sentido" in source
            has_sentido = "sentido" in source
            has_sistematizacion_sentido = False
            
            sist = source.get("sistematizacion", [])
            if isinstance(sist, list) and len(sist) > 0:
                if "sentido" in sist[0] or "sentencia_sentido" in sist[0]:
                    has_sistematizacion_sentido = True
                    
            if has_sentencia_sentido or has_sentido or has_sistematizacion_sentido:
                print(f"Found record {source.get('numero_expediente')}")
                if has_sentencia_sentido: print("Has sentencia_sentido")
                if has_sentido: print("Has sentido directly")
                if has_sistematizacion_sentido: print("Has sistematizacion->sentido")
                
                print(json.dumps(item, indent=2, ensure_ascii=False))
                print("\nExtracted Record:")
                print(json.dumps(client.extract_record_for_csv(item), indent=2, ensure_ascii=False))
                return

if __name__ == "__main__":
    test_api()
