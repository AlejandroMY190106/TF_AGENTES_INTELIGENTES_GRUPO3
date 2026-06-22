"""Test the CORRECT advanced search endpoint: /avanzada (not /avanzado)."""

import requests
import json

url = "https://jurisbackend.sedetc.gob.pe/api/visitor/sentencia/busqueda/avanzada"
params = {
    "page": 1,
    "search": "",
    "numero_expediente": "",
    "nombre_demandante": "",
    "nombre_demandado": "",
    "fecha_publicacion": "2025-05",
    "sentencia_sentido": "",
    "id_sentencia_distrito": "",
    "id_sentencia_sala": "",
    "id_sentencia_tipo": "",
    "palabras_claves": "",
    "palabras": "",
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

r = requests.get(url, params=params, headers=headers, timeout=15)
print(f"GET status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    total = data.get("total")
    pagination = data.get("pagination")
    print(f"Total items: {total}")
    print(f"Pagination: {pagination}")
    if data.get("data"):
        first = data["data"][0]
        source = first.get("_source", {})
        print(f"\nKeys in _source: {list(source.keys())}")
        print(f"\nsentido: {source.get('sentido')}")
        print(f"id_sentencia_sentido: {source.get('id_sentencia_sentido')}")
        print(f"sistematizacion: {type(source.get('sistematizacion'))}")
        print(f"tesaurio: {type(source.get('tesaurio'))}")
        
        # Test extraction
        from tc_pipeline.cleaning.mapping import extract_sentido_resolucion
        result = extract_sentido_resolucion(source)
        print(f"\nextract_sentido_resolucion result: '{result}'")
else:
    print(f"Response: {r.text[:500]}")
