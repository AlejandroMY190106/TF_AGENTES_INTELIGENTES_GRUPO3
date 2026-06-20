import os
import pandas as pd

# Creamos las carpetas por si acaso
os.makedirs("data/interim", exist_ok=True)

# Simulamos 2 expedientes de prueba con la estructura exacta que requiere tu pipeline
datos_prueba = {
    "numero_expediente": ["00001-2026-AI", "00002-2026-AA"],
    "fundamentos": [
        "El Tribunal Constitucional considera que se ha vulnerado el derecho al debido proceso...",
        "Este Colegiado desestima la demanda de amparo al no encontrarse afectación directa..."
    ],
    "sentencia_sala": ["Pleno", "Sala 1"],
    "sentencia_sentido": ["Fundada", "Infundada"],
    "materia": ["Constitucional", "Derechos Fundamentales"]
}

df = pd.DataFrame(datos_prueba)
df.to_parquet("data/interim/expedientes_output.parquet", index=False)
print("¡Archivo Parquet de muestra creado con éxito en data/interim/!")