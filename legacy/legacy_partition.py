import pandas as pd
import os

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
RUTA_EXCEL_MAESTRO = "dataset_tc.xlsx"
CARPETA_DESTINO = "EXCELES_POR_AÑO"
# =========================================================================

os.makedirs(CARPETA_DESTINO, exist_ok=True)

print("Leyendo el archivo maestro de Excel (esto puede demorar unos segundos)...")
df = pd.read_excel(RUTA_EXCEL_MAESTRO, dtype={'PUB_PAGWEB': str})

print(f"¡Archivo cargado con éxito! Total de filas detectadas: {len(df)}")

print("Analizando y extrayendo los años de 'PUB_PAGWEB'...")

def extraer_anio(valor):
    valor_str = str(valor).strip()
    if len(valor_str) == 8 and valor_str.isdigit():
        return int(valor_str[:4]) # Tomamos los 4 primeros dígitos (el año)
    return None

# Aplicamos la función fila por fila
df['AÑO_EXTRAIDO'] = df['PUB_PAGWEB'].apply(extraer_anio)

# 1. Separar y guardar las filas que no tienen fecha válida (los '--')
df_sin_fecha = df[df['AÑO_EXTRAIDO'].isna()]
if not df_sin_fecha.empty:
    ruta_errores = os.path.join(CARPETA_DESTINO, "filas_sin_fecha_valida.xlsx")
    df_sin_fecha.drop(columns=['AÑO_EXTRAIDO']).to_excel(ruta_errores, index=False)
    print(f"⚠️ Alerta: Se encontraron {len(df_sin_fecha)} filas sin fecha ('--'). Guardadas en: {ruta_errores}")

# 2. Filtrar para quedarnos solo con los años reales y válidos
df_con_fecha = df[df['AÑO_EXTRAIDO'].notna()]

# 3. Particionar por año real
print("\nIniciando la partición por años reales...")
años_detectados = sorted(df_con_fecha['AÑO_EXTRAIDO'].unique())

for anio in años_detectados:
    anio_int = int(anio)
    df_anio = df_con_fecha[df_con_fecha['AÑO_EXTRAIDO'] == anio].copy()
    
    # Limpiamos la columna temporal
    df_anio.drop(columns=['AÑO_EXTRAIDO'], inplace=True)
    
    # Ordenamos cronológicamente por el código de 8 dígitos
    df_anio.sort_values(by='PUB_PAGWEB', ascending=True, inplace=True)
    
    nombre_archivo = f"tribunal_{anio_int}.xlsx"
    ruta_salida = os.path.join(CARPETA_DESTINO, nombre_archivo)
    
    df_anio.to_excel(ruta_salida, index=False)
    print(f"    ¡Guardado! Año {anio_int} -> Archivo: {ruta_salida} ({len(df_anio)} filas)")

print("\n¡Partición completada con éxito! Revisa la carpeta:", CARPETA_DESTINO)