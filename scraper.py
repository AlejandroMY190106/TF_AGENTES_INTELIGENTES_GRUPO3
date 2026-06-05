import requests
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================================
# CONFIGURACIÓN GLOBAL CONTROLABLE
# =========================================================================
ANIO_OBJETIVO = "2025"      # Año que deseas procesar
MENSUAL = False             # True: Descarga un solo mes | False: Descarga todo el año completo
MES_OBJETIVO = "01"         # Si MENSUAL es True, indicas el mes aquí ("01", "02", ... "12")

MAX_WORKERS = 10             # Número de descargas en simultáneo
# =========================================================================

# Configuración de Endpoints y Headers
URL_API = "https://jurisbackend.sedetc.gob.pe/api/visitor/sentencia/busqueda/cronologico"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# Lógica de automatización de periodos y creación organizada de carpetas
if MENSUAL:
    periodos_a_descargar = [f"{ANIO_OBJETIVO}-{MES_OBJETIVO}"]
    # Crear estructura: ANIO_OBJETIVO/MES_OBJETIVO (Ej: 2025/01)
    os.makedirs(os.path.join(ANIO_OBJETIVO, MES_OBJETIVO), exist_ok=True)
else:
    periodos_a_descargar = [f"{ANIO_OBJETIVO}-{str(mes).zfill(2)}" for mes in range(1, 13)]
    # Crear la carpeta del año y todas sus subcarpetas del 01 al 12 automáticamente
    for mes in range(1, 13):
        mes_str = str(mes).zfill(2)
        os.makedirs(os.path.join(ANIO_OBJETIVO, mes_str), exist_ok=True)


def descargar_un_pdf(item, periodo):
    """Función independiente para descargar un solo PDF en su carpeta respectiva"""
    source = item.get("_source", {})
    expediente = source.get("numero_expediente")
    url_pdf = source.get("url_archivo")
    
    if not expediente or not url_pdf:
        return "invalido"
        
    # Extraer el año y mes desde el periodo (formato "YYYY-MM") para armar la ruta
    anio, mes = periodo.split("-")
    nombre_limpio = expediente.replace("/", "_").replace(" ", "")
    
    # Ruta dinámica: Año/Mes/expediente.pdf (Ej: 2025/04/01167-2012-AA.pdf)
    ruta_guardado = os.path.join(anio, mes, f"{nombre_limpio}.pdf")
    
    if os.path.exists(ruta_guardado):
        return "existente"
        
    try:
        pdf_response = requests.get(url_pdf, headers=HEADERS, timeout=15)
        if pdf_response.status_code == 200:
            with open(ruta_guardado, "wb") as f:
                f.write(pdf_response.content)
            return "descargado"
        else:
            return f"error_status_{pdf_response.status_code}"
    except Exception:
        return "error_conexion"


# --- Flujo Principal de Extracción ---
print(f"Modo de descarga seleccionado: {'MENSUAL' if MENSUAL else 'ANUAL'}")
print(f"Estructura de directorios organizada en la raíz para el año: {ANIO_OBJETIVO}")

for periodo in periodos_a_descargar:
    print(f"\n=========================================")
    print(f"Procesando periodo: {periodo}")
    print(f"=========================================")
    
    pagina_actual = 1
    total_paginas = 1
    barras_progreso_inicializada = False
    pbar = None
    
    while pagina_actual <= total_paginas:
        params = {"fecha_publicacion": periodo, "page": pagina_actual}
        
        try:
            response = requests.get(URL_API, headers=HEADERS, params=params)
            if response.status_code != 200:
                print(f"[Error API] Status {response.status_code} en la página {pagina_actual}")
                break
                
            res_json = response.json()
            pagination_info = res_json.get("pagination", {})
            total_paginas = pagination_info.get("num_pages", 0)
            total_items = pagination_info.get("total_item", 0)
            
            lista_sentencias = res_json.get("data", [])
            
            if total_items == 0 or not lista_sentencias:
                print(f"El periodo {periodo} no contiene registros públicos en el sistema.")
                break
                
            # Inicializar barra tqdm acumulativa para todo el mes actual
            if not barras_progreso_inicializada:
                print(f"Total expedientes detectados en {periodo}: {total_items}")
                pbar = tqdm(total=total_items, desc=f"Progreso {periodo}", unit="pdf")
                barras_progreso_inicializada = True
            
            # Descargas concurrentes pasando el periodo actual como argumento de ruta
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(descargar_un_pdf, item, periodo): item for item in lista_sentencias}
                
                for future in as_completed(futures):
                    resultado = future.result()
                    if "error" in resultado:
                        item_error = futures[future]
                        exp_error = item_error.get("_source", {}).get("numero_expediente")
                        tqdm.write(f"     X Falló: {exp_error} ({resultado})")
                    
                    if pbar: 
                        pbar.update(1)
            
            pagina_actual += 1
            time.sleep(1)  # Descanso entre páginas de la API
            
        except Exception as e:
            print(f"[Excepción] Error crítico en el bucle principal: {e}")
            break
            
    if pbar:
        pbar.close()

print("\n¡Operación finalizada por completo! Revisa tus carpetas locales.")