# TF Agentes Inteligentes - Grupo 3

## 🛠️ Requisitos e Instalación

El proyecto se ejecuta sobre un entorno de Anaconda. Asegúrate de tener instaladas las librerías necesarias ejecutando en tu terminal:

```bash
pip install pandas openpyxl requests tqdm

```

---

## 📄 Descripción de los Scripts

El repositorio cuenta con dos scripts principales de Python que trabajan de forma secuencial:

### 1. `particionar.py`

* **¿Qué hace?**: Toma el archivo maestro original (`dataset_tc.xlsx - Exportar Hoja de Trabajo.csv`) que contiene cerca de 150,000 registros y lo divide en múltiples archivos Excel pequeños organizados de forma independiente por año de publicación(ej. `tribunal_2024.xlsx`).
* **Características clave**:
* Está configurado con codificación `latin1` para evitar que el script falle debido a tildes o la letra **ñ** que exporta Windows por defecto.
* Analiza la columna numérica de 8 dígitos `PUB_PAGWEB` (formato `YYYYMMDD`) para extraer el año exacto de forma estricta.
* Separa y aísla en un archivo independiente todos los registros que carecen de fecha válida o traen guiones (`--`) para no perder información.
* **Propósito**: Optimiza el rendimiento evitando saturar la memoria RAM de la computadora y permitiendo procesar el lote de descargas por bloques anuales.

### 2. `scraper.py` (En Desarrollo - Modificaciones Planeadas)

* **Objetivo del script**: Este script será el encargado de conectar con la API oficial del Tribunal Constitucional para automatizar la descarga física de los PDFs y enriquecer nuestros archivos de Excel anuales de forma automatizada.

**Funcionalidades Actuales**:
* **Descargas Concurrentes (Multi-threading)**: Implementación de `ThreadPoolExecutor` para gestionar hilos de descarga simultáneos, acelerando drásticamente el almacenamiento local de los archivos PDF.

**Modificaciones Planeadas**:
* **Algoritmo de Match Inteligente**: Como nuestro dataset no cuenta con el número de expediente textual, el script realizará un cruce avanzado comparando la fecha de publicación, la correspondencia de la columna `SALA` y rastreando la palabra exacta de la columna `FALLO` (ej. "FUNDADA") directamente dentro del texto plano del PDF indexado en la API (`attachment.content`).

* **Guardado Incremental en Tiempo Real**: El script actualizará celda por celda directamente en el archivo Excel anual en ejecución. Al finalizar el análisis de cada día, se guardará el progreso para evitar pérdidas de información ante cortes de luz o de internet.
* **Columnas de Enriquecimiento**: Al hacer match, creará e insertará automáticamente dos columnas nuevas al final del Excel: `NUMERO_EXPEDIENTE_TC` (consiguiendo el dato que nos faltaba) y `RUTA_LOCAL_PDF` (con la ubicación del archivo descargado).
* **Tolerancia a Interrupciones**: Estará diseñado para verificar si una fila ya tiene un expediente asignado; si es así, la saltará, permitiendo reanudar el script desde donde se quedó sin duplicar descargas.

---

## 🚀 Flujo de Trabajo (Cómo usarlo)

1. **Paso 1**: Asegúrate de tener el archivo `dataset_tc.xlsx - Exportar Hoja de Trabajo.csv` en la raíz del proyecto.
2. **Paso 2**: Ejecuta el particionador para segmentar tu base de datos:

```bash
python particionar.py
```

*Esto creará la carpeta `EXCELES_POR_AÑO` con los archivos independientes.*
3. **Paso 3**: Mueve o copia el Excel del año que deseas trabajar (ejemplo: `tribunal_2024.xlsx`) a la raíz del proyecto, configura las variables de fecha en la cabecera de `scraper.py` y ejecútalo:

```bash
python scraper.py
```
