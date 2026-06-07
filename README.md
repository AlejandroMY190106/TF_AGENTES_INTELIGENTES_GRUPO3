# TF Agentes Inteligentes - Grupo 3

Sistema Multiagente para análisis de jurisprudencia del Tribunal Constitucional. Este repositorio contiene el pipeline de extracción, limpieza y procesamiento, así como los orquestadores CLI.

---

## 🛠️ Stack Tecnológico y Dependencias Principales

El proyecto utiliza componentes modulares. Puedes revisar el archivo `requirements.txt` para la lista completa, pero aquí destacamos las responsabilidades del stack:

- **Core & Orquestación**: `FastAPI` (Backend/API), `pydantic` (esquemas y validaciones).
- **Procesamiento de Datos**: `pandas`, `openpyxl` (limpieza y cruce de datos Excel), `pyarrow` (formatos columnares).
- **Extracción (Scraping)**: `requests`, `tqdm` (descarga concurrente e interacción con la API del TC).
- **NLP & Embeddings**: `sentence-transformers`, `torch`, `scikit-learn` (chunking, cálculos de similitud, Legal-BERT).
- **Base de Datos Vectorial**: `chromadb` (persistencia y recuperación semántica RAG).
- **Integración LLM**: `anthropic` (generación de briefs ejecutivos con Claude 3), `langchain` (orquestación).
- **Persistencia Operacional**: `sqlite3` nativo (manifiesto de estado de descargas).

---

## 🚀 Requisitos e Instalación (Onboarding)

Para evitar conflictos de dependencias, es obligatorio utilizar un entorno virtual (venv).

### 1. Clonar el repositorio y crear el entorno virtual
En la raíz del proyecto, ejecuta en tu terminal:
```bash
python -m venv venv
```

### 2. Activar el entorno virtual
- **En Windows (PowerShell/CMD)**:
  ```bash
  .\venv\Scripts\activate
  ```
- **En macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 📄 Arquitectura y Responsabilidades

### Fase 1: Pipeline de Extracción (`tc_pipeline/scraping/`)
Se ha migrado de un script monolítico hacia módulos especializados:
- `api_client.py`: Único responsable de consultar la API cronológica del TC, gestionar la paginación, los _timeouts_ (connect=5s, read=20s) y manejar política de reintentos automáticos (backoff exponencial).
- `downloader.py`: Encargado de las descargas de PDFs utilizando multiprocesamiento (`ThreadPoolExecutor`). Verifica si el archivo existe para no re-descargar y guarda los PDFs de forma jerárquica (`EXPEDIENTES/YYYY/MM/xxx.pdf`).
- `manifest.py`: Usa SQLite (`pipeline_state.db`) para llevar auditoría de los expedientes descargados (`success`, `failed`, `pending`). Permite reanudar descargas en caso de caídas de internet sin empezar desde cero.

### Scripts Legacy (Deprecados temporalmente)
- `legacy_scraper.py`: El script original monolítico, conservado como fallback.
- `legacy_partition.py`: Script para dividir el dataset maestro por años.

---

## 💻 Flujo de Trabajo (Cómo usar el orquestador)

El punto de entrada unificado para iniciar descargas físicas desde la API oficial del TC es `scripts/download_pdfs.py`. Este CLI expone varios comandos útiles:

1. **Descargar un mes específico:**
   ```bash
   python scripts/download_pdfs.py --year 2025 --month 01
   ```

2. **Descargar un año completo:**
   ```bash
   python scripts/download_pdfs.py --year 2024
   ```

3. **Descargar un rango histórico (múltiples años):**
   ```bash
   python scripts/download_pdfs.py --start-year 1992 --end-year 2026
   ```

4. **Reanudar descargas fallidas (tolerancia a fallos):**
   Si la extracción se interrumpió o hubo errores temporales de conexión, este comando consultará la base SQLite local y solo reintentará los PDFs marcados como `failed`.
   ```bash
   python scripts/download_pdfs.py --retry-failed
   ```

*(Nota: Opcionalmente, puedes sobrescribir la concurrencia en cualquier comando agregando `--max-workers N`, por ej. `--max-workers 15`)*
