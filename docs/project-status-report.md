# Estado del Proyecto — Sistema Multiagente TC

> **Última actualización:** Junio 2026  
> **Stack:** Python 3.10+ · FastAPI · ChromaDB · XGBoost · Groq (Qwen3-32B) · SentenceTransformers

---

## Resumen Ejecutivo

El sistema es un **agente constitucional inteligente** que predice el sentido de la resolución del Tribunal Constitucional del Perú (`sentido_resolucion`) y argumenta la predicción recuperando jurisprudencia semánticamente similar. El pipeline se divide en cinco capas de responsabilidad bien delimitadas.

---

## Capa 1 — Obtención de Datos (Scraping)

Responsable de extraer la metadata estructurada de las sentencias del TC desde su API oficial y descargar los PDFs correspondientes.

| Archivo | Responsabilidad |
|---|---|
| `tc_pipeline/scraping/api_client.py` | Cliente HTTP del endpoint cronológico del TC (`/busqueda/cronologico`). Pagina resultados año por año, implementa reintentos con backoff exponencial y exporta la metadata a CSV por año. |
| `tc_pipeline/scraping/downloader.py` | Descargador concurrente de PDFs usando `ThreadPoolExecutor`. Gestiona el mapa de IDs (`id_map`) para vincular cada PDF con su número de expediente. |
| `tc_pipeline/scraping/rewrite_csvs.py` | Utilitario de reescritura para normalizar los CSV de salida del cliente API cuando hay cambios en el schema del endpoint. |
| `tc_pipeline/config.py → PipelineConfig` | Fuente única de verdad para URLs de la API, timeouts, headers, rutas de descarga, número de workers y códigos HTTP reintentables. |

**Artefactos producidos:** `data/csv/*.csv` (metadata por año), `data/sentencia-raw/`, `data/auto-resolucion-raw/` (PDFs descargados).

---

## Capa 2 — Limpieza, Extracción y Merge

Responsable de convertir los PDFs y CSVs crudos en un dataset limpio y unificado con las variables finales necesarias para el modelo.

| Archivo | Responsabilidad |
|---|---|
| `tc_pipeline/extraction/pdf_extractor.py` | Extrae el texto estructurado de los PDFs de sentencias y autos de resolución usando `pdfplumber`. Identifica y separa las secciones de `fundamentos` y `motivos_demanda` mediante heurísticas de encabezados. Exporta los resultados a `data/sentencia-Extract/` y `data/auto-resolucion-Extract/`. |
| `tc_pipeline/cleaning/mapping.py` | Módulo de normalización y mapeo de variables. Estandariza `sentido_resolucion` (consolida variantes textuales como "DECLARAR FUNDADA" → "Fundada"), normaliza `tipo_expediente`, y mapea columnas entre el schema del API JSON y el schema final del CSV. |
| `tc_pipeline/cleaning/normalize.py` | Funciones de limpieza de texto a nivel de celda: strip, lowercase selectivo, eliminación de caracteres especiales. |
| `tc_pipeline/cleaning/partition.py` | Lógica de partición y agrupación de registros por año y tipo de documento previo al merge final. |

**Variables finales del dataset consolidado:** `numero_expediente`, `tipo_expediente`, `sentido_resolucion`, `motivos_demanda`, `fundamentos`, `url_archivo_TC`, `url_archivo_original`.

**Artefactos producidos:** `data/merged/expedientes_cleaned_YYYY.csv`

---

## Capa 3 — Chunking, Embeddings e Indexación en ChromaDB

Responsable de transformar el texto limpio en representaciones vectoriales densas y persistirlas en la base de datos vectorial para búsqueda semántica.

| Archivo | Responsabilidad |
|---|---|
| `tc_pipeline/nlp/processing.py` | Preprocesamiento y partición de texto. Extrae y concatena `motivos_demanda` + `fundamentos` por registro, limpia caracteres de control, y aplica **chunking por ventana deslizante** (tokens con solapamiento configurable) para documentos largos. Construye la metadata de cada chunk (`numero_expediente`, `tipo_expediente`, `sentido_resolucion`). |
| `tc_pipeline/nlp/embeddings.py` | Wrapper del modelo `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones). Expone `embed_texts()` para vectorización en lote y `compute_embedding_quality()` para diagnóstico de la calidad del espacio vectorial. |
| `src/indexing/chroma_pipeline.py` | Orquestador ETL completo. Lee los CSV de `data/merged/`, invoca `build_chunks_for_record` de `processing.py`, y carga los vectores en la colección `jurisprudencia_tc` de ChromaDB en lotes de 500. Implementa `SentenceTransformerEmbeddingFunction` como adaptador de ChromaDB. Borra y recrea la colección en cada ejecución para garantizar reinicios limpios. |
| `tc_pipeline/ml-training/data_loader.py` | **Agrupación y Mean Pooling.** Dado que el chunking produce múltiples vectores por expediente, este módulo los agrupa por `numero_expediente` y calcula el **centroide aritmético** de todos los chunks, generando una única representación semántica densa por caso judicial. Retorna las matrices `X` (embeddings consolidados) e `y` (etiquetas) listas para entrenamiento. |

**Artefactos producidos:** `data/chroma_storage/` (base de datos vectorial persistente), `DatasetResult(X, y)` en memoria.

---

## Capa 4 — Entrenamiento y Evaluación del Modelo Predictivo

Responsable de entrenar el clasificador XGBoost sobre los embeddings consolidados y evaluar su fiabilidad con métricas de clasificación multiclase.

| Archivo | Responsabilidad |
|---|---|
| `tc_pipeline/config.py → MLConfig` | Centraliza todos los hiperparámetros del clasificador XGBoost (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `objective='multi:softprob'`, `eval_metric='mlogloss'`), la proporción de split (80/20), el `random_state=42` y las rutas de serialización de artefactos. |
| `tc_pipeline/ml-training/model_trainer.py` | Codifica las etiquetas categóricas con `LabelEncoder`, realiza la partición estratificada 80/20 (`stratify=y_encoded`), entrena el `XGBClassifier`, y serializa los artefactos: `models/xgb_classifier.json` (modelo) y `models/label_encoder.joblib` (encoder). |
| `tc_pipeline/ml-training/model_evaluator.py` | Evaluación rigurosa e independiente: **Accuracy global**, **Matriz de Confusión** completa, **Classification Report** (Precisión/Recall/F1 por clase + promedios macro y ponderado) y **ROC-AUC One-vs-Rest macro**. Exporta el reporte a `models/evaluation_report.txt`. También provee `load_and_evaluate()` para carga de artefactos desde disco en pipelines de CI/CD. |

**Artefactos producidos:** `models/xgb_classifier.json`, `models/label_encoder.joblib`, `models/evaluation_report.txt`.

---

## Capa 5 — Servicios RAG, Orquestación y API

Responsable de exponer los modelos e integrar el agente completo (LLM + predictor + ChromaDB) a través de la API y la interfaz de usuario.

| Archivo | Responsabilidad |
|---|---|
| `src/agent/rag_service.py` | Orquestador RAG completo. Ejecuta búsquedas semánticas sobre ChromaDB (`retrieve_context`), construye el prompt con los fragmentos recuperados, invoca al LLM **Qwen3-32B vía Groq** con salida JSON estructurada, y retorna un `BriefResponse` con el análisis argumentativo y las fuentes. |
| `tc_pipeline/api/schemas.py` | Contratos Pydantic de entrada/salida entre agentes: `MotivosDemandaRequest`, `QueryRequest`, `BriefResponse`, `PrediccionRequest`, `PrediccionResponse`, `ExpedienteBase`, y `HealthResponse`. |
| `tc_pipeline/api/routes.py` | Router FastAPI con los endpoints operativos (scraping masivo, health check, datasets) y los stubs pendientes de integración (`/query`, `/prediccion`). |
| `tc_pipeline/api/main.py` | Aplicación FastAPI con configuración de CORS, lifespan de startup/shutdown, y montaje del router en `/api/v1`. |

---

## Capa 6 — Interfaz de Usuario *(pendiente de implementación)*

La interfaz permitirá al usuario final probar el agente constitucional completo sin acceso al Swagger, con la siguiente experiencia de uso:

1. **Entrada:** El usuario describe el tipo de acción constitucional que quiere interponer y expone los motivos y fundamentos de su demanda.
2. **Procesamiento:** El orquestador ejecuta en paralelo el `PredictorService` (XGBoost → probabilidad de fallo) y el `RAGService` (Qwen3-32B → análisis argumentativo con jurisprudencia recuperada).
3. **Salida:** La UI muestra:
   - Badge visual del **sentido predicho** (Fundada / Infundada / Improcedente) con barra de confianza.
   - **Análisis del LLM** con el fundamento jurídico clave y el resumen del caso.
   - **Casos similares** recuperados de ChromaDB: número de expediente, tipo de proceso, sentido previo y enlace al archivo del TC.

**Archivo a crear:** `src/ui/index.html` — SPA en HTML/CSS/JS puro, sin frameworks, consumiendo el endpoint `POST /api/v1/analizar`.

---

## Diagrama de Flujo del Sistema

```
 CSV (data/merged/)
       │
       ▼
[chroma_pipeline.py]  ──→  ChromaDB (data/chroma_storage/)
 chunking + indexación               │
                                     │
                         ┌───────────┼───────────────┐
                         │           │               │
                    [data_loader]  [RAGService]  [búsqueda
                    mean pooling   semántica]    semántica UI]
                         │
                    [model_trainer]
                         │
                    [model_evaluator]
                         │
                    models/ (xgb_classifier.json
                             label_encoder.joblib)
                         │
              ┌──────────┴──────────────┐
              │                        │
      [PredictorService]        [RAGService]
              │                        │
              └──────────┬─────────────┘
                         │
                  [orchestrator.py]
                         │
                  [FastAPI /analizar]
                         │
                  [src/ui/index.html]
```
