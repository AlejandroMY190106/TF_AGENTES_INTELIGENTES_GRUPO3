# Estado del Proyecto — Sistema Multiagente TC

> **Última actualización:** Junio 2026  
> **Stack:** Python 3.10+ · FastAPI · ChromaDB · XGBoost · Groq (LLaMA-3.3-70b) · SentenceTransformers

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

**Dependencias utilizadas:**

| Dependencia | Operación |
|---|---|
| `requests` | Peticiones HTTP a la API del TC y descarga de PDFs. |
| `tqdm` | Barras de progreso para la descarga concurrente de PDFs. |
| `csv` (stdlib) | Lectura y escritura de archivos CSV de metadata. |
| `concurrent.futures` (stdlib) | `ThreadPoolExecutor` para descarga paralela de PDFs. |

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

**Dependencias utilizadas:**

| Dependencia | Operación |
|---|---|
| `pdfplumber` | Extracción de texto desde los PDFs del TC sin OCR. |
| `re` (stdlib) | Expresiones regulares para capturar secciones de las sentencias (ANTECEDENTES, HA RESUELTO, etc.). |
| `unicodedata` (stdlib) | Normalización Unicode de caracteres del texto extraído. |
| `csv` (stdlib) | Exportación de datos extraídos a CSV. |
| `json` (stdlib) | Parseo de mapas de IDs (`id_map`) para vincular PDFs con expedientes. |

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

**Dependencias utilizadas:**

| Dependencia | Operación |
|---|---|
| `sentence-transformers` | Modelo `paraphrase-multilingual-MiniLM-L12-v2` para generación de embeddings densos de 384 dimensiones. |
| `chromadb` | Base de datos vectorial con persistencia local para almacenar, indexar y buscar embeddings por similitud semántica. |
| `numpy` | Cálculo del centroide aritmético (mean pooling) de chunks por expediente y manipulación de matrices de embeddings. |
| `pandas` | Lectura de los CSVs de `data/merged/` para la fase ETL de indexación. |

---

## Capa 4 — Entrenamiento y Evaluación del Modelo Predictivo

Responsable de entrenar múltiples clasificadores sobre los embeddings consolidados y evaluar su fiabilidad con métricas de clasificación multiclase. Se implementaron cuatro algoritmos de predicción, todos compatibles con la misma interfaz contractual `train(X, y, cfg) → TrainingResult` y evaluables con el `model_evaluator.py` unificado.

| Archivo | Responsabilidad |
|---|---|
| `tc_pipeline/config.py → MLConfig` | Centraliza todos los hiperparámetros de los clasificadores (XGBoost, RandomForest), la proporción de split (80/20), el `random_state=42`, y las rutas de serialización de artefactos para cada modelo (`xgb_classifier.json`, `logreg_classifier.joblib`, `svm_classifier.joblib`, `rf_classifier.joblib`). |
| `tc_pipeline/ml-training/model_trainer_XGBoost.py` | Codifica las etiquetas categóricas con `LabelEncoder`, realiza la partición estratificada 80/20 (`stratify=y_encoded`), entrena el `XGBClassifier` con `objective='multi:softprob'` y `eval_metric='mlogloss'`, y serializa los artefactos: `models/xgb_classifier.json` (modelo) y `models/label_encoder.joblib` (encoder). |
| `tc_pipeline/ml-training/model_trainer_LogReg.py` | Implementa `LogisticRegression` multinomial con `solver='lbfgs'`, `C=1.0`, `max_iter=1000` y `class_weight='balanced'`. Produce probabilidades calibradas de forma nativa. Serializa con `joblib` en `models/logreg_classifier.joblib` y `models/logreg_label_encoder.joblib`. |
| `tc_pipeline/ml-training/model_trainer_SVM.py` | Implementa `SVC` con kernel RBF, `C=10.0`, `gamma='scale'`, `class_weight='balanced'` y `probability=True` (obligatorio para `.predict_proba()`). Serializa con `joblib` en `models/svm_classifier.joblib` y `models/svm_label_encoder.joblib`. |
| `tc_pipeline/ml-training/model_trainer_RandomForest.py` | Implementa `RandomForestClassifier` con `n_estimators=300`, `class_weight='balanced'` y parámetros centralizados en `MLConfig.rf_params`. Serializa con `joblib` en `models/rf_classifier.joblib` y `models/rf_label_encoder.joblib`. |
| `tc_pipeline/ml-training/model_evaluator.py` | Evaluación rigurosa e independiente del algoritmo: **Accuracy global**, **Matriz de Confusión** completa, **Classification Report** (Precisión/Recall/F1 por clase + promedios macro y ponderado) y **ROC-AUC One-vs-Rest macro**. Acepta cualquier clasificador sklearn-compatible (`model: Any`) con parámetro `model_name` dinámico. La función `load_and_evaluate()` soporta carga desde disco tanto de modelos XGBoost (`.load_model()`) como scikit-learn (`joblib.load()`). Incluye menú interactivo `__main__` para seleccionar entre los 4 modelos. Exporta el reporte a `models/evaluation_report_<algoritmo>.txt`. |

**Artefactos producidos:** `models/xgb_classifier.json`, `models/label_encoder.joblib`, `models/logreg_classifier.joblib`, `models/logreg_label_encoder.joblib`, `models/svm_classifier.joblib`, `models/svm_label_encoder.joblib`, `models/rf_classifier.joblib`, `models/rf_label_encoder.joblib`, `models/evaluation_report_<algoritmo>.txt`.

**Dependencias utilizadas:**

| Dependencia | Operación |
|---|---|
| `xgboost` | Clasificador `XGBClassifier` para entrenamiento con gradient boosting multiclase. Serialización en formato JSON nativo. |
| `scikit-learn` | `LabelEncoder` para codificación categórica; `train_test_split` para partición estratificada; `LogisticRegression`, `SVC`, `RandomForestClassifier` como clasificadores alternativos; `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score` para métricas de evaluación. |
| `joblib` | Serialización y deserialización comprimida de modelos scikit-learn y `LabelEncoder` en formato `.joblib`. |
| `numpy` | Manipulación de matrices de features, vectores de etiquetas y cálculos de probabilidades. |

---

## Capa 5 — Servicios RAG, Orquestación y API

Responsable de exponer los modelos e integrar el agente completo (LLM + predictor + ChromaDB) a través de la API y la interfaz de usuario.

| Archivo | Responsabilidad |
|---|---|
| `src/agent/rag_service.py` | Orquestador RAG completo. Ejecuta búsquedas semánticas sobre ChromaDB (`retrieve_context`), construye el prompt con los fragmentos recuperados y la predicción del modelo cuantitativo como anclaje de consistencia, invoca al LLM **LLaMA-3.3-70b vía Groq** con salida JSON estructurada (`temperature=0.3`), y retorna un `BriefResponse` con el análisis argumentativo y las fuentes. Acepta parámetros opcionales `prediccion` y `confianza_prediccion` para alinear el análisis argumentativo con el veredicto del clasificador. |
| `src/agent/predictor_service.py` | Servicio de inferencia que carga el modelo XGBoost y el `LabelEncoder` desde disco, vectoriza el texto de entrada con `EmbeddingModel`, y retorna la clase predicha con probabilidades por clase. |
| `src/agent/orchestrator.py` | Coordinador asíncrono que ejecuta primero el `PredictorService` (inferencia local rápida) y luego pasa la predicción y confianza al `RAGService` para garantizar consistencia entre el veredicto cuantitativo y el análisis argumentativo. Maneja fallos parciales de forma tolerante. |
| `tc_pipeline/api/schemas.py` | Contratos Pydantic de entrada/salida entre agentes: `MotivosDemandaRequest`, `QueryRequest`, `BriefResponse`, `PrediccionRequest`, `PrediccionResponse`, `AnalisisCompletoRequest`, `AnalisisCompletoResponse`, `ExpedienteBase`, y `HealthResponse`. |
| `tc_pipeline/api/routes.py` | Router FastAPI con los endpoints operativos (scraping masivo, health check, datasets) y los endpoints activos: `POST /query` (RAG), `POST /prediccion` (predicción XGBoost), `POST /analizar` (orquestador completo RAG + Predictor). |
| `tc_pipeline/api/main.py` | Aplicación FastAPI con configuración de CORS, lifespan de startup/shutdown, montaje del router en `/api/v1`, servicio de archivos estáticos en `/ui` vía `StaticFiles`, y redirección automática de `GET /` hacia la interfaz de usuario. |
| `src/ui/index.html` | SPA en HTML/CSS/JS puro que permite probar el agente constitucional completo. Consumo del endpoint `POST /api/v1/analizar`. Servida desde FastAPI en `http://localhost:8000/ui`. |

**Dependencias utilizadas:**

| Dependencia | Operación |
|---|---|
| `fastapi` | Framework de API con soporte async, validación automática con Pydantic, `StaticFiles` para servir la UI, y `RedirectResponse` para redirigir `GET /` a la interfaz. |
| `uvicorn` | Servidor ASGI para ejecutar la aplicación FastAPI en desarrollo y producción. |
| `pydantic` | Definición y validación de schemas de entrada/salida (contratos de la API). |
| `groq` | Cliente oficial de la API de Groq para invocar al LLM LLaMA-3.3-70b con salida JSON estructurada. |
| `chromadb` | Búsqueda semántica de precedentes jurisprudenciales por similitud vectorial desde el RAG Service. |
| `sentence-transformers` | Vectorización del texto de entrada del usuario para inferencia predictiva en `PredictorService`. |
| `python-dotenv` | Carga de variables de entorno (como `GROQ_API_KEY`) desde el archivo `.env`. |

---

## Capa 6 — Interfaz de Usuario

La interfaz permite al usuario final probar el agente constitucional completo accediendo a `http://localhost:8000/ui`, con la siguiente experiencia de uso:

1. **Entrada:** El usuario describe el tipo de acción constitucional que quiere interponer y expone los motivos y fundamentos de su demanda.
2. **Procesamiento:** El orquestador ejecuta primero el `PredictorService` (XGBoost → probabilidad de fallo) y luego pasa el resultado al `RAGService` (LLaMA-3.3-70b → análisis argumentativo anclado a la predicción).
3. **Salida:** La UI muestra:
   - Badge visual del **sentido predicho** (Fundada / Infundada / Improcedente) con barra de confianza.
   - **Análisis del LLM** con el fundamento jurídico clave y el resumen del caso.
   - **Casos similares** recuperados de ChromaDB: número de expediente, tipo de proceso, sentido previo y enlace al archivo del TC.

**Archivo:** `src/ui/index.html` — SPA en HTML/CSS/JS puro, servida desde FastAPI vía `StaticFiles` en el path `/ui`.

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
            ┌────────────┼────────────────────────┐
            │            │            │           │
     [trainer_XGBoost] [trainer_LogReg] [trainer_SVM] [trainer_RF]
            │            │            │           │
            └────────────┼────────────────────────┘
                         │
                    [model_evaluator]  ← menú interactivo (4 modelos)
                         │
                    models/ (xgb_classifier.json
                             logreg_classifier.joblib
                             svm_classifier.joblib
                             rf_classifier.joblib
                             + label_encoders)
                         │
              ┌──────────┴──────────────┐
              │                        │
      [PredictorService]        [RAGService]
              │                        │
              │     predicción ──→     │ (anclaje)
              └──────────┬─────────────┘
                         │
                  [orchestrator.py]
                         │
                  [FastAPI /analizar]
                         │
                  [/ui → StaticFiles]
```
