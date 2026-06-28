# Análisis y Próximos Pasos del Proyecto RAG - TC

De acuerdo con el análisis del código actual y la reciente transición del sistema de ingesta de datos (de Parquet/SQLite/API JSON a archivos CSV generados a través de `clean_and_merge.py`), he identificado implementaciones desactualizadas y pasos clave para la refactorización hacia la nueva arquitectura FastAPI y ChromaDB con el modelo "paraphrase-multilingual-MiniLM-L12-v2".

## Archivos a modificar

**tc_pipeline/api/schemas.py**
	Objetivo: Definir los contratos de datos (Pydantic) para la API de FastAPI y la comunicación entre agentes.
	Estado actual: [COMPLETADO] Contiene los esquemas actualizados según el nuevo CSV (`ExpedienteBase`), el `MotivosDemandaRequest` y las bases para RAG (`QueryRequest/BriefResponse`) y Predicciones (`PrediccionRequest/PrediccionResponse`).

**tc_pipeline/nlp/embeddings.py**
	Objetivo: Envolver y manejar el modelo de HuggingFace `SentenceTransformers` para la generación de embeddings.
	Estado actual: [COMPLETADO] El modelo predeterminado ahora es `"paraphrase-multilingual-MiniLM-L12-v2"` y se adaptaron las dimensiones a 384. Incluye un docstring clarificando que es un módulo puramente lógico (sin I/O).

**scripts/process_embeddings.py**
	Objetivo: Orquestar el flujo de lectura, partición de chunks y cálculos de embeddings del corpus de datos.
	Estado actual: [COMPLETADO / MARCADO COMO DEBUG] Se refactorizó para iterar sobre los CSV y generar JSONL; sin embargo, para fines arquitectónicos ha sido marcado explícitamente en su cabecera como un script exclusivo de depuración (debugging) y no interviene en el flujo de indexación principal.

**tc_pipeline/nlp/processing.py**
	Objetivo: Llevar a cabo la limpieza de texto, partición de los textos en "chunks" con solapamiento y la extracción de metadata.
	Estado actual: [COMPLETADO] Refactorizado por completo. Prioriza y concatena `motivos_demanda` y `fundamentos`. Se limpiaron las funciones de regex anidadas y la metadata de cada chunk ahora se construye usando únicamente las llaves de los nuevos CSV (`tipo_expediente`, `sentido_resolucion`, `url_archivo_TC`, etc.).

**src/indexing/chroma_pipeline.py**
	Objetivo: Configurar el pipeline específico para indexar la base vectorial (ChromaDB) de forma persistente.
	Estado actual: [COMPLETADO] Rediseñado como orquestador ETL completo. Extrae leyendo iterativamente del directorio `data/merged/*.csv`, transforma invocando centralizadamente a `build_chunks_for_record` en `processing.py`, e inserta por lotes hacia ChromaDB, la cual computa los vectores usando el motor de embeddings envoltado `SentenceTransformerEmbeddingFunction`. Se agregó comportamiento de *drop collection* para asegurar reinicios limpios.

**ver_indexacion.py**
	Objetivo: Script visual en terminal para comprobar la existencia y veracidad de lo almacenado en la colección de ChromaDB.
	Estado actual: Solicita metadatos obsoletos como `materia` y sus prints de terminal tienen cadenas "hardcodeadas" que refieren a `Legal-BERT`.
	Instrucciones de modificacion: 
		1. Reemplazar toda mención a `Legal-BERT` en consola por `paraphrase-multilingual-MiniLM-L12-v2`.
		2. Ajustar la impresión de `metadatas` para leer e imprimir `tipo_expediente` y `sentido_resolucion` en lugar de `materia`.

**src/agent/rag_service.py**
	Objetivo: Orquestar el agente de Recuperación y Generación (RAG) consultando ChromaDB y utilizando un LLM.
	Estado actual: Es un cascarón que solo contiene la plantilla del prompt. No se conecta a Chroma ni al LLM.
	Instrucciones de modificacion: 
		1. Importar `chromadb.PersistentClient` y conectarlo a la colección `jurisprudencia_tc` usando la función de embeddings adecuada.
		2. Crear una función `retrieve_context` que ejecute búsquedas semánticas sobre ChromaDB.
		3. Integrar un cliente LLM (OpenAI/Anthropic/Ollama) y crear `generate_answer` que procese el query con el contexto recuperado y responda con el esquema `BriefResponse`.

**docs/predictive_model_architecture.md** [NUEVO DOCUMENTO A CREAR]
	Objetivo: Documentar la arquitectura, elección de modelo y flujo de entrenamiento para la IA que predecirá el `sentido_resolucion`.
	Instrucciones de creacion: 
		1. Establecer **XGBoost** sobre los embeddings (generados por Chroma o procesados aparte) como el modelo inicial por su balance entre bajo tiempo de entrenamiento, menor complejidad computacional y alta interpretabilidad.
		2. Documentar las opciones avanzadas (Fine-Tuning de Transformers como RoBERTa o el modelo Multilingüe) detallando su mayor costo computacional y tiempo de implementación como alternativas futuras.
		3. Definir la estrategia de limpieza de clases (`sentido_resolucion`), partición de datos y exportación a la carpeta `models/`.

**src/ml/train_model.py** [NUEVO SCRIPT A CREAR]
	Objetivo: Script para entrenar el modelo XGBoost predictivo utilizando los datos históricos.
	Instrucciones de creacion: 
		1. Leer los vectores (embeddings) y la variable objetivo desde ChromaDB o extrayéndolos de los CSV.
		2. Limpiar las clases de "sentido_resolucion" para unificar criterios.
		3. Entrenar el clasificador XGBoost, evaluar con métricas F1 y exportarlo como `.pkl` para su uso en la API.



## Archivos a eliminar

**docs/plan-desarrollo-mvp-tc.md**
	Objetivo: Plan histórico de la arquitectura MVP, división de Sprints y documentación inicial.
	Estado actual: Mantiene descripciones que orientan a utilizar Scrapy contra un endpoint JSON retirado, partición en parquet y almacenamientos en SQLite, un plan totalmente divorciado de la extracción final por CSVs.
	Justificación: Documento sumamente desactualizado. Con la transición al sistema de archivos directos vía `clean_and_merge.py`, este markdown solo aportará confusión y desvío para tu pipeline actual en FastAPI/RAG. Puede ser retirado o sobreescrito de cero.

**Dockerfile y docker-compose.yml**
	Objetivo: Contenerización y orquestación del backend FastAPI y procesos workers.
	Estado actual: Mencionan rutas de volúmenes, carpetas de extracción y comandos de trabajadores legacy que iban destinados a tareas de "Scraping".
	Instrucciones de modificacion: 
		1. En el `Dockerfile`, eliminar los comandos `RUN mkdir -p` que instancian directorios como `data/csv`, `data/sentencia-raw`, y conservar únicamente la creación de directorios actuales como `data/merged/` y `data/chroma_storage/`.
		2. En el `docker-compose.yml`, simplificar los volúmenes, garantizando acceso a `data/chroma_storage` para la API y, en el worker, actualizar el `command` por defecto para que llame directamente al nuevo script unificado de pipeline (sea refactorizando `process_embeddings.py` o `chroma_pipeline.py`) en lugar de `run_pipeline.py --phase all`.

> [!IMPORTANT]
> ## 🔑 Configuración de la API Key de Groq
>
> El servicio RAG (`src/agent/rag_service.py`) ahora usa **Groq** con el modelo `qwen/qwen3-32b`
> en lugar de Google Generative AI. Debes configurar tu `GROQ_API_KEY` antes de ejecutar el proyecto.
>
> **Obtén tu API Key gratis en:** https://console.groq.com/keys
>
> **Opciones para configurarla:**
>
> **Opción 1 — Archivo `.env` en la raíz del proyecto (recomendado):**
> ```
> GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
> ```
>
> **Opción 2 — PowerShell (sesión actual):**
> ```powershell
> $env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
> ```
>
> **Opción 3 — CMD / bash:**
> ```bash
> set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
> ```
>
> ⚠️ **No compartas tu API Key ni la subas al repositorio.** Asegúrate de que `.env` esté en `.gitignore`.

---