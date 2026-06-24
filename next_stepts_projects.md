# Análisis y Próximos Pasos del Proyecto RAG - TC

De acuerdo con el análisis del código actual y la reciente transición del sistema de ingesta de datos (de Parquet/SQLite/API JSON a archivos CSV generados a través de `clean_and_merge.py`), he identificado implementaciones desactualizadas y pasos clave para la refactorización hacia la nueva arquitectura FastAPI y ChromaDB con el modelo "paraphrase-multilingual-MiniLM-L12-v2".

## Archivos a modificar

**tc_pipeline/api/schemas.py**
	Objetivo: Definir los contratos de datos (Pydantic) para la API de FastAPI y la comunicación entre agentes.
	Estado actual: Contiene modelos Pydantic (`ExpedienteAPI`, `ScrapingRequest`, etc.) que hacen referencia a columnas del antiguo JSON devuelto por la API web o del SQLite.
	Instrucciones de modificacion: 
		1. Limpiar todos los esquemas obsoletos que ya no se usan (`ScrapingRequest`, `ScrapingProgress`, `DatasetListResponse`, y referencias a SQLite en `HealthResponse`).
		2. Actualizar el esquema `ExpedienteBase` para coincidir exactamente con las columnas resultantes de tu nuevo CSV: `numero_expediente`, `url_archivo_TC`, `url_archivo_original`, `tipo_expediente`, `motivos_demanda`, `sentido_resolucion` y `fundamentos`.
		3. Crear un nuevo esquema `MotivosDemandaRequest` que sirva como Input desde la ventana de FastAPI para recibir los "motivos del porqué el usuario coloca la demanda".

**tc_pipeline/nlp/embeddings.py**
	Objetivo: Envolver y manejar el modelo de HuggingFace `SentenceTransformers` para la generación de embeddings.
	Estado actual: Usa como modelo de vectorización predeterminado `all-MiniLM-L6-v2`.
	Instrucciones de modificacion: 
		1. Modificar el parámetro por defecto de `EmbeddingModel` a `"paraphrase-multilingual-MiniLM-L12-v2"`.
		2. Revisar que los métodos de calidad matemática de similitud de cosenos mantengan coherencia con la dimensión del nuevo modelo.

**scripts/process_embeddings.py**
	Objetivo: Orquestar el flujo de lectura, partición de chunks y cálculos de embeddings del corpus de datos.
	Estado actual: Depende de un archivo Parquet central `data/raw/expedientes_tc.parquet` y de librerías como `pyarrow`.
	Instrucciones de modificacion: 
		1. Descartar el uso de dependencias `pyarrow` y `ParquetStore`.
		2. Modificar el `argparse` y la lógica interna para recorrer e iterar sobre los archivos CSV generados por `clean_and_merge.py` en el directorio `data/merged/`.
		3. Sustituir `pd.read_parquet` por `pd.read_csv`, iterando cada uno de los CSVs de `expedientes_cleaned_*.csv` para la extracción y envío a chunking.

**tc_pipeline/nlp/processing.py** **REFACTORIZAR Y UTILIZAR EN chroma_pipeline.py**
	Objetivo: Llevar a cabo la limpieza de texto, partición de los textos en "chunks" con solapamiento y la extracción de metadata.
	Estado actual: La función `extract_text_for_chunking` y `build_chunks_for_record` buscan propiedades asumiendo que procesan el diccionario JSON crudo original o parquets con llaves legacy como `CDES_TIPOPROCESO`, `FALLO` o esquemas en `attachment`.
	Instrucciones de modificacion: 
		1. Simplificar la función `extract_text_for_chunking` para que priorice de manera directa el valor de las columnas `motivos_demanda` y `fundamentos` que vienen de los nuevos CSV.
		2. En la construcción de la `metadata` dentro de `build_chunks_for_record`, utilizar las llaves del nuevo esquema en su lugar (`tipo_expediente` y `sentido_resolucion`), descartando todo rastreo de variables anidadas.
		3. Actualizar los campos al agente con los nuevos campos de los .csv en el fichero merge.




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
