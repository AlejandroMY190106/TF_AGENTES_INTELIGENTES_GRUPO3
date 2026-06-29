# Próximos Pasos — Orquestación de Servicios vía FastAPI

## Análisis del Estado Actual del Sistema

Tras revisar el pipeline completo — desde `chroma_pipeline.py` hasta `model_trainer.py` y `model_evaluator.py` — el estado real del sistema es el siguiente:

| Capa | Módulo(s) | Estado |
|---|---|---|
| Extracción vectorial | `tc_pipeline/ml-training/data_loader.py` | ✅ Completo |
| Entrenamiento XGBoost | `tc_pipeline/ml-training/model_trainer.py` | ✅ Completo |
| Evaluación del modelo | `tc_pipeline/ml-training/model_evaluator.py` | ✅ Completo |
| Servicio RAG | `src/agent/rag_service.py` | ✅ Completo (Groq + ChromaDB) |
| API FastAPI (main) | `tc_pipeline/api/main.py` | ✅ Estructura base lista |
| API FastAPI (routes) | `tc_pipeline/api/routes.py` | ⚠️ Stubs — `/query` y `/prediccion` lanzan HTTP 501 |
| Schemas Pydantic | `tc_pipeline/api/schemas.py` | ✅ Contratos definidos |
| UI de prueba del agente | — | ❌ No existe |

**Conclusión del análisis:** La capa de modelo predictivo está **completa**. El único cuello de botella que bloquea el sistema end-to-end es la **integración del `RAGService` y del predictor XGBoost dentro del router FastAPI**, más la creación de una **interfaz de usuario** que permita probar el agente completo.

---

## Archivos a modificar

**tc_pipeline/api/routes.py**
	Objetivo: Activar los endpoints `/query` y `/prediccion` conectándolos al RAGService y al clasificador XGBoost. Actualmente lanzan `HTTP 501 Not Implemented`.
	Estado actual: Ambos endpoints son stubs que arrojan `HTTPException(501)`. El router ya importa los schemas correctos (`BriefResponse`, `PrediccionResponse`, `PrediccionRequest`, `QueryRequest`).
	Instrucciones de modificacion:
		1. Instanciar `RAGService` como singleton al inicio del módulo (fuera de los handlers), capturando errores de conexión a ChromaDB en el bloque de startup para evitar caídas en runtime.
		2. Reemplazar el stub de `POST /query` para que delegue la llamada a `rag_service.generate_answer(request.query)` y retorne el `BriefResponse` resultante, incluyendo manejo de excepciones con `HTTPException(500)`.
		3. Crear un `PredictorService` interno (o importar desde un módulo nuevo `src/agent/predictor_service.py`) que cargue el modelo `models/xgb_classifier.json` y el encoder `models/label_encoder.joblib` con `joblib`, genere el embedding del texto del `PrediccionRequest.motivos_demanda` usando `SentenceTransformerEmbeddingFunction`, y retorne la clase predicha y las probabilidades por clase como `PrediccionResponse`.
		4. Reemplazar el stub de `POST /prediccion` para que invoque al `PredictorService` y retorne el resultado estructurado.

**tc_pipeline/api/main.py**
	Objetivo: Inicializar los servicios RAG y Predictivo durante el evento `lifespan` (startup) para que los singletons estén disponibles en el momento en que lleguen las primeras peticiones, evitando cold-starts en los primeros requests.
	Estado actual: El lifespan solo crea directorios del pipeline de scraping. No inicializa ningún servicio de inferencia.
	Instrucciones de modificacion:
		1. En el bloque `lifespan`, importar e instanciar `RAGService` desde `src.agent.rag_service` con manejo de excepción (si ChromaDB no existe, logear warning sin romper el arranque).
		2. Guardar la instancia de `RAGService` en `app.state.rag_service` para que sea accesible globalmente desde los handlers sin imports circulares.
		3. Repetir el mismo patrón para el `PredictorService`: intentar cargar los artefactos de `models/` durante el startup; si no existen, logear un warning indicando que hay que correr el pipeline de entrenamiento primero.
		4. Actualizar la descripción de la API (campo `description` de `FastAPI(...)`) para reflejar que el sistema está en **Fase 4 — Agentes activos**.

**tc_pipeline/api/schemas.py**
	Objetivo: Extender los schemas para soportar el input combinado del agente completo (demanda + motivos) y la respuesta enriquecida que combine la predicción del XGBoost con el análisis RAG del LLM.
	Estado actual: `PrediccionRequest` solo acepta campos opcionales sin ninguno requerido. `BriefResponse` no incluye la predicción XGBoost.
	Instrucciones de modificacion:
		1. Agregar el schema `AnalisisCompletoRequest` con dos campos requeridos: `tipo_demanda: str` (descripción de la acción constitucional que se quiere interponer) y `motivos: str` (argumentos y fundamentos del peticionario), con validación `min_length=20`.
		2. Agregar el schema `AnalisisCompletoResponse` que combine los campos de `BriefResponse` (contexto RAG) con los de `PrediccionResponse` (predicción + probabilidades por clase), formando la respuesta unificada del agente.
		3. Hacer que `PrediccionRequest.motivos_demanda` sea `str` requerido (no `Optional`) para garantizar que siempre haya texto que vectorizar.

---

## Archivos a crear

**src/agent/predictor_service.py** [NUEVO]
	Objetivo: Encapsular toda la lógica de inferencia del clasificador XGBoost en un servicio reutilizable con la misma interfaz de ciclo de vida que `RAGService`, manteniendo la separación de responsabilidades fuera de `routes.py`.
	Estado actual: No existe. La lógica de carga del modelo y generación de embeddings para inferencia está dispersa y sin implementar en el router actual.
	Instrucciones de creacion:
		1. Crear la clase `PredictorService` con un `__init__` que reciba las rutas de los artefactos (`model_path`, `encoder_path`) desde `MLConfig`, cargue el `XGBClassifier` con `model.load_model()`, cargue el `LabelEncoder` con `joblib.load()`, e instancie el `EmbeddingModel` para vectorizar texto nuevo en tiempo de inferencia.
		2. Implementar el método `predict(texto: str) -> dict` que: vectorice el texto con `EmbeddingModel.embed_texts([texto])`, obtenga las probabilidades con `model.predict_proba()`, decodifique el índice ganador con `encoder.inverse_transform()`, y retorne un diccionario con `{"prediccion": str, "probabilidades": dict[str, float], "confianza": float}`.
		3. Agregar un bloque `try/except` en el `__init__` que lanza `FileNotFoundError` si los artefactos no se encuentran en disco, con un mensaje de error claro que indique al desarrollador que debe ejecutar `model_trainer.py` primero.

**src/agent/orchestrator.py** [NUEVO]
	Objetivo: Coordinar la ejecución paralela o secuencial del `RAGService` y el `PredictorService` para producir la respuesta unificada `AnalisisCompletoResponse`, manteniendo a `routes.py` libre de lógica de negocio compleja.
	Estado actual: No existe. Los handlers del router contendrían toda la lógica de coordinación directamente, violando el principio de responsabilidad única.
	Instrucciones de creacion:
		1. Crear la función `async def analizar_caso(texto_demanda: str, rag: RAGService, predictor: PredictorService) -> AnalisisCompletoResponse` que ejecute ambos servicios concurrentemente usando `asyncio.gather` para minimizar la latencia total.
		2. Combinar los resultados de `rag.generate_answer(texto_demanda)` y `predictor.predict(texto_demanda)` en una instancia de `AnalisisCompletoResponse` con todos los campos populados.
		3. Incluir manejo de error individual por servicio: si el RAG falla, retornar la predicción sola con un aviso en el campo `brief`; si el predictor falla, retornar el análisis RAG con `prediccion = "No disponible"`.

**src/ui/index.html** [NUEVO]
	Objetivo: Interfaz de usuario web completa para probar el agente constitucional de manera interactiva, permitiendo ingresar la demanda y sus motivos y visualizar tanto la predicción del modelo XGBoost como el análisis argumentativo del LLM con casos similares.
	Estado actual: No existe ninguna interfaz. El único punto de acceso es el Swagger generado por FastAPI en `/docs`.
	Instrucciones de creacion:
		1. Crear una Single Page Application en HTML/CSS/JavaScript puro (sin frameworks) con dos campos de texto: `tipo_demanda` (select o input con sugerencias de los tipos de proceso constitucional: Amparo, Hábeas Corpus, Hábeas Data, Inconstitucionalidad, Conflicto de Competencia) y `motivos` (textarea con `min-length` 50 chars), un botón "Analizar caso" que haga un `fetch POST` al endpoint `/api/v1/analizar`.
		2. Diseñar la sección de resultados con tres bloques visuales diferenciados: (a) **Predicción del Modelo** — badge con el fallo predicho coloreado por clase (verde: Fundada, rojo: Infundada, naranja: Improcedente) + barras de probabilidad por clase; (b) **Análisis del Agente RAG** — card con el brief en Markdown renderizado; (c) **Casos Similares** — lista de expedientes fuente con número, tipo y sentido previo, enlazables a la URL del TC.
		3. Implementar estados de carga (spinner animado), validación de campos en frontend antes del request, y manejo de errores HTTP con mensajes descriptivos para el usuario.

**tc_pipeline/api/routes.py → endpoint POST /analizar** [NUEVO ENDPOINT en archivo existente]
	Objetivo: Exponer el endpoint unificado que invoca al orquestador y retorna el análisis completo del caso constitucional combinando RAG + XGBoost.
	Estado actual: No existe. Los stubs `/query` y `/prediccion` son independientes y no producen la respuesta combinada que necesita la UI.
	Instrucciones de creacion:
		1. Agregar el handler `POST /analizar` con `request: AnalisisCompletoRequest` y `response_model: AnalisisCompletoResponse`, que acceda a los singletons desde `request.app.state` y delegue a `orchestrator.analizar_caso(...)`.
		2. Documentar el endpoint con un ejemplo de request y response en los metadatos de OpenAPI usando `openapi_examples` para que el Swagger sea usable directamente.
		3. Agregar el tag `"Agente Completo"` para que aparezca agrupado en la documentación `/docs`.

---

> [!IMPORTANT]
> ## Orden de implementación recomendado
>
> Para garantizar que cada capa pueda ser probada de manera independiente antes de integrar la siguiente:
>
> 1. `src/agent/predictor_service.py` — servicio de inferencia autónomo
> 2. `src/agent/orchestrator.py` — coordinador de agentes
> 3. `tc_pipeline/api/schemas.py` — extensión de contratos
> 4. `tc_pipeline/api/routes.py` — activación de endpoints + nuevo `/analizar`
> 5. `tc_pipeline/api/main.py` — inicialización de singletons en lifespan
> 6. `src/ui/index.html` — interfaz de usuario final

> [!NOTE]
> ## Requisito previo para inferencia
>
> El `PredictorService` requiere que los artefactos `models/xgb_classifier.json` y `models/label_encoder.joblib` existan en disco.
> Si aún no se han generado, ejecutar desde la raíz del proyecto:
> ```powershell
> python tc_pipeline/ml-training/model_evaluator.py
> ```
> Esto dispara el pipeline completo: carga → entrenamiento → evaluación → serialización.