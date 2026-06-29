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

## Archivos Modificados

**tc_pipeline/api/routes.py**
	Objetivo: Activar los endpoints `/query` y `/prediccion` conectándolos al RAGService y al clasificador XGBoost, y crear el endpoint `/analizar`.
	Estado actual: [COMPLETADO] Los endpoints están activos y conectados. Se ejecuta inferencia en hilos asíncronos para evitar bloqueos del event loop de FastAPI, tolerando la ausencia de ChromaDB/modelos locales.

**tc_pipeline/api/main.py**
	Objetivo: Inicializar los servicios RAG y Predictivo durante el evento `lifespan` (startup) para evitar cold-starts.
	Estado actual: [COMPLETADO] Se implementó el lifespan, importando asíncronamente e instanciando RAGService y PredictorService. Se capturan excepciones y se registran advertencias si no están disponibles las bases de datos o archivos de modelos.

**tc_pipeline/api/schemas.py**
	Objetivo: Extender los schemas para soportar el input combinado del agente completo y la respuesta enriquecida (RAG + XGBoost).
	Estado actual: [COMPLETADO] Se agregaron `AnalisisCompletoRequest` y `AnalisisCompletoResponse`, y se modificó `PrediccionRequest` para requerir motivos.

---

## Archivos Creados

**src/agent/predictor_service.py**
	Objetivo: Encapsular toda la lógica de inferencia del clasificador XGBoost.
	Estado actual: [COMPLETADO] Servicio implementado que carga el clasificador JSON y el codificador joblib, vectoriza textos con MiniLM y calcula predicciones y probabilidades.

**src/agent/orchestrator.py**
	Objetivo: Coordinar la ejecución paralela del RAGService y PredictorService.
	Estado actual: [COMPLETADO] Coordinador asíncrono implementado. Corre los servicios concurrentemente en hilos usando `asyncio.to_thread` y combina los resultados manejando fallas parciales.

**tc_pipeline/api/routes.py → endpoint POST /analizar**
	Objetivo: Exponer el endpoint unificado RAG + Predictivo.
	Estado actual: [COMPLETADO] Endpoint expuesto y documentado con Swagger.

---

## Archivos Pendientes

**src/ui/index.html** [NUEVO]
	Objetivo: Interfaz de usuario web completa para probar el agente constitucional de manera interactiva, permitiendo ingresar la demanda y sus motivos y visualizar tanto la predicción del modelo XGBoost como el análisis argumentativo del LLM con casos similares.
	Estado actual: No existe ninguna interfaz. El único punto de acceso es el Swagger generado por FastAPI en `/docs`.
	Instrucciones de creacion:
		1. Crear una Single Page Application en HTML/CSS/JavaScript puro (sin frameworks) con dos campos de texto: `tipo_demanda` (select o input con sugerencias de los tipos de proceso constitucional: Amparo, Hábeas Corpus, Hábeas Data, Inconstitucionalidad, Conflicto de Competencia) y `motivos` (textarea con `min-length` 50 chars), un botón "Analizar caso" que haga un `fetch POST` al endpoint `/api/v1/analizar`.
		2. Diseñar la sección de resultados con tres bloques visuales diferenciados: (a) **Predicción del Modelo** — badge con el fallo predicho coloreado por clase (verde: Fundada, rojo: Infundada, naranja: Improcedente) + barras de probabilidad por clase; (b) **Análisis del Agente RAG** — card con el brief en Markdown renderizado; (c) **Casos Similares** — lista de expedientes fuente con número, tipo y sentido previo, enlazables a la URL del TC.
		3. Implementar estados de carga (spinner animado), validación de campos en frontend antes del request, y manejo de errores HTTP con mensajes descriptivos para el usuario.



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