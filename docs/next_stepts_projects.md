# Próximos Pasos — Refinamiento y Optimización del Sistema

## Estado Actual del Sistema

Todos los módulos de integración end-to-end han sido implementados satisfactoriamente. El sistema cuenta con pipeline de extracción vectorial, entrenamiento XGBoost, evaluación, servicio RAG con Groq + ChromaDB, orquestador asíncrono, API FastAPI funcional e interfaz de usuario web.

El foco de desarrollo se desplaza ahora hacia la **optimización del desempeño predictivo** y la **consistencia argumentativa del agente RAG**, sin introducir nuevas funcionalidades mayores.

| Capa | Módulo(s) | Estado |
|---|---|---|
| Extracción vectorial | `tc_pipeline/ml-training/data_loader.py` | ✅ Completo |
| Entrenamiento XGBoost | `tc_pipeline/ml-training/model_trainer_XGBoost.py` | ✅ Renombrado (pendiente ejecución) |
| Nuevo algoritmo predicción | `tc_pipeline/ml-training/model_trainer_<nombre>.py` | ⏳ Pendiente implementación |
| Evaluación del modelo | `tc_pipeline/ml-training/model_evaluator.py` | ⚠️ Acoplado a XGBoost — requiere compatibilización |
| Servicio RAG | `src/agent/rag_service.py` | ⚠️ Prompt sin anclaje a predicción del modelo |
| API FastAPI | `tc_pipeline/api/` | ✅ Completo |
| Interfaz de usuario | `src/ui/index.html` | ✅ Completo |

---

## Pendientes

### 1 — Nuevo Algoritmo de Predicción + Renombrado del Trainer XGBoost

**Contexto:** El clasificador XGBoost entrenado sobre embeddings MiniLM alcanzó un **Accuracy del 60 %** en la evaluación (`model_evaluator.py`). Para un sistema de soporte a decisiones jurídicas esto es insuficiente. Se explorarán algoritmos que aprovechen mejor la naturaleza de alta dimensionalidad y la distribución de clases desbalanceadas propia de los expedientes constitucionales.

**Algoritmos sugeridos y justificación:**

| Algoritmo | Justificación para este caso de uso |
|---|---|
| **SVM con kernel RBF** (`sklearn.svm.SVC`) | Excelente desempeño en espacios de alta dimensión como embeddings densos (MiniLM produce vectores de 384 dims). Muy robusto ante clases desbalanceadas con `class_weight='balanced'`. Históricamente supera a árboles de decisión en datasets de texto vectorizado pequeños/medianos. |
| **Regresión Logística Multinomial** (`sklearn.linear_model.LogisticRegression`, `multi_class='multinomial'`) | Establece una línea base probabilística sólida. La regularización L2 previene sobreajuste en embeddings. Produce probabilidades calibradas de forma nativa, lo cual es crítico para mostrar confianza por clase en la UI. Es el primer candidato a probar antes de modelos más complejos. |
| **Random Forest** (`sklearn.ensemble.RandomForestClassifier`) | Robusto ante ruido, no requiere normalización de features, y provee `feature_importances_` para interpretabilidad. Alternativa ensemble menos propensa a sobreajuste que XGBoost en datasets pequeños. |
| **MLP / Red Neuronal ligera** (`sklearn.neural_network.MLPClassifier`) | Los embeddings semánticos son representaciones continuas latentes; una capa oculta puede capturar relaciones no lineales que los modelos superficiales pierden. Viable si el dataset supera ~800 muestras para evitar sobreajuste. |

> **Recomendación de orden de implementación:** Logistic Regression → SVM RBF → Random Forest → MLP. Comparar métricas con `model_evaluator.py` en cada iteración.

---

**tc_pipeline/ml-training/model_trainer.py** → renombrar a **model_trainer_XGBoost.py**
	Objetivo: Renombrar el trainer actual para reflejar que es exclusivo del clasificador XGBoost, liberando el nombre genérico `model_trainer.py` para un posible trainer base o de selección de algoritmo.
	Estado actual: Archivo existente con 264 líneas. Implementa la función `train(X, y, cfg)` que codifica etiquetas con `LabelEncoder`, realiza partición estratificada 80/20, entrena `XGBClassifier` y serializa artefactos en `models/xgb_classifier.json` y `models/label_encoder.joblib`. Importado directamente por `model_evaluator.py` en su bloque `__main__`.
	Instrucciones de modificacion:
		1. Renombrar el archivo físicamente de `model_trainer.py` a `model_trainer_XGBoost.py` conservando todo su contenido intacto.
		2. Actualizar el docstring del módulo en la línea 3 para que refleje la nueva ruta: `tc_pipeline/ml-training/model_trainer_XGBoost.py`.
		3. Actualizar el bloque `__main__` de `model_evaluator.py` (línea 299): cambiar `from model_trainer import train` por `from model_trainer_XGBoost import train` para mantener la cadena de ejecución directa.
		4. Verificar que `predictor_service.py` no importe directamente desde `model_trainer`; si lo hiciera, actualizar también esa referencia.

---

**tc_pipeline/ml-training/model_trainer_<nombre>.py** *(archivo nuevo — uno por algoritmo explorado)*
	Objetivo: Encapsular el entrenamiento del nuevo algoritmo de predicción siguiendo la misma interfaz contractual que `model_trainer_XGBoost.py` (`train(X, y, cfg) -> TrainingResult`) para que sea evaluable con el `model_evaluator.py` actual sin modificarlo.
	Estado actual: No existe. El único trainer disponible es el de XGBoost. El `model_evaluator.py` es agnóstico al algoritmo siempre que el modelo exponga los métodos `.predict(X)` y `.predict_proba(X)`, que todos los clasificadores de scikit-learn implementan por defecto.
	Instrucciones de creacion:
		1. Crear el archivo con el nombre `model_trainer_<algoritmo>.py` (e.g. `model_trainer_SVM.py`, `model_trainer_LogReg.py`) dentro de `tc_pipeline/ml-training/`.
		2. Reutilizar el mismo `@dataclass TrainingResult` de `model_trainer_XGBoost.py` (importarlo o redefinirlo localmente) para garantizar compatibilidad de interfaz con el evaluador y el `predictor_service.py`.
		3. Implementar la función `train(X, y, cfg) -> TrainingResult` siguiendo los mismos pasos de codificación de etiquetas (`LabelEncoder`), filtrado de clases singleton y partición estratificada 80/20 que ya existen en XGBoost trainer (estos pasos son invariantes al algoritmo).
		4. Instanciar el clasificador elegido con hiperparámetros iniciales razonables. Ejemplos:
			- SVM: `SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)` — `probability=True` es obligatorio para que `.predict_proba()` esté disponible.
			- Logistic Regression: `LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, class_weight='balanced')`.
		5. Serializar los artefactos con nombres distintos a los de XGBoost para evitar colisión (e.g. `models/svm_classifier.joblib` y `models/svm_label_encoder.joblib`). Usar `joblib.dump` para ambos artefactos ya que scikit-learn no tiene formato JSON nativo como XGBoost.
		6. Actualizar `MLConfig` en `tc_pipeline/config.py` con las rutas de artefacto del nuevo modelo si se desea centralizar la configuración, o gestionar las rutas localmente con `pathlib.Path`.

---

**tc_pipeline/ml-training/model_evaluator.py**
	Objetivo: Compatibilizar el evaluador para que acepte cualquier clasificador de scikit-learn (o similar) además de `XGBClassifier`, eliminando el acoplamiento de tipo en la firma de `evaluate()`.
	Estado actual: La función `evaluate()` (línea 63) tiene la firma `evaluate(model: xgb.XGBClassifier, ...)`, lo que implica una dependencia de tipo duro a XGBoost. Sin embargo, el cuerpo de la función únicamente llama a `model.predict(X_test)` y `model.predict_proba(X_test)`, métodos presentes en la API estándar de scikit-learn. El encabezado del reporte en línea 170 también referencia literalmente "XGBoost". La función `load_and_evaluate()` instancia explícitamente `xgb.XGBClassifier()` para cargar el modelo, lo que la hace incompatible con modelos joblib de scikit-learn.
	Instrucciones de modificacion:
		1. Cambiar el type hint del parámetro `model` en la firma de `evaluate()` de `xgb.XGBClassifier` a `Any` (importar `from typing import Any`) o a un `Protocol` informal: `model: object`. Esto elimina el acoplamiento estático sin romper el contrato funcional.
		2. Agregar un parámetro opcional `model_name: str = "Clasificador"` a la firma de `evaluate()` y usarlo en el encabezado del reporte (línea 170) en lugar del literal "XGBoost": `f"  📊  REPORTE DE EVALUACIÓN DEL CLASIFICADOR {model_name.upper()}"`.
		3. Modificar `load_and_evaluate()` para recibir un parámetro `model_type: str = "xgboost"` que permita dos ramas de carga:
			- `"xgboost"`: mantiene la lógica actual de `xgb.XGBClassifier().load_model(path)`.
			- `"sklearn"`: carga el artefacto con `joblib.load(model_path)` directamente (los modelos scikit-learn se serializan como objetos completos, no como pesos separados).
		4. Actualizar el bloque `__main__` para que importe desde `model_trainer_XGBoost` en lugar de `model_trainer` (sincronizar con el renombrado del punto anterior).
		5. Verificar que las métricas de Accuracy, Matriz de Confusión, Classification Report y ROC-AUC OvR Macro sigan funcionando correctamente para clasificadores scikit-learn; no se requieren cambios en el cuerpo de cálculo de métricas ya que scikit-learn expone la misma API de predicción.

---

### 2 — Consistencia Argumentativa del RAG Service con la Predicción del Modelo

**Contexto:** Se observó que el LLM (Groq / LLaMA-3.3-70b) generaba un análisis argumentativo que concluía en `"FUNDADA"` cuando el modelo XGBoost había predicho `"INFUNDADA"` con un 47.2 % de confianza. Esto ocurre porque el prompt del sistema no le informa al LLM cuál fue la predicción cuantitativa del modelo predictivo, dejando que el LLM derive su propia conclusión libremente a partir de los precedentes recuperados de ChromaDB. El resultado es una experiencia de usuario contradictoria donde dos componentes del mismo sistema presentan veredictos opuestos.

---

**src/agent/rag_service.py**
	Objetivo: Reestructurar el prompt del sistema y del usuario en `generate_answer()` para que el LLM reciba la predicción del modelo como contexto ancla obligatorio, y su análisis argumental se alinee con dicha predicción en lugar de contradecirla o ignorarla.
	Estado actual: El método `generate_answer(self, query: str)` (línea 118) construye dos prompts: `prompt_sistema` (líneas 128–134) que instruye al LLM sobre su rol y formato JSON, y `prompt_usuario` (líneas 136–144) que le pasa la consulta y el contexto de ChromaDB. Ninguno de los dos prompts incluye la predicción del modelo ni su nivel de confianza. El campo `sentido_sugerido` del schema `StructuredAnalysis` (línea 48) es inferido libremente por el LLM sin restricción de consistencia. La firma del método no acepta parámetros adicionales más allá de `query`. La temperatura de inferencia está configurada en `1` (línea 155), introduciendo alta aleatoriedad.
	Instrucciones de modificacion:
		1. Modificar la firma del método `generate_answer()` para aceptar dos parámetros opcionales adicionales: `prediccion: str | None = None` y `confianza_prediccion: float | None = None`. Esto mantiene retrocompatibilidad con cualquier consumidor que invoque el método sin esos parámetros.
		2. Construir un bloque de texto condicional `anchor_block` que solo se incluya en el prompt cuando `prediccion` no sea `None`:
			```python
			anchor_block = ""
			if prediccion is not None:
			    confianza_str = f"{confianza_prediccion * 100:.1f}%" if confianza_prediccion is not None else "N/D"
			    anchor_block = (
			        f"\n[PREDICCIÓN DEL MODELO CUANTITATIVO]\n"
			        f"El clasificador entrenado ha determinado que la resolución más probable es: "
			        f"**{prediccion.upper()}** (confianza: {confianza_str}).\n"
			        f"Tu análisis argumentativo DEBE ser consistente con esta predicción. "
			        f"Si los precedentes recuperados apuntan en una dirección diferente, "
			        f"debes explicar por qué el caso bajo consulta se aleja de esos precedentes "
			        f"y justificar el resultado '{prediccion.upper()}' con base en sus motivos específicos.\n"
			    )
			```
		3. Insertar `anchor_block` al final del `prompt_usuario`, después del bloque `[CONTEXTO RELEVANTE RECUPERADO (CHROMA DB)]` y antes de la instrucción de generación JSON. Esto garantiza que el LLM tenga la predicción presente en el turno de usuario donde procesa los datos concretos del caso.
		4. Modificar el `prompt_sistema` para añadir una instrucción explícita de consistencia al final: `"Cuando se te proporcione la predicción de un modelo cuantitativo externo, tu campo 'sentido_sugerido' DEBE coincidir con dicha predicción. Tu rol es argumentar y justificar esa predicción con base jurídica, no cuestionarla."`. Esta instrucción en el system prompt tiene mayor peso en el comportamiento del modelo que la misma instrucción en el user prompt.
		5. Actualizar el campo `sentido_sugerido` del schema `StructuredAnalysis` (línea 48) para que su `description` refleje la nueva semántica: `"Sentido de la resolución que DEBE coincidir con la predicción del modelo cuantitativo si fue provista (e.g., FUNDADA, INFUNDADA, IMPROCEDENTE). En ausencia de predicción externa, inferir desde los precedentes."`.
		6. Reducir `temperature` de `1` a `0.3` en la llamada `self.groq_client.chat.completions.create()` (línea 155). Una temperatura alta introduce aleatoriedad que puede hacer que el LLM ignore instrucciones de anclaje. Con `0.3` el modelo es más determinista y obedece mejor las restricciones del prompt sin perder capacidad de razonamiento jurídico.

---

**src/agent/orchestrator.py**
	Objetivo: Propagar la predicción del `PredictorService` hacia el `RAGService` para que el análisis argumentativo esté anclado al veredicto cuantitativo.
	Estado actual: El orquestador ejecuta `PredictorService` y `RAGService` de forma concurrente usando `asyncio.to_thread`. Combina los resultados de ambos servicios en un dict unificado. El llamado a `rag_service.generate_answer()` actualmente solo recibe `query`, sin pasar la predicción ya obtenida del predictor. Esto significa que aunque el orquestador posee ambos resultados en el mismo scope, no los conecta entre sí.
	Instrucciones de modificacion:
		1. Revisar el flujo de ejecución del orquestador para identificar el punto donde se invoca `rag_service.generate_answer(query)` y se dispone ya del resultado del predictor.
		2. Cambiar la estrategia de ejecución: en lugar de correr ambos servicios de forma 100% paralela, ejecutar primero el `PredictorService` (que es más rápido al ser inferencia local) y luego pasar su resultado al `RAGService`. Alternativamente, si se mantiene la ejecución paralela, ejecutar el RAG en una segunda fase secuencial, una vez conocida la predicción.
		3. Extraer `prediccion` (la clase predicha con mayor probabilidad) y `confianza_prediccion` (su probabilidad numérica) del resultado del `PredictorService` y pasarlos como argumentos en la invocación `rag_service.generate_answer(query=query, prediccion=prediccion, confianza_prediccion=confianza_prediccion)`.
		4. Manejar el caso de falla parcial del predictor: si `PredictorService` falla y no hay predicción disponible, invocar `generate_answer()` sin los parámetros opcionales (comportamiento actual) para que el RAG funcione de manera degradada pero no se rompa.

---

> [!IMPORTANT]
> ## Orden de implementación recomendado
>
> Para garantizar trazabilidad y poder evaluar el impacto de cada cambio de forma aislada:
>
> 1. Renombrar `model_trainer.py` → `model_trainer_XGBoost.py` y actualizar las referencias en `model_evaluator.py`.
> 2. Compatibilizar `model_evaluator.py` (type hints + parámetro `model_name` + rama `load_and_evaluate` para sklearn).
> 3. Implementar el primer trainer alternativo (`model_trainer_LogReg.py`) y evaluar con `model_evaluator.py`.
> 4. Comparar Accuracy y ROC-AUC vs XGBoost; si mejora, continuar con SVM RBF.
> 5. Modificar `rag_service.py` (firma + prompts + temperatura) de forma independiente al trabajo del modelo.
> 6. Actualizar `orchestrator.py` para pasar la predicción al RAG en la fase de invocación.
> 7. Probar el flujo completo desde `src/ui/index.html` verificando consistencia predicción ↔ argumentación.

> [!NOTE]
> ## Nota sobre artefactos de modelo
>
> El `PredictorService` (`src/agent/predictor_service.py`) actualmente carga `models/xgb_classifier.json` y `models/label_encoder.joblib`. Al incorporar un nuevo clasificador, será necesario actualizar las rutas de carga en `predictor_service.py` o parametrizarlas mediante `MLConfig` para poder alternar entre modelos sin modificar código.