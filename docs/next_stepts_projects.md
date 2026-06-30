# Próximos Pasos — Refinamiento y Optimización del Sistema

## Estado Actual del Sistema

Todos los módulos de integración end-to-end han sido implementados satisfactoriamente. El sistema cuenta con pipeline de extracción vectorial, entrenamiento multi-modelo (XGBoost, LogisticRegression, SVM, RandomForest), evaluación unificada, servicio RAG con Groq + ChromaDB anclado a la predicción del modelo, orquestador asíncrono con propagación de predicción, API FastAPI funcional con UI servida como archivos estáticos.

| Capa | Módulo(s) | Estado |
|---|---|---|
| Extracción vectorial | `tc_pipeline/ml-training/data_loader.py` | ✅ Completo |
| Entrenamiento XGBoost | `tc_pipeline/ml-training/model_trainer_XGBoost.py` | ✅ Completo (renombrado) |
| Entrenamiento LogReg | `tc_pipeline/ml-training/model_trainer_LogReg.py` | ✅ Completo |
| Entrenamiento SVM | `tc_pipeline/ml-training/model_trainer_SVM.py` | ✅ Completo |
| Entrenamiento RandomForest | `tc_pipeline/ml-training/model_trainer_RandomForest.py` | ✅ Completo |
| Evaluación del modelo | `tc_pipeline/ml-training/model_evaluator.py` | ✅ Completo — compatible con todos los clasificadores |
| Servicio RAG | `src/agent/rag_service.py` | ✅ Completo — prompt anclado a predicción del modelo |
| Orquestador | `src/agent/orchestrator.py` | ✅ Completo — propagación de predicción al RAG |
| API FastAPI — routes | `tc_pipeline/api/routes.py` | ✅ Completo — `/query`, `/prediccion` y `/analizar` activos |
| API FastAPI — main | `tc_pipeline/api/main.py` | ✅ Completo — UI servida vía `StaticFiles` en `/ui` |
| Interfaz de usuario | `src/ui/index.html` | ✅ Completo — accesible en `http://localhost:8000/ui` |

---

## Pendientes

### 1 — Nuevo Algoritmo de Predicción + Renombrado del Trainer XGBoost [COMPLETADO]

### 2 — Consistencia Argumentativa del RAG Service con la Predicción del Modelo [COMPLETADO]

### 3 — Servir la Interfaz de Usuario desde FastAPI [COMPLETADO]

---

> [!NOTE]
> ## Nota sobre artefactos de modelo
>
> El `PredictorService` (`src/agent/predictor_service.py`) actualmente carga `models/xgb_classifier.json` y `models/label_encoder.joblib`. Al incorporar un nuevo clasificador, será necesario actualizar las rutas de carga en `predictor_service.py` o parametrizarlas mediante `MLConfig` para poder alternar entre modelos sin modificar código.