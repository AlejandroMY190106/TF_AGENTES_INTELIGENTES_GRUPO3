# Arquitectura del Modelo Predictivo — `sentido_resolucion`

> **Documento:** `docs/predictive_model_architecture.md`  
> **Alcance:** Decisión de modelo, flujo de entrenamiento, estrategia de datos y alternativas futuras para el clasificador que predice el fallo del Tribunal Constitucional del Perú.

---

## 1. Objetivo del Modelo

Dado el texto de una demanda constitucional (compuesto por `motivos_demanda` y `fundamentos`), el modelo debe clasificar el **sentido de la resolución** esperada en una de las categorías históricamente registradas por el TC:

| Clase | Descripción |
|---|---|
| **Fundada** | La demanda fue amparada por el TC. |
| **Infundada** | La demanda fue rechazada en el fondo. |
| **Improcedente** | La demanda fue rechazada por causas formales o de competencia. |
| *(otras)* | Clases minoritarias que pueden existir según el corpus indexado. |

---

## 2. Modelo Seleccionado: XGBoost sobre Embeddings Densos

### 2.1 Justificación de la Elección

Se seleccionó **XGBoost** (`xgboost.XGBClassifier`) como modelo inicial por el balance óptimo entre tres factores críticos para este proyecto:

| Factor | XGBoost | Transformer Fine-Tuning |
|---|---|---|
| Tiempo de entrenamiento | **Minutos** (CPU) | Horas/días (GPU) |
| Complejidad de implementación | **Baja** — API sklearn compatible | Alta — loops de entrenamiento, schedulers, checkpoints |
| Interpretabilidad | **Alta** — Feature importance nativa | Baja — caja negra |
| Requerimiento de GPU | **No** | Sí (prácticamente obligatorio) |
| Cantidad de datos necesaria | **Moderada** (decenas de documentos) | Alta (miles de ejemplos para fine-tuning estable) |
| Rendimiento en texto tabulado | **Bueno** con buenos embeddings | Mejor con texto crudo |

### 2.2 Arquitectura de Representación

El modelo **no opera sobre texto crudo**. Opera sobre **embeddings vectoriales densos** generados por el modelo `paraphrase-multilingual-MiniLM-L12-v2` de SentenceTransformers (384 dimensiones). Esto permite:

- Capturar semántica multilingüe sin reentrenamiento del modelo de lenguaje.
- Alimentar al clasificador con representaciones de alta calidad ya optimizadas para similitud semántica.
- Separar la capa de representación de la capa de clasificación, facilitando la sustitución de cualquiera de las dos independientemente.

### 2.3 Flujo de Representación por Documento

Dado que el pipeline de indexación aplica **chunking** para manejar documentos largos, un mismo expediente genera múltiples vectores en ChromaDB. Para garantizar que el clasificador reciba **exactamente una muestra por caso judicial**, se aplica **Mean Pooling**:

```
Documento (expediente)
    │
    ├── Chunk 1  ──→  vector₁ (384d)
    ├── Chunk 2  ──→  vector₂ (384d)
    └── Chunk N  ──→  vectorN (384d)
                           │
                    Mean Pooling (eje=0)
                           │
                    vector_consolidado (384d)  ──→  XGBoost
```

El centroide aritmético produce una representación semántica densa que integra la información de todos los fragmentos del expediente sin perder señal de ninguna sección.

---

## 3. Hiperparámetros del Clasificador

Definidos en `tc_pipeline/config.py → MLConfig` como fuente única de verdad:

| Parámetro | Valor | Justificación |
|---|---|---|
| `objective` | `multi:softprob` | Clasificación multiclase con salida de probabilidades por clase |
| `eval_metric` | `mlogloss` | Log-loss multiclase, sensible al desbalance de clases |
| `n_estimators` | `300` | Suficiente para convergencia con datos de tamaño moderado |
| `max_depth` | `6` | Balance entre capacidad expresiva y riesgo de overfitting |
| `learning_rate` | `0.05` | Tasa conservadora que favorece la generalización |
| `subsample` | `0.8` | Submustreo por árbol para reducir varianza |
| `colsample_bytree` | `0.8` | Submustreo de features por árbol |
| `tree_method` | `hist` | Algoritmo basado en histogramas, más eficiente en CPU |
| `n_jobs` | `-1` | Utiliza todos los núcleos disponibles del sistema |
| `random_state` | `42` | Semilla fija para reproducibilidad total |

---

## 4. Estrategia de Limpieza de Clases (`sentido_resolucion`)

Antes del entrenamiento, las etiquetas de `sentido_resolucion` presentes en los metadatos de ChromaDB requieren una fase de normalización, ya que el texto proviene de PDFs con variaciones tipográficas.

### 4.1 Reglas de Consolidación

| Variantes en el corpus | Clase canónica |
|---|---|
| `"FUNDADA"`, `"Fundada"`, `"DECLARA FUNDADA"`, `"fundada"` | `Fundada` |
| `"INFUNDADA"`, `"Infundada"`, `"DECLARA INFUNDADA"`, `"infundada"` | `Infundada` |
| `"IMPROCEDENTE"`, `"Improcedente"`, `"DECLARA IMPROCEDENTE"` | `Improcedente` |
| `"INADMISIBLE"`, `"Inadmisible"` | `Inadmisible` |
| Vacíos, `"N/A"`, `None` | **Excluir de entrenamiento** |

### 4.2 Implementación

La normalización de clases **debe aplicarse en el pipeline de limpieza** (`tc_pipeline/cleaning/mapping.py`) antes de la indexación en ChromaDB, para que los metadatos ya lleguen limpios al `data_loader.py`. Las clases con menos de **5 muestras** deben evaluarse para exclusión o agrupación en una clase `"Otro"` para evitar problemas en la partición estratificada.

---

## 5. Partición de Datos y Validación

```
Dataset consolidado (N expedientes)
           │
    stratify=y_encoded
           │
    ┌──────┴──────┐
    │             │
  80%           20%
Train set      Test set
    │             │
  fit()        evaluate()
    │             │
XGBClassifier  Accuracy · Confusion Matrix
               Classification Report
               ROC-AUC OvR Macro
```

- **Split estratificado:** preserva la distribución original de clases en ambos conjuntos, evitando que la clase minoritaria quede subrepresentada en el test.
- **Codificación de etiquetas:** `sklearn.preprocessing.LabelEncoder` convierte las clases textuales a índices enteros contiguos requeridos por XGBoost (`0, 1, 2, ...`).
- **Semilla fija:** `random_state=42` en todo el pipeline garantiza que los experimentos sean reproducibles.

---

## 6. Exportación de Artefactos

Todos los artefactos producidos durante el entrenamiento se serializan en el directorio `models/` del proyecto:

| Archivo | Formato | Contenido |
|---|---|---|
| `models/xgb_classifier.json` | JSON nativo XGBoost | Modelo entrenado con todos los árboles y parámetros |
| `models/label_encoder.joblib` | joblib comprimido (nivel 3) | Mapeo clase_texto ↔ índice_entero |
| `models/evaluation_report.txt` | Texto plano | Reporte completo de métricas del conjunto de test |

El formato **JSON nativo** de XGBoost es preferible al pickle de sklearn porque:
- Es independiente de la versión de Python y XGBoost.
- Es legible e inspeccionable por humanos.
- Es compatible directamente con implementaciones en otros lenguajes (C++, Java, R).

---

## 7. Integración con el Servicio de Inferencia

Durante la inferencia en producción, el flujo es:

```
texto_demanda (str)
       │
EmbeddingModel.embed_texts([texto])
       │
vector (1 × 384)
       │
XGBClassifier.predict_proba(vector)
       │
probabilidades (1 × n_clases)
       │
LabelEncoder.inverse_transform([argmax])
       │
{"prediccion": "Fundada", "probabilidades": {...}, "confianza": 0.87}
```

El `PredictorService` (`src/agent/predictor_service.py`) encapsula este flujo completo, cargando los artefactos desde disco durante el startup de la aplicación FastAPI.

---

## 8. Alternativas Avanzadas (Implementación Futura)

Si el rendimiento del clasificador XGBoost resulta insuficiente con el corpus actual, se consideran las siguientes alternativas en orden creciente de complejidad computacional:

### 8.1 Fine-Tuning de Modelo Transformer Multilingüe

**Modelo candidato:** `xlm-roberta-base` o `dccuchile/bert-base-spanish-wwm-cased`

| Aspecto | Detalle |
|---|---|
| **Ventaja** | El modelo aprende representaciones específicas del dominio jurídico constitucional peruano |
| **Desventaja** | Requiere GPU (mínimo 8GB VRAM), tiempo de fine-tuning de 2-8 horas, infraestructura de checkpointing |
| **Datos necesarios** | Mínimo 500-1000 expedientes etiquetados de manera limpia para estabilidad |
| **Framework** | HuggingFace `Trainer` API + `datasets` |

### 8.2 Fine-Tuning de RoBERTa en Español

**Modelo candidato:** `PlanTL-GOB-ES/roberta-base-bne` (entrenado sobre el corpus del BOE español)

| Aspecto | Detalle |
|---|---|
| **Ventaja** | Preentrenado en texto legal en español, menor brecha de dominio |
| **Desventaja** | No está optimizado para español peruano ni para texto constitucional |
| **Costo computacional** | Similar al XLM-RoBERTa, con requerimiento adicional de adaptación de vocabulario |

### 8.3 Criterios para Escalar a Transformers

Se recomienda evaluar el salto a fine-tuning de transformers cuando:
- El accuracy del XGBoost sea inferior al **70%** en el test set con el corpus completo.
- El corpus supere los **2 000 expedientes** con etiquetas limpias.
- Se disponga de infraestructura GPU (Google Colab Pro, Kaggle, o instancia cloud con T4/A100).

---

## 9. Métricas de Éxito del Modelo

| Métrica | Umbral mínimo aceptable | Objetivo ideal |
|---|---|---|
| Accuracy global | > 65% | > 80% |
| F1-Score macro | > 0.55 | > 0.75 |
| ROC-AUC OvR macro | > 0.70 | > 0.85 |
| Recall clase minoritaria | > 0.40 | > 0.60 |

> [!NOTE]
> Los umbrales se definieron considerando el desbalance natural de clases en jurisprudencia constitucional, donde "Infundada" e "Improcedente" suelen ser más frecuentes que "Fundada".
