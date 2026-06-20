# Plan de desarrollo MVP — Sistema multiagente para análisis de jurisprudencia del TC

## Sprint de 10 días (16-25 de junio de 2026)

## Supuestos de planificación

- Inicio: martes 16 de junio (Día 1). Fin: jueves 25 de junio (Día 10), 10 días corridos.
- Equipo de 4 desarrolladores con rol principal fijo, pero con colaboración cruzada en los días de cierre de cada fase.
- Los días 5 y 6 (sábado 20 y domingo 21) caen dentro de la Fase 2. Dado que el desarrollo está acelerado con asistencia de IA, se mantienen como días del sprint, pero las tareas asignadas ahí (indexación, pruebas de calidad) son de menor bloqueo para dar flexibilidad al equipo.
- Toda fecha de entregable se entiende como cierre de jornada (EOD) de ese día.

## Equipo y rol principal

| Desarrollador | Rol principal | Responsabilidad central |
|---|---|---|
| Dev A | Ingeniería de datos / Agente de Curación | Pipeline de extracción (Scrapy), limpieza e imputación con Pandas, merge del xlsx histórico con los datos del JSON |
| Dev B | NLP / Embeddings | Chunking del corpus, extracción NLP de campos secundarios, generación y control de calidad de embeddings |
| Dev C | Backend / Orquestación | FastAPI, contratos de API entre agentes, Agente RAG (recuperación semántica + generación de brief) |
| Dev D | ML / Agente Predictivo | Configuración de ChromaDB, feature engineering, entrenamiento y evaluación de modelos |

---

## Fase 1 — Recopilación de datos (Días 1-3 · 16-18 de junio)

**Objetivo:** dejar operativo el pipeline de ingesta sobre una muestra representativa y validar la cobertura del periodo 1992-2012.

**Tareas por desarrollador**

- **Dev A:** implementar el pipeline en Scrapy contra el endpoint `.../sistematizacion-jurisprudencial/avanzado/`, manejando la paginación (una sola consulta genérica ya devuelve 136 páginas) y diseñando combinaciones de filtros (rangos de fecha, tipo de expediente) para una cobertura sistemática 1992-2026. Exportar a almacenamiento intermedio (Parquet o SQLite).
- **Dev B:** construir la tabla de mapeo JSON → variables del xlsx (PUB_PAGWEB ← `fecha_publicacion`, CDES_TIPOPROCESO ← sufijo de `numero_expediente`, SALA ← `sentencia_sala`, FALLO ← `sentencia_sentido`, etc.) y la jerarquía MATERIA/SUB_MATERIA/ESPECIFICA a partir de `palabras`. Confirmar la tabla de equivalencias de `sentencia_tipo` (validar si el código 4 corresponde a "Sentencia" según el combobox "Tipo de Resolución").
- **Dev C:** configurar el repositorio (estructura de carpetas, entorno/Docker), esqueleto de FastAPI y esquemas Pydantic que servirán como contrato de datos entre el Agente de Curación, el Agente RAG y el Agente Predictivo.
- **Dev D:** levantar una instancia persistente de ChromaDB y comparar Legal-BERT vs. Sentence-BERT sobre una muestra de `fundamentos` en español jurídico (calidad de similitud, tiempo de inferencia), seleccionando el modelo de embeddings a usar.

**Entregable — Día 3 (18 jun):** pipeline de extracción funcional con al menos 500 expedientes recolectados, incluyendo una muestra del periodo 1992-2012 para validar su cobertura; diccionario de mapeo consolidado; repositorio y esqueleto de FastAPI operativos; modelo de embeddings seleccionado.

**Riesgo crítico a resolver en esta fase:** si la muestra 1992-2012 muestra `attachment.content` vacío o con ruido de OCR, el equipo debe decidir, a más tardar el Día 3, si ese periodo se trata solo como enriquecimiento parcial de metadatos (sin texto íntegro) o si se excluye del corpus de embeddings, ajustando el alcance de la Fase 2.

---

## Fase 1 (cierre) + Fase 2 — Curación, NLP y embeddings (Días 4-6 · 19-21 de junio)

**Objetivo:** contar con el dataset histórico saneado y enriquecido (v1) y con el corpus indexado en ChromaDB.

**Tareas por desarrollador**

- **Dev A:** como Agente de Curación, aplicar imputación estadística con Pandas usando los datos scrapeados y las reglas de mapeo de la Fase 1 (CDES_TIPOPROCESO, SALA y FALLO se llenan de forma directa; DEPARTAMENTO se obtiene vía una tabla de equivalencia Distrito Judicial → Departamento). Generar el merge final entre el xlsx histórico y los datos enriquecidos del JSON, usando `numero_expediente` como llave.
- **Dev B:** construir el pipeline de chunking usando `fundamentos` como unidad base (con `attachment.content` como respaldo cuando `fundamentos` esté vacío). Implementar la extracción NLP/regex de campos secundarios: FEC_INGRESO (patrón "con fecha [...] interpone demanda"), SALA_ORIGEN (patrón "expedida por la [Sala/Juzgado...]") y una clasificación heurística de TIPO_DEMANDANTE/TIPO_DEMANDADO (persona natural vs. entidad pública) a partir de `nombre_demandante`/`nombre_demandado`.
- **Dev C:** construir los endpoints de FastAPI para ingesta (`/ingest`) y consulta del dataset curado (`/expedientes`), validando contra los esquemas Pydantic definidos en la Fase 1.
- **Dev D:** generar los embeddings sobre el corpus chunked (entregado por Dev B) e indexarlos en ChromaDB junto con su metadata (`sentencia_sala`, `sentencia_sentido`, materia derivada de `palabras`, `numero_expediente`).

**Entregable — Día 6 (21 jun):** dataset histórico v1 saneado y enriquecido (exportado a .xlsx/.parquet); ChromaDB poblado con embeddings y metadata filtrable; endpoints `/ingest` y `/expedientes` operativos.

---

## Fase 3 — Arquitectura del SMA (Días 7-8 · 22-23 de junio)

**Objetivo:** tener el Agente RAG funcionando de extremo a extremo y el dataset de features listo para entrenamiento.

**Tareas por desarrollador**

- **Dev C:** implementar el Agente RAG — búsqueda por similitud de coseno en ChromaDB con filtros de metadata (sala, materia, sentido del fallo) y generación del brief ejecutivo mediante un LLM contextualizado (API de Claude) a partir de los fragmentos recuperados.
- **Dev D:** realizar el feature engineering, combinando variables estructuradas (SALA, CDES_TIPOPROCESO, DEPARTAMENTO, etc.) con embeddings agregados por expediente, y preparar el split de entrenamiento/prueba con datos del periodo 2013-2026 para el Agente Predictivo.
- **Dev A:** dar soporte de integración, asegurando que el dataset curado de la Fase 2 alimente correctamente tanto a ChromaDB como al dataset de features de Dev D, y documentar el pipeline de datos de punta a punta.
- **Dev B:** realizar control de calidad sobre chunking/embeddings (casos de borde como expedientes sin `fundamentos` o contenido truncado), ajustando tamaño y solapamiento de chunks según los resultados del Agente RAG, y dando soporte a Dev C en la calidad de los fragmentos recuperados.

**Entregable — Día 8 (23 jun):** Agente RAG funcional generando briefs ejecutivos de extremo a extremo (consulta → recuperación semántica → brief con LLM); dataset de features del periodo 2013-2026 listo y particionado para entrenamiento.

---

## Fase 4 — Entrenamiento, evaluación e integración (Días 9-10 · 24-25 de junio)

**Objetivo:** MVP integrado con los tres agentes operando vía FastAPI y una comparativa documentada de modelos predictivos.

**Tareas por desarrollador**

- **Dev D + Dev A:** entrenar y evaluar Random Forest, XGBoost y una Red Neuronal Feed-Forward sobre el dataset de features, comparando métricas (accuracy, F1 ponderado, matriz de confusión por clase de FALLO) y seleccionando el modelo de mejor desempeño para el Agente Predictivo.
- **Dev C:** integrar el Agente Predictivo como endpoint (`/prediccion`) en FastAPI y ejecutar las pruebas de integración de extremo a extremo de los tres agentes orquestados (Curación → RAG → Predictivo).
- **Dev B:** documentar técnicamente el pipeline de NLP/embeddings y ejecutar pruebas de regresión sobre el corpus indexado.
- **Todo el equipo:** pruebas de integración finales, preparación de la demo del MVP y documentación de cierre del sprint.

**Entregable — Día 10 (25 jun):** MVP funcional integrado (tres agentes orquestados vía FastAPI), informe comparativo de modelos predictivos con sus métricas, documentación técnica completa y demo lista para presentación.

---

## Resumen de entregables por fecha

| Día | Fecha | Entregable |
|---|---|---|
| 3 | 18 jun | Pipeline de extracción + diccionario de mapeo + esqueleto de FastAPI + modelo de embeddings seleccionado |
| 6 | 21 jun | Dataset histórico v1 saneado/enriquecido + ChromaDB indexado + endpoints de ingesta/consulta |
| 8 | 23 jun | Agente RAG funcional (brief de extremo a extremo) + dataset de features listo |
| 10 | 25 jun | MVP integrado (tres agentes vía FastAPI) + comparativa de modelos + documentación final |

## Dependencias críticas

- La Fase 1 es ruta crítica: todas las fases posteriores dependen de la disponibilidad y calidad del dataset enriquecido.
- La confirmación de la tabla de `sentencia_tipo` (Dev B, Días 1-2) condiciona el llenado de TIPO_RESOLUCION en las Fases 1 y 2.
- La validación de cobertura 1992-2012 (Dev A, Días 1-3) condiciona el alcance del corpus de embeddings en la Fase 2.
- El dataset de features (Dev D, Fase 3) depende del dataset curado v1 (Fase 2); cualquier retraso en el Día 6 impacta directamente el entrenamiento de la Fase 4.
