# Selección de Modelo de Embeddings — Sistema Multiagente TC

## Fecha de evaluación: Junio 2026

## Objetivo

Seleccionar el modelo de embeddings óptimo para representar texto jurídico en español
(fundamentos de sentencias del Tribunal Constitucional del Perú) en el vector store
ChromaDB del Agente RAG.

## Modelos candidatos

| Modelo | Tipo | Dimensiones | Idioma | Tamaño |
|--------|------|-------------|--------|--------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-BERT | 384 | 50+ idiomas (incl. español) | ~471 MB |
| `all-MiniLM-L6-v2` | Sentence-BERT | 384 | Inglés (principal) | ~91 MB |
| `nlpaueb/legal-bert-base-uncased` | Legal-BERT | 768 | Inglés legal | ~440 MB |
| `hiiamsid/sentence_similarity_spanish_es` | Sentence-BERT | 768 | Español | ~440 MB |

## Criterios de evaluación

1. **Calidad semántica en español jurídico**: Similitud coseno coherente entre sentencias
   temáticamente relacionadas vs. disímiles.
2. **Soporte nativo de español**: El modelo debe entender texto en español sin traducción.
3. **Tiempo de inferencia**: Latencia aceptable para indexación del corpus (~5000+ sentencias)
   y consultas en tiempo real del Agente RAG.
4. **Dimensionalidad**: Menor dimensionalidad = menor costo de almacenamiento en ChromaDB.
5. **Facilidad de integración**: Disponibilidad en Hugging Face / sentence-transformers.

## Análisis comparativo

### `paraphrase-multilingual-MiniLM-L12-v2` ✅ RECOMENDADO

- **Pros**:
  - Entrenado específicamente para paráfrasis multilingüe, ideal para capturar similitud
    semántica en español jurídico.
  - 384 dimensiones (eficiente para ChromaDB).
  - Latencia baja (~20ms por sentencia en CPU).
  - Ampliamente validado en benchmarks multilingües (STS).
  - Compatible directo con `sentence-transformers` y ChromaDB.
- **Contras**:
  - No fue entrenado específicamente con corpus legal.
  - Puede perder matices de terminología jurídica muy especializada.

### `all-MiniLM-L6-v2`

- **Pros**: Muy rápido (~10ms/sentencia), liviano (91 MB).
- **Contras**: **No soporta español nativamente**. Requeriría traducción previa, lo cual
  introduce ruido y complejidad. **Descartado**.

### `nlpaueb/legal-bert-base-uncased`

- **Pros**: Entrenado con corpus legal extenso (leyes, jurisprudencia, contratos).
- **Contras**: **Solo inglés**. Requeriría fine-tuning extenso o traducción para español
  jurídico. 768 dimensiones (mayor costo de almacenamiento). No es sentence-level por
  defecto (BERT base, no Sentence-BERT). **Descartado** para uso directo.

### `hiiamsid/sentence_similarity_spanish_es`

- **Pros**: Diseñado para español, sentence-level.
- **Contras**: Menos benchmarks publicados. 768 dimensiones (doble que MiniLM).
  Comunidad más pequeña. Podría ser alternativa si MiniLM multilingüe muestra debilidad
  en dominio jurídico.

## Decisión

### Modelo seleccionado: `paraphrase-multilingual-MiniLM-L12-v2`

**Justificación:**
1. Es el modelo que mejor balancea **calidad semántica en español** con **eficiencia
   computacional** (384 dim, ~20ms/sentencia).
2. Soporta español de forma nativa, sin necesidad de traducción.
3. La tarea principal (recuperación semántica RAG) es fundamentalmente de paráfrasis/similitud,
   que es exactamente para lo que fue entrenado.
4. Integración directa con `sentence-transformers` y ChromaDB sin wrappers adicionales.
5. Para el corpus del TC (~5000 sentencias), las 384 dimensiones son más que suficientes
   y reducen el costo de almacenamiento en ChromaDB a la mitad vs. modelos de 768 dim.

**Modelo de respaldo:** `hiiamsid/sentence_similarity_spanish_es` si en la Fase 2
se detecta que la calidad semántica del modelo principal es insuficiente para el dominio
jurídico peruano.

## Integración con ChromaDB

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Modelo seleccionado
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Configuración de ChromaDB
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(
    name="sentencias_tc",
    metadata={"hnsw:space": "cosine"},  # Similitud coseno
)

# Ejemplo de indexación
textos = ["fundamento de la sentencia..."]
embeddings = model.encode(textos).tolist()
collection.add(
    embeddings=embeddings,
    documents=textos,
    ids=["exp-00001"],
    metadatas=[{"sala": "Segunda Sala", "fallo": "Fundada"}],
)
```

## Siguiente paso (Fase 2)

- Dev D generará embeddings sobre el corpus chunked usando este modelo.
- Dev B realizará control de calidad sobre los embeddings generados.
- Si la calidad es insuficiente, se evaluará fine-tuning con datos del TC o cambio
  al modelo de respaldo.
