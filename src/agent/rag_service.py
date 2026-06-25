"""
src/agent/rag_service.py
─────────────────────────────
Orquestador de Recuperación y Generación (RAG) para Jurisprudencia del TC.

Responsabilidad Arquitectónica:
1. Conectarse de forma persistente a la colección 'jurisprudencia_tc' en ChromaDB.
2. Recuperar contexto relevante mediante búsquedas semánticas (retrieve_context).
3. Invocar al LLM (Gemini-Flash) aplicando ingeniería de prompts estructurada.
4. Forzar y validar el output bajo el esquema de datos pydantic BriefResponse.
"""

import os
import sys
import logging
import chromadb
from pydantic import BaseModel, Field
import google.generativeai as genai

# 🔑 INYECTA TU LLAVE AQUÍ DIRECTAMENTE:
# Reemplaza el texto dentro de las comillas por la API Key que copiaste de Google AI Studio (la que empieza con AIzaSy...)
os.environ["GOOGLE_API_KEY"] = " "

# Asegurar rutas de importación del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tc_pipeline.nlp.embeddings import EmbeddingModel
from src.indexing.chroma_pipeline import SentenceTransformerEmbeddingFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 📋 Esquema Estructurado solicitado por el grupo para la respuesta del Agente
class BriefResponse(BaseModel):
    resumen_caso: str = Field(description="Breve resumen del expediente y la controversia constitucional analizada.")
    fundamento_clave: str = Field(description="El fragmento o razonamiento jurídico más determinante extraído del contexto.")
    sentido_sugerido: str = Field(description="Sentido de la resolución (e.g., FUNDADA, INFUNDADA, IMPROCEDENTE) basado en los precedentes.")
    confianza: float = Field(description="Nivel de confianza de la respuesta (de 0.0 a 1.0).")


class RAGService:
    def __init__(self, db_path: str = "data/chroma_storage", collection_name: str = "jurisprudencia_tc"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        
        logger.info(f"Conectando RAG Service a ChromaDB en {self.db_path}...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        
        # Conexión nativa a la colección con su función de embeddings multilingüe
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        # Configurar la API Key leyendo la variable que pusimos arriba
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "PEGAR_AQUI_TU_API_KEY":
            logger.warning("⚠️ ¡Atención! No has reemplazado el texto 'PEGAR_AQUI_TU_API_KEY' con tu llave real.")
        else:
            genai.configure(api_key=api_key)
            
        self.llm_model_name = "gemini-2.5-flash"

    def retrieve_context(self, query: str, n_results: int = 4) -> str:
        """
        Ejecuta búsquedas semánticas sobre ChromaDB y concatena los fragmentos
        más relevantes para alimentar el contexto del prompt.
        """
        logger.info(f"Buscando contexto semántico para: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            logger.warning("No se encontró contexto relevante en ChromaDB.")
            return "No se encontraron precedentes específicos en la base de datos vectorial."
            
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_blocks = []
        for doc, meta in zip(documents, metadatas):
            exp = meta.get('numero_expediente', 'N/A')
            tipo = meta.get('tipo_expediente', 'N/A')
            sentido = meta.get('sentido_resolucion', 'N/A')
            
            block = f"[Expediente: {exp} | Tipo: {tipo} | Sentido previo: {sentido}]\nTexto: {doc}\n"
            context_blocks.append(block)
            
        return "\n---\n".join(context_blocks)

    def generate_answer(self, query: str) -> BriefResponse:
        """
        Une la consulta con los precedentes recuperados y genera una respuesta 
        estructurada de manera determinista usando Gemini-Flash y Pydantic.
        """
        contexto = self.retrieve_context(query)
        
        prompt_sistema = (
            "Eres un experto agente de inteligencia jurídica especializado en el Tribunal Constitucional (TC) de Perú. "
            "Tu tarea es analizar la consulta del usuario basándote exclusivamente en el contexto de jurisprudencia proveído abajo. "
            "Debes ser riguroso, objetivo y fundamentar tu análisis en los precedentes adjuntos."
        )
        
        prompt_usuario = f"""Analiza la siguiente consulta jurídica utilizando los precedentes vectoriales recuperados de la base de datos.

[CONSULTA DEL USUARIO]
{query}

[CONTEXTO RELEVANTE RECUPERADO (CHROME DB)]
{contexto}

Genera tu dictamen final adaptándote estrictamente al formato JSON requerido por el esquema de salida."""

        logger.info("Invocando a Gemini-Flash con formato de salida estructurado...")
        
        model = genai.GenerativeModel(
            model_name=self.llm_model_name,
            system_instruction=prompt_sistema
        )
        
        response = model.generate_content(
            prompt_usuario,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=BriefResponse,
            ),
        )
        
        return BriefResponse.model_validate_json(response.text)


if __name__ == "__main__":
    try:
        rag = RAGService()
        query_prueba = "Derecho al debido proceso y motivación de resoluciones judiciales en procesos de amparo"
        
        resultado = rag.generate_answer(query=query_prueba)
        print("\n🤖 [RESULTADO DEL AGENTE RAG ÉXITO] 🤖")
        print(resultado.model_dump_json(indent=4))
    except Exception as e:
        logger.error(f"Error ejecutando la prueba del servicio RAG: {str(e)}")