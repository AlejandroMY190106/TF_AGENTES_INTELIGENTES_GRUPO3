"""
src/agent/rag_service.py
────────────────────────
Orquestador de Recuperación y Generación (RAG) para Jurisprudencia del TC.
"""

import os
import sys
import logging
from typing import Any
import chromadb
from pydantic import BaseModel, Field
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", " ")

# Asegurar rutas de importación del proyecto antes de cargar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.indexing.chroma_pipeline import SentenceTransformerEmbeddingFunction
from tc_pipeline.api.schemas import BriefResponse as GlobalBriefResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StructuredAnalysis(BaseModel):
    """Estructura de respuesta que le exigimos al LLM para el análisis del caso."""
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
        
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key.strip() == "":
            logger.warning("⚠️ ¡Atención! No has configurado tu GOOGLE_API_KEY en las variables de entorno.")
        else:
            genai.configure(api_key=api_key)
            
        self.llm_model_name = "gemini-2.5-flash"

    def retrieve_context(self, query: str, n_results: int = 4) -> tuple[str, list[dict[str, Any]]]:
        """
        Ejecuta búsquedas semánticas sobre ChromaDB.
        Retorna una tupla: (texto_contexto_concatenado, lista_de_fuentes_estructuradas)
        """
        logger.info(f"Buscando contexto semántico para: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            logger.warning("No se encontró contexto relevante en ChromaDB.")
            return "No se encontraron precedentes específicos en la base de datos vectorial.", []
            
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_blocks = []
        sources_list = []
        
        for doc, meta in zip(documents, metadatas):
            exp = meta.get('numero_expediente', 'N/A')
            tipo = meta.get('tipo_expediente', 'N/A')
            sentido = meta.get('sentido_resolucion', 'N/A')
            
            block = f"[Expediente: {exp} | Tipo: {tipo} | Sentido previo: {sentido}]\nTexto: {doc}\n"
            context_blocks.append(block)
            
            # Estructuramos la fuente para la respuesta
            sources_list.append({
                "document": doc,
                "numero_expediente": exp,
                "tipo_expediente": tipo,
                "sentido_resolucion": sentido,
                "url_archivo_TC": meta.get('url_archivo_TC', '')
            })
            
        context_text = "\n---\n".join(context_blocks)
        return context_text, sources_list

    def generate_answer(self, query: str) -> GlobalBriefResponse:
        """
        Une la consulta con los precedentes, genera la respuesta estructurada y la 
        adapta al contrato global de BriefResponse de schemas.py.
        """
        contexto_text, sources = self.retrieve_context(query)
        
        prompt_sistema = (
            "Eres un experto agente de inteligencia jurídica especializado en el Tribunal Constitucional (TC) de Perú. "
            "Tu tarea es analizar la consulta del usuario basándote exclusivamente en el contexto de jurisprudencia proveído abajo. "
            "Debes ser riguroso, objetivo y responder estrictamente usando la estructura JSON solicitada."
        )
        
        prompt_usuario = f"""Analiza la siguiente consulta jurídica utilizando los precedentes vectoriales recuperados de la base de datos.

[CONSULTA DEL USUARIO]
{query}

[CONTEXTO RELEVANTE RECUPERADO (CHROMA DB)]
{contexto_text}

Genera tu dictamen final adaptándote exactamente al formato estructurado."""

        logger.info("Invocando a Gemini-Flash con formato de salida estructurado...")
        
        model = genai.GenerativeModel(
            model_name=self.llm_model_name,
            system_instruction=prompt_sistema
        )
        
        # Invocamos forzando Pydantic mediante la API de Google de forma segura
        response = model.generate_content(
            prompt_usuario,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=StructuredAnalysis,
            ),
        )
        
        # Validamos que la respuesta cumpla el esquema estructurado intermedio
        analysis = StructuredAnalysis.model_validate_json(response.text)
        
        # Mapeamos los resultados estructurados del LLM a un formato amigable Markdown para la API
        brief_markdown = (
            f"### 📋 Resumen del Caso\n"
            f"{analysis.resumen_caso}\n\n"
            f"### ⚖️ Fundamento Jurídico Clave\n"
            f"{analysis.fundamento_clave}\n\n"
            f"### 🔮 Sentido Sugerido\n"
            f"**{analysis.sentido_sugerido}**"
        )
        
        # Retornamos el tipo oficial esperado por la API global del backend
        return GlobalBriefResponse(
            query=query,
            brief=brief_markdown,
            sources=sources,
            metadata={
                "confianza": analysis.confianza,
                "sentido_sugerido": analysis.sentido_sugerido
            }
        )


if __name__ == "__main__":
    # Prueba local de ejecución externa
    try:
        rag = RAGService()
        query_prueba = "Derecho al debido proceso y motivación de resoluciones judiciales en procesos de amparo"
        
        resultado = rag.generate_answer(query=query_prueba)
        print("\n🤖 [RESULTADO DEL AGENTE RAG ÉXITO] 🤖")
        print(resultado.model_dump_json(indent=4))
    except Exception as e:
        logger.error(f"Error ejecutando la prueba del servicio RAG: {str(e)}")