"""
src/agent/rag_service.py
────────────────────────
Orquestador de Recuperación y Generación (RAG) para Jurisprudencia del TC.

CONFIGURACIÓN DE API KEY:
─────────────────────────
Debes definir tu GROQ_API_KEY como variable de entorno antes de ejecutar.
Opciones:
  1. En un archivo .env en la raíz del proyecto:
       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  2. Directamente en la terminal (PowerShell):
       $env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  3. Directamente en la terminal (CMD/bash):
       set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Obtén tu API Key gratis en: https://console.groq.com/keys
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Carga automáticamente las variables del archivo .env
import sys
import json
import logging
from typing import Any
import chromadb
from pydantic import BaseModel, Field
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# Asegurar rutas de importación del proyecto antes de cargar módulos locales
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.indexing.chroma_pipeline import SentenceTransformerEmbeddingFunction
from tc_pipeline.api.schemas import BriefResponse as GlobalBriefResponse  # Importación directa al módulo (evita cargar __init__.py → routes.py)

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

        # ── Inicialización del cliente Groq ────────────────────────────────────
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key.strip() == "":
            logger.warning(
                "⚠️  ¡Atención! No has configurado tu GROQ_API_KEY en las variables de entorno.\n"
                "    Consulta el docstring de este archivo para ver cómo configurarla."
            )
        self.groq_client = Groq(api_key=api_key)
        self.llm_model_name = "llama-3.3-70b-versatile"

    def retrieve_context(self, query: str, n_results: int = 2) -> tuple[str, list[dict[str, Any]]]:
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
        Une la consulta con los precedentes, genera la respuesta estructurada con Groq (Qwen3-32B)
        y la adapta al contrato global de BriefResponse de schemas.py.
        """
        contexto_text, sources = self.retrieve_context(query)

        # Schema JSON explícito para instruir al modelo
        json_schema = json.dumps(StructuredAnalysis.model_json_schema(), ensure_ascii=False, indent=2)

        prompt_sistema = (
            "Eres un experto agente de inteligencia jurídica especializado en el Tribunal Constitucional (TC) de Perú. "
            "Tu tarea es analizar la consulta del usuario basándote exclusivamente en el contexto de jurisprudencia proveído abajo. "
            "Debes ser riguroso, objetivo y responder ÚNICAMENTE con un objeto JSON válido que cumpla exactamente con el "
            f"siguiente schema:\n\n{json_schema}\n\n"
            "No incluyas texto adicional fuera del JSON. No uses bloques de código markdown."
        )

        prompt_usuario = f"""Analiza la siguiente consulta jurídica utilizando los precedentes vectoriales recuperados de la base de datos.

[CONSULTA DEL USUARIO]
{query}

[CONTEXTO RELEVANTE RECUPERADO (CHROMA DB)]
{contexto_text}

Genera tu dictamen final en formato JSON, adaptándote exactamente al schema indicado."""

        logger.info(f"Invocando a Groq ({self.llm_model_name}) con formato de salida JSON estructurado...")

        # Llamada al cliente Groq — sin streaming para obtener la respuesta completa en JSON
        completion = self.groq_client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            response_format={"type": "json_object"},
            stream=False,
        )

        raw_json = completion.choices[0].message.content

        # Validamos que la respuesta cumpla el esquema estructurado
        analysis = StructuredAnalysis.model_validate_json(raw_json)

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