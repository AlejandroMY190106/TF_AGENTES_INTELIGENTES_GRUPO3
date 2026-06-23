import os

class LegalRAGAgent:
    def __init__(self, db_path: str = "./data/chroma_storage"):
        """
        Inicializa el Agente utilizando los parámetros unificados del proyecto.
        """
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"🤖 Agente de consulta listo con el modelo NLP: {self.model_name}")

    def generate_prompt(self, query: str, context: str) -> str:
        """
        Ensambla el prompt estructurado con las reglas de negocio del Tribunal Constitucional.
        """
        prompt = f"""
Usted es un Asistente de Inteligencia Artificial experto en la Jurisprudencia del Tribunal Constitucional (TC) del Perú.
Responda la consulta planteada utilizando con rigurosidad los fragmentos del 'CONTEXTO JURÍDICO RELEVANTE'.

REGLAS DE OBLIGATORIO CUMPLIMIENTO:
1. Basarse estrictamente en el contexto proveído. Si no cuenta con datos suficientes, indíquelo de forma educada.
2. Cite de manera obligatoria el número de Expediente al construir su argumento legal.
3. Mantenga un léxico formal, técnico y netamente jurídico.

CONTEXTO JURÍDICO RELEVANTE:
{context}

CONSULTA JURÍDICA:
{query}

RESPUESTA FUNDAMENTADA PERÚ (Cite las fuentes del contexto):
"""
        return prompt

if __name__ == "__main__":
    agent = LegalRAGAgent()
    consulta = "¿Qué criterios se aplican para la protección de la libertad de expresión?"
    contexto_ejemplo = "--- EXPEDIENTE COINCIDENTE #1 ---\nEXPEDIENTE: 00001-2026-AI\nEXTRACTO: La libertad de expresión es pilar fundamental del Estado constitucional..."
    
    prompt_final = agent.generate_prompt(consulta, contexto_ejemplo)
    print("\n======================= PROMPT GENERADO (0 SEGUNDOS) =======================")
    print(prompt_final)