import os
import pandas as pd
from typing import List

class ChromaIndexingPipeline:
    def __init__(self, db_path: str = "./data/chroma_storage", collection_name: str = "jurisprudencia_tc"):
        """
        Configura el esquema del pipeline usando el modelo unificado de la documentación.
        """
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"🗄️ Pipeline de indexación alineado con el modelo: {self.model_name}")
        print(f"📍 Destino de almacenamiento local: {db_path}")

    def chunk_document(self, text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
        """
        [CHUNKING] Divide los extensos expedientes del TC en fragmentos más pequeños por palabras.
        Mantiene un solapamiento (overlap) para asegurar el contexto jurídico entre bloques.
        """
        if not text or not isinstance(text, str):
            return []
            
        words = text.split()
        chunks = []
        
        # Avanza restando el overlap para que el final de un bloque coincida con el inicio del otro
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def index_dataframe(self, df: pd.DataFrame, text_column: str, id_column: str):
        """
        Procesa el conjunto de datos aplicando el chunking respectivo por cada fila.
        """
        print(f"🚀 Iniciando procesamiento e indexación de {len(df)} registros...")
        
        total_chunks = 0
        for idx, row in df.iterrows():
            doc_id = str(row.get(id_column, idx))
            raw_text = row.get(text_column, "")
            
            # Ejecutar el Chunking solicitado
            chunks = self.chunk_document(raw_text)
            total_chunks += len(chunks)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Aquí se genera la estructura limpia mapeada para la base de datos
                _id = f"{doc_id}_chunk_{chunk_idx}"
                _metadata = {
                    "original_id": doc_id,
                    "chunk_index": chunk_idx,
                    "numero_expediente": str(row.get("numero_expediente", doc_id))
                }
                
        print(f"🎯 ¡Procesamiento completado! Se generaron {total_chunks} fragmentos (chunks) listos.")

if __name__ == "__main__":
    pipeline = ChromaIndexingPipeline()
    data_prueba = pd.DataFrame({
        "numero_expediente": ["00001-2026-AI"],
        "texto_legal": ["Sentencia del Tribunal Constitucional sobre la libertad de expresión y los límites del derecho penal en entornos digitales de conformidad con la Constitución."],
    })
    pipeline.index_dataframe(data_prueba, text_column="texto_legal", id_column="numero_expediente")