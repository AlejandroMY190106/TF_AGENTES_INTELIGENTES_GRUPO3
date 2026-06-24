"""
src/indexing/chroma_pipeline.py
─────────────────────────────
Orquestador de Indexación ETL (Extract, Transform, Load).

Responsabilidad Arquitectónica:
Extrae los datos limpios desde los archivos CSV generados, los transforma
en chunks estructurados invocando a `tc_pipeline.nlp.processing`, y finalmente
los carga e indexa de manera nativa en la base de datos vectorial ChromaDB.
ChromaDB se encarga de calcular los embeddings utilizando el modelo multilingüe.
"""

import os
import sys
import glob
import pandas as pd
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import logging

# Inyección dinámica de la raíz del proyecto para evitar errores de ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tc_pipeline.nlp.embeddings import EmbeddingModel
from tc_pipeline.nlp.processing import build_chunks_for_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Clase adaptadora (Wrapper) para inyectar nuestro modelo SentenceTransformer 
    como una función nativa de generación de embeddings dentro de ChromaDB.
    """
    def __init__(self, model_name: str):
        self.model = EmbeddingModel(model_name=model_name)
    
    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB pasa una lista de strings ('Documents') y espera una matriz de floats ('Embeddings')
        return self.model.embed_texts(list(input))

class ChromaIndexingPipeline:
    def __init__(self, db_path: str = "data/chroma_storage", collection_name: str = "jurisprudencia_tc"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        logger.info(f"🗄️ Inicializando ChromaDB en {self.db_path}")
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        
        # Borrado limpio seguro: Verifica las colecciones activas antes de borrar (Evita caídas en Rust)
        try:
            colecciones_existentes = [c.name for c in self.client.list_collections()]
            if self.collection_name in colecciones_existentes:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Colección '{self.collection_name}' eliminada exitosamente para reinicio.")
            else:
                logger.info(f"Colección '{self.collection_name}' no existía. Se creará desde cero.")
        except Exception as e:
            logger.warning(f"No se pudo verificar o eliminar la colección: {str(e)}. Continuamos de todas formas.")
            
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        logger.info(f"Colección '{self.collection_name}' lista para indexación.")

    def run_pipeline(self, input_dir: str = "data/merged", chunk_size: int = 400, overlap: int = 50, batch_size: int = 500):
        # Asegurar que las rutas relativas funcionen desde la raíz del proyecto
        if not os.path.exists(input_dir):
            logger.warning(f"La ruta {input_dir} no existe localmente. Creándola para evitar crasheos.")
            os.makedirs(input_dir, exist_ok=True)

        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        if not csv_files:
            logger.warning(f"No se encontraron archivos CSV en {input_dir}. Coloca los archivos de tu grupo ahí.")
            return
            
        logger.info(f"Iniciando indexación vectorial desde {len(csv_files)} archivos CSV.")
        
        total_chunks_indexed = 0
        
        for file_path in csv_files:
            logger.info(f"Procesando archivo: {os.path.basename(file_path)}")
            try:
                # Se forzan strings para evitar distorsiones en IDs (e.g. numero_expediente)
                df = pd.read_csv(file_path, dtype=str)
                
                # Asegurar que existan las columnas mínimas que pide el pipeline del grupo
                for col in ["fundamentos", "motivos_demanda"]:
                    if col not in df.columns:
                        df[col] = ""
                
                df["fundamentos"] = df["fundamentos"].fillna("")
                df["motivos_demanda"] = df["motivos_demanda"].fillna("")
                
                batch_ids = []
                batch_documents = []
                batch_metadatas = []
                
                for _, row in df.iterrows():
                    record = row.to_dict()
                    
                    # El chunking y lógica de metadata ocurre centralizadamente en 'processing.py'
                    chunks = build_chunks_for_record(record, chunk_size=chunk_size, overlap=overlap)
                    
                    for chunk in chunks:
                        # Ignorar chunks que no tengan texto real
                        if not chunk["text"].strip():
                            continue
                            
                        batch_ids.append(chunk["chunk_id"])
                        batch_documents.append(chunk["text"])
                        batch_metadatas.append(chunk["metadata"])
                        
                        # Inserción por lotes (Batches) hacia ChromaDB
                        if len(batch_ids) >= batch_size:
                            self.collection.add(
                                ids=batch_ids,
                                documents=batch_documents,
                                metadatas=batch_metadatas
                            )
                            total_chunks_indexed += len(batch_ids)
                            batch_ids, batch_documents, batch_metadatas = [], [], []
                            
                # Guardar el residuo final de chunks para el CSV actual
                if batch_ids:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    total_chunks_indexed += len(batch_ids)
                    
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {str(e)}")
                
        logger.info(f"🎯 ¡Indexación completada! Total de chunks almacenados en ChromaDB: {total_chunks_indexed}")

if __name__ == "__main__":
    # Al ejecutar desde la raíz, apuntamos directo a las carpetas base del repositorio
    pipeline = ChromaIndexingPipeline(db_path="data/chroma_storage")
    pipeline.run_pipeline(input_dir="data/merged")