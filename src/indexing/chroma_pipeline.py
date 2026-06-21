import os
import pandas as pd
import chromadb
# Importamos el protocolo base de ChromaDB para funciones de embeddings
from chromadb.api.types import EmbeddingFunction
from typing import List, Dict, Any

def load_local_data(file_path: str) -> pd.DataFrame:
    """
    Lee el archivo intermedio. Soporta formatos CSV, Parquet o SQLite.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.db') or file_path.endswith('.sqlite'):
        import sqlite3
        conn = sqlite3.connect(file_path)
        df = pd.read_sql_query("SELECT * FROM expedientes", conn)
        conn.close()
        return df
    else:
        raise ValueError("Formato de archivo no soportado. Debe ser CSV, Parquet o SQLite.")

def prepare_chroma_payload(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """
    Transforma el DataFrame de Pandas al formato estricto que exige ChromaDB.
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        expediente_id = str(row.get('numero_expediente', f"EXP-{idx}"))
        texto_base = row.get('fundamentos') or row.get('attachment.content', '')
        
        if not str(texto_base).strip() or str(texto_base) == 'nan':
            continue  # Saltamos registros vacíos

        metadata = {
            "numero_expediente": expediente_id,
            "sentencia_sala": str(row.get('sentencia_sala', 'No especificado')),
            "sentencia_sentido": str(row.get('sentencia_sentido', 'No especificado')),
            "materia": str(row.get('materia', 'General'))
        }

        ids.append(expediente_id)
        documents.append(str(texto_base))
        metadatas.append(metadata)

    return {"ids": ids, "documents": documents, "metadatas": metadatas}

def index_to_chromadb(payload: Dict[str, List[Any]], db_path: str = "./data/chroma_storage"):
    """
    Se conecta a la instancia de ChromaDB e inyecta los bloques de texto
    utilizando el modelo oficial de Legal-BERT adaptado formalmente para Chroma.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("\n--- Cargando el modelo oficial de LEGAL-BERT de forma directa ---")
    
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Modificamos la clase para heredar de EmbeddingFunction y resolver el AttributeError
    class LegalBertEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: list) -> list:
            embeddings = []
            for text in input:
                inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Mean Pooling matemático nativo
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                text_embedding = (sum_embeddings / sum_mask).squeeze(0).tolist()
                embeddings.append(text_embedding)
            return embeddings

    embedding_fn = LegalBertEmbeddingFunction()
    
    # Inicialización de la base de datos
    client = chromadb.PersistentClient(path=db_path)
    
    # Si la colección ya existía con otra configuración, la recuperamos limpiamente
    collection = client.get_or_create_collection(
        name="jurisprudencia_tc",
        embedding_function=embedding_fn
    )
    
    if not payload["ids"]:
        print("⚠️ No se encontraron registros válidos con texto para indexar.")
        return

    print(f"Iniciando la indexación de {len(payload['ids'])} expedientes en ChromaDB...")
    collection.add(
        ids=payload['ids'],
        documents=payload['documents'],
        metadatas=payload['metadatas']
    )
    print("¡Indexación completada con éxito en la base de datos vectorial!")

print("el script se está ejecutando...")

if __name__ == "__main__":
    DATA_PATH = "data/muestra_expedientes.csv" 
    CHROMA_DIR = "data/chroma_storage"
    
    if os.path.exists(DATA_PATH):
        dataframe = load_local_data(DATA_PATH)
        chroma_data = prepare_chroma_payload(dataframe)
        index_to_chromadb(chroma_data, db_path=CHROMA_DIR)
    else:
        print(f"Archivo de datos no encontrado en {DATA_PATH}.")