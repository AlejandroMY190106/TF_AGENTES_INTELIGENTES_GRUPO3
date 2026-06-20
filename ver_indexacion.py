import chromadb

# 1. Nos conectamos a la base de datos local que acabas de crear
CHROMA_DIR = "./data/chroma_storage"
client = chromadb.PersistentClient(path=CHROMA_DIR)

# 2. Apuntamos a la colección de jurisprudencia
try:
    collection = client.get_collection(name="jurisprudencia_tc")
    
    # Ver cuántos documentos totales hay registrados
    total_items = collection.count()
    print(f"\n========================================")
    print(f"Conexión exitosa a ChromaDB")
    print(f"Cantidad total de expedientes indexados: {total_items}")
    print(f"========================================\n")
    
    # 3. Traer una muestra de los primeros 2 registros inyectados
    # Pedimos explícitamente que nos muestre documentos, metadatas y los embeddings vectoriales
    resultado = collection.get(limit=2, include=["documents", "metadatas", "embeddings"])
    
    for i in range(len(resultado["ids"])):
        print(f"--- REGISTRO MUESTRA #{i+1} ---")
        print(f"ID / Nro Expediente: {resultado['ids'][i]}")
        print(f"Materia: {resultado['metadatas'][i].get('materia')}")
        print(f"Sentido de la sentencia: {resultado['metadatas'][i].get('sentencia_sentido')}")
        print(f"Fragmento del texto (Primeros 150 caracteres): {resultado['documents'][i][:150]}...")
        
        # Mostramos los primeros 5 valores del vector matemático generado por Legal-BERT
        vector = resultado['embeddings'][i]
        print(f"Vector (Legal-BERT) - Dimensión {len(vector)}: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, {vector[3]:.4f}, ...]")
        print("-" * 40 + "\n")

except Exception as e:
    print(f"Error al conectar o leer la colección: {e}")