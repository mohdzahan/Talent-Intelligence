import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, PointStruct
from sentence_transformers import SentenceTransformer

def initialize_and_index():
    # 1. Connect to local Qdrant
    client = QdrantClient("localhost", port=6333)
    collection_name = "talent_intelligence_cvs"

    # 2. Recreate collection to ensure a clean slate during testing
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # 3. Create the collection with explicit HNSW parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # BGE-M3 requires 1024 dimensions
            distance=Distance.COSINE
        ),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100)
    )

    # Change the model instantiation
    model = SentenceTransformer("BAAI/bge-m3")
    
    cv_texts = [
        "Data Engineer with 13 months of experience building pipelines in PySpark and Delta Lake.",
        "Frontend UI developer specializing in React, Tailwind CSS, and user accessibility.",
        "AI Engineer experienced with FastAPI, Qdrant vector databases, and quantised LLMs."
    ]

    print("Generating embeddings...")
    vectors = model.encode(cv_texts).tolist()

    # 5. Package the vectors and metadata (payload) and upload to Qdrant
    points = [
        PointStruct(
            id=str(uuid.uuid4()), # Qdrant requires a UUID or integer for the ID
            vector=vectors[i],
            payload={"text": cv_texts[i], "document_type": "CV"}
        )
        for i in range(len(cv_texts))
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Successfully indexed {len(cv_texts)} documents into Qdrant.")

if __name__ == "__main__":
    initialize_and_index()