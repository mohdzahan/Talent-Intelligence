import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

COLLECTION_NAME = "talent_intelligence_cvs"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")

client = QdrantClient(host=QDRANT_HOST, port=6333)

# 1. Initialize models
dense_model = SentenceTransformer("BAAI/bge-m3")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# 2. Expanded candidate dataset with specific hard keywords
DOCUMENTS = [
    {
        "id": 1,
        "text": "Data Engineer with 14 months of experience in PySpark, AWS Glue, Delta Lake, and Databricks.",
        "document_type": "CV",
        "skills": ["PySpark", "AWS Glue", "Delta Lake", "Databricks"]
    },
    {
        "id": 2,
        "text": "Senior ML Engineer specializing in FastAPI microservices, Qdrant vector search, Docker, and quantised LLM inference.",
        "document_type": "CV",
        "skills": ["FastAPI", "Qdrant", "Docker", "LLM", "Python"]
    },
    {
        "id": 3,
        "text": "Bilingual Data Analyst fluent in Arabic and English, skilled in SQL, Power BI, and automated ETL workflows.",
        "document_type": "CV",
        "skills": ["SQL", "Power BI", "ETL", "Arabic", "English"]
    },
    {
        "id": 4,
        "text": "Cloud Solutions Architect with AWS Solution Architect Professional certification, Terraform, and Kubernetes expertise.",
        "document_type": "CV",
        "skills": ["AWS", "Terraform", "Kubernetes", "Architecture"]
    }
]

def recreate_hybrid_collection():
    # Delete old single-vector collection if it exists
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # Configure dual named vectors: one Dense (Cosine), one Sparse
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        }
    )
    print(f"Collection '{COLLECTION_NAME}' created with Hybrid (Dense + Sparse) configuration.")

def index_candidates():
    texts = [doc["text"] for doc in DOCUMENTS]
    
    print("Computing dense embeddings with BGE-M3...")
    dense_vectors = dense_model.encode(texts).tolist()
    
    print("Computing sparse embeddings with BM25...")
    sparse_vectors = list(sparse_model.embed(texts))

    points = []
    for doc, d_vec, s_vec in zip(DOCUMENTS, dense_vectors, sparse_vectors):
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": d_vec,
                    "sparse": models.SparseVector(
                        indices=s_vec.indices.tolist(),
                        values=s_vec.values.tolist()
                    )
                },
                payload={
                    "text": doc["text"],
                    "document_type": doc["document_type"],
                    "skills": doc["skills"]
                }
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully upserted {len(points)} candidate profiles with Hybrid representations.")

if __name__ == "__main__":
    recreate_hybrid_collection()
    index_candidates()