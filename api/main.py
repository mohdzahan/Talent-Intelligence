import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

app = FastAPI(title="Enterprise Talent Intelligence API (Hybrid + Reranker)")
executor = ThreadPoolExecutor(max_workers=2)

# Models and Clients
dense_model = SentenceTransformer("BAAI/bge-m3")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder("BAAI/bge-reranker-base")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=6333)
COLLECTION_NAME = "talent_intelligence_cvs"

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Recruiter search query")
    limit: int = Field(default=5, ge=1, le=50, description="Number of CVs to retrieve")
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking")

class SearchResult(BaseModel):
    score: float
    text: str
    document_type: str
    skills: List[str]

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    loop = asyncio.get_running_loop()

    # 1. Generate embeddings in threadpool
    def generate_embeddings():
        d_vec = dense_model.encode(request.query).tolist()
        s_vec = list(sparse_model.embed([request.query]))[0]
        return d_vec, s_vec

    dense_vector, sparse_vector = await loop.run_in_executor(executor, generate_embeddings)

    # 2. Stage 1: Hybrid Retrieval with Qdrant RRF (pull 2x the requested limit)
    fetch_limit = request.limit * 2 if request.rerank else request.limit
    search_response = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=fetch_limit
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist()
                ),
                using="sparse",
                limit=fetch_limit
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_limit,
        with_payload=True
    )

    candidates = [
        {
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "document_type": hit.payload.get("document_type", "unknown"),
            "skills": hit.payload.get("skills", [])
        }
        for hit in search_response.points
    ]

    # 3. Stage 2: Cross-Encoder Reranking
    if request.rerank and candidates:
        def compute_rerank():
            pairs = [[request.query, c["text"]] for c in candidates]
            rerank_scores = reranker.predict(pairs)
            return rerank_scores

        scores = await loop.run_in_executor(executor, compute_rerank)
        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        # Sort descending by cross-encoder score and slice to requested limit
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:request.limit]

    formatted_results = [SearchResult(**c) for c in candidates]
    return SearchResponse(results=formatted_results)