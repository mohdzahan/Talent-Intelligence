from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI(title="Enterprise Talent Intelligence API")
executor = ThreadPoolExecutor(max_workers=2)

# 1. Initialize globals: the model (CPU-heavy) and the async database client (Network I/O)
model = SentenceTransformer("BAAI/bge-m3")

# ...
# Allow Docker to override the Qdrant host
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
qdrant = AsyncQdrantClient(QDRANT_HOST, port=6333)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search string (e.g., a job description)")
    limit: int = Field(default=5, ge=1, le=50, description="How many CVs to retrieve")

class SearchResult(BaseModel):
    score: float
    text: str
    document_type: str

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    loop = asyncio.get_running_loop()
    
    # 2. Offload the blocking CPU math to the threadpool
    def blocking_embed():
        return model.encode(request.query).tolist()
        
    query_vector = await loop.run_in_executor(executor, blocking_embed)
    
    # 3. Await the async network call to Qdrant using the modern API
    search_response = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=request.limit,
        with_payload=True
    )

    # 4. Extract the payload data we injected during indexing
    formatted_results = [
        SearchResult(
            score=hit.score,
            text=hit.payload.get("text", ""),
            document_type=hit.payload.get("document_type", "unknown")
        )
        for hit in search_response.points
    ]
    
    return SearchResponse(results=formatted_results)