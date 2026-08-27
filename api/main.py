from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor



app = FastAPI(title="Talent Intellignence Platform")

executor = ThreadPoolExecutor(max_workers=4)


class EmbedRequest(BaseModel):
    texts: List[str] = Field(...,min_length=1, description="List of strings to embed")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class SimilarityRequest(BaseModel):
    text1: str = Field(...,min_length = 1)
    text2: str = Field(...,min_length=1)

class SimilarityResponse(BaseModel):
    score: float


model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request:EmbedRequest):
    loop = asyncio.get_running_loop()

    def blocking_embed():
        return model.encode(request.texts).tolist()
    
    embeddings = await loop.run_in_executor(executor, blocking_embed)
    return EmbedResponse(embeddings = embeddings)    

@app.post("/similarity",response_model=SimilarityResponse)

async def compute_similarity(request: SimilarityRequest):
    loop = asyncio.get_running_loop()
    
    def blocking_encode_and_score():
        vec1 = model.encode([request.text1])[0]
        vec2 = model.encode([request.text2])[0]

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)


        return 0.0 if norm1==0 or norm2==0 else dot_product/(norm1*norm2)

    score = await loop.run_in_executor(executor, blocking_encode_and_score)
    return SimilarityResponse(score=float(score))