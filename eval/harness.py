import asyncio
import time
import httpx
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TALENT_API_URL = os.getenv("TALENT_API_URL", "http://localhost:8000/search")

# Ground-truth test dataset: queries paired with expected ideal candidate IDs
BENCHMARK_CASES = [
    {
        "query": "Data Engineer with PySpark, Delta Lake, and AWS Glue pipeline experience",
        "expected_id_snippet": "PySpark, AWS Glue, Delta Lake",
        "category": "Data Engineering"
    },
    {
        "query": "Bilingual ETL Developer fluent in Arabic and English with SQL skills",
        "expected_id_snippet": "Bilingual Data Analyst fluent in Arabic",
        "category": "Bilingual / Analytics"
    },
    {
        "query": "Machine Learning Engineer with FastAPI, Docker, and Qdrant vector databases",
        "expected_id_snippet": "FastAPI microservices, Qdrant vector search",
        "category": "ML / Infrastructure"
    },
    {
        "query": "AWS Solutions Architect Professional with Kubernetes and Terraform experience",
        "expected_id_snippet": "AWS Solution Architect Professional certification",
        "category": "Cloud Architecture"
    }
]

async def execute_query(client: httpx.AsyncClient, query: str, rerank: bool, limit: int = 5):
    start = time.perf_counter()
    response = await client.post(
        TALENT_API_URL,
        json={"query": query, "limit": limit, "rerank": rerank}
    )
    latency_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    results = response.json().get("results", [])
    return results, latency_ms

async def run_benchmark():
    records = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for case in BENCHMARK_CASES:
            query = case["query"]
            expected = case["expected_id_snippet"]
            
            # 1. Run Baseline (Dense-Only, no Cross-Encoder rerank)
            dense_results, dense_lat = await execute_query(client, query, rerank=False)
            dense_hit_rank = None
            for idx, res in enumerate(dense_results):
                if expected in res.get("text", ""):
                    dense_hit_rank = idx + 1
                    break

            # 2. Run Hybrid + Reranker
            rerank_results, rerank_lat = await execute_query(client, query, rerank=True)
            rerank_hit_rank = None
            for idx, res in enumerate(rerank_results):
                if expected in res.get("text", ""):
                    rerank_hit_rank = idx + 1
                    break
            
            records.append({
                "Category": case["category"],
                "Dense Hit Rank": dense_hit_rank if dense_hit_rank else "Miss",
                "Dense MRR": 1.0 / dense_hit_rank if dense_hit_rank else 0.0,
                "Dense Latency (ms)": round(dense_lat, 2),
                "Hybrid+Rerank Hit Rank": rerank_hit_rank if rerank_hit_rank else "Miss",
                "Hybrid+Rerank MRR": 1.0 / rerank_hit_rank if rerank_hit_rank else 0.0,
                "Hybrid Latency (ms)": round(rerank_lat, 2)
            })

    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("RETRIEVAL BENCHMARK REPORT: DENSE vs. HYBRID + CROSS-ENCODER")
    print("=" * 80)
    print(df.to_string(index=False))
    
    avg_dense_mrr = df["Dense MRR"].mean()
    avg_hybrid_mrr = df["Hybrid+Rerank MRR"].mean()
    print("\nSUMMARY METRICS:")
    print(f"  Baseline Dense MRR:        {avg_dense_mrr:.3f}")
    print(f"  Hybrid + Cross-Encoder MRR: {avg_hybrid_mrr:.3f}")
    improvement = ((avg_hybrid_mrr - avg_dense_mrr) / avg_dense_mrr * 100) if avg_dense_mrr > 0 else 100.0
    print(f"  MRR Improvement:           +{improvement:.1f}%")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())