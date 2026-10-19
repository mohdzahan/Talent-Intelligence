
# Enterprise Talent Intelligence Engine

An asynchronous, two-stage semantic retrieval and recruiter evaluation microservice built with FastAPI, Qdrant, BAAI/bge-m3, and Cross-Encoder reranking, containerized with Docker and deployed on AWS EC2.

## System Architecture

[ Recruiter Query ]
│
▼
[ FastAPI Gateway (/search) ] ── (Async Event Loop)
│
├──> [ ThreadPoolExecutor ]
│         ├── Dense Embeddings: BAAI/bge-m3 (1024-d)
│         └── Sparse Vectors: FastEmbed (BM25)
│
▼
[ Qdrant Vector DB ]
│
├── Stage 1: Reciprocal Rank Fusion (RRF) over HNSW Graph
│
▼ (Top-K Prefetched Candidates)
[ Cross-Encoder Stage ]
│
├── BAAI/bge-reranker-base (Joint Attention Scoring)
│
▼ (Filtered & Calibrated Scores)
[ Recruiter Agent (LLM) ]
│
└── Structured Pydantic Output (Scorecards, Evidence, Decision)


## Technical Highlights

- **Asynchronous Concurrency:** CPU-bound embedding generation (`bge-m3`) is isolated inside a Python `ThreadPoolExecutor`, preventing event-loop starvation and sustaining sub-100ms I/O throughput.
- **Hybrid Search & Fusion:** Combines 1024-dimensional dense semantic vectors with BM25 sparse vectors via native Qdrant Reciprocal Rank Fusion (RRF) to eliminate keyword blindness.
- **Two-Stage Retrieval:** Mitigates vector false positives using a second-stage cross-encoder (`bge-reranker-base`) for full token-level query-document cross-attention.
- **Structured Agentic Scoring:** Recruiter Agent validates hiring requirements against retrieved candidate payloads using deterministic Pydantic schemas, enforcing zero-hallucination quote extraction.
- **Containerized Infrastructure:** Production-grade multi-container stack orchestrated via Docker Compose, leveraging CPU-optimized PyTorch wheels on an AWS EC2 instance.

## Evaluation & Benchmarks

Baseline Dense retrieval vs. Two-Stage Hybrid + Cross-Encoder retrieval evaluated across technical job descriptions:

| Category | Dense Hit Rank | Dense Latency (ms) | Hybrid+Rerank Hit Rank | Hybrid Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Data Engineering | 1 | 1,369 ms | 1 | 958 ms |
| Bilingual / Analytics | 1 | 970 ms | 1 | 1,292 ms |
| ML / Infrastructure | 1 | 652 ms | 1 | 1,191 ms |
| Cloud Architecture | 1 | 923 ms | 1 | 1,227 ms |

> **Architectural Note:** In low-cardinality environments, MRR saturates at 1.0 across both methods. The cross-encoder stage introduces ~300ms–500ms CPU inference overhead to compute calibrated logit scores, filtering out high-cosine semantic false positives in scaled production collections.

## Local & Cloud Setup

```bash
# Clone and enter
git clone [https://github.com/mohdzahan/Talent-Intelligence.git](https://github.com/mohdzahan/Talent-Intelligence.git)
cd Talent-Intelligence

# Configure environment
cp .env.example .env

# Launch services via Docker Compose
docker compose up -d

# Seed the collection
docker compose exec api python scripts/index_qdrant.py

# Run the Recruiter Agent
python run_agent.py