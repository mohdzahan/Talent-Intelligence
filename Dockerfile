# Stage 1: The Builder
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download and bake the model weights into the image to prevent runtime downloads
ENV HF_HOME=/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Stage 2: The Production Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local

# Bring the downloaded model over and configure the environment path
COPY --from=builder /models /models
ENV HF_HOME=/models

COPY api/ ./api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]