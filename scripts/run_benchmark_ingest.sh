#!/usr/bin/env bash
set -e

# 用法: ./scripts/run_benchmark_ingest.sh fiqa mini
#       ./scripts/run_benchmark_ingest.sh medical_retrieval mini

DATASET="${1:-fiqa}"
VARIANT="${2:-mini}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMBEDDING_PORT="${RAG_EMBEDDING_SERVICE_PORT:-9090}"
EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-mlx-community/Qwen3-Embedding-4B-4bit-DWQ}"
EMBEDDING_URL="http://127.0.0.1:${EMBEDDING_PORT}"
COLLECTION_PREFIX="${RAG_VECTOR_COLLECTION_PREFIX:-${DATASET}_${VARIANT}_qwen4b_v1}"

# 自动拉起 embedding 服务
if ! curl -sSf -m 3 "${EMBEDDING_URL}/health" >/dev/null 2>&1; then
    echo "[*] embedding service not running, starting..."
    screen -dmS "rag_embedding_${EMBEDDING_PORT}" zsh -lc "
        cd '${REPO_ROOT}' &&
        uv run rag embedding-service \
            --model '${EMBEDDING_MODEL}' \
            --port '${EMBEDDING_PORT}'
    "
    echo "[*] waiting for embedding service to be ready..."
    for i in $(seq 1 60); do
        if curl -sSf -m 3 "${EMBEDDING_URL}/health" >/dev/null 2>&1; then
            echo "[*] embedding service ready"
            break
        fi
        sleep 2
    done
fi

curl -sSf -m 3 "${EMBEDDING_URL}/health" >/dev/null

export RAG_EMBEDDING_SERVICE_URL="${EMBEDDING_URL}"

uv run rag benchmark-ingest \
    --dataset "$DATASET" \
    --variant "$VARIANT" \
    --vector-backend milvus \
    --vector-collection-prefix "${COLLECTION_PREFIX}" \
    --summary-provider none \
    --skip-graph-extraction \
    --batch-size 32
