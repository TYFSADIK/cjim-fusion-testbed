#!/usr/bin/env bash
# Runs the whole system on one machine.
#   ./local/run.sh vllm      start vLLM: the chat model on :8000 and the embedding model on :8001
#   ./local/run.sh up        build and start the Docker stack (Redpanda, PostGIS, OpenSearch, Qdrant, workers, API, Grafana)
#   ./local/run.sh replay    play the exercise's five sources into Redpanda (72 hours in about 6 minutes)
#   ./local/run.sh score     score what the pipeline stored against ground truth
#   ./local/run.sh down      stop the stack (vLLM: Ctrl-C in its windows, or ./local/run.sh vllm-stop)
#
# vLLM: on a Mac, vLLM Metal (brew install vllm-project/vllm-metal/vllm-metal); on Linux with an NVIDIA
# GPU, pip install vllm. MODELS points at the downloaded model folders.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-$HOME/models}"
CHAT="${CHAT_MODEL:-$MODELS/Qwen3.5-2B-4bit}"
EMBED="${EMBED_MODEL:-$MODELS/Qwen3-Embedding-0.6B-8bit}"
compose() { docker compose -f stacks/docker-compose.yml "$@"; }
case "${1:-}" in
  vllm)
    mkdir -p results
    # memory shares sized for an 8 GB Mac; on a GPU server raise them (or drop the flags)
    nohup vllm serve "$CHAT" --served-model-name qwen3.5-2b --host 127.0.0.1 --port 8000 \
      --max-model-len "${CHAT_LEN:-4096}" --gpu-memory-utilization "${CHAT_MEM:-0.50}" > results/vllm-chat.log 2>&1 &
    nohup vllm serve "$EMBED" --served-model-name qwen3-embedding --host 127.0.0.1 --port 8001 --runner pooling \
      --max-model-len 512 --gpu-memory-utilization "${EMBED_MEM:-0.30}" > results/vllm-embed.log 2>&1 &
    until curl -sf 127.0.0.1:8000/v1/models >/dev/null && curl -sf 127.0.0.1:8001/v1/models >/dev/null; do sleep 3; done
    echo "vLLM ready: chat :8000, embeddings :8001"
    ;;
  vllm-stop) pkill -f "vllm serve" || true ;;
  up)
    python3 -m scenario.generate
    docker build -t cjim-py:local --build-arg BASE="${BASE:-python:3.12-slim}" -f stacks/Dockerfile .
    compose up -d
    echo "Dashboard http://127.0.0.1:8090, Grafana http://127.0.0.1:3001"
    ;;
  replay) compose run --rm -e SCENARIO=/app/data/scenario normalizer python -m fusion.pipeline produce --speed "${SPEED:-720}" ;;
  score)  compose run --rm normalizer python -m eval.run live --out /app/results/live ;;
  down)   compose down -v ;;
  *) echo "usage: $0 vllm|vllm-stop|up|replay|score|down"; exit 2 ;;
esac
