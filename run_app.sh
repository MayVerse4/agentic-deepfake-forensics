#!/usr/bin/env bash
# Starts the full app:
#   Ollama (Docker ollama-gpu, Gemma 2 9B) -> http://localhost:11435
#   analysis backend (backend/forensics_api.py) -> http://localhost:8001
#   frontend (Next.js)                     -> http://localhost:3333
# Setup once:  python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt \
#                --extra-index-url https://download.pytorch.org/whl/cu130
#              (cd frontend && npm ci)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

docker start ollama-gpu >/dev/null && echo "ollama-gpu container started" \
  || echo "WARNING: could not start ollama-gpu -- explanations will fall back to rule-based text"

cd "$HERE"
.venv/bin/python -m uvicorn backend.forensics_api:app --host 0.0.0.0 --port 8001 &
BACK=$!

cd "$HERE/frontend"
npm run dev &
FRONT=$!

trap 'kill $BACK $FRONT 2>/dev/null' EXIT INT TERM

# Report once the models are loaded, so the first upload isn't the one that pays the warm-up.
(
  for i in $(seq 1 90); do
    H=$(curl -s -m 2 http://localhost:8001/health) && echo "$H" | grep -q '"engine_loaded":true' && break
    sleep 2
  done
  echo "READY -> engine: $(echo "$H" | grep -o '"engine_loaded":[a-z]*')  llm: $(echo "$H" | grep -o '"online":[a-z]*' | head -1)"
) &
echo "Open http://localhost:3333  (Ctrl+C stops backend + frontend)"
wait
