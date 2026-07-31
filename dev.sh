#!/usr/bin/env bash
# Start the API server and the Vite dev server together, and stop both on Ctrl-C.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

if [ ! -x .venv/bin/python ]; then
  echo "no .venv found — create one first:" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt" >&2
  exit 1
fi
if [ ! -d frontend/node_modules ]; then
  echo "installing frontend dependencies..."
  (cd frontend && npm install)
fi

# One BLAS thread per worker: the env subprocesses already saturate the cores,
# and letting torch add its own pool measurably halves throughput.
export OMP_NUM_THREADS=1

pids=()
cleanup() { trap - INT TERM EXIT; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

(cd backend && "$root/.venv/bin/python" scripts/serve.py --port 8000) &
pids+=($!)

(cd frontend && npm run dev) &
pids+=($!)

echo
echo "  backend  http://127.0.0.1:8000"
echo "  frontend http://localhost:5173"
echo
wait
