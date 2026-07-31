# NOVA, packaged for a public deployment.
#
# The image defaults to the only mode that is safe to expose: NOVA_CODE_EXECUTION=off,
# so the server serves the app, trains the built-in templates, and scores uploaded
# .onnx policies - and refuses to run user-supplied Python. Read docs/DEPLOY.md
# before changing that default; behind any reverse proxy the loopback check that
# normally protects the run endpoint can be defeated with a forged header.
#
#   docker build -t nova .
#   docker run --rm -p 8000:8000 -e NOVA_EVAL_SALT=... nova

# ---------------------------------------------------------------------------
# Stage 1: build the frontend. Node is only needed here, never at runtime.
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Dependencies first: this layer is cached until the lockfile actually changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# `npm run build` typechecks before it bundles, so a type error fails the image
# build rather than shipping.
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2: the server, plus the bundle built above.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 is torch's OpenMP runtime and libstdc++6 is onnxruntime's; neither is
# in the slim base, and both fail at import time rather than at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
# torch from PyPI drags in the CUDA runtime on linux/amd64 - roughly 2 GB of
# wheels for hardware no deployment of this has. Installing the CPU build first
# satisfies the pin in requirements.txt, so the second command leaves it alone.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# All mutable state in one place, so a single volume mounted at /data persists
# uploaded policies, trained runs and uploaded robots together. Without a volume
# these live in the container's writable layer and vanish on restart, which is a
# reasonable default for a demo.
RUN mkdir -p /data \
    && ln -s /data/runs /app/backend/runs \
    && ln -s /data/policies /app/backend/policies \
    && ln -s /data/user_robots /app/backend/user_robots \
    && useradd --create-home --uid 10001 nova \
    && chown -R nova:nova /app /data

ENV NOVA_CODE_EXECUTION=off \
    NOVA_FRONTEND_DIST=/app/frontend/dist \
    OMP_NUM_THREADS=1 \
    PORT=8000

USER nova
WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; \
urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/api/health\").read()"

# One process. Training and policy scoring already fan out to child processes,
# and the websocket run state lives in this process's memory - a second uvicorn
# worker would serve requests that know nothing about a run in the first.
CMD ["sh", "-c", "mkdir -p /data/runs /data/policies /data/user_robots && exec python scripts/serve.py --host 0.0.0.0 --port ${PORT:-8000}"]
