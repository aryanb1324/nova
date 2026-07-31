#!/usr/bin/env python
"""Start the API server.

    python scripts/serve.py [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    # Env workers are separate processes; extra BLAS threads inside each one
    # only fight over the same cores. Measured 2x throughput difference.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import uvicorn
    uvicorn.run(
        "nova.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
