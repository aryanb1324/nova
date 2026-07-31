"""Run the user's own training code, locally.

This is the point of NOVA: someone testing their own RL architecture writes
Python here and runs it against a real simulator, instead of clicking a button
that trains a policy we chose for them.

On the security question, plainly: executing a script the user typed on their own
machine is not remote code execution, it is running a script - no different from
`python train.py` in a terminal. The danger is only ever that NOVA gets exposed
to a network, so that is exactly what is gated. Requests from anywhere other than
loopback are refused, and `NOVA_CODE_EXECUTION=off` disables the whole feature.
If you put NOVA behind a reverse proxy, set that variable: the proxy makes every
request look local.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time

USER_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "user_scripts"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}

#: A runaway loop shouldn't hold a core forever if the user closes the tab.
MAX_RUNTIME_S = 3600


def execution_enabled() -> bool:
    return os.environ.get("NOVA_CODE_EXECUTION", "on").strip().lower() not in (
        "0", "off", "false", "no"
    )


def is_local(client_host: str | None) -> bool:
    return client_host in LOOPBACK


#: Headers a reverse proxy adds. Their presence means the peer address cannot be
#: trusted for an authorization decision.
PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host")


def behind_proxy(headers) -> bool:
    """True if this request arrived through something that rewrites the client.

    This exists because the loopback check alone is not sufficient, and the
    reason is sharper than "a proxy looks local". Uvicorn enables
    ProxyHeadersMiddleware by default, trusting 127.0.0.1, and it *overwrites*
    `request.client.host` from `X-Forwarded-For`. Measured against this server:

        curl /api/code/status                       -> enabled: true
        curl -H 'X-Forwarded-For: 8.8.8.8' /...     -> enabled: false

    The address is therefore attacker-controllable. Invert that example and a
    remote request carrying `X-Forwarded-For: 127.0.0.1`, arriving through a
    same-host nginx or Caddy, reads as loopback - and would have been handed
    arbitrary code execution.

    So: any forwarding header at all and we refuse. A genuinely local request
    typed into the editor on this machine carries none of them.
    """
    return any(h in headers for h in PROXY_HEADERS)


class CodeRun:
    """One user script, running as a child process."""

    def __init__(self, run_id: str, path: pathlib.Path, proc: subprocess.Popen):
        self.run_id = run_id
        self.path = path
        self.proc = proc
        self.started = time.time()
        self.exit_code: int | None = None

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "alive": self.alive,
            "elapsed": round(time.time() - self.started, 1),
            "exit_code": self.exit_code,
            "script": self.path.name,
        }


class Runner:
    """Owns at most one running script, and streams its output."""

    def __init__(self) -> None:
        self.current: CodeRun | None = None

    def start(self, script: str, name: str, publish, port: int = 8000) -> CodeRun:
        if self.current and self.current.alive:
            raise RuntimeError(
                "a script is already running - stop it before starting another"
            )

        USER_SCRIPTS.mkdir(parents=True, exist_ok=True)
        run_id = f"code-{time.strftime('%Y%m%d-%H%M%S')}"
        path = USER_SCRIPTS / f"{run_id}.py"
        path.write_text(script)

        env = dict(os.environ)
        # Env workers already saturate the cores; an extra BLAS pool per worker
        # measurably halves throughput.
        env.setdefault("OMP_NUM_THREADS", "1")
        env["NOVA_URL"] = f"http://127.0.0.1:{port}"
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [sys.executable, str(path)],
            cwd=str(USER_SCRIPTS.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        run = CodeRun(run_id, path, proc)
        self.current = run

        threading.Thread(
            target=self._pump, args=(run, publish), daemon=True
        ).start()
        return run

    def _pump(self, run: CodeRun, publish) -> None:
        """Forward the script's output line by line, then report how it ended."""
        try:
            assert run.proc.stdout is not None
            for line in run.proc.stdout:
                publish({
                    "type": "console",
                    "run_id": run.run_id,
                    "line": line.rstrip("\n"),
                })
                if time.time() - run.started > MAX_RUNTIME_S:
                    publish({
                        "type": "console", "run_id": run.run_id,
                        "line": f"[nova] exceeded {MAX_RUNTIME_S}s; stopping",
                    })
                    run.stop()
                    break
        except Exception as exc:  # noqa: BLE001 - never let the pump kill the server
            publish({"type": "console", "run_id": run.run_id,
                     "line": f"[nova] output stream ended: {exc}"})
        finally:
            run.exit_code = run.proc.wait()
            publish({
                "type": "code_exit",
                "run_id": run.run_id,
                "exit_code": run.exit_code,
                "elapsed": round(time.time() - run.started, 1),
            })

    def stop(self) -> bool:
        if self.current and self.current.alive:
            self.current.stop()
            return True
        return False


runner = Runner()
