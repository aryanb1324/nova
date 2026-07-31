# Deploying NOVA

NOVA is local-first by default, and most of what makes it pleasant locally —
writing Python in the browser and having it run — is exactly what you must turn
off before putting it on the internet. This document covers what a public NOVA
is, why the one environment variable is not optional, and the commands to get
there.

**Nothing here has been run for you.** Every command below is yours to run.

---

## 1. What a public NOVA is

Two modes, and they are not the same product.

| | Local (default) | Public (artifact-only) |
|---|---|---|
| Write and run Python in the browser | yes | **no** |
| Design a robot, train the built-in PPO | yes | yes |
| Upload a trained `.onnx` policy and have it scored | yes | yes |
| Upload a custom MJCF robot | yes | yes |
| `NOVA_CODE_EXECUTION` | `on` | **`off`** |

The public mode is *artifact-only*: users bring a policy they trained elsewhere,
as an ONNX file. ONNX is a graph over a fixed operator set — data, not code. It
is loaded by `onnxruntime` inside a separate, resource-limited process
(`nova/policies/evaluate.py::evaluate_isolated`, 180 s wall clock and a 3 GB
address-space cap), so a hostile or merely broken graph takes down a child
process rather than the server.

That is the whole security posture. It works because nothing a user sends is
ever executed as a program.

---

## 2. `NOVA_CODE_EXECUTION=off` is mandatory. Here is the mechanism.

`/api/code/run` executes a Python script the user typed. Locally that is not
remote code execution — it is running a script, the same as `python train.py` in
a terminal. It is gated two ways in `nova/api/code_routes.py::_guard`:

1. `NOVA_CODE_EXECUTION=off` disables the endpoint outright.
2. Otherwise, the request must come from loopback.

**The second gate cannot be relied on behind a proxy, and the reason is sharper
than "the proxy looks local".** Uvicorn enables `ProxyHeadersMiddleware` by
default (`proxy_headers=True`), trusting `X-Forwarded-For` from any peer listed
in `forwarded_allow_ips`, which defaults to `127.0.0.1`. When the header is
trusted, uvicorn **overwrites** `request.client.host` with the value from it —
the value NOVA's loopback check then reads.

Verified on this machine, with execution left enabled:

```
$ curl -s http://127.0.0.1:8124/api/code/status
{"enabled":true, ...}

$ curl -s -H "X-Forwarded-For: 8.8.8.8" http://127.0.0.1:8124/api/code/status
{"enabled":false, ...}
```

A header sent by the client changed the address the gate inspects. Now invert
it. If you run NOVA behind nginx, Caddy, or any sidecar proxy **on the same
host**, the peer is `127.0.0.1`, so the header is trusted — and a request
carrying `X-Forwarded-For: 127.0.0.1` is seen by the gate as loopback. The
loopback check passes for an anonymous request from the internet, and
`/api/code/run` executes attacker-supplied Python as the server user.

So:

- **Always set `NOVA_CODE_EXECUTION=off`.** It is the only gate that does not
  depend on how requests reach the process. The `Dockerfile` in this repo sets
  it as an image default for that reason.
- **Never set `FORWARDED_ALLOW_IPS=*`.** It is common cargo-cult configuration
  on PaaS hosts and it extends the forgeable-header window to every peer.
- Fly.io's proxy connects over the private network, not `127.0.0.1`, so uvicorn
  does not trust the header there and `client.host` stays non-loopback. That is
  a second line of defence, not the first. Set the variable anyway.

Confirm after deploying (section 7) that `/api/code/run` answers `403`.

---

## 3. `NOVA_EVAL_SALT`

Every policy is scored on two seed sets: **public** seeds, published in
`/api/catalog` so anyone can reproduce their own number locally, and **held-out**
seeds derived by HMAC from `NOVA_EVAL_SALT`. The gap between the two is the
interesting result.

Unset, the salt falls back to `"nova-public-default"`, which is in the source.
Held-out seeds are then reproducible by anyone, and a submission can be tuned
against the exact episodes it will be graded on — which makes the held-out score
meaningless and the gap a lie.

Generate one and keep it out of the repo:

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it as a secret, never as a build argument or a line in `fly.toml`. Changing
it later invalidates comparisons between scores computed before and after, so
pick it once.

---

## 4. What users can still send, and what bounds it

Turning off code execution does not leave the surface empty. Everything below is
reachable by an anonymous user; there is **no authentication anywhere in NOVA**.

| Endpoint | Input | What bounds it |
|---|---|---|
| `POST /api/policies` | `.onnx` bytes | 25 MB read cap; validated and scored in a spawned process with a 180 s timeout and 3 GB memory cap |
| `POST /api/robots`, `/api/robots/validate` | MJCF XML | size cap plus the contract in `nova/robots/contract.py`, which rejects `<include>` and any `file`/`meshdir`/`texturedir`/`assetdir` attribute — the file-reference vectors. Compiled **in the server process**, unlike policies |
| `WS /ws/train` | training config | `MAX_STEPS = 2_000_000`, `n_envs` clamped to 32 |
| `POST /api/code/*` | Python | refused entirely when `NOVA_CODE_EXECUTION=off` |
| `DELETE /api/runs/{id}`, `/api/policies/{id}`, `/api/robots/{key}` | id | path traversal blocked by `resolve()` checks; **anyone can delete anything** |

Two consequences worth stating plainly:

- **There are no accounts, no quotas, and no rate limiting.** Any visitor can
  start a training run, upload policies until the disk fills, or delete another
  visitor's uploads. If NOVA is going anywhere other than a demo link, put it
  behind your host's access control (Cloudflare Access, Tailscale, HTTP basic
  auth at the proxy) and treat that as the authentication layer.
- **Trained artifacts are public.** `GET /api/policies/{id}/policy.onnx` returns
  any uploaded policy to anyone who knows the id, and ids are listable.

---

## 5. Resources

Training is CPU-bound and bursty, which is the shape that decides your host.

- A run spawns a worker process, which spawns `n_envs` more for the vectorised
  environments. It saturates every core it is given, for as long as it runs, and
  then drops to nothing.
- Reference throughput on the development machine (10-core M5) is
  12,000–16,000 environment steps/second; the built-in PPO reaches ~96% on the
  reach task in about 30 seconds. **Expect several times slower on a shared-vCPU
  cloud instance** — those numbers are not a deployment forecast.
- The step ceiling is 2,000,000. One request asking for it can occupy the
  machine for well over an hour, and nothing preempts it.
- Policy evaluation is the other burst: up to 180 seconds and up to 3 GB in a
  child process, per upload.

Sizing that follows from the above:

- **Memory: 4 GB minimum**, because a single evaluation is allowed 3 GB and the
  server plus a training worker sit alongside it.
- **CPU: 2 shared vCPUs is enough to demo; 4+ dedicated if runs should feel like
  the README.** Concurrency is what hurts, not any single run — there is no
  queue, so three simultaneous trainings each try to take every core.
- **Disk:** runs and uploaded policies are a few MB each and are never garbage
  collected. 1 GB goes a long way; it is still unbounded growth.
- **Image size:** torch (CPU build), MuJoCo and onnxruntime dominate. Expect
  something in the low gigabytes — worth checking before you pick a plan, since
  it affects cold starts.

There is no GPU path and nothing here wants one: MuJoCo steps these environments
on CPU, and the policies are small MLPs.

---

## 6. Build and run it locally first

The image is self-contained: it builds the frontend, installs the backend, and
serves both from one process on one port.

```sh
docker build -t nova .
docker run --rm -p 8000:8000 \
  -e NOVA_EVAL_SALT="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  nova
```

Then open <http://127.0.0.1:8000>. `NOVA_CODE_EXECUTION=off` is already the
image default; nothing you pass needs to repeat it.

To keep uploaded policies and trained runs across restarts, mount a volume — the
image symlinks `runs/`, `policies/` and `user_robots/` into `/data`:

```sh
docker run --rm -p 8000:8000 -v nova-data:/data -e NOVA_EVAL_SALT=... nova
```

### How the frontend gets served

`backend/nova/api/server.py` mounts `frontend/dist` at `/` **only if the build
exists**, as the last route in the table, so `/api/*` and `/ws/*` are matched
first and development is untouched — with no `dist/` there is no mount at all,
and Vite keeps proxying to the backend exactly as before. Override the location
with `NOVA_FRONTEND_DIST` if your layout differs.

---

## 7. Fly.io

Fly fits NOVA: real VMs (so `multiprocessing` and spawned workers behave),
native websockets for `/ws/train`, and scale-to-zero so an idle demo costs
nothing.

**You run these.** They create resources and, from `fly deploy` onward, cost
money.

```sh
# 1. Once, if you have not: install and sign in.
brew install flyctl
fly auth login

# 2. Generate the app config without deploying anything.
fly launch --no-deploy --name nova-demo --region iad

# 3. Edit fly.toml (see below), then set the secret. Never put it in fly.toml.
fly secrets set NOVA_EVAL_SALT="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# 4. Optional, for persistence. Size and region must match the app.
fly volumes create nova_data --size 1 --region iad

# 5. Deploy.
fly deploy
```

`fly launch` writes a `fly.toml`; the parts that matter:

```toml
app = "nova-demo"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  # Belt and braces: the image already defaults to this. Keep it here so the
  # setting is visible in the config someone reads before changing something.
  NOVA_CODE_EXECUTION = "off"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "suspend"
  auto_start_machines = true
  # Zero, so an idle demo scales to nothing. One long-running websocket keeps a
  # machine up on its own, which is what you want mid-training.
  min_machines_running = 0
  # No shared state between machines: a run lives in the process that started
  # it, and its websocket must stay on that machine.
  max_machines_running = 1

[[vm]]
  memory = "4gb"
  cpu_kind = "shared"
  cpus = 2

# Only if you created the volume in step 4.
[[mounts]]
  source = "nova_data"
  destination = "/data"
```

Two Fly-specific caveats:

- **Volume ownership.** The container runs as uid `10001`; a freshly created Fly
  volume is owned by root, so the first write fails. Fix it once, after the
  first deploy: `fly ssh console -C "chown -R 10001:10001 /data"`. Skip the
  volume entirely if ephemeral state is acceptable — for a demo it usually is.
- **Suspending a machine kills training.** `auto_stop_machines` acts on idle
  HTTP; an open websocket counts as activity, so a run in progress holds the
  machine. A run whose browser tab was closed does not, and will be lost.

### Other hosts

Anything that runs a container with a real process tree works: Railway, Render,
a plain VM with `docker run`. Two requirements are not negotiable — the platform
must allow `fork`/`spawn` of child processes (rules out most function/edge
runtimes), and it must proxy websockets. Set `NOVA_CODE_EXECUTION=off` and
`NOVA_EVAL_SALT` the same way, and read section 2 again before terminating TLS
with a same-host proxy.

---

## 8. Verify the deployment

Replace `$URL` with your deployed origin. The third check is the important one.

```sh
URL=https://nova-demo.fly.dev

# The app is served.
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" $URL/

# The API is alive behind it.
curl -s $URL/api/health
curl -s -o /dev/null -w "catalog %{http_code}\n" $URL/api/catalog

# Code execution is refused. Expect 403 and the "disabled" message.
curl -s -X POST $URL/api/code/run \
  -H 'content-type: application/json' \
  -d '{"script":"print(1)","name":"probe"}'

# And refused even when the request claims to be local.
curl -s -X POST $URL/api/code/run \
  -H 'content-type: application/json' \
  -H 'X-Forwarded-For: 127.0.0.1' \
  -d '{"script":"print(1)","name":"probe"}'

# The reason is reported too: expect "NOVA_CODE_EXECUTION=off".
curl -s $URL/api/code/status
```

The two `code/run` probes must both return:

```json
{"detail":"code execution is disabled (NOVA_CODE_EXECUTION=off). Upload a trained .onnx policy instead."}
```

If either returns anything else — a `409`, a `run_id`, a `403` mentioning the
*machine* rather than the *variable* — the variable did not reach the process.
Stop and fix that before sharing the link.
