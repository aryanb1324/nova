import type {
  Catalog,
  CodeStatus,
  PolicyResult,
  PreviewResponse,
  RobotReport,
  RunDetail,
  RunSummary,
  SavedScript,
  Starter,
  TrainEvent,
  TrainRequest,
  UploadedRobot,
} from "./types";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return (await res.json()) as T;
}

export const getCatalog = () => json<Catalog>("/api/catalog");

export const getPreview = (
  template: string,
  params: Record<string, number>,
  signal?: AbortSignal,
) =>
  json<PreviewResponse>("/api/preview", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ template, params }),
    signal,
  });

export const listRuns = () =>
  json<{ runs: RunSummary[] }>("/api/runs").then((r) => r.runs);

export const getRun = (id: string) => json<RunDetail>(`/api/runs/${id}`);

export const deleteRun = (id: string) =>
  json<{ deleted: string }>(`/api/runs/${id}`, { method: "DELETE" });

// ---- policies ------------------------------------------------------------

export const listPolicies = () =>
  json<{ policies: PolicyResult[] }>("/api/policies").then((r) => r.policies);

export const getPolicy = (id: string) => json<PolicyResult>(`/api/policies/${id}`);

export const deletePolicy = (id: string) =>
  json<{ deleted: string }>(`/api/policies/${id}`, { method: "DELETE" });

/** URL that exports a locally trained run as a portable .onnx. */
export const runPolicyUrl = (runId: string) => `/api/runs/${runId}/policy.onnx`;

/**
 * Upload an .onnx for validation and scoring. The body is the raw file: the
 * manifest lives inside it, so there is nothing else to send.
 */
export async function uploadPolicy(file: File | Blob): Promise<PolicyResult> {
  const res = await fetch("/api/policies", {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: file,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return (await res.json()) as PolicyResult;
}

// ---- your own robot ------------------------------------------------------

/** Dry run — check a model without storing it. Safe to call while typing. */
export const validateRobot = (xml: string, signal?: AbortSignal) =>
  json<RobotReport>("/api/robots/validate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ xml }),
    signal,
  });

export const uploadRobot = (name: string, xml: string, description = "") =>
  json<UploadedRobot & { preview: RobotReport["preview"] }>("/api/robots", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, xml, description }),
  });

export const listUploadedRobots = () =>
  json<{ robots: UploadedRobot[] }>("/api/robots").then((r) => r.robots);

export const deleteRobot = (key: string) =>
  json<{ deleted: string }>(`/api/robots/${encodeURIComponent(key)}`, {
    method: "DELETE",
  });

export const getRobotRequirements = () =>
  json<Record<string, unknown>>("/api/robots/requirements");

// ---- your own code -------------------------------------------------------

export const getStarters = () =>
  json<{ starters: Starter[] }>("/api/code/starters").then((r) => r.starters);

export const getCodeStatus = () => json<CodeStatus>("/api/code/status");

export const runCode = (script: string, name: string) =>
  json<{ run_id: string; script: string }>("/api/code/run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ script, name }),
  });

export const stopCode = () =>
  json<{ stopped: boolean }>("/api/code/stop", { method: "POST" });

export const listSavedScripts = () =>
  json<{ saved: SavedScript[] }>("/api/code/saved").then((r) => r.saved);

export const getSavedScript = (name: string) =>
  json<{ name: string; script: string }>(`/api/code/saved/${encodeURIComponent(name)}`);

export const saveScript = (name: string, script: string) =>
  json<{ name: string; chars: number }>("/api/code/saved", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, script }),
  });

export const deleteSavedScript = (name: string) =>
  json<{ deleted: string }>(`/api/code/saved/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });

// ---- externally attached runs -------------------------------------------

/**
 * Watch training runs driven by code outside this app. Events use the same
 * shapes as `/ws/train`, so the chart and viewer need no separate path.
 */
export function watchAttached(onEvent: (ev: TrainEvent) => void): () => void {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  let ws: WebSocket | null = null;
  let retry: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;

  const connect = () => {
    if (stopped) return;
    ws = new WebSocket(`${proto}//${location.host}/ws/attach`);
    ws.onmessage = (e) => {
      try {
        onEvent(JSON.parse(e.data) as TrainEvent);
      } catch {
        /* ignore a malformed frame rather than tearing down the socket */
      }
    };
    // Keep trying: the point is to notice a training script that starts later.
    ws.onclose = () => {
      if (!stopped) retry = setTimeout(connect, 3000);
    };
  };
  connect();

  return () => {
    stopped = true;
    if (retry) clearTimeout(retry);
    ws?.close();
  };
}

export interface TrainHandle {
  stop(): void;
  close(): void;
}

/**
 * Open a training run. `onEvent` sees every server message in order; `onClose`
 * fires once, whether the run finished, errored, or the socket dropped.
 */
export function startTraining(
  request: TrainRequest,
  onEvent: (ev: TrainEvent) => void,
  onClose?: (reason?: string) => void,
): TrainHandle {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/train`);
  let closed = false;

  ws.onopen = () => ws.send(JSON.stringify(request));
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data) as TrainEvent);
    } catch {
      onEvent({ type: "error", message: "could not parse a server message" });
    }
  };
  ws.onerror = () => {
    if (!closed) onEvent({ type: "error", message: "connection failed — is the backend running?" });
  };
  ws.onclose = () => {
    closed = true;
    onClose?.();
  };

  return {
    stop() {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "stop" }));
      }
    },
    close() {
      closed = true;
      ws.close();
    },
  };
}
