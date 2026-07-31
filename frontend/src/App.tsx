import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import * as api from "./api";
import { CodePane } from "./components/CodePane";
import { ParamEditor } from "./components/ParamEditor";
import { PlaybackBar } from "./components/PlaybackBar";
import { PolicyPanel } from "./components/PolicyPanel";
import { RewardChart, type Series } from "./components/RewardChart";
import { RobotUpload } from "./components/RobotUpload";
import { Viewer, type PlaybackState } from "./components/Viewer";
import type {
  Catalog,
  CodeStatus,
  ConsoleLine,
  EvalResult,
  Frame,
  PolicyResult,
  PreviewResponse,
  ProgressEvent,
  Rollout,
  RunSummary,
  SceneDesc,
  Starter,
  TemplateInfo,
  TrainEvent,
} from "./types";

type Phase = "idle" | "training" | "finished";
type Mode = "design" | "code";

const BUDGETS = [
  { label: "Quick", factor: 0.25 },
  { label: "Standard", factor: 1 },
  { label: "Long", factor: 2 },
];

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [templateKey, setTemplateKey] = useState<string>("reach_arm");
  const [taskKey, setTaskKey] = useState<string>("reach");
  const [values, setValues] = useState<Record<string, number>>({});
  const [budget, setBudget] = useState(1);

  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  /** A robot being uploaded but not yet saved — shown before it exists. */
  const [draft, setDraft] = useState<{
    scene: SceneDesc;
    frame: Frame;
    camera: { distance: number; elevation: number; azimuth?: number };
    label: string;
  } | null>(null);
  const [rollout, setRollout] = useState<Rollout | null>(null);
  const [rolloutLabel, setRolloutLabel] = useState<string>("");

  const [phase, setPhase] = useState<Phase>("idle");
  const [history, setHistory] = useState<ProgressEvent[]>([]);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [policyList, setPolicyList] = useState<PolicyResult[]>([]);
  const [status, setStatus] = useState<string>("");
  const [runId, setRunId] = useState<string | null>(null);
  /** A training loop running outside this app, streaming in over /ws/attach. */
  const [external, setExternal] = useState<{ run_id: string; algo: string } | null>(null);

  const [mode, setMode] = useState<Mode>("design");
  const [starters, setStarters] = useState<Starter[]>([]);
  const [codeStatus, setCodeStatus] = useState<CodeStatus | null>(null);
  const [codeRunning, setCodeRunning] = useState(false);
  const [lines, setLines] = useState<ConsoleLine[]>([]);
  const lineId = useRef(0);

  const pushLine = useCallback((text: string, kind: ConsoleLine["kind"] = "out") => {
    setLines((prev) => {
      const next = [...prev, { id: lineId.current++, text, kind }];
      // A long training run prints thousands of lines; keep the tail bounded so
      // the DOM doesn't become the bottleneck.
      return next.length > 2000 ? next.slice(-2000) : next;
    });
  }, []);

  const trainRef = useRef<api.TrainHandle | null>(null);
  const playback = useRef<PlaybackState>({ index: 0, playing: true, speed: 1, loop: true });
  const [frameIndex, setFrameIndex] = useState(0);
  const [, forceTick] = useState(0);
  const repaint = useCallback(() => forceTick((n) => n + 1), []);

  const template: TemplateInfo | undefined = useMemo(
    () => catalog?.templates.find((t) => t.key === templateKey),
    [catalog, templateKey],
  );
  const task = useMemo(() => catalog?.tasks.find((t) => t.key === taskKey), [catalog, taskKey]);

  // ---- bootstrap -------------------------------------------------------

  useEffect(() => {
    api
      .getCatalog()
      .then((c) => {
        setCatalog(c);
        const first = c.templates[0];
        if (first) {
          setTemplateKey(first.key);
          setTaskKey(first.tasks[0]);
          setValues(Object.fromEntries(first.params.map((p) => [p.key, p.default])));
        }
      })
      .catch((e) => setError(`Could not reach the backend: ${e.message}`));
    refreshRuns();
    refreshPolicies();
    api.getStarters().then(setStarters).catch(() => undefined);
    api.getCodeStatus().then((s) => {
      setCodeStatus(s);
      setCodeRunning(s.running);
    }).catch(() => undefined);
  }, []);

  const refreshRuns = useCallback(() => {
    api.listRuns().then(setRuns).catch(() => undefined);
  }, []);

  const refreshPolicies = useCallback(() => {
    api.listPolicies().then(setPolicyList).catch(() => undefined);
  }, []);

  const refreshCatalog = useCallback(() => {
    // An uploaded robot registers server-side, so the catalog is the source of
    // truth for what exists — including its generated parameter schema.
    api.getCatalog().then(setCatalog).catch(() => undefined);
  }, []);

  async function removeRobot(key: string) {
    try {
      await api.deleteRobot(key);
      if (templateKey === key) {
        const fallback = catalog?.templates.find((t) => t.key !== key);
        if (fallback) selectTemplate(fallback);
      }
      refreshCatalog();
    } catch (e) {
      setError((e as Error).message);
    }
  }


  // ---- live preview ----------------------------------------------------

  useEffect(() => {
    if (!template) return;
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      api
        .getPreview(template.key, values, ctrl.signal)
        .then(setPreview)
        .catch((e) => {
          if (e.name !== "AbortError") setError(e.message);
        });
    }, 120); // debounce so dragging a slider doesn't queue a request per pixel
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [template, values]);

  // ---- template / task selection ---------------------------------------

  function selectTemplate(next: TemplateInfo) {
    if (phase === "training") return;
    setDraft(null);
    setTemplateKey(next.key);
    setValues(Object.fromEntries(next.params.map((p) => [p.key, p.default])));
    if (!next.tasks.includes(taskKey)) setTaskKey(next.tasks[0]);
    setRollout(null);
    setHistory([]);
    setEvalResult(null);
    setPhase("idle");
  }

  function resetParams() {
    if (!template) return;
    setValues(Object.fromEntries(template.params.map((p) => [p.key, p.default])));
  }

  // ---- training --------------------------------------------------------

  function showRollout(r: Rollout, label: string) {
    setRollout(r);
    setRolloutLabel(label);
    playback.current.index = 0;
    playback.current.playing = true;
    setFrameIndex(0);
    repaint();
  }

  /** Shared by the built-in trainer and by externally attached runs. */
  function applyEvent(ev: TrainEvent) {
    switch (ev.type) {
      case "accepted":
        setRunId(ev.run_id);
        setStatus(`run ${ev.run_id}`);
        break;
      case "progress":
        setHistory((h) => [...h, ev]);
        break;
      case "rollout":
        showRollout(ev, ev.final ? "final policy" : `step ${formatCount(ev.step)}`);
        break;
      case "done":
        setStatus(
          `${formatCount(ev.steps_completed ?? 0)} steps in ${ev.wall_time}s` +
            (ev.stopped_early ? " (stopped early)" : ""),
        );
        break;
      case "eval":
        setEvalResult(ev);
        break;
      case "saved":
        refreshRuns();
        break;
      case "error":
        setError(ev.message);
        break;
    }
  }

  // Kept in a ref so the attach socket is opened once and still sees fresh state.
  const applyRef = useRef(applyEvent);
  applyRef.current = applyEvent;

  useEffect(() => {
    // Someone running their own algorithm against these envs appears here, in
    // the same chart and viewer as a built-in run.
    return api.watchAttached((ev) => {
      if (ev.type === "attach_state") return;
      if (ev.type === "console") {
        pushLine(ev.line);
        setCodeRunning(true);
        return;
      }
      if (ev.type === "code_exit") {
        pushLine(
          ev.exit_code === 0
            ? `[nova] finished in ${ev.elapsed}s`
            : `[nova] exited with code ${ev.exit_code} after ${ev.elapsed}s`,
          "exit",
        );
        setCodeRunning(false);
        return;
      }
      if (ev.type === "start") {
        const algo = String((ev.config as { algo?: string })?.algo ?? "custom");
        setExternal({ run_id: (ev as { run_id?: string }).run_id ?? "external", algo });
        setHistory([]);
        setEvalResult(null);
        setRollout(null);
        setRunId(null);
        setStatus(`external run · ${algo}`);
        return;
      }
      applyRef.current(ev);
      if (ev.type === "done" || ev.type === "closed") setExternal(null);
    });
  }, []);

  function startTraining() {
    if (!template || !task) return;
    setHistory([]);
    setEvalResult(null);
    setRollout(null);
    setError(null);
    setPhase("training");
    setStatus("connecting…");

    const totalSteps = Math.round(task.typical_steps * budget);

    trainRef.current = api.startTraining(
      {
        task: task.key,
        template: template.key,
        params: values,
        total_steps: totalSteps,
        rollout_every: 8,
        eval_episodes: 30,
        save: true,
      },
      applyEvent,
      () => {
        setPhase("finished");
        trainRef.current = null;
      },
    );
  }

  function stopTraining() {
    setStatus("stopping…");
    trainRef.current?.stop();
  }

  // ---- running your own code -------------------------------------------

  async function runCode(script: string, name: string) {
    setLines([]);
    setError(null);
    try {
      await api.runCode(script, name);
      setCodeRunning(true);
    } catch (e) {
      pushLine(`[nova] ${(e as Error).message}`, "exit");
      setCodeRunning(false);
    }
  }

  async function stopCode() {
    try {
      await api.stopCode();
      pushLine("[nova] stop requested", "sys");
    } catch (e) {
      pushLine(`[nova] ${(e as Error).message}`, "exit");
    }
  }

  useEffect(() => () => trainRef.current?.close(), []);

  // ---- loading a saved run ---------------------------------------------

  async function loadRun(id: string) {
    if (phase === "training") return;
    try {
      const run = await api.getRun(id);
      setTemplateKey(run.template);
      setTaskKey(run.task);
      setValues(run.params);
      setHistory(run.history ?? []);
      setEvalResult(run.eval ?? null);
      setPhase("finished");
      setRunId(id);
      setStatus(`loaded ${id}`);
      if (run.final_rollout) showRollout(run.final_rollout, "saved policy");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function removeRun(id: string) {
    await api.deleteRun(id).catch(() => undefined);
    refreshRuns();
  }

  // ---- derived ---------------------------------------------------------

  // A draft upload wins: you're looking at it precisely because it isn't saved.
  const scene = draft?.scene ?? rollout?.scene ?? preview?.scene;
  const frames = draft
    ? [draft.frame]
    : rollout?.frames ?? (preview ? [preview.frame] : []);
  const fps = rollout?.fps ?? 50;
  const latest = history[history.length - 1];

  const series: Series[] = useMemo(() => {
    const out: Series[] = [
      {
        label: "mean episode reward",
        color: "#5b9dff",
        points: history
          .filter((h) => h.mean_reward !== undefined)
          .map((h) => ({ x: h.step, y: h.mean_reward as number })),
      },
    ];
    if (history.some((h) => h.is_success !== undefined)) {
      out.push({
        label: "success rate",
        color: "#3ddc84",
        unitAxis: true,
        points: history
          .filter((h) => h.is_success !== undefined)
          .map((h) => ({ x: h.step, y: h.is_success as number })),
      });
    }
    return out;
  }, [history]);

  if (error && !catalog) {
    return (
      <div className="fatal">
        <h1>NOVA</h1>
        <p>{error}</p>
        <pre>cd backend &amp;&amp; python scripts/serve.py</pre>
      </div>
    );
  }

  return (
    <div className={mode === "code" ? "app code-mode" : "app"}>
      <header className="topbar">
        <div className="brand-inline">
          <strong>NOVA</strong>
          <span>next-gen open virtual AI</span>
        </div>
        <div className="modes">
          <button
            className={mode === "design" ? "chip active" : "chip"}
            onClick={() => setMode("design")}
          >
            Design
          </button>
          <button
            className={mode === "code" ? "chip active" : "chip"}
            onClick={() => setMode("code")}
          >
            Your code
          </button>
        </div>
      </header>

      {mode === "code" ? (
        <CodePane
          starters={starters}
          running={codeRunning}
          enabled={codeStatus?.enabled ?? false}
          disabledReason={codeStatus?.reason ?? null}
          lines={lines}
          onRun={runCode}
          onStop={stopCode}
          onClear={() => setLines([])}
        />
      ) : (
      <aside className="sidebar">
        <div className="section">
          <div className="section-head">
            <h3>Robot</h3>
          </div>
          <div className="templates">
            {catalog?.templates.map((t) => (
              <div className="template-row" key={t.key}>
                <button
                  className={t.key === templateKey ? "template active" : "template"}
                  onClick={() => selectTemplate(t)}
                  disabled={phase === "training"}
                >
                  <strong>
                    {t.name}
                    {t.source === "upload" && <em className="tag">yours</em>}
                  </strong>
                  <span>{t.description}</span>
                </button>
                {t.source === "upload" && (
                  <button
                    className="link danger"
                    onClick={() => removeRobot(t.key)}
                    disabled={phase === "training"}
                    title="Delete this robot"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>

          <RobotUpload
            disabled={phase === "training"}
            onPreview={(s, f, cam, label) => setDraft({ scene: s, frame: f, camera: cam, label })}
            onUploaded={() => {
              setDraft(null);
              refreshCatalog();
            }}
          />
        </div>

        {template && (
          <ParamEditor
            params={template.params}
            values={values}
            onChange={(k, v) => setValues((prev) => ({ ...prev, [k]: v }))}
            onReset={resetParams}
            disabled={phase === "training"}
          />
        )}

        <div className="section">
          <div className="section-head">
            <h3>Task</h3>
          </div>
          <div className="tasks">
            {catalog?.tasks
              .filter((t) => template?.tasks.includes(t.key))
              .map((t) => (
                <button
                  key={t.key}
                  className={t.key === taskKey ? "task active" : "task"}
                  onClick={() => setTaskKey(t.key)}
                  disabled={phase === "training"}
                >
                  <strong>{t.name}</strong>
                  <span>{t.description}</span>
                </button>
              ))}
          </div>

          <div className="budget">
            {BUDGETS.map((b) => (
              <button
                key={b.label}
                className={budget === b.factor ? "chip active" : "chip"}
                onClick={() => setBudget(b.factor)}
                disabled={phase === "training"}
              >
                {b.label}
              </button>
            ))}
            <span className="budget-hint">
              {task ? `${formatCount(Math.round(task.typical_steps * budget))} steps` : ""}
            </span>
          </div>

          {phase === "training" ? (
            <button className="primary stop" onClick={stopTraining}>
              Stop training
            </button>
          ) : (
            <button className="primary" onClick={startTraining} disabled={!template || !task}>
              Train
            </button>
          )}
          {status && <p className="status">{status}</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </aside>
      )}

      <main className="stage">
        <div className="viewport">
          {scene ? (
            <Viewer
              scene={scene}
              frames={frames}
              fps={fps}
              playback={playback}
              onIndex={setFrameIndex}
              camera={draft?.camera ?? preview?.camera ?? template?.camera}
              follow={taskKey === "locomotion" && !!rollout}
            />
          ) : (
            <div className="viewport-empty">building the robot…</div>
          )}

          <div className="viewport-tag">
            {draft ? draft.label : rollout ? rolloutLabel : "design preview"}
            {rollout?.stats && (
              <em>
                {rollout.stats.success !== undefined && ` · ${rollout.stats.success ? "reached" : "missed"}`}
                {rollout.stats.final_distance !== undefined &&
                  ` · ${(rollout.stats.final_distance as number).toFixed(2)}m`}
              </em>
            )}
          </div>
        </div>

        <PlaybackBar
          playback={playback}
          index={frameIndex}
          total={frames.length}
          fps={fps}
          onChange={repaint}
        />
      </main>

      <aside className="panel">
        <div className="section">
          <div className="section-head">
            <h3>Training</h3>
            {external ? (
              <span className="badge live">external · {external.algo}</span>
            ) : (
              latest && <span className="muted">{latest.fps.toLocaleString()} steps/s</span>
            )}
          </div>

          {latest && (
            <div className="progressbar">
              <div style={{ width: `${Math.min(100, latest.progress * 100)}%` }} />
            </div>
          )}

          <RewardChart series={series} />

          <div className="metrics">
            <Metric label="steps" value={latest ? formatCount(latest.step) : "—"} />
            <Metric
              label="reward"
              value={latest?.mean_reward !== undefined ? latest.mean_reward.toFixed(1) : "—"}
            />
            <Metric
              label="success"
              value={latest?.is_success !== undefined ? `${Math.round(latest.is_success * 100)}%` : "—"}
            />
            <Metric label="elapsed" value={latest ? `${latest.elapsed.toFixed(0)}s` : "—"} />
          </div>

          {evalResult && (
            <div className="eval">
              <h4>Final evaluation</h4>
              <div className="metrics">
                <Metric label="episodes" value={`${evalResult.episodes}`} />
                <Metric label="return" value={evalResult.mean_return.toFixed(1)} />
                {evalResult.success_rate !== undefined && (
                  <Metric
                    label="success"
                    value={`${Math.round(evalResult.success_rate * 100)}%`}
                    highlight={evalResult.success_rate > 0.8}
                  />
                )}
                {evalResult.mean_final_distance !== undefined && (
                  <Metric label="distance" value={`${evalResult.mean_final_distance.toFixed(2)}m`} />
                )}
              </div>
            </div>
          )}

          {runId && phase !== "training" && (
            <a className="secondary" href={api.runPolicyUrl(runId)} download>
              Download this policy (.onnx)
            </a>
          )}
        </div>

        <PolicyPanel
          policies={policyList}
          onRefresh={refreshPolicies}
          onShowRollout={showRollout}
          disabled={phase === "training"}
        />

        <div className="section runs">
          <div className="section-head">
            <h3>Saved runs</h3>
            <button className="link" onClick={refreshRuns}>
              refresh
            </button>
          </div>
          {runs.length === 0 && <p className="muted">Nothing saved yet — train something.</p>}
          <ul>
            {runs.map((r) => (
              <li key={r.run_id}>
                <button className="run" onClick={() => loadRun(r.run_id)} disabled={phase === "training"}>
                  <strong>
                    {r.template} · {r.task}
                  </strong>
                  <span>
                    {formatCount(r.steps_completed ?? 0)} steps · {r.wall_time}s
                    {r.eval?.success_rate !== undefined &&
                      ` · ${Math.round(r.eval.success_rate * 100)}% success`}
                  </span>
                </button>
                <button className="link danger" onClick={() => removeRun(r.run_id)} title="Delete run">
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={highlight ? "metric good" : "metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatCount(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1000) return `${Math.round(v / 1000)}k`;
  return `${v}`;
}
