import { useCallback, useRef, useState } from "react";

import * as api from "../api";
import type { PolicyResult, Rollout, SeedScore } from "../types";

interface Props {
  policies: PolicyResult[];
  onRefresh: () => void;
  onShowRollout: (rollout: Rollout, label: string) => void;
  disabled?: boolean;
}

/**
 * Bring someone else's policy in.
 *
 * The uploaded file is data, never code: it is an ONNX graph carrying a manifest
 * that says which task and robot it was trained for. The server validates that,
 * then scores it on published seeds and on seeds it could not have been tuned
 * against.
 */
export function PolicyPanel({ policies, onRefresh, onShowRollout, disabled }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [latest, setLatest] = useState<PolicyResult | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      try {
        const result = await api.uploadPolicy(file);
        setLatest(result);
        onRefresh();
        if (result.rollout) onShowRollout(result.rollout, `uploaded · ${file.name}`);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [onRefresh, onShowRollout],
  );

  async function open(policyId: string) {
    try {
      const full = await api.getPolicy(policyId);
      setLatest(full);
      if (full.rollout) onShowRollout(full.rollout, `policy · ${policyId}`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(policyId: string) {
    await api.deletePolicy(policyId).catch(() => undefined);
    if (latest?.policy_id === policyId) setLatest(null);
    onRefresh();
  }

  return (
    <div className="section">
      <div className="section-head">
        <h3>Bring your own policy</h3>
        <button className="link" onClick={onRefresh}>
          refresh
        </button>
      </div>

      <div
        className={dragging ? "dropzone over" : "dropzone"}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file && !disabled) void submit(file);
        }}
        onClick={() => !disabled && !busy && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".onnx"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void submit(file);
            e.target.value = "";
          }}
        />
        {busy ? (
          <span>scoring…</span>
        ) : (
          <>
            <strong>Drop a .onnx policy</strong>
            <span>
              Export one with <code>nova.export()</code>, or download a trained run
              above.
            </span>
          </>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      {latest && <PolicyScores result={latest} />}

      {policies.length > 0 && (
        <ul className="runs-list">
          {policies.map((p) => (
            <li key={p.policy_id}>
              <button className="run" onClick={() => open(p.policy_id)} disabled={disabled}>
                <strong>
                  {p.manifest.algo || "policy"} · {p.manifest.template}
                </strong>
                <span>
                  {p.held_out.success_rate !== undefined
                    ? `${Math.round(p.held_out.success_rate * 100)}% held-out`
                    : `return ${p.held_out.mean_return.toFixed(1)}`}
                  {" · "}
                  {(p.size_bytes / 1024).toFixed(0)} KB
                </span>
              </button>
              <a
                className="link"
                href={`/api/policies/${p.policy_id}/policy.onnx`}
                title="Download"
                download
              >
                ↓
              </a>
              <button className="link danger" onClick={() => remove(p.policy_id)} title="Delete">
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PolicyScores({ result }: { result: PolicyResult }) {
  const { manifest, gap } = result;
  // A large drop on episodes the policy could not have been tuned against is
  // the signal worth surfacing, so call it out rather than burying it.
  const overfit =
    gap.success_rate !== undefined ? gap.success_rate > 0.15 : gap.mean_return > 10;

  return (
    <div className="policy-scores">
      <div className="policy-head">
        <strong>{manifest.algo || "unnamed policy"}</strong>
        <span>
          {manifest.task} · {manifest.template} · {manifest.obs_dim}→{manifest.act_dim}
        </span>
        {manifest.author && <span>by {manifest.author}</span>}
      </div>

      <table className="scores">
        <thead>
          <tr>
            <th />
            <th>public</th>
            <th>held-out</th>
          </tr>
        </thead>
        <tbody>
          <ScoreRow label="return" pub={result.public} priv={result.held_out} field="mean_return" />
          {result.public.success_rate !== undefined && (
            <ScoreRow
              label="success"
              pub={result.public}
              priv={result.held_out}
              field="success_rate"
              percent
            />
          )}
          {result.public.mean_final_distance !== undefined && (
            <ScoreRow
              label="distance"
              pub={result.public}
              priv={result.held_out}
              field="mean_final_distance"
              suffix="m"
            />
          )}
        </tbody>
      </table>

      {overfit && (
        <p className="warn">
          Scores noticeably worse on episodes it couldn't be tuned against — a sign the
          policy fitted specific episodes rather than the task.
        </p>
      )}
    </div>
  );
}

function ScoreRow({
  label,
  pub,
  priv,
  field,
  percent,
  suffix = "",
}: {
  label: string;
  pub: SeedScore;
  priv: SeedScore;
  field: keyof SeedScore;
  percent?: boolean;
  suffix?: string;
}) {
  const fmt = (v: number | undefined) =>
    v === undefined ? "—" : percent ? `${Math.round(v * 100)}%` : `${v.toFixed(2)}${suffix}`;
  return (
    <tr>
      <td>{label}</td>
      <td>{fmt(pub[field] as number | undefined)}</td>
      <td>{fmt(priv[field] as number | undefined)}</td>
    </tr>
  );
}
