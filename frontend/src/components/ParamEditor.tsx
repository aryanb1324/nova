import { useMemo } from "react";

import type { ParamSpec } from "../types";

interface Props {
  params: ParamSpec[];
  values: Record<string, number>;
  onChange: (key: string, value: number) => void;
  onReset: () => void;
  disabled?: boolean;
}

/**
 * Renders whatever knobs the backend declared. Nothing here knows what a
 * quadruped is - adding a template with new parameters needs no frontend change.
 */
export function ParamEditor({ params, values, onChange, onReset, disabled }: Props) {
  const groups = useMemo(() => {
    const byGroup = new Map<string, ParamSpec[]>();
    for (const p of params) {
      const list = byGroup.get(p.group) ?? [];
      list.push(p);
      byGroup.set(p.group, list);
    }
    return [...byGroup.entries()];
  }, [params]);

  const modified = params.some((p) => values[p.key] !== undefined && values[p.key] !== p.default);

  if (params.length === 0) {
    // Uploaded robots are fixed geometry. Say so rather than showing a blank panel.
    return (
      <div className="param-editor">
        <div className="section-head">
          <h3>Design</h3>
        </div>
        <p className="muted">
          Fixed geometry — this robot came from a file, so there is nothing to tune.
          Edit the MJCF and upload it again to change it.
        </p>
      </div>
    );
  }

  return (
    <div className="param-editor">
      <div className="section-head">
        <h3>Design</h3>
        <button className="link" onClick={onReset} disabled={disabled || !modified}>
          reset
        </button>
      </div>

      {groups.map(([group, specs]) => (
        <div className="param-group" key={group}>
          <div className="param-group-label">{group}</div>
          {specs.map((p) => {
            const value = values[p.key] ?? p.default;
            return (
              <label className="param" key={p.key} title={p.help}>
                <div className="param-row">
                  <span className="param-label">{p.label}</span>
                  <span className="param-value">
                    {formatValue(value, p.step)}
                    {p.unit && <em>{p.unit}</em>}
                  </span>
                </div>
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={value}
                  disabled={disabled}
                  onChange={(e) => onChange(p.key, Number(e.target.value))}
                />
              </label>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function formatValue(v: number, step: number): string {
  const decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3;
  return v.toFixed(decimals);
}
