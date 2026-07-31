import { useEffect, useRef } from "react";

export interface Series {
  label: string;
  color: string;
  points: { x: number; y: number }[];
  /** Draw against a fixed 0..1 right-hand axis instead of autoscaling. */
  unitAxis?: boolean;
}

interface Props {
  series: Series[];
  height?: number;
  xLabel?: string;
}

/**
 * Canvas line chart sized for a streaming feed.
 *
 * Deliberately not a charting library: the only requirement is appending a
 * point every couple hundred milliseconds without layout thrash, and a canvas
 * redraw is cheaper than reconciling a few hundred SVG nodes.
 */
export function RewardChart({ series, height = 190, xLabel = "steps" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = height;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 52;
    const padR = series.some((s) => s.unitAxis) ? 42 : 14;
    const padT = 12;
    const padB = 26;
    const plotW = Math.max(1, cssW - padL - padR);
    const plotH = Math.max(1, cssH - padT - padB);

    const withData = series.filter((s) => s.points.length > 0);
    const primary = withData.filter((s) => !s.unitAxis);

    const allX = withData.flatMap((s) => s.points.map((p) => p.x));
    if (allX.length === 0) {
      // Empty state stands alone: axis ticks with nothing to scale against
      // would just be noise behind the message.
      ctx.fillStyle = "#4a5568";
      ctx.font = "12px -apple-system, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("waiting for the first iteration…", cssW / 2, cssH / 2);
      return;
    }
    const xMin = Math.min(...allX);
    const xMax = Math.max(...allX);

    const allY = primary.flatMap((s) => s.points.map((p) => p.y));
    let yMin = allY.length ? Math.min(...allY) : 0;
    let yMax = allY.length ? Math.max(...allY) : 1;
    if (yMin === yMax) {
      yMin -= 1;
      yMax += 1;
    }
    const span = yMax - yMin;
    yMin -= span * 0.08;
    yMax += span * 0.08;

    const sx = (x: number) => padL + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * plotW;
    const sy = (y: number) => padT + (1 - (y - yMin) / Math.max(1e-9, yMax - yMin)) * plotH;
    const syUnit = (y: number) => padT + (1 - Math.min(1, Math.max(0, y))) * plotH;

    // grid + left axis labels
    ctx.strokeStyle = "#1e2836";
    ctx.fillStyle = "#6b7a90";
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      const value = yMax - ((yMax - yMin) * i) / 4;
      ctx.textAlign = "right";
      ctx.fillText(formatTick(value), padL - 8, y + 4);
    }

    if (series.some((s) => s.unitAxis)) {
      ctx.textAlign = "left";
      ctx.fillStyle = "#4d7c5a";
      for (let i = 0; i <= 4; i++) {
        const y = padT + (plotH * i) / 4;
        ctx.fillText(`${100 - i * 25}%`, padL + plotW + 8, y + 4);
      }
    }

    // x labels
    ctx.fillStyle = "#6b7a90";
    ctx.textAlign = "center";
    ctx.fillText(formatCount(xMin), padL, cssH - 8);
    ctx.fillText(formatCount(xMax), padL + plotW, cssH - 8);
    ctx.fillText(xLabel, padL + plotW / 2, cssH - 8);

    for (const s of withData) {
      const project = s.unitAxis ? syUnit : sy;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.unitAxis ? 1.5 : 2;
      if (s.unitAxis) ctx.setLineDash([4, 3]);
      ctx.beginPath();
      s.points.forEach((p, i) => {
        const x = sx(p.x);
        const y = project(p.y);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      const last = s.points[s.points.length - 1];
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(sx(last.x), project(last.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [series, height, xLabel]);

  return (
    <div className="chart">
      <canvas ref={canvasRef} style={{ width: "100%", height }} />
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function formatTick(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) >= 10) return v.toFixed(0);
  return v.toFixed(1);
}

function formatCount(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1000) return `${Math.round(v / 1000)}k`;
  return `${Math.round(v)}`;
}
