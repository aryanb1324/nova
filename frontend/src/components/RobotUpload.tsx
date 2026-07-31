import { useEffect, useRef, useState } from "react";

import * as api from "../api";
import type { Frame, RobotReport, SceneDesc } from "../types";

type Camera = { distance: number; elevation: number; azimuth?: number };

interface Props {
  onPreview: (scene: SceneDesc, frame: Frame, camera: Camera, label: string) => void;
  onUploaded: () => void;
  disabled?: boolean;
}

/**
 * Bring your own robot.
 *
 * Validation runs while you type, and reports the two things that actually
 * matter: whether the viewer can draw it, and which tasks it can be trained on.
 * Neither is declared — both are read off the compiled model, so the answer is
 * about the robot rather than about what someone claimed.
 */
export function RobotUpload({ onPreview, onUploaded, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [xml, setXml] = useState("");
  const [name, setName] = useState("");
  const [report, setReport] = useState<RobotReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Validate as you edit. Cheap on the server — it compiles and steps once.
  useEffect(() => {
    if (!xml.trim()) {
      setReport(null);
      return;
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      setChecking(true);
      api
        .validateRobot(xml, ctrl.signal)
        .then((r) => {
          setReport(r);
          setError(null);
          if (r.ok && r.preview) {
            onPreview(r.preview.scene, r.preview.frame, r.preview.camera, "unsaved robot");
          }
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError((e as Error).message);
        })
        .finally(() => setChecking(false));
    }, 400);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [xml]); // eslint-disable-line react-hooks/exhaustive-deps

  function readFile(file: File) {
    if (!name) setName(file.name.replace(/\.(xml|mjcf)$/i, ""));
    file.text().then(setXml);
  }

  async function save() {
    setSaving(true);
    try {
      await api.uploadRobot(name || "my robot", xml);
      setXml("");
      setName("");
      setReport(null);
      setOpen(false);
      onUploaded();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  if (!open) {
    return (
      <button className="link add-robot" onClick={() => setOpen(true)} disabled={disabled}>
        + Upload your own robot
      </button>
    );
  }

  const tasks = report?.info?.compatible_tasks ?? [];
  const problems = report?.info?.task_problems ?? {};

  return (
    <div className="robot-upload">
      <div className="section-head">
        <h3>Upload a robot</h3>
        <button className="link" onClick={() => setOpen(false)}>
          close
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
          if (file) readFile(file);
        }}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".xml,.mjcf"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) readFile(file);
            e.target.value = "";
          }}
        />
        <strong>Drop an MJCF file</strong>
        <span>or paste it below</span>
      </div>

      <textarea
        className="mjcf-input"
        value={xml}
        spellCheck={false}
        placeholder={"<mujoco>\n  <worldbody>…</worldbody>\n</mujoco>"}
        onChange={(e) => setXml(e.target.value)}
      />

      {checking && <p className="code-note">checking…</p>}

      {report && !report.ok && (
        <ul className="issues">
          {report.errors.map((e, i) => (
            <li key={i} className="bad">
              {e}
            </li>
          ))}
        </ul>
      )}

      {report?.ok && (
        <div className="robot-ok">
          <p className="good-line">
            Valid — {report.info.bodies} bodies, {report.info.actuators} actuators,{" "}
            {report.info.dof} DoF, {report.info.total_mass?.toFixed(2)} kg
          </p>
          <p className="tasks-line">
            {tasks.length ? (
              <>
                Can be trained on: <strong>{tasks.join(", ")}</strong>
              </>
            ) : (
              "No compatible tasks."
            )}
          </p>
          {Object.entries(problems).map(([task, why]) => (
            <p key={task} className="muted small">
              not <strong>{task}</strong>: {why.join(", ")}
            </p>
          ))}
        </div>
      )}

      {report?.warnings?.map((w, i) => (
        <p key={i} className="warn small">
          {w}
        </p>
      ))}

      {error && <p className="error">{error}</p>}

      <div className="upload-actions">
        <input
          className="name-input wide"
          value={name}
          placeholder="name"
          spellCheck={false}
          onChange={(e) => setName(e.target.value)}
        />
        <button
          className="primary compact"
          onClick={save}
          disabled={!report?.ok || saving || tasks.length === 0}
          title={tasks.length === 0 ? "No task can use this robot yet" : "Save"}
        >
          {saving ? "Saving…" : "Save robot"}
        </button>
      </div>
    </div>
  );
}
