import { useEffect, useRef, useState } from "react";

import * as api from "../api";
import type { ConsoleLine, SavedScript, Starter } from "../types";
import { CodeEditor } from "./CodeEditor";

interface Props {
  starters: Starter[];
  running: boolean;
  enabled: boolean;
  disabledReason: string | null;
  lines: ConsoleLine[];
  onRun: (script: string, name: string) => void;
  onStop: () => void;
  onClear: () => void;
}

const DRAFT_KEY = "nova.draft";
const NAME_KEY = "nova.draft.name";

/**
 * Write an algorithm, run it against the simulator, watch it in the same charts
 * as everything else.
 *
 * The script runs as a normal subprocess on this machine — it is not sandboxed,
 * because it is the user's own code on the user's own computer. The server
 * refuses to do this for anything but a loopback request; see api/runner.py.
 */
export function CodePane({
  starters,
  running,
  enabled,
  disabledReason,
  lines,
  onRun,
  onStop,
  onClear,
}: Props) {
  const [code, setCode] = useState<string>(() => localStorage.getItem(DRAFT_KEY) ?? "");
  const [name, setName] = useState<string>(() => localStorage.getItem(NAME_KEY) ?? "untitled");
  const [saved, setSaved] = useState<SavedScript[]>([]);
  const [note, setNote] = useState<string | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const stuck = useRef(true);

  // Seed the editor from the blank starter the first time there's nothing to restore.
  useEffect(() => {
    if (!code && starters.length) {
      const blank = starters.find((s) => s.key === "blank") ?? starters[0];
      setCode(blank.code);
    }
  }, [starters]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    localStorage.setItem(DRAFT_KEY, code);
  }, [code]);
  useEffect(() => {
    localStorage.setItem(NAME_KEY, name);
  }, [name]);

  const refreshSaved = () => {
    api.listSavedScripts().then(setSaved).catch(() => undefined);
  };
  useEffect(refreshSaved, []);

  // Follow the tail, unless the user has scrolled up to read something.
  useEffect(() => {
    const el = consoleRef.current;
    if (el && stuck.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  async function save() {
    try {
      await api.saveScript(name, code);
      setNote(`saved as ${name}`);
      refreshSaved();
    } catch (e) {
      setNote((e as Error).message);
    }
  }

  async function load(scriptName: string) {
    try {
      const s = await api.getSavedScript(scriptName);
      setCode(s.script);
      setName(s.name);
      setNote(`loaded ${s.name}`);
    } catch (e) {
      setNote((e as Error).message);
    }
  }

  return (
    <div className="codepane">
      <div className="code-toolbar">
        <select
          className="select"
          value=""
          onChange={(e) => {
            const s = starters.find((x) => x.key === e.target.value);
            if (s) {
              setCode(s.code);
              setName(s.key);
              setNote(s.description);
            }
            e.target.value = "";
          }}
        >
          <option value="">Start from…</option>
          {starters.map((s) => (
            <option key={s.key} value={s.key}>
              {s.name}
            </option>
          ))}
        </select>

        <input
          className="name-input"
          value={name}
          spellCheck={false}
          onChange={(e) => setName(e.target.value)}
          aria-label="Script name"
        />

        <button className="chip" onClick={save} disabled={!code.trim()}>
          Save
        </button>

        <select
          className="select"
          value=""
          onChange={(e) => {
            if (e.target.value) void load(e.target.value);
            e.target.value = "";
          }}
          disabled={saved.length === 0}
        >
          <option value="">{saved.length ? "Open…" : "No saved scripts"}</option>
          {saved.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
            </option>
          ))}
        </select>

        <div className="spacer" />

        {running ? (
          <button className="primary stop compact" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button
            className="primary compact"
            onClick={() => onRun(code, name)}
            disabled={!enabled || !code.trim()}
            title={enabled ? "Run (⌘↵)" : disabledReason ?? ""}
          >
            Run ▸
          </button>
        )}
      </div>

      {!enabled && (
        <p className="warn code-warn">
          Running code is disabled. {disabledReason ?? ""} You can still upload a trained
          .onnx policy.
        </p>
      )}
      {note && <p className="code-note">{note}</p>}

      <CodeEditor value={code} onChange={setCode} onRun={() => enabled && onRun(code, name)} />

      <div className="console-head">
        <span>Output</span>
        {running && <span className="badge live">running</span>}
        <div className="spacer" />
        <button className="link" onClick={onClear} disabled={lines.length === 0}>
          clear
        </button>
      </div>
      <div
        className="console"
        ref={consoleRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        {lines.length === 0 ? (
          <span className="console-empty">
            Output appears here. Anything your script prints, plus tracebacks.
          </span>
        ) : (
          lines.map((l) => (
            <div key={l.id} className={`console-line ${l.kind}`}>
              {l.text}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
