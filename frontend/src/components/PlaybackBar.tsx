import type { PlaybackState } from "./Viewer";

interface Props {
  playback: React.RefObject<PlaybackState>;
  index: number;
  total: number;
  fps: number;
  /** Bumped by the parent whenever playback state changes, to force a repaint. */
  onChange: () => void;
}

const SPEEDS = [0.25, 0.5, 1, 2];

export function PlaybackBar({ playback, index, total, fps, onChange }: Props) {
  const pb = playback.current;
  const seconds = total > 0 ? (index / fps).toFixed(2) : "0.00";
  const duration = total > 0 ? ((total - 1) / fps).toFixed(2) : "0.00";

  return (
    <div className="playback">
      <button
        className="play"
        onClick={() => {
          // Replaying from the end should start over, not sit on the last frame.
          if (!pb.playing && pb.index >= total - 1) pb.index = 0;
          pb.playing = !pb.playing;
          onChange();
        }}
        disabled={total === 0}
        aria-label={pb.playing ? "Pause" : "Play"}
      >
        {pb.playing ? "❚❚" : "▶"}
      </button>

      <input
        className="scrub"
        type="range"
        min={0}
        max={Math.max(0, total - 1)}
        value={index}
        disabled={total === 0}
        onChange={(e) => {
          pb.index = Number(e.target.value);
          pb.playing = false;
          onChange();
        }}
      />

      <span className="time">
        {seconds}s / {duration}s
      </span>

      <div className="speeds">
        {SPEEDS.map((s) => (
          <button
            key={s}
            className={pb.speed === s ? "chip active" : "chip"}
            onClick={() => {
              pb.speed = s;
              onChange();
            }}
          >
            {s}×
          </button>
        ))}
      </div>

      <label className="loop">
        <input
          type="checkbox"
          checked={pb.loop}
          onChange={(e) => {
            pb.loop = e.target.checked;
            onChange();
          }}
        />
        loop
      </label>
    </div>
  );
}
