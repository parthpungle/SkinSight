import { fmt } from "./ui.jsx";

const TERTILES = ["low", "mid", "high"];

/** AUROC per tertile on a shared axis, with a thin CI whisker. */
export function TertileBars({ per, min = 0.7, max = 1.0, labels }) {
  const pos = (v) => `${((Math.min(max, Math.max(min, v)) - min) / (max - min)) * 100}%`;
  return (
    <div className="space-y-1.5">
      {TERTILES.map((t) => {
        const m = per?.[t];
        if (!m) return null;
        const [lo, hi] = m.ci95?.auroc ?? [null, null];
        // Full literal class names so Tailwind's scanner generates them.
        const tone = t === "high" ? "bg-warn/80" : t === "mid" ? "bg-accent/80" : "bg-good/80";
        return (
          <div key={t} className="grid grid-cols-[52px_minmax(0,1fr)_auto] items-center gap-3">
            <span className="num text-[11px] text-faint">{labels?.[t] ?? t}</span>
            <div className="relative h-2 rounded-full bg-surface-hover" aria-hidden="true">
              {m.auroc !== null && (
                <div className={`absolute inset-y-0 left-0 rounded-full ${tone}`} style={{ width: pos(m.auroc) }} />
              )}
              {lo !== null && hi !== null && (
                <div
                  className="absolute top-1/2 h-px -translate-y-1/2 bg-fg/70"
                  style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }}
                />
              )}
            </div>
            <span className="num whitespace-nowrap text-xs text-muted">
              <span className="text-fg">{fmt(m.auroc)}</span>{" "}
              <span className="text-faint">
                [{fmt(lo, 2)}–{fmt(hi, 2)}] n={m.n}/{m.n_pos}
              </span>
            </span>
          </div>
        );
      })}
      <div className="num grid grid-cols-[52px_minmax(0,1fr)_auto] gap-3 text-[10px] text-faint">
        <span />
        <span className="flex justify-between">
          <span>{min.toFixed(1)}</span>
          <span>AUROC</span>
          <span>{max.toFixed(1)}</span>
        </span>
        <span className="invisible">0.000 [0.00–0.00] n=000/000</span>
      </div>
    </div>
  );
}

/** Paired delta with CI on a symmetric axis around zero. */
export function DeltaWhisker({ delta, ci, range = 0.2 }) {
  const pos = (v) => `${((Math.max(-range, Math.min(range, v)) + range) / (2 * range)) * 100}%`;
  const [lo, hi] = ci ?? [null, null];
  const includesZero = lo !== null && hi !== null && lo <= 0 && hi >= 0;
  const color = includesZero ? "bg-muted" : delta < 0 ? "bg-bad" : "bg-good";
  return (
    <div className="relative h-4 w-full" aria-hidden="true">
      <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
      <div className="absolute top-0 h-4 w-px bg-faint" style={{ left: "50%" }} />
      {lo !== null && hi !== null && (
        <div
          className={`absolute top-1/2 h-0.5 -translate-y-1/2 ${color}`}
          style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }}
        />
      )}
      {delta !== null && (
        <div className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ${color}`} style={{ left: pos(delta) }} />
      )}
    </div>
  );
}
