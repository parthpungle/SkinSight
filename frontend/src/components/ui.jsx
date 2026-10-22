/* Shared presentational primitives for the dark dashboard. */

export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

const TONES = {
  neutral: "text-muted border-line bg-surface",
  accent: "text-accent border-accent/40 bg-accent/10",
  good: "text-good border-good/40 bg-good/10",
  warn: "text-warn border-warn/40 bg-warn/10",
  bad: "text-bad border-bad/40 bg-bad/10",
};

export function Chip({ tone = "neutral", mono = false, children, title }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center whitespace-nowrap rounded border px-1.5 py-px text-xs font-medium ${
        mono ? "num font-normal" : ""
      } ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

export function Button({ variant = "outline", className = "", ...props }) {
  const styles =
    variant === "primary"
      ? "bg-accent text-black border-accent hover:brightness-110"
      : variant === "ghost"
        ? "border-transparent text-muted hover:text-fg hover:bg-surface-hover"
        : "border-line text-fg hover:bg-surface-hover";
  return (
    <button
      type="button"
      className={`rounded-md border px-3.5 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${className}`}
      {...props}
    />
  );
}

export function MiniButton(props) {
  return (
    <button
      type="button"
      className="num rounded border border-line px-2 py-0.5 text-xs text-muted hover:bg-surface-hover hover:text-fg"
      {...props}
    />
  );
}

export function Toggle({ checked, onChange, label }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2.5 text-sm text-fg">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full border transition-colors ${
          checked ? "border-accent/60 bg-accent/30" : "border-line bg-surface"
        }`}
      >
        <span
          className={`absolute top-0.5 h-3.5 w-3.5 rounded-full transition-all ${
            checked ? "left-[18px] bg-accent" : "left-0.5 bg-faint"
          }`}
        />
      </button>
      {label}
    </label>
  );
}

export function SectionLabel({ children, right }) {
  return (
    <div className="mb-3 flex items-baseline justify-between">
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-faint">{children}</h3>
      {right}
    </div>
  );
}

export function StatRow({ label, value, hint, tone }) {
  const color = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : "text-fg";
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5 text-sm">
      <span className="text-muted" title={hint}>
        {label}
      </span>
      <span className={`num ${color}`}>{value}</span>
    </div>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex rounded-md border border-line bg-surface p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`flex-1 rounded px-2 py-1 text-xs font-medium ${
            value === o.value ? "bg-surface-hover text-fg" : "text-muted hover:text-fg"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Horizontal score bar with a vertical tick at the decision threshold. */
export function ScoreBar({ value, threshold, width = "w-44" }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const over = value >= threshold;
  return (
    <div className={width}>
      <div className="relative h-1.5 rounded-full bg-rust/60" aria-hidden="true">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${over ? "bg-bad" : "bg-good"}`}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute -top-1 h-3.5 w-0.5 rounded bg-fg/80"
          style={{ left: `calc(${threshold * 100}% - 1px)` }}
          title={`threshold ${fmt(threshold)}`}
        />
      </div>
      <div className="num mt-1.5 flex gap-3 text-[11px] text-faint">
        <span>
          <span className={over ? "text-bad" : "text-good"}>■</span> score {fmt(value)}
        </span>
        <span>▏thr {fmt(threshold)}</span>
      </div>
    </div>
  );
}

/** Multi-segment bar (like the reference's green/blue/amber/rust strip). */
export function SegmentBar({ parts, width = "w-44" }) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  return (
    <div className={width}>
      <div className="flex h-1.5 overflow-hidden rounded-full bg-surface-hover" aria-hidden="true">
        {parts.map((p) => (
          <div key={p.label} className={p.color} style={{ width: `${(p.value / total) * 100}%` }} />
        ))}
      </div>
      <div className="num mt-1.5 flex flex-wrap gap-x-3 text-[11px] text-faint">
        {parts.map((p) => (
          <span key={p.label}>
            <span className={p.text}>■</span> {p.caption ?? fmt(p.value)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Card({ selected, onClick, children, className = "" }) {
  const interactive = Boolean(onClick);
  const Tag = interactive ? "button" : "div";
  return (
    <Tag
      type={interactive ? "button" : undefined}
      onClick={onClick}
      aria-pressed={interactive ? Boolean(selected) : undefined}
      className={`print-plain group block w-full rounded-md border bg-surface text-left transition-colors ${
        selected ? "border-line-strong bg-surface-hover" : "border-line"
      } ${interactive ? "hover:bg-surface-hover" : ""} ${className}`}
    >
      {children}
    </Tag>
  );
}

export function EmptyState({ children }) {
  return <p className="text-sm leading-relaxed text-muted">{children}</p>;
}

export function Spinner({ className = "" }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-faint border-t-accent ${className}`}
    />
  );
}

export function ErrorNote({ children }) {
  return (
    <p role="alert" className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">
      {children}
    </p>
  );
}

/** Three-column shell: sidebar | ranked list | evidence panel. Columns scroll independently on desktop. */
export function Shell({ sidebar, main, panel }) {
  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_minmax(380px,520px)]">
      <aside data-noprint className="border-b border-line bg-panel px-5 py-6 lg:overflow-y-auto lg:border-r lg:border-b-0">
        {sidebar}
      </aside>
      <section className="min-w-0 px-4 py-5 sm:px-7 lg:overflow-y-auto">{main}</section>
      <aside className="border-t border-line bg-panel px-5 py-6 lg:overflow-y-auto lg:border-t-0 lg:border-l">{panel}</aside>
    </div>
  );
}
