import { useMemo, useState } from "react";
import { figureUrl } from "../api.js";
import { TertileBars } from "../components/charts.jsx";
import { Card, Chip, EmptyState, ErrorNote, SectionLabel, Segmented, Shell, StatRow, fmt } from "../components/ui.jsx";

const NAMES = {
  small_cnn: "Small CNN (paper-style)",
  efficientnet_b0: "EfficientNet-B0",
  mobilenet_v2: "MobileNetV2",
  efficientnet_b0_cc: "EfficientNet-B0 + colour constancy",
};
const METRICS = [
  ["sensitivity", "sens", "Malignant lesions predicted malignant: TP / (TP + FN)"],
  ["specificity", "spec", "Benign lesions predicted benign: TN / (TN + FP)"],
  ["balanced_accuracy", "bal", "Mean of sensitivity and specificity"],
];

function MetricBars({ values }) {
  return (
    <div className="w-48 space-y-1">
      {METRICS.map(([k, short, def]) => (
        <div key={k} className="grid grid-cols-[30px_minmax(0,1fr)_40px] items-center gap-2" title={def}>
          <span className="num text-[10px] text-faint">{short}</span>
          <div className="h-1.5 rounded-full bg-surface-hover" aria-hidden="true">
            <div className="h-1.5 rounded-full bg-good/80" style={{ width: `${(values[k] ?? 0) * 100}%` }} />
          </div>
          <span className="num text-right text-[11px] text-muted">{fmt(values[k])}</span>
        </div>
      ))}
    </div>
  );
}

export default function ResultsPage({ metrics, error }) {
  const [view, setView] = useState("mean");
  const [selected, setSelected] = useState(null);
  const s = metrics?.summary;

  const rows = useMemo(() => {
    if (!s) return [];
    return Object.entries(s.aggregate)
      .map(([variant, agg]) => {
        const run = s.runs[s.served_checkpoints[variant]];
        const values =
          view === "mean"
            ? Object.fromEntries(["auroc", ...METRICS.map((m) => m[0])].map((k) => [k, agg[k].mean]))
            : run;
        return { variant, agg, run, values };
      })
      .sort((a, b) => b.values.auroc - a.values.auroc);
  }, [s, view]);

  if (error) return <Shell sidebar={null} main={<ErrorNote>{error}</ErrorNote>} panel={null} />;
  if (!metrics) return <Shell sidebar={null} main={<EmptyState>Loading saved results…</EmptyState>} panel={null} />;
  if (!metrics.available)
    return (
      <Shell
        sidebar={null}
        main={
          <EmptyState>
            No evaluation results yet. Train the models, then run <code className="num">python -m ml.evaluate</code>.
            This page only shows saved outputs.
          </EmptyState>
        }
        panel={null}
      />
    );

  const anyRun = Object.values(s.runs)[0];
  const current = rows.find((r) => r.variant === selected) ?? rows[0];
  const strat = metrics.stratified_test?.scores;

  const sidebar = (
    <div className="space-y-8">
      <div>
        <SectionLabel>View</SectionLabel>
        <Segmented
          label="Metric view"
          value={view}
          onChange={setView}
          options={[
            { value: "mean", label: "Seed mean" },
            { value: "served", label: "Served checkpoint" },
          ]}
        />
        <p className="mt-2 text-xs text-faint">
          {view === "mean" ? "Mean ± SD over 3 training seeds." : "The seed with the best validation AUROC; this is the one the app serves."}
        </p>
      </div>
      <div>
        <SectionLabel>Test set</SectionLabel>
        <StatRow label="Dataset" value={s.dataset} />
        <StatRow label="Images" value={s.n_test_images.toLocaleString()} />
        <StatRow label="Lesions" value={s.n_test_lesions.toLocaleString()} />
        <StatRow label="Malignant" value={anyRun.n_pos} />
        <StatRow label="Split seed" value={s.split_seed} />
      </div>
      <div>
        <SectionLabel>Method</SectionLabel>
        <p className="text-sm leading-relaxed text-muted">
          Positive class = malignant. Threshold: {s.threshold_rule}. CIs: {s.ci_method}. Scores are{" "}
          <span className="num text-xs">{s.score_semantics}</span>.
        </p>
      </div>
    </div>
  );

  const main = (
    <div className="space-y-3">
      <SectionLabel right={<span className="num text-xs text-muted">ranked by test AUROC</span>}>Models</SectionLabel>
      {rows.map((r, i) => (
        <Card key={r.variant} selected={current?.variant === r.variant} onClick={() => setSelected(r.variant)}>
          <div className="grid grid-cols-[36px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 px-5 py-4 md:grid-cols-[36px_minmax(0,1fr)_auto_96px]">
            <span className={`num text-sm ${i === 0 ? "text-accent" : "text-faint"}`}>{String(i + 1).padStart(2, "0")}</span>
            <div className="min-w-0">
              <div className="font-semibold">{NAMES[r.variant] ?? r.variant}</div>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                <Chip mono>{(r.run.n_params / 1e6).toFixed(2)}M params</Chip>
                <Chip mono>{r.run.artifact_mb} MB</Chip>
                <Chip mono title={r.run.latency_method}>{fmt(r.run.latency_ms_median, 1)} ms</Chip>
                {r.variant.endsWith("_cc") && <Chip tone="warn">novelty fix</Chip>}
              </div>
            </div>
            <div className="col-span-2 md:col-span-1">
              <MetricBars values={r.values} />
            </div>
            <div className="num col-start-2 text-right md:col-start-auto">
              <div className={`text-2xl font-semibold ${current?.variant === r.variant ? "text-accent" : ""}`}>{fmt(r.values.auroc)}</div>
              <div className="text-[11px] text-faint">{view === "mean" ? `± ${fmt(r.agg.auroc.sd)}` : "AUROC"}</div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );

  const run = current?.run;
  const panel = run && (
    <div className="space-y-6">
      <div>
        <SectionLabel>Served checkpoint</SectionLabel>
        <h2 className="text-lg font-semibold">{NAMES[current.variant] ?? current.variant}</h2>
        <p className="num mt-0.5 text-xs text-faint">
          {run.run} · epoch {run.best_epoch} · thr {fmt(run.threshold)}
        </p>
      </div>
      <div>
        <SectionLabel>Test metrics · 95% CI</SectionLabel>
        <div className="divide-y divide-line/60">
          {["auroc", "sensitivity", "specificity", "balanced_accuracy"].map((k) => (
            <StatRow
              key={k}
              label={k.replace("_", " ")}
              value={`${fmt(run[k])}  [${fmt(run.ci95[k][0])}–${fmt(run.ci95[k][1])}]`}
            />
          ))}
          <StatRow label="F1" value={fmt(run.f1)} />
          <StatRow label="TP / FN / FP / TN" value={`${run.tp} / ${run.fn} / ${run.fp} / ${run.tn}`} />
        </div>
      </div>
      <div>
        <SectionLabel>Confusion matrix</SectionLabel>
        <img
          src={figureUrl(`confusion_${run.run}.png`)}
          alt={`Confusion matrix for ${run.run}`}
          className="w-full max-w-sm rounded border border-line bg-white"
        />
      </div>
      {strat &&
        Object.entries(strat).map(([score, block]) =>
          block.runs[run.run] ? (
            <div key={score}>
              <SectionLabel>{score === "lab_sd" ? "By colour inhomogeneity" : "By border haziness"}</SectionLabel>
              <TertileBars per={block.runs[run.run]} />
            </div>
          ) : null,
        )}
    </div>
  );

  return <Shell sidebar={sidebar} main={main} panel={panel} />;
}
