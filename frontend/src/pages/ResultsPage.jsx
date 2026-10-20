import { useEffect, useState } from "react";
import { figureUrl, getMetrics } from "../api.js";

const KEYS = [
  ["auroc", "AUROC", "Area under the ROC curve; threshold-free ranking quality (0.5 = chance)."],
  ["sensitivity", "Sensitivity", "Share of malignant lesions predicted malignant (TP / (TP + FN))."],
  ["specificity", "Specificity", "Share of benign lesions predicted benign (TN / (TN + FP))."],
  ["balanced_accuracy", "Balanced acc.", "Mean of sensitivity and specificity."],
  ["f1", "F1", "Harmonic mean of precision and sensitivity."],
];
const SCORE_LABELS = {
  border_haziness: "Border haziness (1 / edge strength)",
  lab_sd: "Colour inhomogeneity (Lab SD)",
};

const fmt = (v, d = 3) => (v === null || v === undefined ? "–" : v.toFixed(d));
const ci = (c) => (c && c[0] !== null ? `${fmt(c[0])}–${fmt(c[1])}` : "");

function Card({ title, children, note }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5">
      <h2 className="text-base font-semibold">{title}</h2>
      {note && <p className="mt-1 text-sm text-slate-500">{note}</p>}
      <div className="mt-4 overflow-x-auto">{children}</div>
    </section>
  );
}

export default function ResultsPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetrics().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error)
    return (
      <p role="alert" className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
        {error}
      </p>
    );
  if (!data) return <p className="text-sm text-slate-500">Loading saved results…</p>;
  if (!data.available)
    return (
      <Card title="No results yet">
        <p className="text-sm text-slate-600">
          Evaluation has not been run. Train the models, then run <code>python -m ml.evaluate</code>. This page shows
          only saved evaluation outputs.
        </p>
      </Card>
    );

  const s = data.summary;
  const served = new Set(Object.values(s.served_checkpoints));
  const strat = data.stratified_test;

  return (
    <div className="space-y-6">
      <Card
        title="Test-set performance (mean ± SD across seeds)"
        note={`${s.dataset}, ${s.n_test_images} test images from ${s.n_test_lesions} lesions; positive class = malignant; threshold: ${s.threshold_rule}.`}
      >
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">Seeds</th>
              {KEYS.map(([k, label, def]) => (
                <th key={k} className="py-2 pr-4 font-medium" title={def}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(s.aggregate).map(([variant, agg]) => (
              <tr key={variant} className="border-b last:border-0">
                <td className="py-2 pr-4 font-medium">{variant}</td>
                <td className="py-2 pr-4 tabular-nums">{agg.n_seeds}</td>
                {KEYS.map(([k]) => (
                  <td key={k} className="py-2 pr-4 tabular-nums">
                    {fmt(agg[k].mean)}
                    {agg[k].sd !== null && <span className="text-slate-400"> ± {fmt(agg[k].sd)}</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Served checkpoints (95% lesion-level bootstrap CI)" note={s.ci_method}>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2 pr-4 font-medium">Run</th>
              <th className="py-2 pr-4 font-medium">AUROC</th>
              <th className="py-2 pr-4 font-medium">Sensitivity</th>
              <th className="py-2 pr-4 font-medium">Specificity</th>
              <th className="py-2 pr-4 font-medium">Params</th>
              <th className="py-2 pr-4 font-medium">Latency</th>
            </tr>
          </thead>
          <tbody>
            {Object.values(s.runs)
              .filter((r) => served.has(r.run))
              .map((r) => (
                <tr key={r.run} className="border-b last:border-0">
                  <td className="py-2 pr-4 font-medium">{r.run}</td>
                  {["auroc", "sensitivity", "specificity"].map((k) => (
                    <td key={k} className="py-2 pr-4 tabular-nums">
                      {fmt(r[k])} <span className="text-slate-400">({ci(r.ci95[k])})</span>
                    </td>
                  ))}
                  <td className="py-2 pr-4 tabular-nums">{(r.n_params / 1e6).toFixed(2)}M</td>
                  <td className="py-2 pr-4 tabular-nums" title={r.latency_method}>
                    {fmt(r.latency_ms_median, 1)} ms
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </Card>

      {strat ? (
        Object.entries(strat.scores).map(([score, block]) => (
          <Card
            key={score}
            title={`By ${SCORE_LABELS[score] || score}`}
            note="Tertile cut-points fitted on the training split. Differences show association, not causation."
          >
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-slate-500">
                  <th className="py-2 pr-4 font-medium">Run</th>
                  {["low", "mid", "high"].map((t) => (
                    <th key={t} className="py-2 pr-4 font-medium">
                      {t}: AUROC / sens. (n)
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(block.runs)
                  .filter(([run]) => served.has(run))
                  .map(([run, per]) => (
                    <tr key={run} className="border-b last:border-0">
                      <td className="py-2 pr-4 font-medium">{run}</td>
                      {["low", "mid", "high"].map((t) => (
                        <td key={t} className="py-2 pr-4 tabular-nums">
                          {per[t] ? (
                            <>
                              {fmt(per[t].auroc)} / {fmt(per[t].sensitivity)}{" "}
                              <span className="text-slate-400">({per[t].n})</span>
                            </>
                          ) : (
                            "–"
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
              </tbody>
            </table>
          </Card>
        ))
      ) : (
        <Card title="Border and colour analysis">
          <p className="text-sm text-slate-600">
            Stratified analysis not generated yet (<code>python -m ml.stratified_analysis --split test</code>).
          </p>
        </Card>
      )}

      {data.figures.length > 0 && (
        <Card title="Figures">
          <div className="grid gap-4 sm:grid-cols-2">
            {data.figures.map((f) => (
              <figure key={f}>
                <img src={figureUrl(f)} alt={f.replace(/[_-]/g, " ").replace(".png", "")} className="w-full rounded-md border" />
                <figcaption className="mt-1 text-xs text-slate-500">{f}</figcaption>
              </figure>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
