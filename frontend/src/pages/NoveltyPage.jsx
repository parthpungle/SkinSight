import { useEffect, useMemo, useState } from "react";
import { getNovelty } from "../api.js";
import { DeltaWhisker, TertileBars } from "../components/charts.jsx";
import { Card, Chip, EmptyState, ErrorNote, SectionLabel, Segmented, Shell, StatRow, fmt } from "../components/ui.jsx";

const SCORES = {
  lab_sd: { label: "Colour inhomogeneity", short: "colour", low: "uniform", high: "varied" },
  border_haziness: { label: "Border haziness", short: "border", low: "clear", high: "hazy" },
};
const TERTILES = ["low", "mid", "high"];
const OUTCOMES = ["TP", "FP", "TN", "FN"];

const gapOf = (per) => (per?.high?.auroc != null && per?.low?.auroc != null ? per.high.auroc - per.low.auroc : null);
const signed = (v) => (v === null || v === undefined ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`);

function EnrichmentCard({ enrichment }) {
  const runs = Object.entries(enrichment ?? {});
  if (!runs.length) return null;
  return (
    <Card>
      <div className="px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-semibold">Where does Grad-CAM look?</h3>
          <span className="num text-[11px] text-faint">median CAM share in lesion ÷ lesion area share · 1 = chance</span>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="num w-full text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wider text-faint">
                <th className="py-1.5 pr-4 font-medium">run</th>
                {OUTCOMES.map((o) => (
                  <th key={o} className="py-1.5 pr-4 text-right font-medium">
                    {o}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map(([run, e]) => (
                <tr key={run} className="border-t border-line/60">
                  <td className="py-1.5 pr-4 text-muted">{run}</td>
                  {OUTCOMES.map((o) => {
                    const v = e[o]?.median;
                    return (
                      <td key={o} className={`py-1.5 pr-4 text-right ${o === "FP" && v !== null && v < 1 ? "text-bad" : "text-fg"}`}>
                        {fmt(v, 2)} <span className="text-[10px] text-faint">n={e[o]?.n}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-faint">
          Red: false positives whose evidence sits mostly <em>outside</em> the lesion. Grad-CAM is qualitative, so this is
          a lead to investigate, not proof of a shortcut.
        </p>
      </div>
    </Card>
  );
}

function FixComparison({ comparison }) {
  if (!comparison) return null;
  const rows = [...TERTILES, "all"];
  return (
    <div>
      <SectionLabel right={<span className="num text-[11px] text-faint">fix − baseline, paired</span>}>
        Colour-constancy fix
      </SectionLabel>
      <p className="mb-3 text-xs text-muted">
        <span className="num">{comparison.base}</span> → <span className="num">{comparison.fix}</span>, per seed. Axis
        ±0.20; the vertical line is zero.
      </p>
      {["auroc", "sensitivity"].map((key) => (
        <div key={key} className="mb-4">
          <div className="num mb-1.5 text-[11px] uppercase tracking-wider text-faint">Δ {key}</div>
          <div className="space-y-1">
            {rows.map((t) =>
              Object.entries(comparison.by_tertile[t]?.[key] ?? {}).map(([seed, d]) => {
                const inc = d.ci95[0] <= 0 && d.ci95[1] >= 0;
                return (
                  <div key={`${t}-${seed}`} className="grid grid-cols-[64px_minmax(0,1fr)_120px] items-center gap-3">
                    <span className="num text-[11px] text-faint">
                      {t} {seed}
                    </span>
                    <DeltaWhisker delta={d.delta} ci={d.ci95} />
                    <span className={`num text-right text-[11px] ${inc ? "text-muted" : d.delta < 0 ? "text-bad" : "text-good"}`}>
                      {signed(d.delta)} {inc ? "· CI∋0" : ""}
                    </span>
                  </div>
                );
              }),
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function NoveltyPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [score, setScore] = useState("lab_sd");
  const [split, setSplit] = useState("test");
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    getNovelty().then(setData).catch((e) => setError(e.message));
  }, []);

  const strat = data?.available ? (split === "test" ? data.stratified_test : data.stratified_val) : null;
  const block = strat?.scores?.[score];
  const cards = useMemo(() => {
    if (!block) return [];
    return Object.entries(block.runs)
      .map(([run, per]) => ({ run, per, gap: gapOf(per) }))
      .sort((a, b) => (a.gap ?? 0) - (b.gap ?? 0));
  }, [block]);

  if (error) return <Shell sidebar={null} main={<ErrorNote>{error}</ErrorNote>} panel={null} />;
  if (!data) return <Shell sidebar={null} main={<EmptyState>Loading saved analysis…</EmptyState>} panel={null} />;
  if (!data.available)
    return (
      <Shell
        sidebar={null}
        main={
          <EmptyState>
            No novelty analysis yet. Run <code className="num">python -m ml.lesion_features</code>, then{" "}
            <code className="num">python -m ml.stratified_analysis --split test</code>.
          </EmptyState>
        }
        panel={null}
      />
    );

  const meta = SCORES[score];
  const labels = { low: meta.low, mid: "mid", high: meta.high };
  const current = cards.find((c) => c.run === selected) ?? cards[0];
  const variant = current?.run.replace(/_s\d+$/, "");
  const comparison = strat?.comparisons?.[score];
  const showFix = comparison && (variant === comparison.base || variant === comparison.fix);

  const sidebar = (
    <div className="space-y-8">
      <div>
        <SectionLabel>Score</SectionLabel>
        <Segmented
          label="Lesion score"
          value={score}
          onChange={setScore}
          options={Object.entries(SCORES).map(([v, m]) => ({ value: v, label: m.label }))}
        />
        <div className="mt-2">
          <Segmented
            label="Split"
            value={split}
            onChange={setSplit}
            options={[
              { value: "test", label: "Test" },
              ...(data.stratified_val ? [{ value: "val", label: "Validation" }] : []),
            ]}
          />
        </div>
      </div>
      <div>
        <SectionLabel>Hypotheses</SectionLabel>
        <div className="space-y-3">
          {data.verdicts.map((v) => (
            <div key={v.id}>
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className="text-muted">{v.label}</span>
                <Chip tone={v.tone}>{v.verdict}</Chip>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-faint">{v.summary}</p>
            </div>
          ))}
          {data.verdicts.length === 0 && <EmptyState>No verdicts recorded.</EmptyState>}
        </div>
      </div>
      <div>
        <SectionLabel>Cut-points (train)</SectionLabel>
        <StatRow label={`${meta.low} | mid`} value={fmt(block?.cutpoints?.[0])} />
        <StatRow label={`mid | ${meta.high}`} value={fmt(block?.cutpoints?.[1])} />
      </div>
      <div>
        <SectionLabel>Method</SectionLabel>
        <p className="text-sm leading-relaxed text-muted">
          Tertiles fitted on the training split. Lesion-level bootstrap, 1,000 resamples. Differences show association,
          not causation.
        </p>
      </div>
    </div>
  );

  const main = (
    <div className="space-y-3">
      <SectionLabel right={<span className="num text-xs text-muted">ranked by AUROC gap ({meta.high} − {meta.low})</span>}>
        {meta.label} · {split}
      </SectionLabel>
      <EnrichmentCard enrichment={data.enrichment?.[split]} />
      {cards.map((c, i) => (
        <Card key={c.run} selected={current?.run === c.run} onClick={() => setSelected(c.run)}>
          <div className="grid grid-cols-[36px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 px-5 py-4 md:grid-cols-[36px_minmax(0,1fr)_96px]">
            <span className={`num text-sm ${i === 0 ? "text-accent" : "text-faint"}`}>{String(i + 1).padStart(2, "0")}</span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="num font-semibold">{c.run}</span>
                {/_cc_s\d+$/.test(c.run) && <Chip tone="warn">fix</Chip>}
              </div>
              <div className="mt-3">
                <TertileBars per={c.per} labels={labels} />
              </div>
            </div>
            <div className="num col-start-2 text-right md:col-start-auto">
              <div className={`text-2xl font-semibold ${c.gap !== null && c.gap < 0 ? "text-bad" : "text-good"}`}>{signed(c.gap)}</div>
              <div className="text-[11px] text-faint">AUROC gap</div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );

  const per = current?.per;
  const panel = current && (
    <div className="space-y-6">
      <div>
        <SectionLabel>Selected run</SectionLabel>
        <h2 className="num text-lg font-semibold">{current.run}</h2>
        <p className="mt-0.5 text-xs text-faint">
          {meta.label} tertiles · {split} split
        </p>
      </div>
      <div>
        <SectionLabel>Per tertile</SectionLabel>
        <table className="num w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wider text-faint">
              <th className="py-1.5 font-medium">tertile</th>
              <th className="py-1.5 text-right font-medium">sens</th>
              <th className="py-1.5 text-right font-medium">spec</th>
              <th className="py-1.5 text-right font-medium" title="Grad-CAM enrichment (served checkpoints only)">
                cam
              </th>
            </tr>
          </thead>
          <tbody>
            {TERTILES.map((t) =>
              per?.[t] ? (
                <tr key={t} className="border-t border-line/60">
                  <td className="py-1.5 text-muted">{labels[t]}</td>
                  <td className="py-1.5 text-right">{fmt(per[t].sensitivity, 2)}</td>
                  <td className="py-1.5 text-right">{fmt(per[t].specificity, 2)}</td>
                  <td className="py-1.5 text-right">{fmt(per[t].cam_enrichment_median, 2)}</td>
                </tr>
              ) : null,
            )}
          </tbody>
        </table>
      </div>
      {showFix ? (
        <FixComparison comparison={comparison} />
      ) : (
        comparison && (
          <p className="text-xs text-faint">
            Select an EfficientNet-B0 run to see the paired colour-constancy comparison.
          </p>
        )
      )}
    </div>
  );

  return <Shell sidebar={sidebar} main={main} panel={panel} />;
}
