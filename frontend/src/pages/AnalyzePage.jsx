import { useEffect, useMemo, useRef, useState } from "react";
import { compare, predict } from "../api.js";
import EvidencePanel from "../components/EvidencePanel.jsx";
import ImageDrop from "../components/ImageDrop.jsx";
import LesionProfile from "../components/LesionProfile.jsx";
import { Card, Chip, EmptyState, ErrorNote, ScoreBar, SectionLabel, Shell, Spinner, fmt } from "../components/ui.jsx";

function ModelCard({ rank, result, info, aggregate, sample, selected, onSelect }) {
  const mal = result.class_scores.malignant;
  const isMal = result.predicted_class === "malignant";
  const agrees = sample ? isMal === (sample.label === 1) : null;
  return (
    <Card selected={selected} onClick={onSelect}>
      <div className="grid grid-cols-[36px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 px-5 py-4 md:grid-cols-[36px_minmax(0,1fr)_auto_auto_88px]">
        <span className={`num text-sm ${rank === 1 ? "text-accent" : "text-faint"}`}>{String(rank).padStart(2, "0")}</span>
        <div className="min-w-0">
          <div className="font-semibold">{result.model_name}</div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {info?.val_auroc != null && <Chip mono>val {fmt(info.val_auroc)}</Chip>}
            {aggregate && (
              <Chip mono title="Test AUROC, mean ± SD over 3 seeds">
                test {fmt(aggregate.auroc.mean)} ± {fmt(aggregate.auroc.sd)}
              </Chip>
            )}
            <Chip tone={isMal ? "bad" : "good"}>{result.predicted_class}</Chip>
            {agrees !== null && <Chip tone={agrees ? "good" : "bad"}>{agrees ? "agrees with label" : "disagrees with label"}</Chip>}
          </div>
        </div>
        <div className="col-span-2 md:col-span-1">
          <ScoreBar value={mal} threshold={result.decision_threshold} />
        </div>
        <div className="hidden md:block">
          <span
            aria-hidden="true"
            className="num rounded border border-line px-2 py-0.5 text-xs text-muted opacity-0 transition-opacity group-hover:opacity-100"
          >
            details
          </span>
        </div>
        <div className={`num text-right text-2xl font-semibold ${selected ? "text-accent" : "text-fg"} col-start-2 md:col-start-auto`}>
          {fmt(mal)}
        </div>
      </div>
    </Card>
  );
}

export default function AnalyzePage({ image, setImage, models, modelsError, metrics, gradcam, onScored, openGallery }) {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [cams, setCams] = useState({});
  const reqId = useRef(0);

  // Score every ready model whenever the image changes; ignore responses for stale images.
  useEffect(() => {
    const id = ++reqId.current;
    setResults(null);
    setSelected(null);
    setCams({});
    setError(null);
    if (!image) {
      setLoading(false);
      return;
    }
    setLoading(true);
    compare(image.file)
      .then((r) => {
        if (id !== reqId.current) return;
        setResults(r);
        onScored();
        const top = [...r].sort((a, b) => b.class_scores.malignant - a.class_scores.malignant)[0];
        setSelected(top?.model_id ?? null);
      })
      .catch((e) => id === reqId.current && setError(e.message))
      .finally(() => id === reqId.current && setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image]);

  // Grad-CAM is fetched lazily, only for the selected model.
  useEffect(() => {
    if (!gradcam || !selected || !image || !results || cams[selected]) return;
    const id = reqId.current;
    setCams((c) => ({ ...c, [selected]: { loading: true } }));
    predict(image.file, selected, true)
      .then((data) => id === reqId.current && setCams((c) => ({ ...c, [selected]: { data } })))
      .catch((e) => id === reqId.current && setCams((c) => ({ ...c, [selected]: { error: e.message } })));
  }, [gradcam, selected, image, results, cams]);

  const ranked = useMemo(
    () => (results ? [...results].sort((a, b) => b.class_scores.malignant - a.class_scores.malignant) : []),
    [results],
  );
  const nMal = ranked.filter((r) => r.predicted_class === "malignant").length;
  const infoById = Object.fromEntries((models ?? []).map((m) => [m.id, m]));
  const aggregate = metrics?.summary?.aggregate ?? {};
  const sample = image?.source === "gallery" ? image.sample : null;
  const selectedResult = ranked.find((r) => r.model_id === selected) ?? null;

  const sidebar = (
    <div className="space-y-8">
      <div>
        <SectionLabel>This image</SectionLabel>
        <ImageDrop
          image={image}
          onFile={(file) => setImage({ file, previewUrl: URL.createObjectURL(file), source: "upload" })}
          onPickGallery={openGallery}
        />
      </div>
      <LesionProfile image={image} />
      <div>
        <SectionLabel>Models</SectionLabel>
        {modelsError && <ErrorNote>{modelsError}</ErrorNote>}
        {!models && !modelsError && <EmptyState>Connecting to the API…</EmptyState>}
        <div className="divide-y divide-line/60">
          {(models ?? []).map((m) => (
            <div key={m.id} className="flex items-center justify-between py-1.5 text-sm">
              <span className={m.ready ? "text-muted" : "text-faint line-through"}>{m.name}</span>
              <span className="num text-xs text-faint">{m.ready ? m.checkpoint : "not trained"}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const main = (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-1">
        <SectionLabel>Model comparison</SectionLabel>
        {ranked.length > 0 && (
          <span className="num text-xs text-muted">
            {nMal === 0 || nMal === ranked.length
              ? `${ranked.length} models agree: ${nMal ? "malignant" : "benign"}`
              : `split ${nMal} malignant : ${ranked.length - nMal} benign`}
          </span>
        )}
      </div>
      {!image && (
        <Card>
          <div className="px-5 py-10 text-center">
            <p className="text-sm text-muted">Upload a dermoscopic image or pick one from the test-set gallery.</p>
            <p className="num mt-1 text-xs text-faint">every trained model scores it; ranked by malignant score</p>
          </div>
        </Card>
      )}
      {loading && (
        <p className="flex items-center gap-2 py-4 text-sm text-muted">
          <Spinner /> Scoring with every model…
        </p>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
      {ranked.map((r, i) => (
        <ModelCard
          key={r.model_id}
          rank={i + 1}
          result={r}
          info={infoById[r.model_id]}
          aggregate={aggregate[r.model_id]}
          sample={sample}
          selected={selected === r.model_id}
          onSelect={() => setSelected(r.model_id)}
        />
      ))}
    </div>
  );

  return (
    <Shell
      sidebar={sidebar}
      main={main}
      panel={
        <EvidencePanel
          result={selectedResult}
          image={image}
          cam={cams[selected]}
          gradcamOn={gradcam}
          stratified={metrics?.stratified_test}
        />
      }
    />
  );
}
