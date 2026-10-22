import { EmptyState, ErrorNote, SectionLabel, Spinner, StatRow, fmt } from "./ui.jsx";

const SCORE_TEXT = { lab_sd: "highly inhomogeneous", border_haziness: "very hazy-bordered" };
const SCORE_LOW = { lab_sd: "uniform", border_haziness: "clear-bordered" };

function weakSpots(stratified, run) {
  if (!stratified?.scores || !run) return [];
  const notes = [];
  for (const [score, block] of Object.entries(stratified.scores)) {
    const per = block.runs?.[run];
    if (!per?.low || !per?.high || per.low.auroc === null || per.high.auroc === null) continue;
    notes.push({
      score,
      text: `On ${SCORE_TEXT[score] ?? score} lesions this checkpoint's test AUROC is ${fmt(per.high.auroc)}, vs ${fmt(
        per.low.auroc,
      )} on ${SCORE_LOW[score] ?? "low-score"} ones.`,
      worse: per.high.auroc < per.low.auroc,
    });
  }
  return notes;
}

export default function EvidencePanel({ result, image, cam, gradcamOn, stratified }) {
  if (!result) {
    return (
      <div>
        <SectionLabel>Evidence</SectionLabel>
        <EmptyState>Select a model to see the evidence behind its score.</EmptyState>
      </div>
    );
  }
  const notes = weakSpots(stratified, result.checkpoint);
  return (
    <div className="space-y-6">
      <div>
        <SectionLabel>Evidence</SectionLabel>
        <h2 className="text-lg font-semibold">{result.model_name}</h2>
        <p className="num mt-0.5 text-xs text-faint">checkpoint {result.checkpoint}</p>
        <p className="mt-3 text-sm text-muted">
          Predicts <span className="text-fg">malignant</span> when the malignant score is{" "}
          <span className="num text-fg">≥ {fmt(result.decision_threshold)}</span> (threshold chosen on the validation
          set, Youden J).
        </p>
      </div>

      <div>
        <SectionLabel>Model scores</SectionLabel>
        <div className="divide-y divide-line/60">
          {Object.entries(result.class_scores).map(([cls, v]) => (
            <StatRow key={cls} label={cls} value={fmt(v, 4)} tone={cls === result.predicted_class ? (cls === "malignant" ? "bad" : "good") : undefined} />
          ))}
        </div>
        <p className="mt-2 text-xs text-faint">Uncalibrated model scores, not probabilities of disease.</p>
      </div>

      <div>
        <SectionLabel right={<span className="num text-[11px] text-faint">target: {result.predicted_class}</span>}>
          Grad-CAM
        </SectionLabel>
        {!gradcamOn && <EmptyState>Grad-CAM is switched off in the top bar.</EmptyState>}
        {gradcamOn && cam?.loading && (
          <p className="flex items-center gap-2 text-sm text-muted">
            <Spinner /> Computing Grad-CAM…
          </p>
        )}
        {gradcamOn && cam?.error && <ErrorNote>{cam.error}</ErrorNote>}
        {gradcamOn && cam?.data && (
          cam.data.explanation_available ? (
            <div className="grid grid-cols-2 gap-3">
              <figure>
                <img src={image.previewUrl} alt="Original image" className="aspect-square w-full rounded border border-line object-fill" />
                <figcaption className="num mt-1.5 text-[11px] text-faint">original (resized as the model sees it)</figcaption>
              </figure>
              <figure>
                <img
                  src={`data:image/png;base64,${cam.data.gradcam_png_base64}`}
                  alt={`Grad-CAM overlay for the ${result.predicted_class} prediction`}
                  className="aspect-square w-full rounded border border-line"
                />
                <figcaption className="num mt-1.5 text-[11px] text-faint">jet colormap · red = higher</figcaption>
              </figure>
            </div>
          ) : (
            <EmptyState>{cam.data.explanation_message || "Explanation unavailable."}</EmptyState>
          )
        )}
        <p className="mt-2 text-xs text-faint">
          A qualitative visualisation of regions that influenced the score. Not a segmentation and not a medical
          explanation.
        </p>
      </div>

      {notes.length > 0 && (
        <div>
          <SectionLabel>Known weak spots (test set)</SectionLabel>
          <ul className="space-y-2">
            {notes.map((n) => (
              <li key={n.score} className={`rounded-md border px-3 py-2 text-sm ${n.worse ? "border-warn/30 bg-warn/5 text-fg" : "border-line text-muted"}`}>
                {n.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
