import { useEffect, useRef, useState } from "react";
import { getModels, predict } from "../api.js";

const MAX_MB = 10;
const ACCEPTED = ["image/jpeg", "image/png"];

function Spinner() {
  return (
    <span
      className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
      aria-hidden="true"
    />
  );
}

function ErrorMessage({ children }) {
  return (
    <p role="alert" className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
      {children}
    </p>
  );
}

function ScoreBar({ label, value, emphasis }) {
  const pct = Math.round(value * 1000) / 10;
  return (
    <div>
      <div className="flex justify-between text-sm">
        <span className={emphasis ? "font-semibold" : ""}>{label}</span>
        <span className="tabular-nums">{value.toFixed(3)}</span>
      </div>
      <div className="mt-1 h-2 rounded-full bg-slate-200" aria-hidden="true">
        <div className={`h-2 rounded-full ${emphasis ? "bg-teal-600" : "bg-slate-400"}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function AnalyzePage() {
  const [models, setModels] = useState([]);
  const [modelsError, setModelsError] = useState(null);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    getModels()
      .then((ms) => {
        setModels(ms);
        const first = ms.find((m) => m.ready);
        if (first) setModelId(first.id);
      })
      .catch((e) => setModelsError(e.message));
  }, []);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function chooseFile(f) {
    setResult(null);
    setError(null);
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      setFileError("Please choose a JPEG or PNG image.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFileError(`Image is larger than ${MAX_MB} MB.`);
      return;
    }
    setFileError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function analyze() {
    if (!file || !modelId || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult({ ...(await predict(file, modelId, true)), fileName: file.name });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const readyModels = models.filter((m) => m.ready);

  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-5" aria-labelledby="input-heading">
        <h2 id="input-heading" className="text-base font-semibold">
          1. Choose an image and model
        </h2>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            chooseFile(e.dataTransfer.files?.[0]);
          }}
          className={`flex min-h-48 flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-4 text-center ${
            dragOver ? "border-teal-500 bg-teal-50" : "border-slate-300"
          }`}
        >
          {preview ? (
            <img src={preview} alt="Selected dermoscopic image" className="max-h-56 rounded-md object-contain" />
          ) : (
            <p className="text-sm text-slate-500">Drag a dermoscopic image here (JPEG or PNG, up to {MAX_MB} MB)</p>
          )}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50"
          >
            {preview ? "Choose a different image" : "Browse files"}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="sr-only"
            aria-label="Upload image"
            onChange={(e) => chooseFile(e.target.files?.[0])}
          />
        </div>
        {fileError && <ErrorMessage>{fileError}</ErrorMessage>}

        <div>
          <label htmlFor="model" className="mb-1 block text-sm font-medium">
            Model
          </label>
          {modelsError ? (
            <ErrorMessage>{modelsError}</ErrorMessage>
          ) : (
            <select
              id="model"
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value);
                setResult(null);
              }}
              className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              {models.length === 0 && <option value="">Loading models…</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.ready}>
                  {m.name}
                  {m.ready ? "" : " (not trained yet)"}
                </option>
              ))}
            </select>
          )}
          {models.length > 0 && readyModels.length === 0 && (
            <p className="mt-2 text-sm text-slate-500">No trained checkpoints yet. Train a model first.</p>
          )}
        </div>

        <button
          type="button"
          onClick={analyze}
          disabled={!file || !modelId || loading}
          className="flex w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading && <Spinner />}
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </section>

      <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-5" aria-labelledby="result-heading" aria-live="polite">
        <h2 id="result-heading" className="text-base font-semibold">
          2. Model output
        </h2>
        {error && <ErrorMessage>{error}</ErrorMessage>}
        {!result && !error && (
          <p className="text-sm text-slate-500">
            {loading ? "Running the model…" : "Results will appear here after you analyze an image."}
          </p>
        )}
        {result && (
          <>
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500">Model prediction</p>
              <p className="text-2xl font-semibold capitalize">{result.predicted_class}</p>
              <p className="mt-1 text-xs text-slate-500">
                {result.model_name} · {result.checkpoint} · {result.fileName}
              </p>
            </div>
            <div className="space-y-3">
              {Object.entries(result.class_scores).map(([cls, v]) => (
                <ScoreBar key={cls} label={cls} value={v} emphasis={cls === result.predicted_class} />
              ))}
            </div>
            <p className="text-xs text-slate-500">
              Uncalibrated model scores, <em>not</em> probabilities of disease. The prediction is "malignant" when the
              malignant score ≥ {result.decision_threshold.toFixed(3)} (threshold chosen on the validation set).
            </p>

            <div>
              <h3 className="mb-2 text-sm font-semibold">Grad-CAM</h3>
              {result.explanation_available ? (
                <div className="grid grid-cols-2 gap-3">
                  <figure>
                    <img src={preview} alt="Original image" className="aspect-square w-full rounded-md object-fill" />
                    <figcaption className="mt-1 text-xs text-slate-500">Original (resized as the model sees it)</figcaption>
                  </figure>
                  <figure>
                    <img
                      src={`data:image/png;base64,${result.gradcam_png_base64}`}
                      alt={`Grad-CAM overlay for the ${result.predicted_class} prediction`}
                      className="aspect-square w-full rounded-md"
                    />
                    <figcaption className="mt-1 text-xs text-slate-500">
                      Grad-CAM for "{result.predicted_class}" (red = higher)
                    </figcaption>
                  </figure>
                </div>
              ) : (
                <p className="text-sm text-slate-500">{result.explanation_message || "Explanation unavailable."}</p>
              )}
              <p className="mt-2 text-xs text-slate-500">
                Grad-CAM is a qualitative visualisation of image regions that influenced the score. It is not a
                segmentation and not a medical explanation.
              </p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
