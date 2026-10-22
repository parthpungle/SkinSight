import { useCallback, useEffect, useMemo, useState } from "react";
import { getMetrics, getModels, getSamples, withRetry } from "./api.js";
import TopBar from "./components/TopBar.jsx";
import Disclaimer from "./components/Disclaimer.jsx";
import AnalyzePage from "./pages/AnalyzePage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import NoveltyPage from "./pages/NoveltyPage.jsx";
import GalleryPage from "./pages/GalleryPage.jsx";

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [models, setModels] = useState(null);
  const [modelsError, setModelsError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState(null);
  const [galleryTotal, setGalleryTotal] = useState(null);
  const [gradcam, setGradcam] = useState(true);
  const [scored, setScored] = useState(0);
  // { file, previewUrl, source: "upload" | "gallery", sample? }
  const [image, setImageState] = useState(null);

  useEffect(() => {
    getModels().then(setModels).catch((e) => setModelsError(e.message));
    getMetrics().then(setMetrics).catch((e) => setMetricsError(e.message));
    withRetry(() => getSamples({ limit: 1 }))
      .then((b) => setGalleryTotal(b.available ? b.total : 0))
      .catch(() => setGalleryTotal(0));
  }, []);

  const setImage = useCallback((next) => {
    setImageState((prev) => {
      if (prev?.previewUrl) URL.revokeObjectURL(prev.previewUrl);
      return next;
    });
  }, []);

  const bestAuroc = useMemo(() => {
    const agg = metrics?.summary?.aggregate;
    if (!agg) return null;
    return Math.max(...Object.values(agg).map((a) => a.auroc.mean));
  }, [metrics]);

  const tabs = [
    { id: "analyze", label: "Analyze" },
    { id: "results", label: "Results", count: metrics?.summary ? Object.keys(metrics.summary.aggregate).length : null },
    { id: "novelty", label: "Novelty", count: 3 },
    { id: "gallery", label: "Gallery", count: galleryTotal || null },
  ];

  return (
    <div className="flex min-h-screen flex-col lg:h-screen">
      <TopBar
        bestAuroc={bestAuroc}
        scored={scored}
        gradcam={gradcam}
        onGradcam={setGradcam}
        onClear={() => {
          setImage(null);
          setScored(0);
          setTab("analyze");
        }}
      />
      <nav aria-label="Main" className="flex gap-1 overflow-x-auto border-b border-line bg-panel px-3 sm:px-6" data-noprint>
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            aria-current={tab === t.id ? "page" : undefined}
            className={`-mb-px whitespace-nowrap border-b-2 px-3.5 py-3 text-[15px] font-medium transition-colors ${
              tab === t.id ? "border-accent text-accent" : "border-transparent text-muted hover:text-fg"
            }`}
          >
            {t.label}
            {t.count ? <span className="num ml-1.5 text-xs text-faint">{t.count}</span> : null}
          </button>
        ))}
      </nav>
      <Disclaimer />

      {tab === "analyze" && (
        <AnalyzePage
          image={image}
          setImage={setImage}
          models={models}
          modelsError={modelsError}
          metrics={metrics}
          gradcam={gradcam}
          onScored={() => setScored((n) => n + 1)}
          openGallery={() => setTab("gallery")}
        />
      )}
      {tab === "results" && <ResultsPage metrics={metrics} error={metricsError} />}
      {tab === "novelty" && <NoveltyPage />}
      {tab === "gallery" && (
        <GalleryPage
          onAnalyze={(next) => {
            setImage(next);
            setTab("analyze");
          }}
        />
      )}
    </div>
  );
}
