import { useState } from "react";
import AnalyzePage from "./pages/AnalyzePage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import Disclaimer from "./components/Disclaimer.jsx";

const TABS = [
  { id: "analyze", label: "Analyze image" },
  { id: "results", label: "Results" },
];

export default function App() {
  const [tab, setTab] = useState("analyze");
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">SkinSight</h1>
            <p className="text-sm text-slate-500">
              Benign vs malignant dermoscopy classification: a research prototype
            </p>
          </div>
          <nav aria-label="Main" className="flex gap-1 rounded-lg bg-slate-100 p-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                  tab === t.id ? "bg-white shadow-sm text-slate-900" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        <Disclaimer />
        {tab === "analyze" ? <AnalyzePage /> : <ResultsPage />}
      </main>
    </div>
  );
}
