import { Button, Toggle, fmt } from "./ui.jsx";

function Stat({ value, label }) {
  return (
    <div className="text-center">
      <div className="num text-lg font-semibold leading-tight text-fg">{value}</div>
      <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-faint">{label}</div>
    </div>
  );
}

export default function TopBar({ bestAuroc, scored, gradcam, onGradcam, onClear }) {
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-line bg-panel px-5 py-3 sm:px-8">
      <div className="flex items-center gap-2.5">
        <span className="grid h-7 w-7 place-items-center rounded bg-accent text-xs font-bold text-black">SS</span>
        <span className="text-lg font-semibold tracking-tight">SkinSight</span>
      </div>
      <div className="hidden h-6 w-px bg-line sm:block" />
      <div className="min-w-0">
        <span className="font-semibold">Benign vs malignant dermoscopy</span>{" "}
        <span className="text-sm text-muted">HAM10000 · research prototype</span>
      </div>
      <div className="ml-auto flex flex-wrap items-center gap-x-6 gap-y-3" data-noprint>
        <Stat value={bestAuroc === null ? "–" : fmt(bestAuroc)} label="Best test AUROC" />
        <Stat value={scored} label="Images scored" />
        <Toggle checked={gradcam} onChange={onGradcam} label="Grad-CAM" />
        <Button onClick={() => window.print()}>Export summary</Button>
        <Button onClick={onClear}>Clear</Button>
      </div>
    </header>
  );
}
