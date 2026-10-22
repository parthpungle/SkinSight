import { useEffect, useRef, useState } from "react";
import { fetchSampleFile, getSamples, sampleImageUrl } from "../api.js";
import { Button, Card, Chip, EmptyState, ErrorNote, SectionLabel, Segmented, Shell, Spinner, StatRow, Toggle, fmt } from "../components/ui.jsx";

const PAGE = 50;
const TERTILE_OPTS = [
  { value: "", label: "all" },
  { value: "low", label: "low" },
  { value: "mid", label: "mid" },
  { value: "high", label: "high" },
];

export default function GalleryPage({ onAnalyze }) {
  const [filters, setFilters] = useState({ label: "", dx: "", haziness_tertile: "", colour_tertile: "", disagree: false });
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [opening, setOpening] = useState(false);
  const [openError, setOpenError] = useState(null);
  const reqId = useRef(0);

  useEffect(() => {
    const id = ++reqId.current;
    setLoading(true);
    setError(null);
    getSamples({ ...filters, offset, limit: PAGE })
      .then((d) => id === reqId.current && setData(d))
      .catch((e) => id === reqId.current && setError(e.message))
      .finally(() => id === reqId.current && setLoading(false));
  }, [filters, offset]);

  const update = (patch) => {
    setFilters((f) => ({ ...f, ...patch }));
    setOffset(0);
  };

  async function analyze(item) {
    setOpening(true);
    setOpenError(null);
    try {
      const file = await fetchSampleFile(item.image_id);
      onAnalyze({ file, previewUrl: URL.createObjectURL(file), source: "gallery", sample: item });
    } catch (e) {
      setOpenError(e.message);
    } finally {
      setOpening(false);
    }
  }

  if (data && !data.available)
    return (
      <Shell
        sidebar={null}
        main={<EmptyState>The sample index is not available. Build the manifest and lesion features first.</EmptyState>}
        panel={null}
      />
    );

  const sidebar = (
    <div className="space-y-7">
      <div>
        <SectionLabel>True label</SectionLabel>
        <Segmented
          label="True label"
          value={filters.label}
          onChange={(v) => update({ label: v })}
          options={[
            { value: "", label: "all" },
            { value: "0", label: "benign" },
            { value: "1", label: "malignant" },
          ]}
        />
      </div>
      <div>
        <SectionLabel>Diagnosis</SectionLabel>
        <select
          aria-label="Diagnosis"
          value={filters.dx}
          onChange={(e) => update({ dx: e.target.value })}
          className="num w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm text-fg"
        >
          <option value="">all</option>
          {(data?.dx_values ?? []).map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </div>
      <div>
        <SectionLabel>Border haziness</SectionLabel>
        <Segmented label="Border haziness tertile" value={filters.haziness_tertile} onChange={(v) => update({ haziness_tertile: v })} options={TERTILE_OPTS} />
      </div>
      <div>
        <SectionLabel>Colour inhomogeneity</SectionLabel>
        <Segmented label="Colour tertile" value={filters.colour_tertile} onChange={(v) => update({ colour_tertile: v })} options={TERTILE_OPTS} />
      </div>
      <div>
        <Toggle checked={filters.disagree} onChange={(v) => update({ disagree: v })} label="Only reference-model errors" />
        <p className="mt-2 text-xs text-faint">
          Reference: <span className="num">{data?.reference_run ?? "–"}</span> saved test predictions.
        </p>
      </div>
    </div>
  );

  const total = data?.total ?? 0;
  const main = (
    <div className="space-y-2">
      <SectionLabel
        right={
          <span className="num text-xs text-muted">
            {total ? `${offset + 1}–${Math.min(offset + PAGE, total)} of ${total}` : "0 images"}
          </span>
        }
      >
        Test set
      </SectionLabel>
      {error && <ErrorNote>{error}</ErrorNote>}
      {loading && !data && (
        <p className="flex items-center gap-2 text-sm text-muted">
          <Spinner /> Loading…
        </p>
      )}
      {(data?.items ?? []).map((it) => {
        const wrong = it.ref_pred !== null && it.ref_pred !== it.label;
        return (
          <Card key={it.image_id} selected={selected?.image_id === it.image_id} onClick={() => setSelected(it)}>
            <div className="grid grid-cols-[minmax(0,1fr)_80px] items-center gap-3 px-4 py-2.5">
              <div className="min-w-0">
                <div className="num text-sm font-semibold">{it.image_id}</div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  <Chip tone={it.label === 1 ? "bad" : "good"}>{it.label === 1 ? "malignant" : "benign"}</Chip>
                  <Chip mono>{it.dx}</Chip>
                  {it.haziness_tertile && it.haziness_tertile !== "nan" && <Chip mono>border {it.haziness_tertile}</Chip>}
                  {it.colour_tertile && it.colour_tertile !== "nan" && <Chip mono>colour {it.colour_tertile}</Chip>}
                  {wrong && <Chip tone="warn">reference wrong</Chip>}
                </div>
              </div>
              <div className="num text-right">
                <div className="text-lg font-semibold">{fmt(it.ref_score)}</div>
                <div className="text-[10px] text-faint">ref score</div>
              </div>
            </div>
          </Card>
        );
      })}
      {total > PAGE && (
        <div className="flex items-center justify-between pt-2">
          <Button disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            Previous
          </Button>
          <Button disabled={offset + PAGE >= total || loading} onClick={() => setOffset(offset + PAGE)}>
            Next
          </Button>
        </div>
      )}
    </div>
  );

  const panel = selected ? (
    <div className="space-y-5">
      <div>
        <SectionLabel>Preview</SectionLabel>
        <img src={sampleImageUrl(selected.image_id)} alt={`Dermoscopic image ${selected.image_id}`} className="w-full rounded border border-line" />
      </div>
      <div className="divide-y divide-line/60">
        <StatRow label="Image" value={selected.image_id} />
        <StatRow label="Lesion" value={selected.lesion_id} />
        <StatRow label="True label" value={selected.label === 1 ? "malignant" : "benign"} tone={selected.label === 1 ? "bad" : "good"} />
        <StatRow label="Diagnosis" value={selected.dx} />
        <StatRow label="Border haziness" value={`${fmt(selected.border_haziness, 2)} · ${selected.haziness_tertile ?? "–"}`} />
        <StatRow label="Colour SD (Lab)" value={`${fmt(selected.lab_sd, 1)} · ${selected.colour_tertile ?? "–"}`} />
        <StatRow label="Reference score" value={`${fmt(selected.ref_score)} (thr ${fmt(selected.ref_threshold)})`} />
      </div>
      <Button variant="primary" className="w-full" disabled={opening} onClick={() => analyze(selected)}>
        {opening ? "Loading…" : "Analyze this image"}
      </Button>
      {openError && <ErrorNote>{openError}</ErrorNote>}
    </div>
  ) : (
    <div>
      <SectionLabel>Preview</SectionLabel>
      <EmptyState>Select an image to preview it and send it to Analyze.</EmptyState>
    </div>
  );

  return <Shell sidebar={sidebar} main={main} panel={panel} />;
}
