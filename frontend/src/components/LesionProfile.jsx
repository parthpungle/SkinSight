import { EmptyState, SectionLabel, StatRow, fmt } from "./ui.jsx";

const DX_NAMES = {
  mel: "melanoma",
  bcc: "basal cell carcinoma",
  akiec: "actinic keratosis / IEC",
  nv: "melanocytic nevus",
  bkl: "benign keratosis",
  df: "dermatofibroma",
  vasc: "vascular lesion",
};

export default function LesionProfile({ image }) {
  const s = image?.source === "gallery" ? image.sample : null;
  return (
    <div>
      <SectionLabel>Lesion profile</SectionLabel>
      {!image && <EmptyState>Choose an image to see its profile.</EmptyState>}
      {image && !s && (
        <EmptyState>Border and colour scores need a lesion mask, so they are only available for gallery images.</EmptyState>
      )}
      {s && (
        <div className="divide-y divide-line/60">
          <StatRow
            label="True label"
            value={s.label === 1 ? "malignant" : "benign"}
            tone={s.label === 1 ? "bad" : "good"}
          />
          <StatRow label="Diagnosis" value={s.dx} hint={DX_NAMES[s.dx]} />
          <StatRow
            label="Border haziness"
            value={`${fmt(s.border_haziness, 2)} · ${s.haziness_tertile ?? "–"}`}
            hint="1 / edge strength at the mask contour (higher = more indistinct)"
          />
          <StatRow
            label="Colour SD (Lab)"
            value={`${fmt(s.lab_sd, 1)} · ${s.colour_tertile ?? "–"}`}
            hint="Colour standard deviation inside the lesion (higher = more inhomogeneous)"
          />
          <StatRow label="Lesion/skin contrast" value={fmt(s.contrast_L, 1)} hint="|L* inside − L* outside|" />
          <StatRow label="Hair fraction" value={`${fmt((s.hair_frac ?? 0) * 100, 1)}%`} />
        </div>
      )}
    </div>
  );
}
